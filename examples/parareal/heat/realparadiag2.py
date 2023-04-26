
import firedrake as fd
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

from utils.serial import SerialMiniApp
import asQ

global_comm = fd.COMM_WORLD

T = 0.4
nt = 512
nx = 32
cfl = 1.
theta = 1.0
nu = 1.
alpha = 1e-4

dx = 1./nx
dtf = T/nt

cfl = nu*dtf/(dx*dx)
Print(cfl, dtf)
verbose = False
serial = False
tol = 1e-16

ntf = 16
ntc = nt//ntf
nits = ntc

assert (nt % ntc) == 0

dtc = ntf*dtf

assert ntf == global_comm.size

time_partition = tuple(1 for _ in range(ntf))
ensemble = asQ.create_ensemble(time_partition, global_comm)
rank = ensemble.ensemble_comm.rank
is_root = (rank == 0)

mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=True, comm=ensemble.comm)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

Nu = fd.Constant(nu)


def form_mass(u, v):
    return fd.inner(u, v)*fd.dx


def form_function(u, v):
    return Nu*fd.inner(fd.grad(u), fd.grad(v))*fd.dx


initial_expr = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)
uinitial = fd.Function(V).interpolate(initial_expr)

serial_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-12,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-12,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

block_sparams = {
    'ksp': {
        'rtol': 1e-12,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

pdg_sparams = {
    'snes_type': 'ksponly',
    'snes': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-12,
    },
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-12,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_block': block_sparams
}


if verbose and is_root:
    serial_sparams['snes']['monitor'] = None
    pdg_sparams['snes']['monitor'] = None
    pdg_sparams['snes']['converged_reason'] = None
    pdg_sparams['ksp']['monitor'] = None
    pdg_sparams['ksp']['converged_reason'] = None

miniapp = SerialMiniApp(dtc, theta,
                        uinitial,
                        form_mass, form_function,
                        serial_sparams)

pdg = asQ.paradiag(ensemble=ensemble,
                   form_mass=form_mass,
                   form_function=form_function,
                   w0=uinitial, dt=dtf, theta=theta,
                   alpha=alpha, circ=None,
                   time_partition=time_partition,
                   solver_parameters=pdg_sparams)

# ## define F and G


def G(u, uout, **kwargs):
    miniapp.dt.assign(dtc)
    miniapp.solve(1, ics=u, **kwargs)
    uout.assign(miniapp.w0)


def F(u, uout, serial=serial, **kwargs):
    if serial:
        miniapp.dt.assign(dtf)
        miniapp.solve(ntf, ics=u, **kwargs)
        uout.assign(miniapp.w0)
    else:
        pdg.aaos.next_window(u)
        pdg.solve(nwindows=1)

        end_rank = ntf-1
        pdg.aaos.get_field(-1, wout=uout)
        ensemble.bcast(uout, root=end_rank)


Print('### === --- Timestepping loop --- === ###')
linear_its = 0
nonlinear_its = 0

ofile = fd.File("output/heat.pvd")
ofile.write(miniapp.w0, time=0)


def preproc(app, step, t):
    if verbose:
        Print('')
        Print(f'=== --- Timestep {step} --- ===')
        Print('')


def postproc(app, step, t):
    global linear_its
    global nonlinear_its

    linear_its += app.nlsolver.snes.getLinearSolveIterations()
    nonlinear_its += app.nlsolver.snes.getIterationNumber()

    ofile.write(app.w0, time=t)


def coarse_series():
    return [fd.Function(V) for _ in range(ntc+1)]


def copy_series(dst, src):
    for d, s in zip(dst, src):
        d.assign(s)


def series_norm(series):
    norm = 0.
    for s in series:
        norm += fd.norm(s)
    return norm


def series_error(exact, series):
    norm = 0.
    for e, s in zip(exact, series):
        norm += fd.errornorm(e, s)
    return norm


# ## find "exact" fine solution at coarse points in serial
userial = coarse_series()
userial[0].assign(uinitial)

for i in range(ntc):
    F(userial[i], userial[i+1], serial=True)

uparallel = coarse_series()
uparallel[0].assign(uinitial)

for i in range(ntc):
    F(uparallel[i], uparallel[i+1], serial=False)

err = series_error(userial, uparallel)
Print(rank, err)

# from sys import exit
# exit()

# ## initialise coarse points

Gk = coarse_series()
Uk = coarse_series()
Fk = coarse_series()

Gk1 = coarse_series()
Uk1 = coarse_series()

Gk[0].assign(uinitial)
for i in range(ntc):
    G(Gk[i], Gk[i+1])

copy_series(Uk, Gk)
copy_series(Gk1, Gk)
copy_series(Uk1, Gk)


# ## parareal iterations

for it in range(nits):
    copy_series(Uk, Uk1)
    copy_series(Gk, Gk1)

    for i in range(ntc):
        F(Uk[i], Fk[i+1])

    Uk1[0].assign(uinitial)

    for i in range(ntc):
        G(Uk1[i], Gk1[i+1])

        Uk1[i+1].assign(Fk[i+1] + Gk1[i+1] - Gk[i+1])

    err = series_error(userial, Uk1)
    res = series_error(Uk, Uk1)
    if is_root:
        Print(rank, it, err, res, comm=ensemble.comm)
    if err < tol:
        break

# Print('')
# Print('### === --- Iteration counts --- === ###')
# Print('')
#
# Print(f'linear iterations: {linear_its} | iterations per timestep: {linear_its/nt}')
# Print(f'nonlinear iterations: {nonlinear_its} | iterations per timestep: {nonlinear_its/nt}')
# Print('')
