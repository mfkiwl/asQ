
import firedrake as fd
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print

import asQ

global_comm = fd.COMM_WORLD

T = 0.5
nt = 16
nx = 16
cfl = 1.
theta = 0.5
nu = 1.
alpha = 1.e-4

nwindows = 1

dx = 1./nx
dt = T/nt

cfl = nu*dt/(dx*dx)
Print(cfl, dt)
verbose = False

nslices = global_comm.size
assert (nt % nslices) == 0
slice_length = nt//nslices

time_partition = tuple(slice_length for _ in range(nslices))
ensemble = asQ.create_ensemble(time_partition)

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

# parameters

block_sparams = {
    'ksp': {
        # 'monitor': None,
        # 'converged_reason': None,
        'rtol': 1e-5,
    },
    'ksp_type': 'preonly',
    'pc_type': 'lu',
}

sparameters = {
    'snes_type': 'ksponly',
    'snes': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-8,
    },
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'rtol': 1e-8,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
    'diagfft_block': block_sparams
}

pdg = asQ.paradiag(ensemble=ensemble,
                   form_function=form_function,
                   form_mass=form_mass,
                   w0=uinitial, dt=dt, theta=theta,
                   alpha=alpha, time_partition=time_partition,
                   solver_parameters=sparameters)


def window_preproc(pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


pdg.solve(nwindows,
          preproc=window_preproc)

Print('')
Print('### === --- Iteration counts --- === ###')
Print('')
# paradiag collects a few solver diagnostics for us to inspect
nw = nwindows

# Number of nonlinear iterations, total and per window.
# (1 for fgmres and # picard iterations for preonly)
PETSc.Sys.Print(f'nonlinear iterations: {pdg.nonlinear_iterations}  |  iterations per window: {pdg.nonlinear_iterations/nw}')

# Number of linear iterations, total and per window.
# (# of gmres iterations for fgmres and # picard iterations for preonly)
PETSc.Sys.Print(f'linear iterations: {pdg.linear_iterations}  |  iterations per window: {pdg.linear_iterations/nw}')

# Number of iterations needed for each block in step-(b), total and per block solve
# The number of iterations for each block will usually be different because of the different eigenvalues
PETSc.Sys.Print(f'block linear iterations: {pdg.block_iterations._data}  |  iterations per block solve: {pdg.block_iterations._data/pdg.linear_iterations}')
