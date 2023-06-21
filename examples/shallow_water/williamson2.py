
import numpy as np
import firedrake as fd
from petsc4py import PETSc
import asQ

from utils import mg
from utils import units
from utils.planets import earth
import utils.shallow_water as swe
import utils.shallow_water.williamson1992.case2 as case2

PETSc.Sys.popErrorHandler()

# get command arguments
import argparse
parser = argparse.ArgumentParser(
    description='Williamson 2 testcase for ParaDiag solver using fully implicit SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve.')
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid.')
parser.add_argument('--nwindows', type=int, default=1, help='Number of time-windows.')
parser.add_argument('--nslices', type=int, default=2, help='Number of time-slices per time-window.')
parser.add_argument('--slice_length', type=int, default=2, help='Number of timesteps per time-slice.')
parser.add_argument('--nspatial_domains', type=int, default=2, help='Size of spatial partition.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Circulant coefficient.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='williamson2', help='Name of output vtk files')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

PETSc.Sys.Print('')
PETSc.Sys.Print('### === --- Setting up --- === ###')
PETSc.Sys.Print('')

# time steps

time_partition = [args.slice_length for _ in range(args.nslices)]
window_length = sum(time_partition)
nsteps = args.nwindows*window_length

dt = args.dt*units.hour
# multigrid mesh set up

ensemble = asQ.create_ensemble(time_partition)

distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

# mesh set up
mesh = mg.icosahedral_mesh(R0=earth.radius,
                           base_level=args.base_level,
                           degree=args.coords_degree,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level,
                           comm=ensemble.comm)
x = fd.SpatialCoordinate(mesh)

# Mixed function space for velocity and depth
V1 = swe.default_velocity_function_space(mesh, degree=args.degree)
V2 = swe.default_depth_function_space(mesh, degree=args.degree)
W = fd.MixedFunctionSpace((V1, V2))

# initial conditions
w0 = fd.Function(W)
un = w0.subfunctions[0]
hn = w0.subfunctions[1]

f = case2.coriolis_expression(*x)
b = case2.topography_function(*x, V2, name="Topography")
H = case2.H0

un.project(case2.velocity_expression(*x))
etan = case2.elevation_function(*x, V2, name="Elevation")
hn.assign(H + etan - b)


# nonlinear swe forms

def form_function(u, h, v, q, t):
    return swe.nonlinear.form_function(mesh, earth.Gravity, b, f, u, h, v, q, t)


def form_mass(u, h, v, q):
    return swe.nonlinear.form_mass(mesh, u, h, v, q)


# parameters for the implicit diagonal solve in step-(b)

sparameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'atol': 1e-5,
        'rtol': 1e-5,
        'max_it': 50,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': {
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': {
                'pc_patch_save_operators': True,
                'pc_patch_partition_of_unity': True,
                'pc_patch_sub_mat_type': 'seqdense',
                'pc_patch_construct_dim': 0,
                'pc_patch_construct_type': 'vanka',
                'pc_patch_local_type': 'additive',
                'pc_patch_precompute_element_tensors': True,
                'pc_patch_symmetrise_sweep': False,
                'sub_ksp_type': 'preonly',
                'sub_pc_type': 'lu',
                'sub_pc_factor_shift_type': 'nonzero',
            },
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'assembled_pc_factor_mat_solver_type': 'mumps',
        },
    }
}

atol = 1e-0
sparameters_diag = {
    'snes': {
        'linesearch_type': 'basic',
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
        'ksp_ew_threshold': 1e-2,
    },
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': atol,
        'rtol': 1e-5,
    },
    'pc_type': 'python',
    'pc_python_type': 'asQ.DiagFFTPC',
}

PETSc.Sys.Print('### === --- Calculating parallel solution --- === ###')
PETSc.Sys.Print('')

sparameters_diag['diagfft_block_'] = sparameters

# non-petsc information for block solve
block_ctx = {}

# mesh transfer operators
transfer_managers = []
nlocal_timesteps = time_partition[ensemble.ensemble_comm.rank]
for _ in range(nlocal_timesteps):
    tm = mg.manifold_transfer_manager(W)
    transfer_managers.append(tm)

block_ctx['diag_transfer_managers'] = transfer_managers

PD = asQ.paradiag(ensemble=ensemble,
                  form_function=form_function,
                  form_mass=form_mass, w0=w0,
                  dt=dt, theta=0.5,
                  alpha=args.alpha,
                  time_partition=time_partition, solver_parameters=sparameters_diag,
                  circ=None, tol=1.0e-6, maxits=None,
                  ctx={}, block_ctx=block_ctx, block_mat_type="aij")


def window_preproc(pdg, wndw):
    PETSc.Sys.Print('')
    PETSc.Sys.Print(f'### === --- Calculating time-window {wndw} --- === ###')
    PETSc.Sys.Print('')


# check against initial conditions
wcheck = w0.copy(deepcopy=True)
ucheck = wcheck.subfunctions[0]
hcheck = wcheck.subfunctions[1]
hcheck.assign(hcheck - H + b)


def steady_state_test(w):
    up = w.subfunctions[0]
    hp = w.subfunctions[1]
    hp.assign(hp - H + b)

    uerr = fd.errornorm(ucheck, up)/fd.norm(ucheck)
    herr = fd.errornorm(hcheck, hp)/fd.norm(hcheck)

    return uerr, herr


# check each timestep against steady state
def window_postproc(pdg, wndw):
    errors = np.zeros((window_length, 2))
    local_errors = np.zeros_like(errors)

    # collect errors for this slice
    def for_each_callback(window_index, slice_index, w):
        nonlocal local_errors
        local_errors[window_index, :] = steady_state_test(w)

    pdg.aaos.for_each_timestep(for_each_callback)

    # collect and print errors for full window
    ensemble.ensemble_comm.Reduce(local_errors, errors, root=0)
    if pdg.time_rank == 0:
        for window_index in range(window_length):
            timestep = wndw*window_length + window_index
            uerr = errors[window_index, 0]
            herr = errors[window_index, 1]
            PETSc.Sys.Print(f"timestep={timestep}, uerr={uerr}, herr={herr}",
                            comm=ensemble.comm)


PD.solve(nwindows=args.nwindows,
         preproc=window_preproc,
         postproc=window_postproc)
