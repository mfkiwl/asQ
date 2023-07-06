import asQ
import firedrake as fd
import pytest
from petsc4py import PETSc
from functools import reduce
from operator import mul


@pytest.mark.parallel(nprocs=4)
def test_Nitsche_BCs():
    # test the linear equation u_t - Delta u = 0, with u_ex = exp(0.5*x + y + 1.25*t) and weakly imposing Dirichlet BCs
    nspatial_domains = 2
    degree = 1
    nx = 10
    dx = 1/nx
    dt = dx
    ensemble = fd.Ensemble(fd.COMM_WORLD, nspatial_domains)
    mesh = fd.UnitSquareMesh(nx, nx, quadrilateral=False, comm=ensemble.comm)

    x, y = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)
    V = fd.FunctionSpace(mesh, "CG", degree)

    w0 = fd.Function(V)
    w0.interpolate(fd.exp(0.5*x + y))

    def form_mass(q, phi):
        return phi*q*fd.dx

    def form_function(q, phi, t):
        return fd.inner(fd.grad(q), fd.grad(phi))*fd.dx - fd.inner(phi, fd.inner(fd.grad(q), n))*fd.ds - fd.inner(q-fd.exp(0.5*x + y + 1.25*t), fd.inner(fd.grad(phi), n))*fd.ds + 20*nx*fd.inner(q-fd.exp(0.5*x + y + 1.25*t), phi)*fd.ds

    # Parameters for the diag
    sparameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'}

    solver_parameters_diag = {
        "snes_linesearch_type": "basic",
        'snes_atol': 1e-8,
        'ksp_rtol': 1e-8,
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }

    M = [2, 2]
    solver_parameters_diag["diagfft_block_"] = sparameters

    theta = 0.5

    PD = asQ.Paradiag(ensemble=ensemble,
                      form_function=form_function,
                      form_mass=form_mass, ics=w0,
                      dt=dt, theta=theta,
                      time_partition=M, bcs=[],
                      solver_parameters=solver_parameters_diag,
                      )
    PD.solve()
    q_exact = fd.Function(V)
    qp = fd.Function(V)
    errors = asQ.SharedArray(M, comm=ensemble.ensemble_comm)
    times = asQ.SharedArray(M, comm=ensemble.ensemble_comm)

    for step in range(2):
        if PD.aaoform.layout.is_local(step):
            local_step = PD.aaofunc.transform_index(step, from_range='window')
            t = PD.aaoform.time[local_step]
            q_exact.interpolate(fd.exp(.5*x + y + 1.25*t))
            PD.aaofunc.get_field(local_step, uout=qp)

            errors.dlocal[local_step] = fd.errornorm(qp, q_exact)
            times.dlocal[local_step] = t

    errors.synchronise()
    times.synchronise()

    for step in range(4):
        assert (errors.dglobal[step] < (dx)**(3/2))


@pytest.mark.parallel(nprocs=4)
def test_Nitsche_heat_timeseries():
    from utils.serial import ComparisonMiniapp
    from copy import deepcopy

    nwindows = 1
    nslices = 2
    slice_length = 2
    dt = 0.5
    theta = 0.5

    time_partition = [slice_length for _ in range(nslices)]
    ensemble = asQ.create_ensemble(time_partition)
    nx = 10
    mesh = fd.UnitSquareMesh(nx, nx, comm=ensemble.comm)
    x, y = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)

    W = fd.FunctionSpace(mesh, 'CG', 1)

    # initial conditions
    w_initial = fd.Function(W)
    w_initial.interpolate(fd.exp(0.5*x + y))

    # Heat equaion with Nitsch BCs.
    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx - fd.inner(v, fd.inner(fd.grad(u), n))*fd.ds - fd.inner(u-fd.exp(0.5*x + y + 1.25*t), fd.inner(fd.grad(v), n))*fd.ds + 20*nx*fd.inner(u-fd.exp(0.5*x + y + 1.25*t), v)*fd.ds

    def form_mass(u, v):
        return u*v*fd.dx

    block_sparameters = {
        'ksp_type': 'preonly',
        'ksp': {
            'atol': 1e-5,
            'rtol': 1e-5,
        },
        'pc_type': 'lu',
    }

    snes_sparameters = {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-10,
        'rtol': 1e-12,
        'stol': 1e-12,
    }

    # solver parameters for serial method
    serial_sparameters = {
        'snes': snes_sparameters
    }
    serial_sparameters.update(deepcopy(block_sparameters))
    serial_sparameters['ksp']['monitor'] = None
    serial_sparameters['ksp']['converged_reason'] = None

    # solver parameters for parallel method
    parallel_sparameters = {
        'snes': snes_sparameters,
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp': {
            'monitor': None,
            'converged_reason': None,
        },
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }

    for i in range(sum(time_partition)):
        parallel_sparameters['diagfft_block_'+str(i)] = block_sparameters
    appctx = {}

    miniapp = ComparisonMiniapp(ensemble, time_partition,
                                form_mass,
                                form_function,
                                w_initial,
                                dt, theta,
                                serial_sparameters,
                                parallel_sparameters, appctx=appctx)

    norm0 = fd.norm(w_initial)

    def preproc(serial_app, paradiag, wndw):
        PETSc.Sys.Print('')
        PETSc.Sys.Print(f'### === --- Time window {wndw} --- === ###')
        PETSc.Sys.Print('')
        PETSc.Sys.Print('=== --- Parallel solve --- ===')
        PETSc.Sys.Print('')

    def parallel_postproc(pdg, wndw, rhs):
        PETSc.Sys.Print('')
        PETSc.Sys.Print('=== --- Serial solve --- ===')
        PETSc.Sys.Print('')
        return

    PETSc.Sys.Print('')
    PETSc.Sys.Print('### === --- Timestepping loop --- === ###')

    errors = miniapp.solve(nwindows=nwindows,
                           preproc=preproc,
                           parallel_postproc=parallel_postproc)

    PETSc.Sys.Print('')
    PETSc.Sys.Print('### === --- Errors --- === ###')

    for it, err in enumerate(errors):
        PETSc.Sys.Print(f'Timestep {it} error: {err/norm0}')

    for err in errors:
        assert err/norm0 < 1e-5


@pytest.mark.parallel(nprocs=4)
def test_galewsky_timeseries():
    from utils import units
    from utils import mg
    from utils.planets import earth
    import utils.shallow_water as swe
    from utils.shallow_water import galewsky
    from utils.serial import ComparisonMiniapp
    from copy import deepcopy

    ref_level = 2
    nwindows = 1
    nslices = 2
    slice_length = 2
    dt = 0.5
    theta = 0.5
    degree = swe.default_degree()

    time_partition = [slice_length for _ in range(nslices)]

    dt = dt*units.hour

    ensemble = asQ.create_ensemble(time_partition)

    # icosahedral mg mesh
    mesh = swe.create_mg_globe_mesh(ref_level=ref_level,
                                    comm=ensemble.comm,
                                    coords_degree=1)
    x = fd.SpatialCoordinate(mesh)

    # shallow water equation function spaces (velocity and depth)
    W = swe.default_function_space(mesh, degree=degree)

    # parameters
    gravity = earth.Gravity

    topography = galewsky.topography_expression(*x)
    coriolis = swe.earth_coriolis_expression(*x)

    # initial conditions
    w_initial = fd.Function(W)
    u_initial = w_initial.subfunctions[0]
    h_initial = w_initial.subfunctions[1]

    u_initial.project(galewsky.velocity_expression(*x))
    h_initial.project(galewsky.depth_expression(*x))

    # shallow water equation forms
    def form_function(u, h, v, q, t):
        return swe.nonlinear.form_function(mesh,
                                           gravity,
                                           topography,
                                           coriolis,
                                           u, h, v, q, t)

    def form_mass(u, h, v, q):
        return swe.nonlinear.form_mass(mesh, u, h, v, q)

    # vanka patch smoother
    patch_parameters = {
        'pc_patch': {
            'save_operators': True,
            'partition_of_unity': True,
            'sub_mat_type': 'seqdense',
            'construct_dim': 0,
            'construct_type': 'vanka',
            'local_type': 'additive',
            'precompute_element_tensors': True,
            'symmetrise_sweep': False,
        },
        'sub': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_shift_type': 'nonzero',
        }
    }

    # mg with patch smoother
    mg_parameters = {
        'levels': {
            'ksp_type': 'gmres',
            'ksp_max_it': 5,
            'pc_type': 'python',
            'pc_python_type': 'firedrake.PatchPC',
            'patch': patch_parameters
        },
        'coarse': {
            'pc_type': 'python',
            'pc_python_type': 'firedrake.AssembledPC',
            'assembled_pc_type': 'lu',
            'assembled_pc_factor_mat_solver_type': 'mumps',
        }
    }

    # parameters for the implicit solves at:
    #   each Newton iteration of serial method
    #   each diagonal block solve in step-(b) of parallel method
    block_sparameters = {
        'mat_type': 'matfree',
        'ksp_type': 'fgmres',
        'ksp': {
            'atol': 1e-5,
            'rtol': 1e-5,
        },
        'pc_type': 'mg',
        'pc_mg_cycle_type': 'v',
        'pc_mg_type': 'multiplicative',
        'mg': mg_parameters
    }

    # nonlinear solver options
    snes_sparameters = {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
        'stol': 1e-12,
        'ksp_ew': None,
        'ksp_ew_version': 1,
    }

    # solver parameters for serial method
    serial_sparameters = {
        'snes': snes_sparameters
    }
    serial_sparameters.update(deepcopy(block_sparameters))
    serial_sparameters['ksp']['monitor'] = None
    serial_sparameters['ksp']['converged_reason'] = None

    # solver parameters for parallel method
    parallel_sparameters = {
        'snes': snes_sparameters,
        'mat_type': 'matfree',
        'ksp_type': 'fgmres',
        'ksp': {
            'monitor': None,
            'converged_reason': None,
        },
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft_alpha': 1e-3,
    }

    for i in range(sum(time_partition)):
        parallel_sparameters['diagfft_block_'+str(i)] = block_sparameters

    appctx = {}
    transfer_managers = []
    for _ in range(time_partition[ensemble.ensemble_comm.rank]):
        tm = mg.manifold_transfer_manager(W)
        transfer_managers.append(tm)
    appctx['diag_transfer_managers'] = transfer_managers

    miniapp = ComparisonMiniapp(ensemble, time_partition,
                                form_mass,
                                form_function,
                                w_initial, dt, theta,
                                serial_sparameters,
                                parallel_sparameters,
                                appctx=appctx)

    miniapp.serial_app.nlsolver.set_transfer_manager(
        mg.manifold_transfer_manager(W))

    norm0 = fd.norm(w_initial)

    def preproc(serial_app, paradiag, wndw):
        PETSc.Sys.Print('')
        PETSc.Sys.Print(f'### === --- Time window {wndw} --- === ###')
        PETSc.Sys.Print('')
        PETSc.Sys.Print('=== --- Parallel solve --- ===')
        PETSc.Sys.Print('')

    def parallel_postproc(pdg, wndw, rhs):
        PETSc.Sys.Print('')
        PETSc.Sys.Print('=== --- Serial solve --- ===')
        PETSc.Sys.Print('')
        return

    PETSc.Sys.Print('')
    PETSc.Sys.Print('### === --- Timestepping loop --- === ###')

    errors = miniapp.solve(nwindows=nwindows,
                           preproc=preproc,
                           parallel_postproc=parallel_postproc)

    PETSc.Sys.Print('')
    PETSc.Sys.Print('### === --- Errors --- === ###')

    for it, err in enumerate(errors):
        PETSc.Sys.Print(f'Timestep {it} error: {err/norm0}')

    for err in errors:
        assert err/norm0 < 1e-5


@pytest.mark.parallel(nprocs=4)
def test_steady_swe():
    # test that steady-state is maintained for shallow water eqs
    import utils.units as units
    import utils.planets.earth as earth
    import utils.shallow_water.nonlinear as swe
    import utils.shallow_water.williamson1992.case2 as case2

    # set up the ensemble communicator for space-time parallelism
    ref_level = 2
    degree = 1

    nslices = fd.COMM_WORLD.size//2
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    mesh = fd.IcosahedralSphereMesh(radius=earth.radius,
                                    refinement_level=ref_level,
                                    degree=degree,
                                    comm=ensemble.comm)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
    V2 = fd.FunctionSpace(mesh, "DG", degree)
    W = fd.MixedFunctionSpace((V1, V2))

    # initial conditions
    f = case2.coriolis_expression(*x)

    g = earth.Gravity
    H = case2.H0
    b = fd.Constant(0)

    # W = V1 * V2
    w0 = fd.Function(W)
    un = w0.subfunctions[0]
    hn = w0.subfunctions[1]
    un.project(case2.velocity_expression(*x))
    hn.project(H - b + case2.elevation_expression(*x))

    # finite element forms

    def form_function(u, h, v, q, t):
        return swe.form_function(mesh, g, b, f, u, h, v, q, t)

    def form_mass(u, h, v, q):
        return swe.form_mass(mesh, u, h, v, q)

    # Parameters for the diag
    sparameters = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'}

    solver_parameters_diag = {
        "snes_linesearch_type": "basic",
        'snes_atol': 1e3,
        'snes_monitor': None,
        'snes_converged_reason': None,
        'ksp_rtol': 1e-3,
        'ksp_monitor': None,
        'ksp_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'aaos_jacobian_state': 'initial',
        'diagfft_state': 'initial',
        'diagfft_alpha': 1e-3,
    }

    for i in range(sum(time_partition)):
        solver_parameters_diag["diagfft_block_"+str(i)] = sparameters

    dt = 0.2*units.hour

    theta = 0.5

    pdg = asQ.Paradiag(ensemble=ensemble,
                       form_function=form_function,
                       form_mass=form_mass,
                       ics=w0, dt=dt, theta=theta,
                       time_partition=time_partition,
                       solver_parameters=solver_parameters_diag)
    pdg.solve()

    # check against initial conditions
    hn.assign(hn - H + b)

    hmag = fd.norm(hn)
    umag = fd.norm(un)

    for step in range(pdg.nlocal_timesteps):

        up = pdg.aaofunc.get_component(step, 0, index_range='slice')
        hp = pdg.aaofunc.get_component(step, 1, index_range='slice')
        hp.assign(hp-H+b)

        herr = fd.errornorm(hn, hp)/hmag
        uerr = fd.errornorm(un, up)/umag

        htol = pow(10, -ref_level)
        utol = pow(10, -ref_level)

        assert (abs(herr) < htol)
        assert (abs(uerr) < utol)


bc_opts = ["no_bcs", "homogeneous_bcs", "inhomogeneous_bcs"]

extruded_mixed = [pytest.param(False, id="standard_mesh"),
                  pytest.param(True, id="extruded_mesh",
                               marks=pytest.mark.xfail(reason="fd.split for TensorProductElements in unmixed spaces broken by ufl PR#122."))]


@pytest.mark.parallel(nprocs=6)
@pytest.mark.parametrize("bc_opt", bc_opts)
@pytest.mark.parametrize("extruded", extruded_mixed)
def test_solve_para_form(bc_opt, extruded):
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the NONLINEAR heat equation as an example by
    # solving the all-at-once system and comparing with the sequential

    # set up the ensemble communicator for space-time parallelism
    nslices = fd.COMM_WORLD.size//2
    slice_length = 2

    time_partition = tuple((slice_length for _ in range(nslices)))
    ensemble = asQ.create_ensemble(time_partition, comm=fd.COMM_WORLD)

    if extruded:
        mesh1D = fd.UnitIntervalMesh(4, comm=ensemble.comm)
        mesh = fd.ExtrudedMesh(mesh1D, 4, layer_height=0.25)
    else:
        mesh = fd.UnitSquareMesh(4, 4, quadrilateral=True, comm=ensemble.comm)

    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    time = .01
    theta = 0.5
    c = fd.Constant(1)
    time_partition = [2, 2, 2]
    ntimesteps = sum(time_partition)

    # Parameters for the diag
    sparameters = {
        "ksp_type": "preonly",
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    solver_parameters_diag = {
        "snes_linesearch_type": "basic",
        'snes_monitor': None,
        'snes_stol': 1.0e-100,
        'snes_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_type': 'gmres',
        'ksp_monitor': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
    }

    for i in range(ntimesteps):
        solver_parameters_diag[f"diagfft_block_{i}_"] = sparameters

    def form_function(u, v, t):
        return fd.inner((1.+c*fd.inner(u, u))*fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    if bc_opt == "inhomogeneous_bcs":
        bcs = [fd.DirichletBC(V, fd.sin(2*fd.pi*x), "on_boundary")]
    elif bc_opt == "homogeneous_bcs":
        bcs = [fd.DirichletBC(V, 0., "on_boundary")]
    else:
        bcs = []

    pdg = asQ.Paradiag(ensemble=ensemble,
                       form_function=form_function,
                       form_mass=form_mass,
                       ics=u0, dt=dt, theta=theta,
                       time_partition=time_partition, bcs=bcs,
                       solver_parameters=solver_parameters_diag)
    pdg.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v, time - dt)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v, time)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1, bcs=bcs)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    # Calculation of time slices in serial:
    VFull = reduce(mul, (V for _ in range(ntimesteps)))
    vfull = fd.Function(VFull)
    vfull_list = vfull.subfunctions

    for i in range(ntimesteps):
        ssolver.solve()
        time += dt
        vfull_list[i].assign(unp1)
        un.assign(unp1)

    for i in range(pdg.nlocal_timesteps):
        fidx = pdg.aaofunc.transform_index(i, from_range='slice', to_range='window')
        assert (fd.errornorm(vfull.sub(fidx), pdg.aaofunc.get_field(i, index_range='slice')) < 1.0e-9)


@pytest.mark.parallel(nprocs=6)
def test_diagnostics():
    # tests that the diagnostics recording is accurate
    ensemble = fd.Ensemble(fd.COMM_WORLD, 2)

    mesh = fd.UnitSquareMesh(4, 4, comm=ensemble.comm)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5 ** 2))
    dt = 0.01
    theta = 0.5
    time_partition = [2, 2, 2]

    block_sparameters = {
        "ksp_type": "preonly",
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    diag_sparameters = {
        'snes_converged_reason': None,
        'ksp_converged_reason': None,
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft_alpha': 1e-3,
    }

    for i in range(sum(time_partition)):
        diag_sparameters["diagfft_block_" + str(i) + "_"] = block_sparameters

    def form_function(u, v, t):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    pdg = asQ.Paradiag(ensemble=ensemble,
                       form_function=form_function,
                       form_mass=form_mass,
                       ics=u0, dt=dt, theta=theta,
                       time_partition=time_partition,
                       solver_parameters=diag_sparameters)

    pdg.solve(nwindows=1)

    pdg.sync_diagnostics()

    assert pdg.total_timesteps == pdg.ntimesteps
    assert pdg.total_windows == 1
    assert pdg.linear_iterations == pdg.solver.snes.getLinearSolveIterations()
    assert pdg.nonlinear_iterations == pdg.solver.snes.getIterationNumber()

    # direct block solve
    for i in range(pdg.ntimesteps):
        assert pdg.block_iterations.dglobal[i] == pdg.linear_iterations

    linear_iterations0 = pdg.linear_iterations
    nonlinear_iterations0 = pdg.nonlinear_iterations

    pdg.solve(nwindows=1)

    assert pdg.total_timesteps == 2*pdg.ntimesteps
    assert pdg.total_windows == 2
    assert pdg.linear_iterations == linear_iterations0 + pdg.solver.snes.getLinearSolveIterations()
    assert pdg.nonlinear_iterations == nonlinear_iterations0 + pdg.solver.snes.getIterationNumber()

    for i in range(pdg.ntimesteps):
        assert pdg.block_iterations.dglobal[i] == pdg.linear_iterations
