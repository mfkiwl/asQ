import asQ
import firedrake as fd
import numpy as np

def test_set_para_form():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # substituting the sequential solution and evaluating the residual

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0,
                      dt=dt, theta=theta,
                      alpha=alpha,
                      M=M, solver_parameters=solver_parameters,
                      circ="none")

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v/dt*fd.dx
    eqn += fd.Constant((1-theta))*form_function(un, v)
    eqn += fd.Constant(theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    for i in range(M):
        ssolver.solve()
        PD.w_all.sub(i).assign(unp1)
        un.assign(unp1)

    Pres = fd.assemble(PD.para_form)
    for i in range(M):
        assert(dt*np.abs(Pres.sub(i).dat.data[:]).max() < 1.0e-16)


def test_set_para_form_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'gmres', 'pc_type': 'none',
                         'ksp_rtol': 1.0e-8, 'ksp_atol': 1.0e-8,
                         'ksp_monitor': None}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="none")

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = (1.0/dt)*form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= (1.0/dt)*form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant((1-theta))*form_function(*(fd.split(un)),
                                                *(fd.split(v)))
    eqn += fd.Constant(theta)*form_function(*(fd.split(unp1)),
                                            *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    for i in range(M):
        ssolver.solve()
        for k in range(2):
            PD.w_all.sub(2*i+k).assign(unp1.sub(k))
        un.assign(unp1)

    Pres = fd.assemble(PD.para_form)
    for i in range(M):
        assert(dt*np.abs(Pres.sub(i).dat.data[:]).max() < 1.0e-16)


def test_solve_para_form():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the heat equation as an example by
    # solving the all-at-once system and comparing with the sequential
    # solution

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="none")
    PD.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


def test_solve_para_form_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="none")
    PD.solve()

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    err = fd.Function(W, name="err")
    pun = fd.Function(W, name="pun")
    puns = pun.split()
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        walls = PD.w_all.split()[2*i:2*i+2]
        for k in range(2):
            puns[k].assign(walls[k])
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


def test_relax():
    # tests the relaxation method
    # using the heat equation as an example

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    # Solving U_t + F(U) = 0
    # defining F(U)
    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    # defining the structure of U_t
    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12)
    PD.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


def test_relax_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12)
    PD.solve(verbose=True)

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    err = fd.Function(W, name="err")
    pun = fd.Function(W, name="pun")
    puns = pun.split()
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        walls = PD.w_all.split()[2*i:2*i+2]
        for k in range(2):
            puns[k].assign(walls[k])
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


def test_diag_precon():
    # Test PCDIAGFFT by using it
    # within the relaxation method
    # using the heat equation as an example
    # we compare one iteration using just the diag PC
    # with the direct solver

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.01
    M = 4

    mass_options = {
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'pc_sub_type': 'icc',
        'ksp_atol': 1.0e-50,
        'ksp_rtol': 1.0e-12
    }

    diagfft_options = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'mass': mass_options}

    solver_parameters = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp_rtol': 1.0e-10,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft': diagfft_options}

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12, maxits=1)
    PD.solve()
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    PDe = asQ.paradiag(form_function=form_function,
                       form_mass=form_mass, W=V, w0=u0, dt=dt,
                       theta=theta, alpha=alpha, M=M,
                       solver_parameters=solver_parameters,
                       circ="picard", tol=1.0e-12, maxits=1)
    PDe.solve()
    unD = fd.Function(V, name='diag')
    un = fd.Function(V, name='full')
    err = fd.Function(V, name='error')
    unD.assign(u0)
    un.assign(u0)
    for i in range(M):
        walls = PD.w_all.split()[i]
        wallsE = PDe.w_all.split()[i]
        unD.assign(walls)
        un.assign(wallsE)
        err.assign(un-unD)
        assert(fd.norm(err) < 1.0e-13)


def test_diag_precon_mixed():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual

    mesh = fd.PeriodicUnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "BDM", 1)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    W = V * Q

    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4

    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}

    def form_function(uu, up, vu, vp):
        return (fd.div(vu)*up - fd.div(uu)*vp)*fd.dx

    def form_mass(uu, up, vu, vp):
        return (fd.inner(uu, vu) + up*vp)*fd.dx

    mass_options = {
        'ksp_type': 'preonly',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'additive',
        'fieldsplit_0_ksp_type': 'cg',
        'fieldsplit_0_pc_type': 'bjacobi',
        'fieldsplit_0_sub_pc_type': 'icc',
        'fieldsplit_0_ksp_atol': 1.0e-50,
        'fieldsplit_0_ksp_rtol': 1.0e-12,
        'fieldsplit_0_ksp_type': 'preonly',
        'fieldsplit_0_pc_type': 'bjacobi',
        'fieldsplit_0_sub_pc_type': 'icc'
    }
    
    diagfft_options = {'ksp_type': 'gmres', 'pc_type': 'lu',
                       'ksp_monitor': None,
                       'pc_factor_mat_solver_type': 'mumps',
                       'mat_type': 'aij'}

    solver_parameters_diag = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp_rtol': 1.0e-10,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft': diagfft_options}

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters_diag,
                      circ="picard", tol=1.0e-12)
    PD.solve(verbose=True)

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    err = fd.Function(W, name="err")
    pun = fd.Function(W, name="pun")
    puns = pun.split()
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        walls = PD.w_all.split()[2*i:2*i+2]
        for k in range(2):
            puns[k].assign(walls[k])
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-15)


def test_diag_precon_nl():
    # Test PCDIAGFFT by using it within the relaxation method
    # using the NONLINEAR heat equation as an example
    # we compare one iteration using just the diag PC
    # with the direct solver

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.01
    M = 4
    c = fd.Constant(0.1)

    mass_options = {
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'pc_sub_type': 'icc',
        'ksp_atol': 1.0e-50,
        'ksp_rtol': 1.0e-12
    }
    
    diagfft_options = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij',
        'mass': mass_options
    }

    solver_parameters = {
        'snes_monitor': None,
        'snes_converged_reason': None,
        'mat_type': 'matfree',
        'ksp_rtol': 1.0e-10,
        'ksp_max_it': 12,
        'ksp_converged_reason': None,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft': diagfft_options}

    def form_function(u, v):
        return fd.inner((1.+c*fd.inner(u, u))*fd.grad(u), fd.grad(v))*fd.dx

    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="picard", tol=1.0e-12, maxits=1)
    PD.solve(verbose=True)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'
                         }
    PDe = asQ.paradiag(form_function=form_function,
                       form_mass=form_mass, W=V, w0=u0, dt=dt,
                       theta=theta, alpha=alpha, M=M,
                       solver_parameters=solver_parameters,
                       circ="picard", tol=1.0e-12, maxits=1)
    PDe.solve(verbose=True)
    unD = fd.Function(V, name='diag')
    un = fd.Function(V, name='full')
    err = fd.Function(V, name='error')
    unD.assign(u0)
    un.assign(u0)
    for i in range(M):
        walls = PD.w_all.split()[i]
        wallsE = PDe.w_all.split()[i]
        unD.assign(walls)
        un.assign(wallsE)
        err.assign(un-unD)
        assert(fd.norm(err) < 1.0e-12)


def test_quasi():
    # tests the quasi-Newton option
    # using the heat equation as an example

    mesh = fd.UnitSquareMesh(20, 20)
    V = fd.FunctionSpace(mesh, "CG", 1)

    x, y = fd.SpatialCoordinate(mesh)
    u0 = fd.Function(V).interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4

    mass_options = {
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'pc_sub_type': 'icc',
        'ksp_atol': 1.0e-50,
        'ksp_rtol': 1.0e-12
    }

    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij',
                         'snes_monitor': None,
                         'mass': mass_options}

    # Solving U_t + F(U) = 0
    # defining F(U)
    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    # defining the structure of U_t
    def form_mass(u, v):
        return u*v*fd.dx

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=V, w0=u0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters,
                      circ="quasi")
    PD.solve()

    # sequential solver
    un = fd.Function(V)
    unp1 = fd.Function(V)

    un.assign(u0)
    v = fd.TestFunction(V)

    eqn = (unp1 - un)*v*fd.dx
    eqn += fd.Constant(dt*(1-theta))*form_function(un, v)
    eqn += fd.Constant(dt*theta)*form_function(unp1, v)

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)

    err = fd.Function(V, name="err")
    pun = fd.Function(V, name="pun")
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        pun.assign(PD.w_all.sub(i))
        err.assign(un-pun)
        assert(fd.norm(err) < 1.0e-10)


def test_diag_precon_mixed_helmpc():
    # checks that the all-at-once system is the same as solving
    # timesteps sequentially using the mixed wave equation as an
    # example by substituting the sequential solution and evaluating
    # the residual
    # using the Helm PC

    import petsc4py.PETSc as PETSc
    PETSc.Sys.popErrorHandler() 
    
    mesh = fd.PeriodicUnitSquareMesh(10, 10)
    V = fd.FunctionSpace(mesh, "BDM", 2)
    Q = fd.FunctionSpace(mesh, "DG", 1)
    W = V * Q

    gamma = fd.Constant(1.0e5)
    x, y = fd.SpatialCoordinate(mesh)
    w0 = fd.Function(W)
    u0, p0 = w0.split()
    p0.interpolate(fd.exp(-((x-0.5)**2 + (y-0.5)**2)/0.5**2))
    dt = 0.01
    theta = 0.5
    alpha = 0.001
    M = 4

    def form_function(uu, up, vu, vp):
        eqn = (- fd.div(vu)*up + fd.div(uu)*vp)*fd.dx
        eqn += gamma*fd.div(vu)*fd.div(uu)*fd.dx
        return eqn

    def form_mass(uu, up, vu, vp):
        eqn = (fd.inner(uu, vu) + up*vp)*fd.dx
        eqn += gamma*fd.div(vu)*up*fd.dx
        return eqn

    diag_parameters = {
        "mat_type": "matfree",
        "ksp_type": "fgmres",
        "ksp_converged_reason": None,
        "ksp_atol": 1.0e-12,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_off_diag_use_amat": True,
    }

    Hparameters = {
        "ksp_type":"preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }

    mass_options = {
        'ksp_type': 'preonly',
        'pc_type': 'fieldsplit',
        'pc_fieldsplit_type': 'additive',
        'fieldsplit_0_ksp_type': 'cg',
        'fieldsplit_0_pc_type': 'bjacobi',
        'fieldsplit_0_sub_pc_type': 'icc',
        'fieldsplit_0_ksp_atol': 1.0e-50,
        'fieldsplit_0_ksp_rtol': 1.0e-12,
        'fieldsplit_0_ksp_type': 'preonly',
        'fieldsplit_0_pc_type': 'bjacobi',
        'fieldsplit_0_sub_pc_type': 'icc'
    }
    
    bottomright = {
        "ksp_type": "gmres",
        "ksp_gmres_modifiedgramschmidt": None,
        "ksp_max_it": 3,
        "pc_type": "python",
        "pc_python_type": "asQ.HelmholtzPC",
        "Hp": Hparameters,
        "mass": mass_options
    }

    diag_parameters["fieldsplit_1"] = bottomright

    topleft_LU = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "lu",
        "assembled_pc_factor_mat_solver_type": "mumps"
    }

    diag_parameters["fieldsplit_0"] = topleft_LU
    
    solver_parameters_diag = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'ksp_rtol': 1.0e-10,
        'pc_type': 'python',
        'pc_python_type': 'asQ.DiagFFTPC',
        'diagfft': diag_parameters}

    PD = asQ.paradiag(form_function=form_function,
                      form_mass=form_mass, W=W, w0=w0, dt=dt,
                      theta=theta, alpha=alpha, M=M,
                      solver_parameters=solver_parameters_diag,
                      circ="picard", tol=1.0e-12)
    PD.solve(verbose=True)

    # sequential solver
    un = fd.Function(W)
    unp1 = fd.Function(W)

    un.assign(w0)
    v = fd.TestFunction(W)

    def form_function(uu, up, vu, vp):
        eqn = (- fd.div(vu)*up + fd.div(uu)*vp)*fd.dx
        return eqn

    def form_mass(uu, up, vu, vp):
        eqn = (fd.inner(uu, vu) + up*vp)*fd.dx
        return eqn

    
    eqn = form_mass(*(fd.split(unp1)), *(fd.split(v)))
    eqn -= form_mass(*(fd.split(un)), *(fd.split(v)))
    eqn += fd.Constant(dt*(1-theta))*form_function(*(fd.split(un)),
                                                   *(fd.split(v)))
    eqn += fd.Constant(dt*theta)*form_function(*(fd.split(unp1)),
                                               *(fd.split(v)))

    sprob = fd.NonlinearVariationalProblem(eqn, unp1)
    solver_parameters = {'ksp_type': 'preonly', 'pc_type': 'lu',
                         'pc_factor_mat_solver_type': 'mumps',
                         'mat_type': 'aij'}
    ssolver = fd.NonlinearVariationalSolver(sprob,
                                            solver_parameters=solver_parameters)
    ssolver.solve()

    err = fd.Function(W, name="err")
    pun = fd.Function(W, name="pun")
    puns = pun.split()
    for i in range(M):
        ssolver.solve()
        un.assign(unp1)
        walls = PD.w_all.split()[2*i:2*i+2]
        for k in range(2):
            puns[k].assign(walls[k])
        err.assign(un-pun)
        print(fd.norm(err))
        assert(fd.norm(err) < 1.0e-13)
