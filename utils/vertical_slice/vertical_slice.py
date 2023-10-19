import firedrake as fd
from firedrake.petsc import PETSc
from utils.misc import curl0, cross0, cross1


def pi_formula(rho, theta, gas=None, R_d=None, p_0=None, kappa=None):
    if gas is None:
        if any(x is None for x in (R_d, p_0, kappa)):
            raise ValueError("R_d, p_0, and kappa must be specified if GasProperties not given")
    else:
        R_d = gas.R_d
        p_0 = gas.p_0
        kappa = gas.kappa
    return (rho * R_d * theta / p_0) ** (kappa / (1 - kappa))


def rho_formula(pi, theta, gas=None, R_d=None, p_0=None, kappa=None):
    if gas is None:
        if any(x is None for x in (R_d, p_0, kappa)):
            raise ValueError("R_d, p_0, and kappa must be specified if GasProperties not given")
    else:
        R_d = gas.R_d
        p_0 = gas.p_0
        kappa = gas.kappa
    return p_0*pi**((1-kappa)/kappa)/R_d/theta


def hydrostatic_rho(Vv, V2, mesh, thetan, rhon, pi_boundary,
                    gas, Up, top=False, Pi=None, verbose=0):
    # Calculate hydrostatic Pi, rho
    W_h = Vv * V2
    wh = fd.Function(W_h)
    n = fd.FacetNormal(mesh)
    dv, drho = fd.TestFunctions(W_h)

    v, Pi0 = fd.TrialFunctions(W_h)

    Pieqn = (
        gas.cp*(fd.inner(v, dv) - fd.div(dv*thetan)*Pi0)*fd.dx
        + drho*fd.div(thetan*v)*fd.dx
    )

    if top:
        bmeasure = fd.ds_t
        bstring = "bottom"
    else:
        bmeasure = fd.ds_b
        bstring = "top"

    zeros = []
    for i in range(Up.ufl_shape[0]):
        zeros.append(fd.Constant(0.))

    L = -gas.cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
    L -= gas.g*fd.inner(dv, Up)*fd.dx
    bcs = [fd.DirichletBC(W_h.sub(0), zeros, bstring)]

    PiProblem = fd.LinearVariationalProblem(Pieqn, L, wh, bcs=bcs)

    lu_params = {
        'snes_stol': 1e-10,
        'mat_type': 'aij',
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_ordering_type": "rcm",
        "pc_factor_mat_solver_type": "mumps",
    }

    verbose_options = {
        'snes_monitor': None,
        'snes_converged_reason': None,
        'ksp_monitor': None,
        'ksp_converged_reason': None,
    }

    if verbose == 0:
        pass
    elif verbose == 1:
        lu_params['snes_converged_reason'] = None
    elif verbose >= 2:
        lu_params.update(verbose_options)

    PiSolver = fd.LinearVariationalSolver(PiProblem,
                                          solver_parameters=lu_params,
                                          options_prefix="pisolver")
    PiSolver.solve()
    v = wh.subfunctions[0]
    Pi0 = wh.subfunctions[1]
    if Pi:
        Pi.assign(Pi0)

    if rhon:
        rhon.interpolate(rho_formula(Pi0, thetan, gas))
        v = wh.subfunctions[0]
        rho = wh.subfunctions[1]
        rho.assign(rhon)
        v, rho = fd.split(wh)

        Pif = pi_formula(rho, thetan, gas)

        rhoeqn = gas.cp*(
            (fd.inner(v, dv) - fd.div(dv*thetan)*Pif)*fd.dx
            + drho*fd.div(thetan*v)*fd.dx
        )

        if top:
            bmeasure = fd.ds_t
            bstring = "bottom"
        else:
            bmeasure = fd.ds_b
            bstring = "top"

        zeros = []
        for i in range(Up.ufl_shape[0]):
            zeros.append(fd.Constant(0.))

        rhoeqn += gas.cp*fd.inner(dv, n)*thetan*pi_boundary*bmeasure
        rhoeqn += gas.g*fd.inner(dv, Up)*fd.dx
        bcs = [fd.DirichletBC(W_h.sub(0), zeros, bstring)]

        RhoProblem = fd.NonlinearVariationalProblem(rhoeqn, wh, bcs=bcs)

        RhoSolver = fd.NonlinearVariationalSolver(RhoProblem,
                                                  solver_parameters=lu_params,
                                                  options_prefix="rhosolver")

        RhoSolver.solve()
        v = wh.subfunctions[0]
        Rho0 = wh.subfunctions[1]
        rhon.assign(Rho0)
        del RhoSolver
    del PiSolver
    import gc
    gc.collect()
    PETSc.garbage_cleanup(mesh._comm)


def theta_tendency(q, u, theta, n, Up, c_pen):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))

    # the basic consistent equation with horizontal upwinding
    eqn = (
        q*fd.inner(u, fd.grad(theta))*fd.dx
        + fd.jump(q)*(unn('+')*theta('+')
                      - unn('-')*theta('-'))*fd.dS_v
        - fd.jump(q*u*theta, n)*fd.dS_v
    )
    # jump stabilisation
    mesh = u.ufl_domain()
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)

    eqn += (
        h**2*c_pen*abs(fd.inner(u('+'), n('+')))
        * fd.inner(fd.jump(fd.grad(theta)),
                   fd.jump(fd.grad(q)))*(fd.dS_v+fd.dS_h))
    return eqn


def theta_mass(q, theta):
    return q*theta*fd.dx


def rho_mass(q, rho):
    return q*rho*fd.dx


def rho_tendency(q, rho, u, n):
    unn = 0.5*(fd.inner(u, n) + abs(fd.inner(u, n)))
    return (
        - fd.inner(fd.grad(q), u*rho)*fd.dx
        + fd.jump(q)*(unn('+')*rho('+')
                      - unn('-')*rho('-'))*(fd.dS_v + fd.dS_h)
    )


def u_mass(u, w):
    return fd.inner(u, w)*fd.dx


def both(u):
    return 2*fd.avg(u)


def u_tendency(w, n, u, theta, rho,
               cp, g, R_d, p_0, kappa, Up,
               mu=None, f=None, F=None):
    """
    Written in a dimension agnostic way
    """
    Pi = pi_formula(rho, theta, R_d=R_d, p_0=p_0, kappa=kappa)

    K = fd.Constant(0.5)*fd.inner(u, u)
    Upwind = 0.5*(fd.sign(fd.dot(u, n))+1)

    eqn = (
        + fd.inner(u, curl0(cross1(u, w)))*fd.dx
        - fd.inner(both(Upwind*u),
                   both(cross0(n, cross1(u, w))))*(fd.dS_h + fd.dS_v)
        - fd.div(w)*K*fd.dx
        - cp*fd.div(theta*w)*Pi*fd.dx(degree=6)
        + cp*fd.jump(w*theta, n)*fd.avg(Pi)*fd.dS_v(degree=6)
        + fd.inner(w, Up)*g*fd.dx
    )

    if mu:  # Newtonian dissipation in vertical
        eqn += mu*fd.inner(w, Up)*fd.inner(u, Up)*fd.dx
    if f:  # Coriolis term
        eqn += f*fd.inner(w, fd.cross(Up, u))*fd.dx
    if F:  # additional source term
        eqn += fd.inner(w, F)*fd.dx
    return eqn


def get_form_mass():
    def form_mass(u, rho, theta, du, drho, dtheta):
        return u_mass(u, du) + rho_mass(rho, drho) + theta_mass(theta, dtheta)
    return form_mass


def get_form_function(n, Up, c_pen,
                      cp, g, R_d, p_0, kappa, mu,
                      f=None, F=None,
                      viscosity=None, diffusivity=None):
    def form_function(u, rho, theta, du, drho, dtheta, t):
        eqn = theta_tendency(dtheta, u, theta, n, Up, c_pen)
        eqn += rho_tendency(drho, rho, u, n)
        eqn += u_tendency(du, n, u, theta, rho,
                          cp, g, R_d, p_0, kappa, Up, mu, f, F)
        if viscosity:
            eqn += form_viscosity(u, du, viscosity)
        if diffusivity:
            eqn += form_viscosity(theta, dtheta, diffusivity)
        return eqn
    return form_function


def form_viscosity(u, v, kappa, mu=fd.Constant(10.0)):
    mesh = v.ufl_domain()
    n = fd.FacetNormal(mesh)
    a = fd.inner(fd.grad(u), fd.grad(v))*fd.dx
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)

    def get_flux_form(dS):
        fluxes = (-fd.inner(2*fd.avg(fd.outer(v, n)), fd.avg(fd.grad(u)))
                  - fd.inner(fd.avg(fd.grad(v)), 2*fd.avg(fd.outer(u, n)))
                  + mu*h*fd.inner(2*fd.avg(fd.outer(v, n)),
                                  2*fd.avg(fd.outer(u, n))))*dS
        return fluxes

    a += kappa*get_flux_form(fd.dS_v)
    a += kappa*get_flux_form(fd.dS_h)
    return a
