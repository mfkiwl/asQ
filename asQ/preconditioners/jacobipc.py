import firedrake as fd
from firedrake.petsc import PETSc
from asQ.profiling import profiler
from asQ.parallel_arrays import SharedArray
from asQ.allatonce import time_average
from asQ.preconditioners.base import AllAtOnceBlockPCBase

__all__ = ['JacobiPC']


class JacobiPC(AllAtOnceBlockPCBase):
    """
    PETSc options:

    'aaojacobi_linearisation': <'consistent', 'user'>
        Which form to linearise when constructing the block Jacobians.
        Default is 'consistent'.

        'consistent': use the same form used in the AllAtOnceForm residual.
        'user': use the alternative forms given in the appctx.
            If this option is specified then the appctx must contain form_mass
            and form_function entries with keys 'pc_form_mass' and 'pc_form_function'.

    'aaojacobi_state': <'current', 'window', 'slice', 'linear', 'initial', 'reference', 'user'>
        Which state to linearise around when constructing the block Jacobians.
        Default is 'current'.

        'window': use the time average over the entire AllAtOnceFunction.
        'slice': use the time average over timesteps on the local Ensemble member.
        'linear': the form linearised is already linear, so no update to the state is needed.
        'initial': the initial condition of the AllAtOnceFunction is used for all timesteps.
        'reference': use the reference state of the AllAtOnceJacobian for all timesteps.

    'aaojacobi_block_%d': <LinearVariationalSolver options>
        The solver options for the %d'th block, enumerated globally.
        Use 'aaojacobi_block' to set options for all blocks.
        Default is the Firedrake default options.

    'aaojacobi_dt': <float>
        The timestep size to use in the preconditioning matrix.
        Defaults to the timestep size used in the AllAtOnceJacobian.

    'aaojacobi_theta': <float>
        The implicit theta method parameter to use in the preconditioning matrix.
        Defaults to the implicit theta method parameter used in the AllAtOnceJacobian.

    If the AllAtOnceSolver's appctx contains a 'block_appctx' dictionary, this is
    added to the appctx of each block solver.  The appctx of each block solver also
    contains the following:
        'blockid': index of the block.
        'dt': timestep of the block.
        'theta': implicit theta value of the block.
        'u0': state around which the block is linearised.
        't0': time at which the block is linearised.
        'bcs': block boundary conditions.
        'form_mass': function used to build the block mass matrix.
        'form_function': function used to build the block stiffness matrix.
    """
    prefix = "aaojacobi_"

    valid_jacobian_states = tuple(('current', 'window', 'slice', 'linear',
                                   'initial', 'reference', 'user'))

    @profiler()
    def __init__(self):
        r"""A block diagonal Jacobi preconditioner for all-at-once systems.
        """
        self.initialized = False

    @profiler()
    def setUp(self, pc):
        """Setup method called by PETSc."""
        if not self.initialized:
            self.initialize(pc)
        self.update(pc)

    @profiler()
    def initialize(self, pc):
        super().initialize(pc, final_initialize=False)

        aaofunc = self.aaofunc

        # all-at-once reference state
        self.state_func = aaofunc.copy()

        # single timestep function space
        self.blockV = aaofunc.field_function_space

        self.time = tuple(fd.Constant(0)
                          for _ in range(self.nlocal_timesteps))

        # Building the nonlinear operator
        self.block_solvers = []

        # zero out bc dofs
        self.block_bcs = tuple(
            fd.DirichletBC(self.blockV, 0*bc.function_arg, bc.sub_domain)
            for bc in self.aaoform.field_bcs)

        # user appctx for the blocks
        block_appctx = self.appctx.get('block_appctx', {})

        dt1 = fd.Constant(1/self.dt)
        tht = fd.Constant(self.theta)

        self.block_sol = aaofunc.copy(copy_values=False)

        # building the block problem solvers
        for i in range(self.nlocal_timesteps):

            # The block rhs is timestep i of the AllAtOnceCofunction
            L = self.x[i]

            # the reference states
            u0 = self.state_func[i]
            t0 = self.time[i]

            # the form
            vs = fd.TestFunctions(self.blockV)
            us = fd.split(u0)
            M = self.form_mass(*us, *vs)
            K = self.form_function(*us, *vs, t0)

            F = dt1*M + tht*K
            A = fd.derivative(F, u0)

            # pass parameters into PC:
            appctx_h = {
                "blockid": i,
                "dt": self.dt,
                "theta": self.theta,
                "t0": t0,
                "u0": u0,
                "bcs": self.block_bcs,
                "form_mass": self.form_mass,
                "form_function": self.form_function,
            }

            appctx_h.update(block_appctx)

            # Options with prefix 'aaojacobi_block_' apply to all blocks by default
            # If any options with prefix 'aaojacobi_block_{i}' exist, where i is the
            # block number, then this prefix is used instead (like pc fieldsplit)

            ii = aaofunc.transform_index(i, from_range='slice', to_range='window')

            block_prefix = f"{self.full_prefix}block_"
            for k, v in PETSc.Options().getAll().items():
                if k.startswith(f"{block_prefix}{str(ii)}_"):
                    block_prefix = f"{block_prefix}{str(ii)}_"
                    break

            block_problem = fd.LinearVariationalProblem(A, L, self.block_sol[i],
                                                        bcs=self.block_bcs)
            block_solver = fd.LinearVariationalSolver(block_problem,
                                                      appctx=appctx_h,
                                                      options_prefix=block_prefix)
            # multigrid transfer manager
            if f'{self.full_prefix}transfer_managers' in self.appctx:
                tm = self.appctx[f'{self.prefix}transfer_managers'][i]
                block_solver.set_transfer_manager(tm)

            self.block_solvers.append(block_solver)

        self.block_iterations = SharedArray(self.time_partition,
                                            dtype=int,
                                            comm=self.ensemble.ensemble_comm)
        self.initialized = True

    @profiler()
    def update(self, pc):
        """
        Update the state to linearise around according to aaojacobi_state.
        """

        aaofunc = self.aaofunc
        state_func = self.state_func
        jacobian_state = self.jacobian_state()

        if jacobian_state == 'linear':
            pass

        elif jacobian_state == 'current':
            state_func.assign(aaofunc)

        elif jacobian_state in ('window', 'slice'):
            time_average(aaofunc, state_func.initial_condition,
                         state_func.uprev, average=jacobian_state)
            state_func.assign(state_func.initial_condition)

        elif jacobian_state == 'initial':
            state_func.assign(aaofunc.initial_condition)

        elif jacobian_state == 'reference':
            aaofunc.assign(self.jacobian.reference_state)

        elif jacobian_state == 'user':
            pass

        return

    @profiler()
    def apply_impl(self, pc, x, y):

        # x is already rhs of block solvers
        self.block_sol.zero()
        for i in range(self.nlocal_timesteps):
            self.block_solvers[i].solve()

        # block_sol is aaofunc but y is aaocofunc so
        # we have to manually copy Vecs
        with self.block_sol.global_vec_ro as bvec:
            with y.global_vec_wo() as yvec:
                bvec.copy(yvec)

    @profiler()
    def applyTranspose(self, pc, x, y):
        raise NotImplementedError
