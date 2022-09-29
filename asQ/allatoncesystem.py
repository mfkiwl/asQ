import firedrake as fd
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from functools import reduce
from operator import mul


class JacobianMatrix(object):
    def __init__(self, aaos):
        r"""
        Python matrix for the Jacobian of the all at once system
        :param aaos: The AllAtOnceSystem object
        """
        self.aaos = aaos
        self.u = fd.Function(self.aaos.function_space_all)  # for the input function
        self.F = fd.Function(self.aaos.function_space_all)  # for the output residual
        self.F_prev = fd.Function(self.aaos.function_space_all)  # Where we compute the
        # part of the output residual from neighbouring contributions
        self.u0 = fd.Function(self.aaos.function_space_all)  # Where we keep the state

        self.Fsingle = fd.Function(self.aaos.function_space)
        self.urecv = fd.Function(self.aaos.function_space)  # will contain the previous time value i.e. 3*r-1
        self.ualls = self.u.split()
        # Jform missing contributions from the previous step
        # Find u1 s.t. F[u1, u2, u3; v] = 0 for all v
        # definition:
        # dF_{u1}[u1, u2, u3; delta_u, v] =
        #  lim_{eps -> 0} (F[u1+eps*delta_u,u2,u3;v]
        #                  - F[u1,u2,u3;v])/eps
        # Newton, solves for delta_u such that
        # dF_{u1}[u1, u2, u3; delta_u, v] = -F[u1,u2,u3; v], for all v
        # then updates u1 += delta_u
        self.Jform = fd.derivative(self.aaos.aao_form, self.aaos.w_all)
        # Jform contributions from the previous step
        self.Jform_prev = fd.derivative(self.aaos.aao_form,
                                        self.aaos.w_recv)

    @PETSc.Log.EventDecorator()
    def mult(self, mat, X, Y):

        self.aaos.update(X, wall=self.u, wrecv=self.urecv, blocking=True)

        # Set the flag for the circulant option
        if self.aaos.circ in ["quasi", "picard"]:
            self.aaos.Circ.assign(1.0)
        else:
            self.aaos.Circ.assign(0.0)

        # assembly stage
        fd.assemble(fd.action(self.Jform, self.u), tensor=self.F)
        fd.assemble(fd.action(self.Jform_prev, self.urecv),
                    tensor=self.F_prev)
        self.F += self.F_prev

        # unset flag if alpha-circulant approximation only in Jacobian
        if self.aaos.circ not in ["picard"]:
            self.aaos.Circ.assign(0.0)

        # Apply boundary conditions
        # assumes aaos.w_all contains the current state we are
        # interested in
        # For Jacobian action we should just return the values in X
        # at boundary nodes
        for bc in self.aaos.boundary_conditions_all:
            bc.homogenize()
            bc.apply(self.F, u=self.u)
            bc.restore()

        with self.F.dat.vec_ro as v:
            v.copy(Y)


class AllAtOnceSystem(object):
    def __init__(self,
                 ensemble, time_partition,
                 dt, theta,
                 form_mass, form_function,
                 w0, bcs=[],
                 circ="", alpha=1e-3):
        """
        The all-at-once system representing multiple timesteps of a time-dependent finite-element problem.

        :arg ensemble: time-parallel ensemble communicator.
        :arg time_partition: a list of integers for the number of timesteps stored on each ensemble rank.
        :arg w0: a Function containing the initial data.
        :arg bcs: a list of DirichletBC boundary conditions on w0.function_space.
        """

        # check that the ensemble communicator is set up correctly
        if isinstance(time_partition, int):
            time_partition = [time_partition]
        nsteps = len(time_partition)
        ensemble_size = ensemble.ensemble_comm.size
        if nsteps != ensemble_size:
            raise ValueError(f"Number of timesteps {nsteps} must equal size of ensemble communicator {ensemble_size}")

        self.ensemble = ensemble
        self.time_partition = time_partition
        self.time_rank = ensemble.ensemble_comm.rank
        self.nlocal_timesteps = self.time_partition[self.time_rank]

        self.initial_condition = w0
        self.function_space = w0.function_space()
        self.boundary_conditions = bcs
        self.ncomponents = len(self.function_space.split())

        self.dt = dt
        self.theta = theta

        self.form_mass = form_mass
        self.form_function = form_function

        self.circ = circ
        self.alpha = alpha
        self.Circ = fd.Constant(0.0)

        self.max_indices = {
            'component': self.ncomponents,
            'slice': self.nlocal_timesteps,
            'window': sum(self.time_partition)
        }

        # function pace for the slice of the all-at-once system on this process
        self.function_space_all = reduce(mul, (self.function_space
                                               for _ in range(self.nlocal_timesteps)))

        self.w_all = fd.Function(self.function_space_all)
        self.w_alls = self.w_all.split()

        for i in range(self.nlocal_timesteps):
            self.set_field(i, self.initial_condition, index_range='slice')

        self.boundary_conditions_all = self.set_boundary_conditions(bcs)

        for bc in self.boundary_conditions_all:
            bc.apply(self.w_all)

        # function to assemble the nonlinear residual
        self.F_all = fd.Function(self.function_space_all)

        # functions containing the last and next steps for parallel
        # communication timestep
        # from the previous iteration
        self.w_recv = fd.Function(self.function_space)
        self.w_send = fd.Function(self.function_space)

        self._set_aao_form()
        self.jacobian = JacobianMatrix(self)

    def set_boundary_conditions(self, bcs):
        """
        Set the boundary conditions onto each solution in the all-at-once system
        """
        is_mixed_element = isinstance(self.function_space.ufl_element(), fd.MixedElement)

        bcs_all = []
        for bc in bcs:
            for step in range(self.nlocal_timesteps):
                if is_mixed_element:
                    i = bc.function_space().index
                    index = step*self.ncomponents + i
                else:
                    index = step
                bc_all = fd.DirichletBC(self.function_space_all.sub(index),
                                        bc.function_arg,
                                        bc.sub_domain)
                bcs_all.append(bc_all)

        return bcs_all

    def check_index(self, i, index_range='slice'):
        '''
        Check that timestep index is in range
        :arg i: timestep index to check
        :arg index_range: range that index is in. Either slice or window or component
        '''
        # set valid range
        if index_range not in self.max_indices.keys():
            raise ValueError("index_range must be one of "+" or ".join(self.max_indices.keys()))

        maxidx = self.max_indices[index_range]

        # allow for pythonic negative indices
        minidx = -maxidx

        if not (minidx <= i < maxidx):
            raise ValueError(f"index {i} outside {index_range} range {maxidx}")

    def shift_index(self, i, from_range='slice', to_range='slice'):
        '''
        Shift timestep index from one range to another, and accounts for -ve indices
        :arg i: timestep index to shift
        :arg from_range: range of i. Either slice or window
        :arg to_range: range to shift i to. Either slice or window
        '''
        if from_range == 'component' or to_range == 'component':
            raise ValueError('Component indices cannot be shifted')

        self.check_index(i, index_range=from_range)

        # deal with -ve indices
        i = i % self.max_indices[from_range]

        # no shift needed
        if to_range == from_range:
            return i

        # index of first timestep in slice
        index0 = sum(self.time_partition[:self.time_rank])

        if to_range == 'slice':  # 'from_range' == 'window'
            i -= index0

        if to_range == 'window':  # 'from_range' == 'slice'
            i += index0

        self.check_index(i, index_range=to_range)

        return i

    @PETSc.Log.EventDecorator()
    def set_component(self, step, cpt, wnew, index_range='slice', f_alls=None):
        '''
        Set component of solution at a timestep to new value

        :arg step: index of timestep
        :arg cpt: index of component
        :arg wout: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg f_alls: an all-at-once function to set timestep in. If None, self.w_alls is used
        '''
        step_local = self.shift_index(step, from_range=index_range, to_range='slice')
        self.check_index(cpt, index_range='component')

        if f_alls is None:
            f_alls = self.w_alls

        # index of first component of this step
        index0 = self.ncomponents*step_local

        f_alls[index0 + cpt].assign(wnew)

    @PETSc.Log.EventDecorator()
    def get_component(self, step, cpt, index_range='slice', wout=None, name=None, f_alls=None, deepcopy=False):
        '''
        Get component of solution at a timestep

        :arg step: index of timestep to get
        :arg cpt: index of component
        :arg index_range: is index in window or slice?
        :arg wout: function to set to component (component returned if None)
        :arg name: name of returned function if deepcopy=True. Ignored if wout is not None
        :arg f_alls: an all-at-once function to get timestep from. If None, self.w_alls is used
        :arg deepcopy: if True, new function is returned. If false, handle to component of f_alls is returned. Ignored if wout is not None
        '''
        step_local = self.shift_index(step, from_range=index_range, to_range='slice')
        self.check_index(cpt, index_range='component')

        if f_alls is None:
            f_alls = self.w_alls

        # index of first component of this step
        index0 = self.ncomponents*step_local

        # required component
        wget = f_alls[index0 + cpt]

        if wout is not None:
            wout.assign(wget)
            return wout

        if deepcopy is False:
            return wget
        else:  # deepcopy is True
            wreturn = fd.Function(self.function_space.sub(cpt), name=name)
            wreturn.assign(wget)
            return wreturn

    @PETSc.Log.EventDecorator()
    def set_field(self, step, wnew, index_range='slice', f_alls=None):
        '''
        Set solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg wnew: new solution for timestep
        :arg index_range: is index in window or slice?
        :arg f_alls: an all-at-once function to set timestep in. If None, self.w_alls is used
        '''
        for cpt in range(self.ncomponents):
            self.set_component(step, cpt, wnew.sub(cpt),
                               index_range=index_range, f_alls=f_alls)

    @PETSc.Log.EventDecorator()
    def get_field(self, step, index_range='slice', wout=None, name=None, f_alls=None):
        '''
        Get solution at a timestep to new value

        :arg step: index of timestep to set.
        :arg index_range: is index in window or slice?
        :arg wout: function to set to timestep (timestep returned if None)
        :arg name: name of returned function. Ignored if wout is not None
        :arg f_alls: an all-at-once function to get timestep from. If None, self.w_alls is used
        '''
        if wout is None:
            wget = fd.Function(self.function_space, name=name)
        else:
            wget = wout

        for cpt in range(self.ncomponents):
            wcpt = self.get_component(step, cpt, index_range=index_range, f_alls=f_alls)
            wget.sub(cpt).assign(wcpt)

        return wget

    @PETSc.Log.EventDecorator()
    def for_each_timestep(self, callback):
        '''
        call callback for each timestep in each slice in the current window
        callback arguments are: timestep index in window, timestep index in slice, Function at timestep

        :arg callback: the function to call for each timestep
        '''

        w = fd.Function(self.function_space)
        for slice_index in range(self.nlocal_timesteps):
            window_index = self.shift_index(slice_index,
                                            from_range='slice',
                                            to_range='window')
            self.get_field(slice_index, wout=w, index_range='slice')
            callback(window_index, slice_index, w)

    @PETSc.Log.EventDecorator()
    def next_window(self, w1=None):
        """
        Reset all-at-once-system ready for next time-window

        :arg w1: initial solution for next time-window.If None,
                 will use the final timestep from previous window
        """
        rank = self.time_rank
        ncomm = self.ensemble.ensemble_comm.size

        if w1 is not None:  # use given function
            self.initial_condition.assign(w1)
        else:  # last rank broadcasts final timestep
            if rank == ncomm-1:
                # index of start of final timestep
                self.get_field(-1, wout=self.initial_condition, index_range='slice')

            with self.initial_condition.dat.vec as vec:
                self.ensemble.ensemble_comm.Bcast(vec.array, root=ncomm-1)

        # persistence forecast
        for i in range(self.nlocal_timesteps):
            self.set_field(i, self.initial_condition, index_range='slice')

        return

    @PETSc.Log.EventDecorator()
    def update_time_halos(self, wsend=None, wrecv=None, walls=None, blocking=True):
        '''
        Update wrecv with the last step from the previous slice (periodic) of walls

        :arg wsend: Function to send last step of current slice to next slice. if None self.w_send is used
        :arg wrecv: Function to receive last step of previous slice. if None self.w_recv is used
        :arg walls: all at once function list to update wrecv from. if None self.w_alls is used
        :arg blocking: Whether to blocking until MPI communications have finished. If false then a list of MPI requests is returned
        '''
        n = self.ensemble.ensemble_comm.size
        r = self.time_rank

        if wsend is None:
            wsend = self.w_send
        if wrecv is None:
            wrecv = self.w_recv
        if walls is None:
            walls = self.w_alls

        # Communication stage
        mpi_requests = []

        self.get_field(-1, wout=wsend, index_range='slice', f_alls=walls)

        # these should be replaced with isendrecv once ensemble updates are pushed to Firedrake
        request_send = self.ensemble.isend(wsend, dest=((r+1) % n), tag=r)
        mpi_requests.extend(request_send)

        request_recv = self.ensemble.irecv(wrecv, source=((r-1) % n), tag=r-1)
        mpi_requests.extend(request_recv)

        if blocking:
            # wait for the data [we should really do this after internal
            # assembly but have avoided that for now]
            MPI.Request.Waitall(mpi_requests)
            return
        else:
            return mpi_requests

    @PETSc.Log.EventDecorator()
    def update(self, X, wall=None, wsend=None, wrecv=None, blocking=True):
        '''
        Update self.w_alls and self.w_recv from PETSc Vec X.
        The local parts of X are copied into self.w_alls
        and the last step from the previous slice (periodic)
        is copied into self.u_prev
        '''
        if wall is None:
            wall = self.w_all
        if wsend is None:
            wsend = self.w_send
        if wrecv is None:
            wrecv = self.w_recv

        with wall.dat.vec_wo as v:
            v.array[:] = X.array_r

        return self.update_time_halos(wsend=wsend, wrecv=wrecv, walls=wall.split(), blocking=True)

    @PETSc.Log.EventDecorator()
    def _assemble_function(self, snes, X, Fvec):
        r"""
        This is the function we pass to the snes to assemble
        the nonlinear residual.
        """
        self.update(X)

        # Set the flag for the circulant option
        if self.circ == "picard":
            self.Circ.assign(1.0)
        else:
            self.Circ.assign(0.0)
        # assembly stage
        fd.assemble(self.aao_form, tensor=self.F_all)

        # apply boundary conditions
        for bc in self.boundary_conditions_all:
            bc.apply(self.F_all, u=self.w_all)

        with self.F_all.dat.vec_ro as v:
            v.copy(Fvec)

    def _set_aao_form(self):
        """
        Constructs the bilinear form for the all at once system.
        Specific to the theta-centred Crank-Nicholson method
        """

        w_alls = fd.split(self.w_all)
        test_fns = fd.TestFunctions(self.function_space_all)

        dt = fd.Constant(self.dt)
        theta = fd.Constant(self.theta)
        alpha = fd.Constant(self.alpha)

        def get_cpts(i, buf):
            return [self.get_component(i, cpt, f_alls=buf)
                    for cpt in range(self.ncomponents)]

        def get_step(i):
            return get_cpts(i, w_alls)

        def get_test(i):
            return get_cpts(i, test_fns)

        for n in range(self.nlocal_timesteps):

            # previous time level
            if n == 0:
                if self.time_rank == 0:
                    # need the initial data
                    w0list = fd.split(self.initial_condition)

                    # circulant option for quasi-Jacobian
                    wrecvlist = fd.split(self.w_recv)

                    w0s = [w0list[i] + self.Circ*alpha*wrecvlist[i]
                           for i in range(self.ncomponents)]
                else:
                    # self.w_recv will contain the data from the previous slice
                    w0s = fd.split(self.w_recv)
            else:
                w0s = get_step(n-1)

            # current time level
            w1s = get_step(n)
            dws = get_test(n)

            # time derivative
            if n == 0:
                aao_form = (1.0/dt)*self.form_mass(*w1s, *dws)
            else:
                aao_form += (1.0/dt)*self.form_mass(*w1s, *dws)
            aao_form -= (1.0/dt)*self.form_mass(*w0s, *dws)

            # vector field
            aao_form += theta*self.form_function(*w1s, *dws)
            aao_form += (1-theta)*self.form_function(*w0s, *dws)

        self.aao_form = aao_form