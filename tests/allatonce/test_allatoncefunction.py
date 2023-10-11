import asQ
import firedrake as fd
import pytest
import numpy as np
from functools import reduce
from operator import mul


def random_aaof(aaof):
    for dat in aaof.initial_condition.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))
    for dat in aaof.function.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))
    aaof.update_time_halos()


nprocs = 4


max_ncpts = 3
ncpts = [pytest.param(i, id=f"{i}component") for i in range(1, max_ncpts+1)]


@pytest.fixture
def slice_length():
    if fd.COMM_WORLD.size == 1:
        return
    return 3


@pytest.fixture
def nspatial_ranks():
    if fd.COMM_WORLD.size == 1:
        return
    return 2


@pytest.fixture
def time_partition(slice_length, nspatial_ranks):
    if fd.COMM_WORLD.size == 1:
        return
    assert (fd.COMM_WORLD.size % nspatial_ranks == 0)
    nslices = fd.COMM_WORLD.size//nspatial_ranks
    return tuple((slice_length for _ in range(nslices)))


@pytest.fixture
def ensemble():
    if fd.COMM_WORLD.size == 1:
        return

    return fd.Ensemble(fd.COMM_WORLD, 2)


@pytest.fixture
def mesh(ensemble):
    if fd.COMM_WORLD.size == 1:
        return

    return fd.UnitSquareMesh(4, 4, comm=ensemble.comm)


@pytest.fixture
def V(mesh):
    if fd.COMM_WORLD.size == 1:
        return

    return fd.FunctionSpace(mesh, "CG", 1)


# mixed space
@pytest.fixture(params=ncpts)
def W(request, V):
    if fd.COMM_WORLD.size == 1:
        return

    return reduce(mul, [V for _ in range(request.param)])


@pytest.fixture()
def aaof(ensemble, time_partition, W):
    if fd.COMM_WORLD.size == 1:
        return
    return asQ.AllAtOnceFunction(ensemble, time_partition, W)


@pytest.mark.parallel(nprocs=nprocs)
def test_transform_index(aaof):
    '''
    test transforming between window, slice, and component indexes
    '''
    ncpts = aaof.ncomponents
    window_length = aaof.ntimesteps
    time_rank = aaof.time_rank
    local_timesteps = aaof.nlocal_timesteps
    nslices = len(aaof.time_partition)

    max_indices = {
        'slice': aaof.nlocal_timesteps,
        'window': aaof.ntimesteps
    }

    # from_range == to_range
    for index_type in max_indices.keys():

        max_index = max_indices[index_type]
        index = max_index - 1

        # +ve index unchanged
        pos_shift = aaof.transform_index(index, from_range=index_type, to_range=index_type)
        assert (pos_shift == index)

        # -ve index changed to +ve
        neg_shift = aaof.transform_index(-index, from_range=index_type, to_range=index_type)
        assert (neg_shift == max_index - index)

    # slice_range -> window_range

    slice_index = 1

    # +ve index in slice range
    window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')
    assert (window_index == time_rank*local_timesteps + slice_index)

    # -ve index in slice range
    window_index = aaof.transform_index(-slice_index, from_range='slice', to_range='window')
    assert (window_index == (time_rank+1)*local_timesteps - slice_index)

    slice_index = local_timesteps + 1

    # +ve index out of slice range
    with pytest.raises(IndexError):
        window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')

    # -ve index out of slice range
    with pytest.raises(IndexError):
        window_index = aaof.transform_index(-slice_index, from_range='slice', to_range='window')

    # window_range -> slice_range

    # +ve index in range
    window_index = time_rank*local_timesteps + 1
    slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    assert (slice_index == 1)

    # -ve index in range
    window_index = -window_length + time_rank*local_timesteps + 1
    slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    assert (slice_index == 1)

    # +ve index out of slice range
    window_index = ((time_rank + 1) % nslices)*local_timesteps + 1
    # for some reason pytest.raises doesn't work with this call
    try:
        slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    except IndexError:
        pass

    # -ve index out of slice range
    window_index = -window_index
    # for some reason pytest.raises doesn't work with this call
    try:
        slice_index = aaof.transform_index(-window_index, from_range='window', to_range='slice')
    except IndexError:
        pass

    # component indices

    # +ve slice and +ve component indices
    slice_index = 1
    cpt_index = ncpts-1
    aao_index = aaof.transform_index(slice_index, cpt_index, from_range='slice')
    check_index = ncpts*slice_index + cpt_index
    assert (aao_index == check_index)

    # +ve window and -ve component indices
    window_index = time_rank*local_timesteps + 1
    slice_index = aaof.transform_index(window_index, from_range='window', to_range='slice')
    cpt_index = -1
    aao_index = aaof.transform_index(window_index, cpt_index, from_range='window')
    check_index = ncpts*(slice_index+1) + cpt_index

    # reject component index out of range
    with pytest.raises(IndexError):
        aaof.transform_index(0, 100, from_range='slice')
    with pytest.raises(IndexError):
        window_index = aaof.transform_index(0, from_range='slice', to_range='window')
        aaof.transform_index(window_index, -100, from_range='window')


@pytest.mark.parallel(nprocs=nprocs)
def test_subfunctions(aaof):
    '''
    test setting the allatonce function values from views over each timestep
    '''
    np.random.seed(572046)
    aaof.zero()

    def randu(u):
        for dat in u.dat:
            dat.data[:] = np.random.rand(*(dat.data.shape))

    u = fd.Function(aaof.field_function_space)

    cpt_idx = 0
    for i in range(aaof.nlocal_timesteps):

        randu(u)
        aaof[i].assign(u)

        for c in range(aaof.ncomponents):
            err = fd.errornorm(u.subfunctions[c],
                               aaof.function.subfunctions[cpt_idx])
            assert (err < 1e-12)
            cpt_idx += 1

    aaof.zero()

    cpt_idx = 0
    for i in range(aaof.nlocal_timesteps):
        norm = fd.norm(aaof[i])
        assert (norm < 1e-12)

        for c in range(aaof.ncomponents):
            aaof.function.subfunctions[cpt_idx].assign(cpt_idx)

            err = fd.errornorm(aaof[i].subfunctions[c],
                               aaof.function.subfunctions[cpt_idx])
            assert (err < 1e-12)
            cpt_idx += 1


@pytest.mark.parallel(nprocs=nprocs)
def test_set_get_component(aaof):
    '''
    test setting a specific component of a timestep
    '''
    ncpts = aaof.ncomponents

    vc = fd.Function(aaof.field_function_space, name="vc")
    v0 = fd.Function(aaof.field_function_space, name="v0")
    v1 = fd.Function(aaof.field_function_space, name="v1")

    # two random solutions
    np.random.seed(572046)
    random_aaof(aaof)

    for slice_index in range(aaof.nlocal_timesteps):
        window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')

        for cpt in range(ncpts):
            for dat in v0.dat:
                dat.data[:] = np.random.rand(*(dat.data.shape))
            for dat in v1.dat:
                dat.data[:] = np.random.rand(*(dat.data.shape))

            # set component using slice index
            vc.assign(0)
            aaof.set_component(slice_index, cpt, v0.subfunctions[cpt], index_range='slice')
            aaof.get_component(slice_index, cpt, vc.subfunctions[cpt], index_range='slice')
            assert (fd.errornorm(v0.subfunctions[cpt], vc.subfunctions[cpt]) < 1e-12)

            # set component using window index
            vc.assign(0)
            aaof.set_component(window_index, cpt, v1.subfunctions[cpt], index_range='window')
            aaof.get_component(window_index, cpt, vc.subfunctions[cpt], index_range='window')
            assert (fd.errornorm(v1.subfunctions[cpt], vc.subfunctions[cpt]) < 1e-12)

            # get handle to component index in aaof.function
            vcpt = aaof.get_component(slice_index, cpt)
            for datcpt, dat0 in zip(vcpt.dat, v0.subfunctions[cpt].dat):
                datcpt.data[:] = dat0.data[:]
            aaof.get_component(slice_index, cpt, vc.subfunctions[cpt], index_range='slice')
            assert (fd.errornorm(v0.subfunctions[cpt], vc.subfunctions[cpt]) < 1e-12)


@pytest.mark.parallel(nprocs=nprocs)
def test_set_get_field(aaof):
    '''
    test getting a specific timestep
    only valid if test_set_field passes
    '''
    # prep aaof setup
    vc = fd.Function(aaof.field_function_space, name="vc")
    v0 = fd.Function(aaof.field_function_space, name="v1")
    v1 = fd.Function(aaof.field_function_space, name="v1")

    # two random solutions
    np.random.seed(572046)
    random_aaof(aaof)

    # get each step using slice index
    v = fd.Function(aaof.field_function_space)
    for step in range(aaof.nlocal_timesteps):
        windx = aaof.transform_index(step, from_range='slice', to_range='window')

        for dat in v0.dat:
            dat.data[:] = np.random.rand(*(dat.data.shape))
        for dat in v1.dat:
            dat.data[:] = np.random.rand(*(dat.data.shape))

        aaof.set_field(step, v0, index_range='slice')
        aaof.get_field(step, vc, index_range='slice')
        err = fd.errornorm(v0, vc)
        assert (err < 1e-12)

        aaof.set_field(step, v1, index_range='slice')
        aaof.get_field(step, v, index_range='slice')
        err = fd.errornorm(v1, v)
        assert (err < 1e-12)

        aaof.set_field(windx, v0, index_range='window')
        aaof.get_field(windx, vc, index_range='window')
        err = fd.errornorm(v0, vc)
        assert (err < 1e-12)

        aaof.set_field(windx, v1, index_range='window')
        aaof.get_field(windx, v, index_range='window')
        err = fd.errornorm(v1, v)
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=nprocs)
def test_bcast_field(aaof):
    '''
    test broadcasting the solution at a particular timestep to all ensemble ranks
    '''
    u = fd.Function(aaof.field_function_space, name="u")
    v = fd.Function(aaof.field_function_space, name="v")

    # solution at timestep i is i
    for slice_index in range(aaof.nlocal_timesteps):
        window_index = aaof.transform_index(slice_index, from_range='slice', to_range='window')
        v.assign(window_index)
        aaof[slice_index].assign(v)

    for i in range(aaof.ntimesteps):
        v.assign(-1)
        u.assign(i)
        aaof.bcast_field(i, v)
        assert (fd.errornorm(v, u) < 1e-12)


@pytest.mark.parallel(nprocs=nprocs)
def test_copy(aaof):
    """
    test setting all timesteps/ics to given function.
    """
    # two random solutions
    np.random.seed(572046)

    # set next window from new solution
    random_aaof(aaof)
    aaof1 = aaof.copy()

    # layout is the same
    assert aaof.nlocal_timesteps == aaof1.nlocal_timesteps
    assert aaof.ntimesteps == aaof1.ntimesteps

    # check all timesteps are equal

    assert aaof.function_space == aaof1.function_space

    for step in range(aaof.nlocal_timesteps):
        err = fd.errornorm(aaof[step], aaof1[step])
        assert (err < 1e-12)

    err = fd.errornorm(aaof.initial_condition, aaof1.initial_condition)
    assert (err < 1e-12)

    err = fd.errornorm(aaof.uprev, aaof1.uprev)
    assert (err < 1e-12)


@pytest.mark.parallel(nprocs=nprocs)
def test_assign(aaof):
    """
    test setting all timesteps/ics to given function.
    """
    v0 = fd.Function(aaof.field_function_space, name="v0")

    # two random solutions
    np.random.seed(572046)

    aaof1 = aaof.copy()

    # assign from another aaof
    random_aaof(aaof)
    aaof1.assign(aaof)

    for step in range(aaof.nlocal_timesteps):
        err = fd.errornorm(aaof[step], aaof1[step])
        assert (err < 1e-12)

    err = fd.errornorm(aaof.initial_condition,
                       aaof1.initial_condition)
    assert (err < 1e-12)

    err = fd.errornorm(aaof.uprev, aaof1.uprev)
    assert (err < 1e-12)

    # set from PETSc Vec
    random_aaof(aaof)
    with aaof.global_vec() as gvec0:
        aaof1.assign(gvec0)

    for step in range(aaof.nlocal_timesteps):
        err = fd.errornorm(aaof[step], aaof1[step])
        assert (err < 1e-12)

    err = fd.errornorm(aaof.uprev, aaof1.uprev)
    assert (err < 1e-12)

    # set from field function

    for dat in v0.dat:
        dat.data[:] = np.random.rand(*(dat.data.shape))

    aaof.assign(v0)

    for step in range(aaof.nlocal_timesteps):
        err = fd.errornorm(v0, aaof[step])
        assert (err < 1e-12)

    # set from allatonce.function
    random_aaof(aaof1)
    aaof.assign(aaof1.function)

    for step in range(aaof.nlocal_timesteps):
        err = fd.errornorm(aaof[step], aaof1[step])
        assert (err < 1e-12)


@pytest.mark.parallel(nprocs=nprocs)
def test_zero(aaof):
    """
    test setting all timesteps/ics to given function.
    """
    # two random solutions
    np.random.seed(572046)

    # assign from another aaof
    random_aaof(aaof)
    aaof.zero()

    norm = fd.norm(aaof.initial_condition)
    assert (norm < 1e-12)

    norm = fd.norm(aaof.uprev)
    assert (norm < 1e-12)

    norm = fd.norm(aaof.unext)
    assert (norm < 1e-12)

    for step in range(aaof.nlocal_timesteps):
        norm = fd.norm(aaof[step])
        assert (norm < 1e-12)


@pytest.mark.parallel(nprocs=nprocs)
def test_global_vec(aaof):
    """
    test synchronising the global Vec with the local Functions
    """
    v = fd.Function(aaof.field_function_space, name="v")

    def all_equal(func, val):
        v.assign(val)
        for step in range(func.nlocal_timesteps):
            err = fd.errornorm(v, func[step])
            assert (err < 1e-12)

    # read only
    aaof.initial_condition.assign(10)
    aaof.assign(aaof.initial_condition)

    with aaof.global_vec_ro() as rvec:
        assert np.allclose(rvec.array, 10)
        rvec.array[:] = 20

    all_equal(aaof, 10)

    # write only
    aaof.initial_condition.assign(30)
    aaof.assign(aaof.initial_condition)

    with aaof.global_vec_wo() as wvec:
        assert np.allclose(wvec.array, 20)
        wvec.array[:] = 40

    all_equal(aaof, 40)

    aaof.initial_condition.assign(50)
    aaof.assign(aaof.initial_condition)

    with aaof.global_vec() as vec:
        assert np.allclose(vec.array, 50)
        vec.array[:] = 60

    all_equal(aaof, 60)


@pytest.mark.parallel(nprocs=nprocs)
def test_update_time_halos(aaof):
    """
    test updating the time halo functions
    """
    v0 = fd.Function(aaof.field_function_space, name="v0")
    v1 = fd.Function(aaof.field_function_space, name="v1")

    # test updating own halos
    aaof.zero()

    rank = aaof.ensemble.ensemble_comm.rank
    size = aaof.ensemble.ensemble_comm.size

    # solution on this slice
    v0.assign(rank)
    # solution on previous slice
    v1.assign((rank - 1) % size)

    # set last field from each slice
    aaof[-1].assign(v0)

    aaof.update_time_halos()

    assert (fd.errornorm(aaof.uprev, v1) < 1e-12)
