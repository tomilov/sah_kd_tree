from dumper import DumperBase, Children
from utils import DisplayFormat


def _dump_pointer(d, value, count=1, is_array=False):
    try:
        if value.type.name.startswith('thrust::device_ptr'):
            value = value.members(includeBases=True)[0]
        d.check(value.type.name.startswith('thrust::pointer'))
        iterator = value['m_iterator']
        innerType = value.type[0]
        size = innerType.size() * count
        def _dump(p):
            if is_array:
                d.putItemCount(count)
                d.putPlotData(p, count, innerType)
            else:
                d.putItem(d.createValue(p, innerType))
        p = iterator.pointer()
        if p == 0 or count == 0:
            d.putEmptyValue()
        elif value.type[1].name == 'thrust::cuda_cub::tag':
            h = d.parseAndEvaluate(f'(void *)malloc({size})').pointer()
            d.parseAndEvaluate(f'(void)cudaMemcpy({h}, {p}, {size}, cudaMemcpyDeviceToHost)')
            _dump(h)
            d.parseAndEvaluate(f'(void)free({h})')
        else:
            _dump(p)
    except Exception as e:
        import traceback
        DumperBase.warn(f'ERROR: {e} {traceback.format_exc()}')


def qdump__thrust__tuple(d, value):
    d.putExpandable()
    if d.isExpanded():
        with Children(d):
            child = 0
            while True:
                head = value['head']
                d.putSubItem(f'<{child}> ({head.type.size()} bytes)', head)
                if value.type[1].name == 'thrust::null_type':
                    break
                child += 1
                value = value['tail']


def qdump__thrust__pointer(d, value):
    _dump_pointer(d, value)
    d.putBetterType(value.type)


def qdump__thrust__device_ptr(d, value):
    _dump_pointer(d, value)
    d.putBetterType(value.type)


def qdump__thrust__reference(d, value):
    _dump_pointer(d, value['ptr'])
    d.putBetterType(value.type)


def qdump__thrust__tagged_reference(d, value):
    _dump_pointer(d, value['ptr'])
    d.putBetterType(value.type)


def qdump__thrust__cuda_cub__reference(d, value):
    _dump_pointer(d, value['ptr'])
    d.putBetterType(value.type)


def qdump__thrust__device_reference(d, value):
    _dump_pointer(d, value['ptr'])
    d.putBetterType(value.type)


def qform__thrust__device_vector():
    return [DisplayFormat.ArrayPlot]


def qdump__thrust__device_vector(d, value):
    count = value['m_size'].integer()
    iterator = value['m_storage']['m_begin']['m_iterator']
    _dump_pointer(d, iterator, count, True)
    d.putBetterType(value.type)


def qform__thrust__host_vector():
    return [DisplayFormat.ArrayPlot]


def qdump__thrust__host_vector(d, value):
    count = value['m_size'].integer()
    p = value['m_storage']['m_begin']['m_iterator'].pointer()
    d.putItemCount(count)
    d.putPlotData(p, count, value.type[0])
    d.putBetterType(value.type)


def qdump__thrust__cuda_cub__vector(d, value):
    return qdump__thrust__device_vector(d, value)


def qdump__thrust__system__omp__vector(d, value):
    return qdump__thrust__device_vector(d, value)


def qdump__thrust__system__tbb__vector(d, value):
    return qdump__thrust__device_vector(d, value)


def qdump__thrust__system__cpp__vector(d, value):
    return qdump__thrust__device_vector(d, value)


def qdump__thrust__zip_iterator(d, value):
    d.putItem(value['m_iterator_tuple'])
    d.putBetterType(value.type)


def qdump__thrust__constant_iterator(d, value):
    d.putItem(value['m_value'])
    d.putBetterType(value.type)


def qdump__thrust__discard_iterator(d, value):
    d.putItem(value['m_iterator']['m_iterator'])
    d.putBetterType(value.type)


def qdump__thrust__detail__normal_iterator(d, value):
    _dump_pointer(d, value['m_iterator'])
    d.putBetterType(value.type)


def qdump__thrust__counting_iterator(d, value):
    d.putItem(value['m_iterator'])
    d.putBetterType(value.type)


def qdump__thrust__reverse_iterator(d, value):
    d.putItem(value['m_iterator'])
    d.putBetterType(value.type)


def qdump__thrust__transform_input_output_iterator(d, value):
    d.putItem(value['m_iterator'])
    d.putBetterType(value.type)


def qdump__thrust__transform_iterator(d, value):
    d.putItem(value['m_iterator'])
    d.putBetterType(value.type)


def qdump__thrust__transform_output_iterator(d, value):
    d.putItem(value['m_iterator'])
    d.putBetterType(value.type)


def qdump__thrust__complex(d, value):
    d.putItem(value['data'])
    d.putBetterType(value.type)
