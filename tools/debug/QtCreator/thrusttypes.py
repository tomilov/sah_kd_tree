from dumper import Children
from utils import DisplayFormat


def qform__thrust__device_vector():
    return [DisplayFormat.ArrayPlot]


def qdump__thrust__device_vector(d, value):
    size = value['m_size'].integer()
    d.putItemCount(size)

    start = value['m_storage']['m_begin']['m_iterator']['m_iterator'].pointer()
    d.putPlotData(start, size, value.type[0])


def qdump__thrust__tuple(d, value):
    #d.putBetterType(value.type.stripTypedefs())
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
