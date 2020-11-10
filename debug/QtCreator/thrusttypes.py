import dumper


def qdump__thrust__device_vector(d, value):
    innerType = value.type[0]

    size = int(value["m_size"])
    start = value["m_storage"]["m_begin"]["m_iterator"]["m_iterator"].pointer()

    d.putItemCount(size)
    d.putPlotData(start, size, innerType)
