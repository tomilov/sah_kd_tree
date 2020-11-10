import dumper


def qdump__thrust__device_vector(d, value):
    innerType = value.type[0]

    value = value["m_storage"]
    start = value["m_begin"]["m_iterator"]["m_iterator"].pointer()
    size = int(value["m_size"])

    d.putItemCount(size)
    d.putPlotData(start, size, innerType)
