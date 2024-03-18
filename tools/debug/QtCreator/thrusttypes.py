import re

from dumper import DumperBase, Children

THRUST_TYPE_NAMES = "|".join(
    [
        "tuple",
        "complex",
        "pointer",
        "device_ptr",
        "reference",
        "tagged_reference",
        "cuda_cub::reference",
        "device_reference",
        "device_vector",
        "cuda_cub::vector",
        "cuda_cub::tag",
        "system::omp::vector",
        "system::tbb::vector",
        "system::cpp::vector",
        "host_vector",
        "zip_iterator",
        "constant_iterator",
        "discard_iterator",
        "iterator_adaptor",
        "detail::normal_iterator",
        "counting_iterator",
    ]
).replace("::", "(?:__|::)")

THRUST_TYPE_REGEX = re.compile(
    rf"""
    ^
    thrust(?:__|::)
    (?:  # inline namespace
        (?:[A-Z0-9]+_)*[A-Z0-9]+(?:__|::)
    )?
    (?P<name>
        # (?:[a-z_]+::)*[a-z_]+
        {THRUST_TYPE_NAMES}
    )
    (?:<.*>)?
    $
""",
    re.VERBOSE,
)


def _get_thrust_type_name(name):
    match = THRUST_TYPE_REGEX.fullmatch(name)
    if match is None:
        return None
    return match.group("name")


def _traverse(value, i=0):
    result = ""
    for member in value.members(includeBases=True):
        result += f"{'  ' * i}"
        result += "BASE" if member.isBaseClass else member.name
        result += "\n"
        result += _traverse(member, i=i + 1)
    if i == 0:
        print(result)
    else:
        return result


def _dump_pointer(d, value, offset=0, count=1, is_array=False):
    try:
        if _get_thrust_type_name(value.type.name) == "device_ptr":
            (value,) = value.members(includeBases=True)
            d.check(value.isBaseClass)
        d.check(_get_thrust_type_name(value.type.name) == "pointer")
        iterator = value["m_iterator"]
        device_ptr = iterator.pointer()
        if device_ptr == 0 or count == 0:
            if is_array:
                d.putItemCount(0)
            else:
                d.putEmptyValue()
            return
        inner_type = value.type[0]
        size = inner_type.size() * count
        device_ptr += size * offset

        def _dump(h):
            if is_array:
                d.putItemCount(count)
                d.putPlotData(h, count, inner_type)
            else:
                d.putItem(d.createValue(h, inner_type))

        if _get_thrust_type_name(value.type[1].name) == "cuda_cub::tag":
            host_ptr = d.parseAndEvaluate(
                f"({iterator.type.name})malloc({size})"
            ).pointer()
            try:
                d.parseAndEvaluate(
                    f"(void)cudaMemcpy({host_ptr}, {device_ptr}, {size}, cudaMemcpyDeviceToHost)"
                )
                _dump(host_ptr)
            finally:
                d.parseAndEvaluate(f"(void)free({host_ptr})")
        else:
            _dump(device_ptr)
    except Exception as e:
        import traceback

        DumperBase.warn(f"ERROR: {e} {traceback.format_exc()}")


def qdump__thrust__tuple(d, value):
    d.putExpandable()
    if d.isExpanded():
        with Children(d):
            for i, member in enumerate(value["__base_"].members(includeBases=True)):
                child = member["__value_"]
                d.putSubItem(f"<{i}>", child)


def qdump__thrust__complex(d, value):
    d.putExpandable()
    if d.isExpanded():
        with Children(d):
            (x, y) = value["data"]
            d.putSubItem("x", x)
            d.putSubItem("y", y)


def qdump__thrust__pointer(d, value):
    _dump_pointer(d, value)
    d.putBetterType(value.type)


def qdump__thrust__reference(d, value):
    return qdump__thrust__pointer(d, value["ptr"])


def qdump__thrust__device_vector(d, value):
    count = value["m_size"].integer()
    iterator = value["m_storage"]["m_begin"]["m_iterator"]
    _dump_pointer(d, iterator, count=count, is_array=True)
    d.putBetterType(value.type)


def qdump__thrust__host_vector(d, value):
    count = value["m_size"].integer()
    p = value["m_storage"]["m_begin"]["m_iterator"].pointer()
    d.putItemCount(count)
    d.putPlotData(p, count, value.type[0])
    d.putBetterType(value.type)


def qdump__thrust__constant_iterator(d, value):
    d.putValue(value["m_value"].value())
    d.putExpandable()
    if d.isExpanded():
        with Children(d):
            d.putSubItem("<iterator>", value["m_iterator"])
    d.putBetterType(value.type)


def qdump__thrust__discard_iterator(d, value):
    d.putValue(value["m_element"].value())
    d.putBetterType(value.type)


def qdump__thrust__iterator_adaptor(d, value):
    d.putItem(value["m_iterator"])
    d.putBetterType(value.type)


def qdump__thrust__detail__normal_iterator(d, value):
    (value,) = value.members(includeBases=True)
    d.check(value.isBaseClass)
    d.putItem(value)


def qdump__thrust__counting_iterator(d, value):
    d.putItem(value["m_iterator"])
    d.putBetterType(value.type)


def qdump__thrust__(d, value, regex=THRUST_TYPE_REGEX):
    thrust_type_name = _get_thrust_type_name(value.type.name)
    if thrust_type_name in ("tuple", "zip_iterator"):
        return qdump__thrust__tuple(d, value)
    elif thrust_type_name == "complex":
        return qdump__thrust__complex(d, value)
    elif thrust_type_name in ("pointer", "device_ptr"):
        return qdump__thrust__pointer(d, value)
    elif thrust_type_name in (
        "reference",
        "device_reference",
        "tagged_reference",
        "cuda_cub::reference",
    ):
        return qdump__thrust__reference(d, value)
    elif thrust_type_name in (
        "device_vector",
        "cuda_cub::vector",
        "system::omp::vector",
        "system::tbb::vector",
        "system::cpp::vector",
    ):
        return qdump__thrust__device_vector(d, value)
    elif thrust_type_name == "host_vector":
        return qdump__thrust__host_vector(d, value)
    elif thrust_type_name == "constant_iterator":
        return qdump__thrust__constant_iterator(d, value)
    elif thrust_type_name == "discard_iterator":
        return qdump__thrust__discard_iterator(d, value)
    elif thrust_type_name == "iterator_adaptor":
        return qdump__thrust__iterator_adaptor(d, value)
    elif thrust_type_name.startswith("detail::normal_iterator"):
        return qdump__thrust__detail__normal_iterator(d, value)
    elif thrust_type_name == "counting_iterator":
        return qdump__thrust__counting_iterator(d, value)
    else:
        DumperBase.warn(f"ERROR: unknown thurst type name {thrust_type_name}")
