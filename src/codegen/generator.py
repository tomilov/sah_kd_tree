import re
import sys
import subprocess
from jinja2 import Environment, select_autoescape, FileSystemLoader
import difflib
from termcolor import colored
import argparse
import xml.etree.ElementTree as etree

FORMAT_REGEX = re.compile(r'''
^VK_FORMAT
(?:
        # NVIDIA optical flow
        _R16G16_S10_5_NV
    |
        # PVRTC
        _(?P<pvrtc>PVRTC[12])
        _(?P<pvrtc_bpp>[24])BPP
        _(?P<pvrtc_numeric_format>UNORM|SRGB)
        _BLOCK
        _IMG
    |
        # ASTC
        _(?P<astc>ASTC)
        _(?P<astc_block_width>[4568]|10|12)x(?P<astc_block_height>[4568]|10|12)
        _(?P<astc_numeric_format>UNORM|SRGB|SFLOAT)
        _BLOCK
    |
        # EAC
        _(?P<eac>EAC)
        _(?P<eac_components>R11|R11G11)
        _(?P<eac_numeric_format>UNORM|SNORM)
        _BLOCK
    |
        # ETC2
        _(?P<etc2>ETC2)
        _(?P<etc2_components>R8G8B8|R8G8B8A[18])
        _(?P<etc2_numeric_format>UNORM|SRGB)
        _BLOCK
    |
        # BC
        _(?P<bc>BC(?:[123457]|6H))
        (?:_(?P<bc_components>RGBA?))?
        _(?P<bc_numeric_format>UNORM|SRGB|SNORM|UFLOAT|SFLOAT)
        _BLOCK
    |
        # Depth and|or stencil
        (?:
            (?:_X(?P<depth_padding>8))?
            _D(?P<depth_bits>16|24|32)
            _(?P<depth_numeric_format>UNORM|SFLOAT)
        )?
        (?:
            _S(?P<stencil_bits>8)
            _(?P<stencil_numeric_format>UINT)
        )?
        (?:_PACK(?P<depth_pack>32))?
    |
        # all image formats including multiplane
        _(?P<component0>[ERGBA])(?P<component0bits>[1245689]|10|12|16|32|64)(?:X(?P<component0padding>[46]))?
        (?:(?P<plane1>_)?(?P<component1>[RGB])(?P<component1bits>[45689]|10|11|12|16|32|64)(?:X(?P<component1padding>[46]))?)?
        (?:(?P<plane2>_)?(?P<component2>[RGB])(?P<component2bits>[45689]|10|11|12|16|32|64)(?:X(?P<component2padding>[46]))?)?
        (?:(?P<component3>[RGBA])(?P<component3bits>[145689]|10|12|16|32|64)(?:X(?P<component3padding>[46]))?)?
        (?:_(?P<plane_count>[23])PLANE)?
        (?:_(?P<chroma>420|422|444))?
        _(?P<numeric_format>USCALED|UINT|UFLOAT|SINT|SFLOAT|SSCALED|SRGB|SNORM|UNORM)
        (?:_(?P<batch>[234])?PACK(?P<pack>8|16|32))?
)$
''', re.VERBOSE)


def print_diff(unformatted, formatted, /, *, file_name=''):
    a = unformatted.splitlines(keepends=False)
    b = formatted.splitlines(keepends=False)
    for line in difflib.unified_diff(a, b, fromfile=f'{file_name} <unformatted>', tofile=f'{file_name} <formatted>', n=3, lineterm=''):
        if len(line) == 0:
            continue
        if line[0] == '-':
            color = 'red'
        elif line[0] == '+':
            color = 'green'
        elif line[0] == '@':
            color = 'cyan'
        else:
            color = None
        print(colored(line, color), file=sys.stderr)


def gen_spirv_format_context(args):
    sys.path.append(args.spirv_headers)
    from spirv.unified1.spirv import spv

    def prefix_to_lower(m):
        g1 = m.group(1)
        if g1:
            return g1[:-1].lower() + g1[-1]
        return m.group().lower()

    split_typename_regex = re.compile(r'^(?:([A-Z]{2,})|([A-Z]))')

    def to_variable_name(name):
        return split_typename_regex.sub(prefix_to_lower, name)

    spv_enums = list()
    for key, value in spv.items():
        if not isinstance(value, dict):
            continue
        spv_enum = dict()

        spv_enum['enum_typename'] = f'Spv{key}'
        spv_enum['variable_name'] = to_variable_name(key)

        enum_values = sorted(value.items(), key=lambda item: (item[1], item[0]))
        unique_enum_underlying_values = set()
        unique_enum_values = list()
        for enum_value_name, enum_underlying_value in enum_values:
            assert isinstance(enum_underlying_value, int), type(enum_underlying_value).__name__
            if not enum_underlying_value in unique_enum_underlying_values:
                unique_enum_underlying_values.add(enum_underlying_value)
                enum_value = {
                    'enum_value_name': enum_value_name,
                    'enum_underlying_value': enum_underlying_value,
                }
                unique_enum_values.append(enum_value)
        spv_enum['enum_values'] = unique_enum_values

        spv_enums.append(spv_enum)

    filters = {
        'to_variable_name': to_variable_name,
    }
    context = {
        'spv_enums': spv_enums,
    }
    return context, filters


def gen_vulkan_utils_context(args):
    registry = etree.parse(args.vulkan_registry).getroot()

    def cpp_case(m):
        g1 = m.group(1)
        if g1:
            return g1
        if m.group(2):
            return ''
        s = m.group()
        return s[0] + s[1:].lower()

    tags = registry.findall('tags')
    assert len(tags) == 1, len(tags)
    vendors = '|'.join(tag.get('name') for tag in tags[0].findall('tag'))
    split_c_regex = re.compile(f'(?:_({vendors})$)|(_)|([A-Z]+)')

    def to_cpp_case(identifier):
        return split_c_regex.sub(cpp_case, identifier)


    def c_enum_to_cpp(identifier, prefix):
        vk_len = len('VK_')
        assert prefix[:vk_len] == 'VK_', prefix
        assert identifier[:len(prefix)] == prefix, identifier[:len(prefix)]
        camel_case = to_cpp_case(identifier[vk_len:])
        type_len = sum(1 for c in prefix[vk_len:] if c != '_')
        return f'vk::{camel_case[:type_len]}::e{camel_case[type_len:]}'


    reformat_identifier_regex = re.compile(r'[_\ \-]|\bASTC_(\d+x\d+)\b|([A-Za-z]+)')


    def reformat_identifier(identifier):
        def reformat(m):
            g1 = m.group(1)
            if g1:
                return f'Astc{g1}'
            g2 = m.group(2)
            if g2:
                return g2[0].upper() + g2[1:].lower()
        return reformat_identifier_regex.sub(reformat, identifier)


    compatibility_classes = set()
    component_types = set()
    max_component_count = 0
    numeric_formats = set()
    compression_types = set()
    chroma_kinds = set()
    max_plane_count = 0

    formats = registry.findall('formats')
    assert len(formats) == 1, len(formats)
    output_formats = list()
    for format in formats[0]:
        assert format.tag == 'format', format.tag
        output_format = dict()

        assert len(format) > 0, len(format)
        for key, value in format.attrib.items():
            if key == 'name':
                format_name = value
                assert FORMAT_REGEX.match(format_name), format_name
                output_format['format_name'] = format_name
                output_format['format_cpp_name'] = c_enum_to_cpp(format_name, 'VK_FORMAT')
            elif key == 'blockExtent':
                if value != '1,1,1':
                    block_extent = tuple(map(int, value.split(',')))
                    assert len(block_extent) == 3
                    output_format['block_extent'] = block_extent
            elif key == 'blockSize':
                output_format['block_size'] = value
            elif key == 'class':
                output_format['compatibility_class'] = value
                compatibility_class = reformat_identifier(value)
                output_format['compatibility_cpp_class'] = compatibility_class
                compatibility_classes.add((value, compatibility_class))
            elif key == 'packed':
                output_format['packed'] = value
            elif key == 'compressed':
                output_format['compression_type'] = value
                compression_type = reformat_identifier(value)
                output_format['compression_cpp_type'] = compression_type
                compression_types.add((value, compression_type))
            elif key == 'chroma':
                output_format['chroma_kind'] = value
                chroma_kind = reformat_identifier(value)
                output_format['chroma_kind_cpp'] = chroma_kind
                chroma_kinds.add((value, chroma_kind))
            elif key == 'texelsPerBlock':
                if value != '1':
                    output_format['texels_per_block'] = value
            else:
                assert False, key

        assert 'format_name' in output_format
        assert 'block_size' in output_format
        assert 'compatibility_class' in output_format

        assert all(child.tag in {'component', 'spirvimageformat', 'plane'} for child in format)

        components = format.findall('component')
        planes = format.findall('plane')
        spirvimageformats = format.findall('spirvimageformat')

        assert components
        max_component_count = max(max_component_count, len(components))
        output_components = list()
        for component in components:
            output_component = dict()
            for key, value in component.attrib.items():
                if key == 'name':
                    assert len(value) == 1, 'Identifiers should be reformatted'
                    output_component['component_type'] = value
                    component_types.add(value)
                elif key == 'numericFormat':
                    output_component['numeric_format'] = value
                    numeric_format = reformat_identifier(value)
                    output_component['numeric_cpp_format'] = numeric_format
                    numeric_formats.add((value, numeric_format))
                elif key == 'bits':
                    output_component['bitsize'] = value if value == "compressed" else int(value)
                elif key == 'planeIndex':
                    output_component['plane_index'] = int(value)
                else:
                    assert False, key
            assert 'component_type' in output_component
            assert 'numeric_format' in output_component
            assert 'bitsize' in output_component
            if output_component['component_type'] in ('D', 'S'):
                assert output_component['bitsize'] % 8 == 0
            output_components.append(output_component)

        output_format['components'] = output_components

        if planes:
            assert len(planes) != 1, len(planes)
            max_plane_count = max(max_plane_count, len(planes))
            output_planes = list()
            for plane in planes:
                output_plane = dict()
                for key, value in plane.attrib.items():
                    if key == 'index':
                        output_plane['index'] = int(value)
                    elif key == 'compatible':
                        output_plane['compatible_format_name'] = value
                        output_plane['compatible_cpp_format_name'] = c_enum_to_cpp(value, 'VK_FORMAT')
                    elif key == 'heightDivisor':
                        output_plane['height_divisor'] = int(value)
                    elif key == 'widthDivisor':
                        output_plane['width_divisor'] = int(value)
                    else:
                        assert False, key
                assert 'index' in output_plane
                assert 'compatible_format_name' in output_plane
                assert 'compatible_cpp_format_name' in output_plane
                assert 'height_divisor' in output_plane
                assert 'width_divisor' in output_plane
                output_planes.append(output_plane)
            output_planes.sort(key=lambda output_plane: output_plane['index'])
            for output_plane in output_planes:
                del output_plane['index']

            output_format['planes'] = output_planes

        assert len(spirvimageformats) < 2, len(spirvimageformats)
        if len(spirvimageformats) == 1:
            for key, value in spirvimageformats[0].attrib.items():
                if key == 'name':
                    output_format['spirv_image_format'] = value
                else:
                    assert False, key
            assert 'spirv_image_format' in output_format

        output_formats.append(output_format)

    assert compatibility_classes
    assert component_types
    assert max_component_count > 0
    assert numeric_formats
    assert compression_types
    assert chroma_kinds
    assert max_plane_count > 0
    assert output_formats

    context = {
        'compatibility_classes': sorted(compatibility_classes),
        'component_types': sorted(component_types),
        'max_component_count': max_component_count,
        'numeric_formats': sorted(numeric_formats),
        'compression_types': sorted(compression_types),
        'chroma_kinds': sorted(chroma_kinds),
        'max_plane_count': max_plane_count,
        'formats': sorted(output_formats, key=lambda x: x['format_name']),
    }
    filters = {
        'c_enum_to_cpp': c_enum_to_cpp,
    }
    return context, filters


def clang_format(unformatted):
    popenargs = [args.clang_format_executable, f'-style=file:{args.clang_format_config}']
    completed_process = subprocess.run(popenargs, input=unformatted.encode(), stdout=subprocess.PIPE)
    completed_process.check_returncode()
    return completed_process.stdout.decode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, help='Source directory')
    parser.add_argument('--clang-format-executable', required=True, help='Filepath of clang-format executable')
    parser.add_argument('--clang-format-config', required=True, help='Filepath of .clang-format config file')
    parser.add_argument('--fail-on-format-mismatch', action='store_true', help='Abort if formatted differs from unformatted')

    subparsers = parser.add_subparsers(dest='subparser_name')

    spirv_parser = subparsers.add_parser('spirv_format')
    spirv_parser.add_argument('--spirv-headers', required=True, help='SPIR-V Headers dirpath')
    spirv_parser.set_defaults(handler=gen_spirv_format_context)

    vulkan_parser = subparsers.add_parser('vulkan_utils')
    vulkan_parser.add_argument('--vulkan-registry', required=True, help='Vulkan Registry filepath')
    vulkan_parser.set_defaults(handler=gen_vulkan_utils_context)

    args = parser.parse_args()
    context, filters = args.handler(args)

    env = Environment(
        loader=FileSystemLoader(args.source_dir),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=select_autoescape(
            default=False,
        )
    )
    env.filters.update(filters)

    for ext in ['hpp', 'cpp']:
        file_name = f'{args.subparser_name}.{ext}'
        unformatted = env.get_template(f'{file_name}.jinja2').render(context)
        formatted = clang_format(unformatted)
        if unformatted != formatted:
            print_diff(unformatted, formatted, file_name=file_name)
            assert not args.fail_on_format_mismatch, 'Failed on formats mismatch'
        with open(f'{file_name}.tmp', 'wb') as tmp:
            tmp.write(unformatted.encode())
