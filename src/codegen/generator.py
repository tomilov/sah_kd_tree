import re
import sys
import subprocess
from jinja2 import Environment, FileSystemLoader
from difflib import context_diff
from termcolor import colored
import argparse
import xml.etree.ElementTree as etree

FORMAT_REGEX = re.compile(r'''
VK_FORMAT
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
        _(?P<astc_width>[4568]|10|12)x(?P<astc_height>[4568]|10|12)
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
        _(?P<etc>ETC2)
        _(?P<etc_components>R8G8B8|R8G8B8A[18])
        _(?P<etc_numeric_format>UNORM|SRGB)
        _BLOCK
    |
        # BC
        _(?P<bc>BC(?:[123457]|6H))
        (?:_(?P<bc_components>RGB|RGBA))?
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


def cpp_case(m):
    g1 = m.group(1)
    if g1:
        return g1
    if m.group(2):
        return ''
    s = m.group()
    return s[0] + s[1:].lower()

def c_format_name_to_cpp(format):
    return re.sub(r'(?:_(IMG|NV)$)|(_)|([A-Z]+)', cpp_case, format[len('VK_FORMAT_'):])


def print_diff(unformatted, formatted):
    a = unformatted.splitlines(keepends=True)
    b = formatted.splitlines(keepends=True)
    sys.stderr.writelines(context_diff(a, b, n=0, fromfile='unformatted', tofile='formatted'))


def gen_spirv_format_context(args):
    sys.path.append(args.spirv_headers)
    from spirv.unified1.spirv import spv

    spv_enums = list()
    for key, value in spv.items():
        if not isinstance(value, dict):
            continue
        spv_enum = dict()

        spv_enum['enum_type'] = f'Spv{key}'

        prefix = -1
        for c in key:
            if not c.isupper():
                break
            prefix += 1
        prefix = max(prefix, 1)
        spv_enum['variable_name'] = key[:prefix].lower() + key[prefix:]

        enum_values = sorted(list(value.items()), key=lambda item: (item[1], item[0]))
        unique_enum_underlying_values = set()
        unique_enum_values = list()
        for enum_value_name, enum_underlying_value in enum_values:
            assert isinstance(enum_underlying_value, int)
            if not enum_underlying_value in unique_enum_underlying_values:
                unique_enum_underlying_values.add(enum_underlying_value)
                enum_value = {
                    'enum_value_name': enum_value_name,
                    'enum_underlying_value': enum_underlying_value,
                }
                unique_enum_values.append(enum_value)
        spv_enum['enum_values'] = unique_enum_values

        spv_enums.append(spv_enum)

    return {'spv_enums': spv_enums}


def gen_vulkan_utils_context(args):
    registry = etree.parse(args.vulkan_registry).getroot()

    name = set()
    blockExtent = set()
    blockSize = set()
    class_ = set()
    packed = set()
    compressed = set()
    chroma = set()
    texelsPerBlock = set()
    multiplane_formats = list()
    depth_formats = list()
    stencil_formats = list()
    numeric_format = set()
    mp_numeric_format = set()
    plane_attribs = set()

    tags = set()
    formats = registry.findall('formats')
    output_formats = list()
    assert len(formats) == 1
    for format in formats[0]:
        assert format.tag == 'format'
        format_name = format.get('name')
        planes = format.findall('plane')
        if len(planes) > 0:
            assert len(planes) != 1
            output_planes = list()
            for plane in planes:
                output_plane = {
                    'index': plane.get('index'),
                    'compatible_format': c_format_name_to_cpp(plane.get('compatible')),
                    'height_divisor': plane.get('heightDivisor'),
                    'width_divisor': plane.get('widthDivisor'),
                }
                output_planes.append(output_plane)
            output_format['planes'] = output_planes

            multiplane_formats.append(enum_value_name)

        print(format_name)
        print(format.tag, format.attrib)
        assert len(format) > 0
        output_format = dict()
        enum_value_name = c_format_name_to_cpp(format_name)
        output_format['enum_value_name'] = enum_value_name
        for key, value in format.attrib.items():
            if key == 'name':
                name.add(value)
            elif key == 'blockExtent':
                blockExtent.add(value)
            elif key == 'blockSize':
                blockSize.add(value)
            elif key == 'class':
                class_.add(value)
            elif key == 'packed':
                packed.add(value)
            elif key == 'compressed':
                compressed.add(value)
            elif key == 'chroma':
                chroma.add(value)
            elif key == 'texelsPerBlock':
                texelsPerBlock.add(value)
            else:
                assert False

        for property in format:
            assert len(property) == 0

            print('\t', property.tag, property.attrib)
            tags.add(property.tag) # 'component', 'spirvimageformat', 'plane'

            if property.tag == 'component':
                numeric_format.add(property.get('numericFormat'))
                if len(planes) > 0:
                    mp_numeric_format.add(property.get('numericFormat'))
                #plane_attribs.update(property.attrib.keys())
                # 'numericFormat', 'bits', 'planeIndex', 'name'
                if property.get('name') == 'D':
                    depth_formats.append(enum_value_name)
                if property.get('name') == 'S':
                    stencil_formats.append(enum_value_name)
                pass
            elif property.tag == 'spirvimageformat':
                #plane_attribs.update(property.attrib.keys())
                # 'name'
                pass
            elif property.tag == 'plane':
                #plane_attribs.update(property.attrib.keys())
                # 'compatible', 'heightDivisor', 'index', 'widthDivisor'
                pass
            else:
                assert False
        output_formats.append(output_format)

        m = FORMAT_REGEX.match(format_name)
        print('<Match: %r, groupdict=%r>' % (m.group(), m.groupdict()))

    print('name', name)
    print('numeric_format', numeric_format)
    print('mp_numeric_format', mp_numeric_format)
    print('blockExtent', blockExtent)
    print('blockSize', blockSize)
    print('class_', class_)
    print('packed', packed)
    print('compressed', compressed)
    print('chroma', chroma)
    print('texelsPerBlock', texelsPerBlock)
    print('tags', tags)
    print('plane_attribs', plane_attribs)
    print('depth_formats', '|'.join(depth_formats))
    print('stencil_formats', '|'.join(stencil_formats))

    return {'formats': output_formats}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True, help='Source directory')
    parser.add_argument('--clang-format-executable', required=True, help='Filepath of clang-format executable')
    parser.add_argument('--clang-format', required=True, help='Filepaht of .clang-format config file')
    parser.add_argument('--ignore-format-mismatch', action='store_true', help='Do not print diff if formatted differs from unformatted')

    subparsers = parser.add_subparsers(dest='subparser_name')

    spirv_parser = subparsers.add_parser('spirv_format')
    spirv_parser.add_argument('--spirv-headers', required=True, help='SPIR-V Headers dirpath')
    spirv_parser.set_defaults(handler=gen_spirv_format_context)

    vulkan_parser = subparsers.add_parser('vulkan_utils')
    vulkan_parser.add_argument('--vulkan-registry', required=True, help='Vulkan Registry filepath')
    vulkan_parser.set_defaults(handler=gen_vulkan_utils_context)

    args = parser.parse_args()
    context = args.handler(args)

    env = Environment(
        loader=FileSystemLoader(args.source_dir),
        keep_trailing_newline=True,
        #line_statement_prefix='//%',
    )
    for ext in ['hpp', 'cpp']:
        unformatted = env.get_template(f'{args.subparser_name}.{ext}.jinja2').render(context)
        popenargs = [args.clang_format_executable, f'-style=file:{args.clang_format}']
        completed_process = subprocess.run(popenargs, input=unformatted.encode(), stdout=subprocess.PIPE)
        completed_process.check_returncode()
        formatted = completed_process.stdout.decode()
        if unformatted != formatted:
            #print(unformatted)
            if not args.ignore_format_mismatch:
                print_diff(unformatted, formatted)
        with open(f'{args.subparser_name}.{ext}.tmp', 'wb') as tmp:
            tmp.write(formatted.encode())
