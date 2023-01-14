import argparse
import xml.dom.minidom
import sys
import os

def gen_spirv_format(args):
    sys.path.append(args.spirv_headers)
    from spirv.unified1.spirv import spv

    fn = "spirv_format"

    hpp_path = os.path.join(args.output_dir, f"{fn}_unformatted.hpp")
    cpp_path = os.path.join(args.output_dir, f"{fn}_unformatted.cpp")
    with open(hpp_path, "wt") as hpp_file, open(cpp_path, "wt") as cpp_file:
        def hpp(text=''):
            print(text, file=hpp_file)

        def cpp(text=''):
            print(text, file=cpp_file)

        hpp('#pragma once')
        hpp()
        hpp('#include <codegen/codegen_export.h>')
        hpp()
        hpp('#include <../SPIRV-Reflect/spirv_reflect.h>')
        hpp()
        hpp('namespace codegen::spv')
        hpp('{')
        hpp()

        cpp(f'#include <codegen/{fn}.hpp>')
        cpp()
        cpp('#include <type_traits>')
        cpp()
        cpp('namespace codegen::spv')
        cpp('{')
        cpp()

        for key, value in spv.items():
            if not isinstance(value, dict):
                continue
            enum_name = f'Spv{key}'
            prefix = -1
            for c in key:
                if not c.isupper():
                    break
                prefix += 1
            prefix = max(prefix, 1)
            var_name = key[:prefix].lower() + key[prefix:]
            hpp(f'[[nodiscard]] const char * toString({enum_name} {var_name}) CODEGEN_EXPORT;')
            cpp(f'const char * toString({enum_name} {var_name})')
            cpp('{')
            cpp(f'switch (static_cast<std::underlying_type_t<{enum_name}>>({var_name})) {{')
            enum = sorted(list(value.items()), key=lambda item: (item[1], item[0]))
            values = set()
            for enum_value_name, value in enum:
                assert isinstance(value, int)
                if not value in values:
                    values.add(value)
                    cpp(f'case {value}: return "{enum_value_name}";')
            cpp('}')
            cpp('return nullptr;')
            cpp(f'}}  // toString({enum_name})')
            cpp()

        cpp('}  // namespace codegen::spv')

        hpp()
        hpp('}  // namespace codegen::spv')


def gen_vk_format_utils(args):
    fn = "vk_format_utils"

    cpp_path = os.path.join(args.output_dir, f"{fn}_unformatted.cpp")
    hpp_path = os.path.join(args.output_dir, f"{fn}_unformatted.hpp")
    with open(hpp_path, "wt") as hpp_file, open(cpp_path, "wt") as cpp_file:
        def hpp(text=''):
            print(text, file=hpp_file)

        def cpp(text=''):
            print(text, file=cpp_file)

        hpp('#pragma once')
        hpp()
        hpp('#include <vulkan/vulkan.hpp>')
        hpp()
        hpp('#include <codegen/codegen_export.h>')
        hpp()
        hpp('namespace codegen::vk')
        hpp('{')
        hpp()

        cpp(f'#include <codegen/{fn}.hpp>')
        cpp()
        cpp('#include <type_traits>')
        cpp()
        cpp('namespace codegen::vk')
        cpp('{')
        cpp()

        #

        cpp('}  // namespace codegen::vk')

        hpp()
        hpp('}  // namespace codegen::vk')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--vulkan-registry", required=True, help='Vulkan Registry path')
    arg_parser.add_argument("--spirv-headers", required=True, help='SPIR-V Headers path')
    arg_parser.add_argument("--output-dir", required=True, help='Output directory')
    args = arg_parser.parse_args()

    gen_spirv_format(args)
    #gen_vk_format_utils(args)
