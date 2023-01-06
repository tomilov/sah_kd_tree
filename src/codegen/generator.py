import argparse
import xml.dom.minidom
import sys
import os

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--vulkan-registry", required=True, help='Vulkan Registry path')
    arg_parser.add_argument("--spirv-headers", required=True, help='SPIR-V Headers path')
    arg_parser.add_argument("--output-dir", required=True, help='Output directory')
    args = arg_parser.parse_args()

    sys.path.append(args.spirv_headers)

    from spirv.unified1.spirv import spv

    spirv_format_hpp_path = os.path.join(args.output_dir, "spirv_format_raw.hpp")
    spirv_format_cpp_path = os.path.join(args.output_dir, "spirv_format_raw.cpp")
    with open(spirv_format_hpp_path, "wt") as spirv_format_hpp, open(spirv_format_cpp_path, "wt") as spirv_format_cpp:
        def println_hpp(text=''):
            print(text, file=spirv_format_hpp)

        def println_cpp(text=''):
            print(text, file=spirv_format_cpp)

        println_hpp('#pragma once')
        println_hpp()
        println_hpp('#include <codegen/codegen_export.h>')
        println_hpp()
        println_hpp('#include <spirv_reflect.h>')
        println_hpp()
        println_hpp('namespace codegen::spv')
        println_hpp('{')
        println_hpp()

        println_cpp('#include <codegen/spirv_format.hpp>')
        println_cpp()
        println_cpp('#include <type_traits>')
        println_cpp()
        println_cpp('namespace codegen::spv')
        println_cpp('{')
        println_cpp()

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
            println_hpp(f'const char * toString({enum_name} {var_name}) CODEGEN_EXPORT;')
            println_cpp(f'const char * toString({enum_name} {var_name})')
            println_cpp('{')
            println_cpp(f'switch (static_cast<std::underlying_type_t<{enum_name}>>({var_name})) {{')
            enum = sorted(list(value.items()), key=lambda item: (item[1], item[0]))
            values = set()
            for enum_value_name, value in enum:
                if not value in values:
                    values.add(value)
                    println_cpp(f'case {value}: return "{enum_value_name}";')
            println_cpp('}')
            println_cpp('return nullptr;')
            println_cpp(f'}}  // toString({enum_name})')
            println_cpp()

        println_cpp('}  // namespace codegen')

        println_hpp()
        println_hpp('}  // namespace codegen')
