find_package(OpenMP REQUIRED)
string(REPLACE " " ";" OpenMP_CXX_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
foreach(OpenMP_CXX_FLAG IN LISTS OpenMP_CXX_FLAGS_LIST)
    target_compile_options(
        OpenMP::OpenMP_CXX
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAG}>
    )
endforeach()
