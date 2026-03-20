find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust FROM_OPTIONS)

find_package(TBB REQUIRED)
target_link_libraries(
    Thrust
    INTERFACE
        OpenMP::OpenMP_CUDA
        TBB::tbb
)
