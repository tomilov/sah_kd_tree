#set(THRUST_ENABLE_MULTICONFIG ON)
#set(THRUST_MULTICONFIG_ENABLE_SYSTEM_CPP ON)
#set(THRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA ON)
#set(THRUST_MULTICONFIG_WORKLOAD FULL)
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust FROM_OPTIONS)

find_package(TBB REQUIRED)
target_link_libraries(
    Thrust
    INTERFACE
        OpenMP::OpenMP_CXX
        TBB::tbb)
