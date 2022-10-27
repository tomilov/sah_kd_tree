find_package(
    assimp 5
    REQUIRED)
set_target_properties(
    assimp::assimp
    PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)
