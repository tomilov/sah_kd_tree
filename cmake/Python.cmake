set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_ROOT "${CMAKE_SOURCE_DIR}/venv/")
find_package(
    Python3
    REQUIRED COMPONENTS
        Interpreter
)
