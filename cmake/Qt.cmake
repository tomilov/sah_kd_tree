set(QT_NO_PRIVATE_MODULE_WARNING ON)
set(QT_CREATOR_SKIP_MAINTENANCE_TOOL_PROVIDER ON)
find_package(
    Qt6 6.7.2
    REQUIRED
    COMPONENTS
        Core
        Gui
        GuiPrivate
        Widgets
        Qml
        Quick
        QuickControls2
        Svg
        Xml
        Quick3D
        Concurrent)

qt6_standard_project_setup()
set(CMAKE_AUTORCC ON)

add_compile_definitions(
    QT_NO_KEYWORDS
    QT_NO_FOREACH
    QT_RESTRICTED_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    QT_NO_CAST_FROM_BYTEARRAY
    QT_NO_NARROWING_CONVERSIONS_IN_CONNECT
    QT_MESSAGELOGCONTEXT)
