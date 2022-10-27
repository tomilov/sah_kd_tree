find_package(
    Qt6
    REQUIRED
    COMPONENTS
        Core
        Gui
        Widgets
        Qml
        Quick
        QuickControls2)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_compile_definitions(
    QT_NO_KEYWORDS
    QT_NO_FOREACH
    QT_RESTRICTED_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    QT_NO_CAST_FROM_BYTEARRAY
    QT_NO_NARROWING_CONVERSIONS_IN_CONNECT
    QT_MESSAGELOGCONTEXT)
