import QtQuick
import QtQuick.Controls
import QtQuick.Window
import SahKdTree

ApplicationWindow {
    id: root

    objectName: "rootWindow"

    visible: true
    visibility: Window.AutomaticVisibility

    SahKdTreeViewer {
        SequentialAnimation on t {
            NumberAnimation {
                to: 1
                duration: 2500
                easing.type: Easing.InQuad
            }
            NumberAnimation {
                to: 0
                duration: 2500
                easing.type: Easing.OutQuad
            }
            loops: Animation.Infinite
            running: true
        }
    }
}
