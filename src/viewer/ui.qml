import QtQuick
import QtQuick.Controls 2
import SahKdTreeViewer

ApplicationWindow {
    id: root

    visible: true

    Viewer {
        SequentialAnimation on t {
            NumberAnimation { to: 1; duration: 2500; easing.type: Easing.InQuad }
            NumberAnimation { to: 0; duration: 2500; easing.type: Easing.OutQuad }
            loops: Animation.Infinite
            running: true
        }
    }
}
