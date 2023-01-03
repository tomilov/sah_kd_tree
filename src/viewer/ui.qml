import QtQuick
import QtQuick.Controls
import QtQuick.Window

import SahKdTree

ApplicationWindow {
    id: root

    objectName: Qt.application.name

    visible: true
    visibility: Window.AutomaticVisibility

    SahKdTreeViewer {
        anchors.fill: parent

        engine: SahKdTreeEngine
        SequentialAnimation on t {
            NumberAnimation {
                to: 1
                duration: 1500
                easing.type: Easing.InQuad
            }
            NumberAnimation {
                to: 0
                duration: 1500
                easing.type: Easing.OutQuad
            }
            loops: Animation.Infinite
            running: true
        }
    }

    Rectangle {
        color: Qt.rgba(1, 1, 1, 0.7)
        radius: 10
        border.width: 1
        border.color: "white"
        anchors.fill: label
        anchors.margins: -10
    }

    Text {
        id: label
        color: "black"
        wrapMode: Text.WordWrap
        text: "The background here is a squircle rendered with raw Vulkan using the beforeRendering() and beforeRenderPassRecording() signals in QQuickWindow. This text label and its border is rendered using QML"
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20
    }

    Shortcut {
        sequences: [StandardKey.Cancel] // "Escape"
        autoRepeat: false
        onActivated: root.close()
    }
}
