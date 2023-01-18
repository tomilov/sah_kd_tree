import QtQuick
import QtQuick.Controls
import QtQuick.Window
import QtQuick.Layouts

import SahKdTree

ApplicationWindow {
    id: root

    objectName: Qt.application.name

    visible: true
    visibility: Window.AutomaticVisibility

    Shortcut {
        sequences: [StandardKey.Cancel] // "Escape"
        autoRepeat: false
        onActivated: root.close()
    }

    Component {
        id: sahKdTreeViewer

        SahKdTreeViewer {
            scale: 1.0
            opacity: 1.0

            Layout.fillWidth: true
            Layout.fillHeight: true

            Layout.column: index % gridLayout.columns
            Layout.row: Math.trunc(index / gridLayout.columns)

            //visible: index !== 1

            engine: SahKdTreeEngine

            SequentialAnimation on t {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    to: 1.0
                    duration: 1500
                    easing.type: Easing.InQuad
                }
                NumberAnimation {
                    to: 0.0
                    duration: 1500
                    easing.type: Easing.OutQuad
                }
            }

            Rectangle {
                anchors.fill: parent
                color: "transparent"
                border.color: "red"
                border.width: 10
                radius: 16
            }
        }
    }

    header: Rectangle {
        height: 128
        color: "blue"
        opacity: 0.4
    }

    Rectangle {
        color: Qt.rgba(1, 1, 1, 0.7)
        radius: 10
        border.width: 1
        border.color: "white"
        anchors.fill: label
        anchors.margins: -10

        z: label.z
    }

    Text {
        id: label
        color: "black"
        wrapMode: Text.WordWrap
        horizontalAlignment: Text.AlignHCenter
        text: "The background here is a squircle rendered with raw Vulkan using the beforeRendering() and beforeRenderPassRecording() signals in QQuickWindow. This text label and its border is rendered using QML"
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20

        z: 1.0
    }

    GridLayout {
        id: gridLayout

        anchors.fill: parent
        anchors.margins: 16

        rows: 4
        columns: 4

        Repeater {
            model: gridLayout.rows * gridLayout.columns
            delegate: sahKdTreeViewer
        }
    }

    footer: Rectangle {
        height: 128
        color: "green"
        opacity: 0.4
    }
}
