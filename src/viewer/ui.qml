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

    GridLayout {
        id: gridLayout

        anchors.fill: parent

        readonly property int rows: 4
        readonly property int cols: 3

        Repeater {
            model: gridLayout.rows * gridLayout.cols

            delegate: SahKdTreeViewer {
                id: sahKdTreeViewer

                Layout.fillWidth: true
                Layout.fillHeight: true

                Layout.column: index % gridLayout.cols
                Layout.row: Math.trunc(index / gridLayout.cols)

                engine: SahKdTreeEngine

                t: index / (gridLayout.rows * gridLayout.cols - 1)

                SequentialAnimation {
                    loops: Animation.Infinite
                    running: true
                    NumberAnimation {
                        target: sahKdTreeViewer
                        property: "t"
                        to: 1
                        duration: 1000
                        easing.type: Easing.InQuad
                    }
                    NumberAnimation {
                        target: sahKdTreeViewer
                        property: "t"
                        to: 0
                        duration: 1000
                        easing.type: Easing.OutQuad
                    }
                }
            }
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
        horizontalAlignment: Text.AlignHCenter
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
