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

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Layout.column: index % gridLayout.columns
            Layout.row: Math.trunc(index / gridLayout.columns)

            transformOrigin: Item.TopLeft

            SequentialAnimation on rotation {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: 0.0
                    to: 360.0
                    duration: 10000
                }
            }

            SequentialAnimation on scale {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: 1.0
                    to: 0.5
                    duration: 1000
                }
                NumberAnimation {
                    from: 0.5
                    to: 1.0
                    duration: 1000
                }
            }

            SequentialAnimation on opacity {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: 1.0
                    to: 0.2
                    duration: 1500
                }
                NumberAnimation {
                    from: 0.2
                    to: 1.0
                    duration: 1500
                }
            }

            SahKdTreeViewer {
                engine: SahKdTreeEngine

                anchors.fill: parent
                visible: index !== 1
                scale: 0.8
                opacity: 1.0

                transformOrigin: Item.BottomRight

                SequentialAnimation on rotation {
                    loops: Animation.Infinite
                    running: true
                    NumberAnimation {
                        from: 0.0
                        to: 360.0
                        duration: 3333
                    }
                }

                SequentialAnimation on t {
                    loops: Animation.Infinite
                    running: true
                    NumberAnimation {
                        to: 1.0
                        duration: 100
                        easing.type: Easing.InQuad
                    }
                    NumberAnimation {
                        to: 0.0
                        duration: 100
                        easing.type: Easing.OutQuad
                    }
                }

                Rectangle {
                    anchors.fill: parent
                    anchors.margins: -2

                    border.color: "red"
                    border.width: 2

                    color: "transparent"
                }
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
        text: "The quick brown fox jumped over the lazy dog's back 1234567890"
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
        columns: 3

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
