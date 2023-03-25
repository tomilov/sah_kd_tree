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
            engine: SahKdTreeEngine

            activeFocusOnTab: true

            scale: 0.8
            opacity: 1.0

            //transformOrigin: Item.BottomRight

            /*
            SequentialAnimation on rotation {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    from: 0.0
                    to: 360.0
                    duration: 3333
                }
            }
            */

            SequentialAnimation on t {
                loops: Animation.Infinite
                running: true
                NumberAnimation {
                    to: 1.0
                    duration: 1000
                    easing.type: Easing.InQuad
                }
                NumberAnimation {
                    to: 0.0
                    duration: 1000
                    easing.type: Easing.OutQuad
                }
            }

            Rectangle {
                anchors.fill: parent
                anchors.margins: -4

                border.color: parent.activeFocus ? "red" : "green"
                border.width: 3

                color: "transparent"
            }

            Text {
                id: objectNameText

                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter

                text: parent.objectName
                color: parent.activeFocus ? "red" : "green"
            }

            Component.onCompleted: console.log("created")
            Component.onDestruction: console.log("destroyed")
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

    ColumnLayout {
        anchors.fill: parent

        TabBar {
            id: tabBar

            Layout.fillWidth: true

            TabButton {
                text: qsTr("Single")

                //width: implicitWidth
            }

            TabButton {
                text: qsTr("Swipe")

                //width: implicitWidth
            }

            TabButton {
                text: qsTr("Grid")

                //width: implicitWidth
            }
        }

        StackLayout {
            currentIndex: tabBar.currentIndex

            Layout.fillWidth: true
            Layout.fillHeight: true

            Loader {
                onItemChanged: if (item) item.objectName = "Main"

                active: StackLayout.isCurrentItem

                sourceComponent: sahKdTreeViewer
            }

            ColumnLayout {
                SwipeView {
                    id: swipeView

                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    Repeater {
                        model: 50

                        delegate: Loader {
                            onItemChanged: if (item) item.objectName = "Swipe %1".arg(SwipeView.index)

                            active: SwipeView.isCurrentItem || SwipeView.isNextItem || SwipeView.isPreviousItem

                            rotation: -5.0
                            scale: 0.9

                            sourceComponent: sahKdTreeViewer
                        }
                    }
                }

                PageIndicator {
                    id: indicator

                    Layout.alignment: Qt.AlignHCenter

                    count: swipeView.count
                    currentIndex: swipeView.currentIndex
                }
            }

            Loader {
                active: StackLayout.isCurrentItem

                sourceComponent: GridLayout {
                    id: gridLayout

                    columns: 4

                    anchors.margins: 16

                    Repeater {
                        //id: repeater

                        model: 11
                        delegate: Item {
                            required property int index

                            Layout.fillWidth: true
                            Layout.fillHeight: true

                            //Layout.column: index % gridLayout.columns
                            //Layout.row: Math.trunc(index / gridLayout.columns)

                            transformOrigin: Item.TopLeft

                            /*
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
                            */

                            Loader {
                                onItemChanged: if (item) item.objectName = "Grid %1".arg(index)

                                anchors.fill: parent

                                focus: index === 0
                                active: index !== 1

                                //KeyNavigation.priority: KeyNavigation.BeforeItem
                                //KeyNavigation.up: print(index, gridLayout.columns)//repeater.itemAt((index + count - gridLayout.columns) % count)

                                sourceComponent: sahKdTreeViewer
                            }
                        }
                    }
                }
            }
        }
    }

    footer: Rectangle {
        height: 128
        color: "green"
        opacity: 0.4
    }
}
