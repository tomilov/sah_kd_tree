import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Dialogs as Dialogs

import Qt.labs.folderlistmodel

import SahKdTree 1.0

ApplicationWindow {
    id: root

    objectName: Qt.application.name

    visible: true
    visibility: Window.AutomaticVisibility

    title: {
        qsTr("%1 (dt %2ms) (screen refresh rate %3) - %4")
        .arg(Qt.application.displayName)
        .arg((mainSahKdTreeViewer.dt * 1000.0).toFixed(3))
        .arg(app.primaryScreen.refreshRate.toFixed(3))
        .arg(mainSahKdTreeViewer.scenePath)
    }

    CenteredDialog {
        id: confirmationDialog

        title: qsTr("Close application")

        Label {
            anchors.fill: parent

            text: qsTr("Are you sure?")
        }

        standardButtons: Dialog.Yes | Dialog.No

        onAccepted: root.close()
    }

    onClosing: (close) => {
        if (visibility === Window.FullScreen) {
            show()
            confirmationDialog.open()
            close.accepted = false
        }
    }

    onActiveFocusItemChanged: {
        if (activeFocusItem instanceof SahKdTreeViewer) {
            print("activeFocusItem", activeFocusItem)
        }
    }

    Shortcut {
        sequences: [StandardKey.Cancel] // "Escape"
        context: Qt.WindowShortcut
        autoRepeat: false
        onActivated: confirmationDialog.open()
    }

    Menu {
        id: contextMenu

        Action {
            text: qsTr("Open")
            onTriggered: {
                sceneOpenDialog.open()
            }
        }
    }

    /*SahKdTreeViewer {
        anchors.fill: parent

        focus: true

        id: mainSahKdTreeViewer
        objectName: "Main"

        engine: SahKdTreeEngine
    }*/

    Shortcut {
        sequences: [StandardKey.Open]
        context: Qt.WindowShortcut
        autoRepeat: false
        onActivated: {
            if (root.activeFocusItem instanceof SahKdTreeViewer) {
                sceneOpenDialog.item = root.activeFocusItem
                sceneOpenDialog.open()
            }
        }
    }

    Dialogs.FileDialog {
        id: sceneOpenDialog2

        title: qsTr("Open scene")

        nameFilters: ["All files (*)"]

        onAccepted: {
            print(fileUrl)
        }
    }

    CenteredDialog {
        id: sceneOpenDialog

        width: Math.min(384, root.width)
        height: Math.min(384, root.height)

        title: qsTr("Open scene file")

        property url folder
        property SahKdTreeViewer item

        ColumnLayout {
            anchors.fill: parent

            Frame {
                Label {
                    anchors.fill: parent

                    text: sceneOpenDialog.folder + "/"
                }

                Layout.fillWidth: true
            }
            ListView {
                clip: true

                Layout.fillWidth: true
                Layout.fillHeight: true

                flickableDirection: Flickable.AutoFlickIfNeeded

                model: FolderListModel {
                    folder: sceneOpenDialog.folder

                    nameFilters: SahKdTreeEngine.supportedSceneFileExtensions

                    showDirsFirst: true
                    showOnlyReadable: true
                    showDotAndDotDot: true
                }

                delegate: Label {
                    text: fileName + (fileIsDir ? "/" : "")

                    MouseArea {
                        anchors.fill: parent
                        onDoubleClicked: (mouse) => {
                            if (fileIsDir) {
                                sceneOpenDialog.folder = fileURL
                            } else {
                                sceneOpenDialog.item.scenePath = fileURL
                                sceneOpenDialog.item = null
                                sceneOpenDialog.accept()
                            }
                            mouse.accepted = true
                        }
                    }
                }

                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AlwaysOn
                }
            }
        }

        standardButtons: Dialog.Close

        Settings {
            property alias sceneOpenDialogFolder: sceneOpenDialog.folder
        }
    }

    Component {
        id: sahKdTreeViewerComponent

        SahKdTreeViewer {
            id: sahKdTreeViewer

            engine: SahKdTreeEngine

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
            //*/

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

            MouseArea {
                anchors.fill: parent

                acceptedButtons: Qt.RightButton | Qt.LeftButton

                cursorShape: parent.cursor

                onPressed: (mouse) => {
                    parent.forceActiveFocus()
                    switch (mouse.button) {
                    case Qt.LeftButton: {
                        mouse.accepted = false
                        break
                    }
                    case Qt.RightButton: {
                        mouse.accepted = true
                        break
                    }
                    }
                }

                onClicked: (mouse) => {
                    switch (mouse.button) {
                    case Qt.RightButton: {
                        mouse.accepted = true

                        sceneOpenDialog.item = parent
                        contextMenu.x = mouse.x
                        contextMenu.y = mouse.y
                        contextMenu.popup()
                        break
                    }
                    }
                }
            }

            Settings {
                category: "%1".arg(sahKdTreeViewer.objectName)
                property alias scenePath: sahKdTreeViewer.scenePath
                property alias cameraPosition: sahKdTreeViewer.cameraPosition
                property alias eulerAngles: sahKdTreeViewer.eulerAngles
                property alias fieldOfView: sahKdTreeViewer.fieldOfView
            }

            Component.onCompleted: console.log("created")
            Component.onDestruction: console.log("destroyed")
        }
    }

    header: RowLayout {
        height: 128

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true

            color: "blue"
            opacity: 0.4
        }
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
        text: "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG'S BACK 1234567890"
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

                width: implicitWidth
            }

            TabButton {
                text: qsTr("Swipe")

                width: implicitWidth
            }

            TabButton {
                text: qsTr("Grid")

                width: implicitWidth
            }
        }

        StackLayout {
            currentIndex: tabBar.currentIndex

            Layout.fillWidth: true
            Layout.fillHeight: true

            SahKdTreeViewer {
                id: mainSahKdTreeViewer
                objectName: "Main"

                engine: SahKdTreeEngine

                //rotation: 30.0
                //scale: 0.7

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
                SequentialAnimation on scale {
                    loops: Animation.Infinite
                    running: true
                    NumberAnimation {
                        from: 0.2
                        to: 1.2
                        duration: 2000
                    }
                    NumberAnimation {
                        from: 1.2
                        to: 0.2
                        duration: 2000
                    }
                }
                SequentialAnimation on t {
                    loops: Animation.Infinite
                    running: true
                    NumberAnimation {
                        to: 1.0
                        duration: 200
                        easing.type: Easing.InQuad
                    }
                    NumberAnimation {
                        to: 0.0
                        duration: 200
                        easing.type: Easing.OutQuad
                    }
                }
                //*/
                Rectangle {
                    anchors.fill: parent
                    anchors.margins: -4

                    border.color: parent.activeFocus ? "red" : "green"
                    border.width: 3

                    color: "transparent"
                }

                focus: StackLayout.isCurrentItem

                Settings {
                    category: "%1".arg(mainSahKdTreeViewer.objectName)
                    property alias scenePath: mainSahKdTreeViewer.scenePath
                    property alias cameraPosition: mainSahKdTreeViewer.cameraPosition
                    property alias eulerAngles: mainSahKdTreeViewer.eulerAngles
                    property alias fieldOfView: mainSahKdTreeViewer.fieldOfView
                }
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

                            sourceComponent: sahKdTreeViewerComponent

                            activeFocusOnTab: true
                            onActiveFocusChanged: {
                                if (item && activeFocus) {
                                    item.forceActiveFocus()
                                }
                            }
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
                        delegate: Loader {
                            onItemChanged: if (item) item.objectName = "Grid %1".arg(index)

                            Layout.fillWidth: true
                            Layout.fillHeight: true

                            required property int index

                            active: index !== 1

                            //KeyNavigation.priority: KeyNavigation.BeforeItem
                            //KeyNavigation.up: print(index, gridLayout.columns)//repeater.itemAt((index + count - gridLayout.columns) % count)

                            sourceComponent: sahKdTreeViewerComponent

                            activeFocusOnTab: true
                            onActiveFocusChanged: {
                                if (item && activeFocus) {
                                    item.forceActiveFocus()
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    footer: RowLayout {
        height: 128

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true

            color: "green"
            opacity: 0.4
        }
        Label {
            width: implicitWidth
            Layout.fillHeight: true
            text: {
                var pos = mainSahKdTreeViewer.cameraPosition
                var dir = mainSahKdTreeViewer.eulerAngles
                var fov = mainSahKdTreeViewer.fieldOfView
                qsTr("pos(%1, %2, %3) dir(%4, %5, %6) fov(%7)")
                .arg(pos.x.toFixed(3)).arg(pos.y.toFixed(3)).arg(pos.z.toFixed(3))
                .arg(dir.x.toFixed(3)).arg(dir.y.toFixed(3)).arg(dir.z.toFixed(3))
                .arg(fov.toFixed(3))
            }
        }
    }
}
