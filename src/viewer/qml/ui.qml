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
        .arg((sahKdTreeViewer.dt * 1000.0).toFixed(3))
        .arg(app.primaryScreen.refreshRate.toFixed(3))
        .arg(sahKdTreeViewer.scenePath)
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
    CenteredDialog {
        id: sceneOpenDialog
        width: Math.min(384, root.width)
        height: Math.min(384, root.height)
        title: qsTr("Open scene file")
        property url folder
        property SahKdTreeViewer item: sahKdTreeViewer
        ColumnLayout {
            anchors.fill: parent
            Frame {
                height: labelCurrentOpenPath.implicitHeight
                Label {
                    id: labelCurrentOpenPath
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
    Dialogs.FileDialog {
        id: sceneOpenDialog2
        title: qsTr("Open scene")
        nameFilters: ["All files (*)"]
        property SahKdTreeViewer item: sahKdTreeViewer
        onAccepted: {
            sceneOpenDialog2.item.scenePath = fileURL
        }
    }
    onClosing: (close) => {
        if (visibility === Window.FullScreen) {
            show()
            //confirmationDialog.open()
            close.accepted = false
        }
    }
    Action {
        id: actionOpenScene
        text: qsTr("&Open (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Open
        onTriggered: {
            sceneOpenDialog.open()
        }
    }
    Action {
        id: actionCloseScene
        text: qsTr("&Close")
        onTriggered: {
            sahKdTreeViewer.scenePath = undefined
        }
    }
    Action {
        id: actionExit
        text: qsTr("&Exit (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Cancel
        onTriggered: {
            root.close()
            //confirmationDialog.open()
        }
    }
    Action {
        id: actionUseOffscreenTexture
        text: qsTr("Offscreen (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        shortcut: "F2"
    }
    Action {
        id: actionWireFrame
        text: qsTr("Wireframe (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        shortcut: "F3"
    }
    Action {
        id: actionResetContentOrientation
        text: qsTr("Reset view orientation")
        onTriggered: {
            itemContent.rotation = 0
            itemContent.scale = 1
            sahKdTreeViewer.update()
        }
    }
    Action {
        id: actionRotatePos
        text: qsTr("Rotate view CCW")
        onTriggered: {
            itemContent.rotation -= 5
            sahKdTreeViewer.update()
        }
    }
    Action {
        id: actionRotateNeg
        text: qsTr("Rotate view CW")
        onTriggered: {
            itemContent.rotation += 5
            sahKdTreeViewer.update()
        }
    }
    Action {
        id: actionScaleInc
        text: qsTr("Inc view scale")
        onTriggered: {
            itemContent.scale += 0.125
        }
    }
    Action {
        id: actionScaleDec
        text: qsTr("Dec view scale")
        onTriggered: {
            if (itemContent.scale <= 0.125) {
                return
            }
            itemContent.scale -= 0.125
        }
    }
    Menu {
        id: contextMenu
        title: "Context menu"
        MenuItem {
            action: actionOpenScene
        }
        MenuSeparator {}
        MenuItem {
            action: actionUseOffscreenTexture
        }
        MenuItem {
            action: actionWireFrame
        }
    }
    menuBar: MenuBar {
        visible: visibility !== Window.FullScreen
        Menu {
            title: qsTr("&File")
            MenuItem {
                action: actionOpenScene
            }
            MenuItem {
                action: actionCloseScene
            }
            MenuSeparator {}
            MenuItem {
                action: actionExit
            }
        }
        Menu {
            title: qsTr("&Mode")
            MenuItem {
                action: actionUseOffscreenTexture
            }
            MenuItem {
                action: actionWireFrame
            }
        }
    }
    header: ToolBar {
        visible: visibility !== Window.FullScreen
        RowLayout {
            anchors.fill: parent
            Label {
                text: qsTr("Camera:")
            }
            ToolButton {
                text: qsTr("Reset")
                onClicked: sahKdTreeViewer.resetCamera()
            }
            ToolButton {
                text: qsTr("Align")
                onClicked: sahKdTreeViewer.alignCameraDirection()
            }
            ToolButton {
                text: qsTr("Reflect")
                onClicked: sahKdTreeViewer.reflectCameraDirection()
            }
            ToolButton {
                text: qsTr("Origin")
                onClicked: sahKdTreeViewer.setCameraPosition(Qt.vector3d(0.0, 0.0, 0.0))
            }
            ToolSeparator {}
            Label {
                text: qsTr("View:")
            }
            ToolButton {
                text: qsTr("Reset")
                action: actionResetContentOrientation
            }
            ToolButton {
                text: qsTr("CW")
                action: actionRotateNeg
            }
            ToolButton {
                text: qsTr("CCW")
                action: actionRotatePos
            }
            ToolButton {
                text: qsTr("+")
                action: actionScaleInc
            }
            ToolButton {
                text: qsTr("-")
                action: actionScaleDec
            }
            Item {
                Layout.fillWidth: true
            }
        }
    }
    footer: ToolBar {
        visible: visibility !== Window.FullScreen
        RowLayout {
            anchors.fill: parent
            Label {
                text: "Mode:"
            }
            Label {
                textFormat: Text.StyledText
                text: sahKdTreeViewer.modeString
            }
            Item {
                Layout.fillWidth: true
            }
            Label {
                text: {
                    "View: rotation(%1) scale(%2)"
                    .arg(itemContent.rotation)
                    .arg(itemContent.scale)
                }
            }
            Item {
                Layout.fillWidth: true
            }
            Label {
                text: "Camera:"
            }
            Label {
                textFormat: Text.StyledText
                text: sahKdTreeViewer.statusString
            }
        }
    }
    background: Rectangle {
        color: "lightgreen"
    }
    Item {
        id: itemContent
        anchors.fill: parent
        SahKdTreeViewer {
            id: sahKdTreeViewer
            objectName: "sahKdTreeViewer"
            anchors.fill: parent
            engine: SahKdTreeEngine
            useOffscreenTexture: actionUseOffscreenTexture.checked
            wireFrame: actionWireFrame.checked
            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.RightButton | Qt.LeftButton
                cursorShape: parent.cursor
                onPressed: (mouse) => {
                    parent.forceActiveFocus()
                    switch (mouse.button) {
                    case Qt.RightButton: {
                        mouse.accepted = true
                        break
                    }
                    case Qt.LeftButton: {
                        mouse.accepted = false
                    }
                    }
                }
                onClicked: (mouse) => {
                    switch (mouse.button) {
                    case Qt.RightButton: {
                        mouse.accepted = true
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
                property alias cameraPosition: sahKdTreeViewer.cameraPosition
                property alias eulerAngles: sahKdTreeViewer.eulerAngles
                property alias fieldOfView: sahKdTreeViewer.fieldOfView
                property alias scenePath: sahKdTreeViewer.scenePath
                property alias useOffscreenTexture: actionUseOffscreenTexture.checked
                property alias wireFrame: actionWireFrame.checked
            }
        }
        Rectangle {
            color: "transparent"
            x: sahKdTreeViewer.x
            y: sahKdTreeViewer.y
            width: sahKdTreeViewer.width
            height: sahKdTreeViewer.height
            anchors.margins: 4
            border.color: Qt.alpha("yellow", 0.5)
            border.width: anchors.margins
            scale: sahKdTreeViewer.scale
            transformOrigin: sahKdTreeViewer.transformOrigin
            rotation: sahKdTreeViewer.rotation
        }
        Settings {
            category: "ContentItem"
            property alias rotation: itemContent.rotation
            property alias scale: itemContent.scale
        }
    }
}
