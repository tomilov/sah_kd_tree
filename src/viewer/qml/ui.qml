import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Dialogs as Dialogs

import SahKdTree 1.0

ApplicationWindow {
    id: root
    objectName: Qt.application.name
    visible: true
    visibility: Window.AutomaticVisibility
    function pprops(item) {
        console.log("PPROPS:")
        for (var p in item)
            //if (typeof item[p] != "function")
                console.log(p + ": " + item[p]);
    }
    readonly property var sahKdTreeViewer: swipeView.currentItem?.sahKdTreeViewerRef
    title: {
        qsTr("%1 (dt %2ms) (screen refresh rate %3) - [%4]")
        .arg(Qt.application.displayName)
        .arg(sahKdTreeViewer ? (sahKdTreeViewer.dt * 1000.0).toFixed(3) : "?")
        .arg(app.primaryScreen.refreshRate.toFixed(3))
        .arg(sahKdTreeViewer?.scenePath || "-")
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
    SceneOpenDialog {
        id: sceneOpenDialog
        width: Math.min(384, root.width)
        height: Math.min(384, root.height)
        title: qsTr("Open scene file")
        onAccepted: {
            for (var i = 0; i < listModel.count; ++i) {
                if (listModel.get(i).fileUrl === fileUrl.toString()) {
                    swipeView.setCurrentIndex(i)
                    return
                }
            }
            var listItem = {
                "filePath": filePath,
                "fileName": fileName,
                "fileUrl": fileUrl.toString(),
            }
            var currentIndex = Math.min(swipeView.currentIndex + 1, swipeView.count)
            listModel.insert(currentIndex, listItem)
            swipeView.setCurrentIndex(currentIndex)
        }
        Settings {
            category: "sceneOpenDialog"
            property alias folderUrl: sceneOpenDialog.folderUrl
        }
    }
    Dialogs.FileDialog {
        id: sceneOpenDialog2
        title: qsTr("Open scene")
        nameFilters: ["All files (*)"]
        onAccepted: if (sahKdTreeViewer) sahKdTreeViewer.scenePath = fileURL
    }
    onClosing: close => {
        if (visibility === Window.FullScreen) {
            show()
            //confirmationDialog.open()
            close.accepted = false
        }
    }
    function removeCurrentTab() {
        var currentIndex = swipeView.currentIndex
        listModel.remove(currentIndex)
        swipeView.setCurrentIndex(currentIndex - 1)
    }
    Action {
        id: actionOpenScene
        text: qsTr("&Open (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Open
        onTriggered: sceneOpenDialog.open()
    }
    Action {
        id: actionCloseScene
        text: qsTr("&Close")
        enabled: swipeView.count > 0
        onTriggered: removeCurrentTab()
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
    Settings {
        category: "root"
        property alias useOffscreenTexture: actionUseOffscreenTexture.checked
        property alias wireFrame: actionWireFrame.checked
    }
    Menu {
        id: contextMenu
        title: "Context menu"
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
    header: TabBar {
        id: tabBar
        visible: visibility !== Window.FullScreen
        currentIndex: swipeView.currentIndex
        Repeater {
            model: listModel
            TabButton {
                required property string fileName
                required property url fileUrl
                required property string index
                readonly property SwipeView swipeViewRef: swipeView
                text: fileName
                onDoubleClicked: removeCurrentTab()
                hoverEnabled: true
                ToolTip.delay: 1000
                ToolTip.timeout: 5000
                ToolTip.visible: hovered
                ToolTip.text: fileUrl

            }
        }
    }
    SwipeView {
        id: swipeView
        anchors.fill: parent
        interactive: false
        currentIndex: tabBar.currentIndex
        onCurrentItemChanged: if (currentItem) currentItem.sahKdTreeViewerRef.forceActiveFocus()
        property string jsonModel
        Settings {
            id: swipeViewSettings
            property alias jsonModel: swipeView.jsonModel
            property int currentIndex
        }
        Component.onCompleted: Qt.callLater(() => setCurrentIndex(swipeViewSettings.currentIndex))
        Component.onDestruction: swipeViewSettings.currentIndex = currentIndex
        Repeater {
            anchors.fill: parent
            model: ListModel {
                id: listModel
                Component.onCompleted: {
                    if (swipeView.jsonModel) {
                        var items = JSON.parse(swipeView.jsonModel)
                        for (var i in items) {
                            listModel.append(items[i])
                        }
                    }
                }
                Component.onDestruction: {
                    var items = []
                    for (var i = 0; i < listModel.count; ++i)
                        items.push(listModel.get(i))
                    swipeView.jsonModel = JSON.stringify(items)
                    console.log("JSON model:", swipeView.jsonModel)
                }
            }
            delegate: Component {
                Page {
                    id: page
                    required property url fileUrl
                    required property string filePath
                    required property string fileName
                    readonly property SahKdTreeViewer sahKdTreeViewerRef: sahKdTreeViewer
                    readonly property string fileUrlHash: Qt.md5(fileUrl)
                    Action {
                        id: actionContentVisibility
                        text: qsTr("Content visibility")
                        checkable: true
                        checked: true
                    }
                    Action {
                        id: actionResetContentOrientation
                        text: qsTr("Reset view orientation")
                        onTriggered: {
                            content.rotation = 0
                            content.scale = 1
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionRotatePos
                        text: qsTr("Rotate view CCW")
                        onTriggered: {
                            content.rotation -= 5
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionRotateNeg
                        text: qsTr("Rotate view CW")
                        onTriggered: {
                            content.rotation += 5
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionScaleInc
                        text: qsTr("Inc view scale")
                        onTriggered: {
                            content.scale += 0.125
                        }
                    }
                    Action {
                        id: actionScaleDec
                        text: qsTr("Dec view scale")
                        onTriggered: {
                            if (content.scale <= 0.125) {
                                return
                            }
                            content.scale -= 0.125
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
                            Switch {
                                text: qsTr("Show/Hide")
                                action: actionContentVisibility
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
                                    .arg(content.rotation)
                                    .arg(content.scale)
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
                    Item {
                        id: content
                        anchors.fill: parent
                        visible: actionContentVisibility.checked
                        Rectangle {
                            anchors.fill: parent
                            border.color: "yellow"
                            border.width: sahKdTreeViewer.anchors.margins
                            color: "transparent"
                        }
                        SahKdTreeViewer {
                            id: sahKdTreeViewer
                            anchors.fill: parent
                            anchors.margins: 4
                            engine: SahKdTreeEngine
                            scenePath: page.fileUrl
                            useOffscreenTexture: actionUseOffscreenTexture.checked
                            wireFrame: actionWireFrame.checked
                            MouseArea {
                                anchors.fill: parent
                                acceptedButtons: Qt.RightButton | Qt.LeftButton
                                cursorShape: parent.cursor
                                onPressed: mouse => {
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
                                onClicked: mouse => {
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
                                category: "SahKdTreeItem %1".arg(page.fileUrlHash)
                                property alias cameraPosition: sahKdTreeViewer.cameraPosition
                                property alias eulerAngles: sahKdTreeViewer.eulerAngles
                                property alias fieldOfView: sahKdTreeViewer.fieldOfView
                            }
                        }
                        Settings {
                            category: "Content %1".arg(page.fileUrlHash)
                            property alias rotation: content.rotation
                            property alias scale: content.scale
                            property alias visible: actionContentVisibility.checked
                        }
                    }
                }
            }
        }
    }
}
