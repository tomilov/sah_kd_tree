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
            console.log(p + ": " + item[p]);
    }
    readonly property var sahKdTreeViewer: stackLayout.children[stackLayout.currentIndex]?.sahKdTreeViewerRef
    title: {
        qsTr("%1 (dt %2ms) (screen refresh rate %3) - [%4]")
        .arg(Qt.application.displayName)
        .arg(sahKdTreeViewer ? (sahKdTreeViewer.dt * 1000.0).toFixed(3) : "?")
        .arg(app.primaryScreen.refreshRate.toFixed(3))
        .arg(sahKdTreeViewer?.sceneUrl || "-")
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
                    Qt.callLater(tabBar.setCurrentIndex, i)
                    return
                }
            }
            var listItem = {
                "filePath": filePath,
                "fileName": fileName,
                "fileUrl": fileUrl.toString(),
            }
            var currentIndex = Math.min(stackLayout.currentIndex + 1, stackLayout.count)
            listModel.insert(currentIndex, listItem)
            Qt.callLater(tabBar.setCurrentIndex, currentIndex)
        }
        Settings {
            property alias folderUrl: sceneOpenDialog.folderUrl
        }
    }
    onClosing: close => {
        if (visibility === Window.FullScreen) {
            show()
            //confirmationDialog.open()
            close.accepted = false
        }
    }
    function removeCurrentTab() {
        var currentIndex = stackLayout.currentIndex
        listModel.remove(currentIndex)
        Qt.callLater(tabBar.setCurrentIndex, currentIndex - 1)
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
        enabled: stackLayout.count > 0
        onTriggered: removeCurrentTab()
        icon.name: "close-symbolic"
    }
    Action {
        id: actionExit
        text: qsTr("&Exit (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Cancel
        onTriggered: {
            root.close()
            //confirmationDialog.open()
        }
        icon.name: "window-close-symbolic"
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
        background: Pane {}
        Repeater {
            model: listModel
            TabButton {
                required property string fileName
                required property url fileUrl
                required property string index
                readonly property StackLayout stackLayoutRef: stackLayout
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
    StackLayout {
        id: stackLayout
        anchors.fill: parent
        currentIndex: tabBar.currentIndex
        property string jsonModel
        Settings {
            id: stackLayoutSettings
            property alias jsonModel: stackLayout.jsonModel
            property int currentIndex
        }
        Component.onCompleted: Qt.callLater(tabBar.setCurrentIndex, stackLayoutSettings.currentIndex)
        Component.onDestruction: stackLayoutSettings.currentIndex = currentIndex
        Repeater {
            anchors.fill: parent
            model: ListModel {
                id: listModel
                Component.onCompleted: {
                    if (stackLayout.jsonModel) {
                        var items = JSON.parse(stackLayout.jsonModel)
                        for (var i in items)
                            listModel.append(items[i])
                    }
                }
                Component.onDestruction: {
                    var items = []
                    for (var i = 0; i < listModel.count; ++i)
                        items.push(listModel.get(i))
                    stackLayout.jsonModel = JSON.stringify(items)
                    console.log("JSON model:", stackLayout.jsonModel)
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
                        icon.name: "zoom-original-symbolic"
                        onTriggered: {
                            rotationSlider.value = 0
                            scaleSlider.value = 1
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionRotatePos
                        text: qsTr("Rotate view CCW")
                        icon.name: "object-rotate-left-symbolic"
                        onTriggered: {
                            rotationSlider.decrease()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionRotateNeg
                        text: qsTr("Rotate view CW")
                        icon.name: "object-rotate-right-symbolic"
                        onTriggered: {
                            rotationSlider.increase()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionScaleDec
                        text: qsTr("Dec view scale")
                        icon.name: "zoom-out-symbolic"
                        onTriggered: {
                            scaleSlider.decrease()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionScaleInc
                        text: qsTr("Inc view scale")
                        icon.name: "zoom-in-symbolic"
                        onTriggered: {
                            scaleSlider.increase()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionAlphaDec
                        text: qsTr("Dec view opacity")
                        icon.name: "path-combine-symbolic"
                        onTriggered: {
                            alphaSlider.decrease()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionAlphaInc
                        text: qsTr("Inc view opacity")
                        icon.name: "path-difference-symbolic"
                        onTriggered: {
                            alphaSlider.increase()
                            sahKdTreeViewer.update()
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
                                onClicked: sahKdTreeViewer.resetCameraPosition()
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
                                text: qsTr("")
                                action: actionRotatePos
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Slider {
                                id: rotationSlider
                                from: -180
                                value: 0
                                to: 180
                                stepSize: 5
                                snapMode: Slider.SnapAlways
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionRotateNeg
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionScaleDec
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Slider {
                                id: scaleSlider
                                from: 0.125
                                value: 1
                                to: 1.25
                                stepSize: 0.125
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionScaleInc
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionAlphaDec
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Slider {
                                id: alphaSlider
                                from: 0.0
                                value: 1.0
                                to: 1.0
                                stepSize: 0.1
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionAlphaInc
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
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
                    background: Rectangle {
                        color: "deepskyblue"
                    }
                    Item {
                        id: content
                        anchors.fill: parent
                        visible: actionContentVisibility.checked
                        scale: scaleSlider.value
                        rotation: rotationSlider.value
                        opacity: alphaSlider.value
                        Rectangle {
                            anchors.fill: parent
                            border.color: "gold"
                            border.width: sahKdTreeViewer.anchors.margins
                            color: "transparent"
                        }
                        SahKdTreeViewer {
                            id: sahKdTreeViewer
                            anchors.fill: parent
                            anchors.margins: 4
                            engine: SahKdTreeEngine
                            sceneUrl: page.fileUrl
                            useOffscreenTexture: actionUseOffscreenTexture.checked
                            wireFrame: actionWireFrame.checked
                            focusPolicy: Qt.WheelFocus
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
                        }
                        Settings {
                            category: fileUrlHash
                            property alias cameraPosition: sahKdTreeViewer.cameraPosition
                            property alias eulerAngles: sahKdTreeViewer.eulerAngles
                            property alias fieldOfView: sahKdTreeViewer.fieldOfView
                        }
                    }
                    Settings {
                        category: fileUrlHash
                        property alias rotation: rotationSlider.value
                        property alias scale: scaleSlider.value
                        property alias visible: actionContentVisibility.checked
                    }
                }
            }
        }
    }
}
