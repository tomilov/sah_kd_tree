import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Window
import QtQuick.Layouts

import SahKdTree 1.0

ApplicationWindow {
    id: root
    objectName: Qt.application.name
    visible: true
    visibility: Window.AutomaticVisibility
    function pprops(item) {
        console.log("PPROPS:")
        for (let p in item)
            console.log(p + ": " + item[p]);
    }
    readonly property var sahKdTreeViewer: stackLayout.children[stackLayout.currentIndex]?.sahKdTreeViewerRef
    title: {
        qsTr("%1 (dt %2ms) (screen refresh rate %3) - [%4]")
        .arg(Application.displayName)
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
        property bool shouldReplaceScene
        function appendScene() {
            shouldReplaceScene = false
            open()
        }
        function replaceScene() {
            shouldReplaceScene = true
            open()
        }
        onAccepted: {
            for (let i = 0; i < listModel.count; ++i) {
                if (listModel.get(i).fileUrl === fileUrl.toString()) {
                    Qt.callLater(tabBar.setCurrentIndex, i)
                    return
                }
            }
            let listItem = {
                filePath: filePath,
                fileBaseName: fileBaseName,
                fileUrl: fileUrl.toString(),
            }
            if (!shouldReplaceScene || stackLayout.currentIndex < 0) {
                let currentIndex = listModel.count
                listModel.append(listItem)
                Qt.callLater(tabBar.setCurrentIndex, currentIndex)
            } else {
                listModel.set(stackLayout.currentIndex, listItem)
            }
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
        let currentIndex = tabBar.currentIndex
        if (currentIndex < 0) {
            return
        }
        listModel.remove(currentIndex)
    }
    Action {
        id: actionOpenScene
        text: qsTr("&Open (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Open
        onTriggered: sceneOpenDialog.replaceScene()
        icon.name: "document-open-symbolic"
    }
    Action {
        id: actionReplaceScene
        text: qsTr("&Replace (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.AddTab
        onTriggered: sceneOpenDialog.appendScene()
        icon.name: "edit-find-replace-symbolic"
    }
    Action {
        id: actionSaveSceneScreenshot
        text: qsTr("Screenshot (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Copy
        enabled: sahKdTreeViewer !== undefined
        onTriggered: sahKdTreeViewer?.grabToImage(result => app.setClipboardImage(result.image))
        icon.name: "edit-copy-symbolic"
    }
    Action {
        id: actionCloseScene
        text: qsTr("&Close")
        enabled: listModel.count > 0
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
        id: actionNextTab
        text: qsTr("Next tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.NextChild
        enabled: tabBar.count !== 0
        onTriggered: {
            if (tabBar.currentIndex + 1 == tabBar.count) {
                tabBar.setCurrentIndex(0)
            } else {
                tabBar.incrementCurrentIndex()
            }
        }
    }
    Action {
        id: actionPreviosTab
        text: qsTr("Previous tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.PreviousChild
        enabled: tabBar.count !== 0
        onTriggered: {
            if (tabBar.currentIndex == 0) {
                tabBar.decrementCurrentIndex()
            } else {
                tabBar.setCurrentIndex(tabBar.count - 1)
            }
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
        shortcut: "F4"
    }
    Action {
        id: actionShowAboutQt
        text: qsTr("About Qt")
        enabled: app.showAboutQt !== undefined
        onTriggered: Qt.callLater(app.showAboutQt)
        icon.source: app.getQtLogoUrl()
    }
    Menu {
        id: contextMenu
        title: "Context menu"
        parent: Overlay.overlay
        MenuItem {
            action: actionSaveSceneScreenshot
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
                action: actionReplaceScene
            }
            MenuItem {
                action: actionSaveSceneScreenshot
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
            title: qsTr("&Navigation")
            MenuItem {
                action: actionNextTab
            }
            MenuItem {
                action: actionPreviosTab
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
        Menu {
            title: qsTr("&Help")
            MenuItem {
                action: actionShowAboutQt
            }
        }
    }
    ListModel {
        id: listModel
        Component.onCompleted: {
            if (settings.jsonModel) {
                let items = JSON.parse(settings.jsonModel)
                for (let i in items)
                    listModel.append(items[i])
            }
        }
        Component.onDestruction: {
            let items = []
            for (let i = 0; i < listModel.count; ++i)
                items.push(listModel.get(i))
            settings.jsonModel = JSON.stringify(items)
            console.log("JSON model:", settings.jsonModel)
        }
    }
    header: TabBar {
        id: tabBar
        visible: visibility !== Window.FullScreen
        background: Pane {}
        Repeater {
            model: listModel
            TabButton {
                required property string fileBaseName
                required property url fileUrl
                required property string index
                text: fileBaseName
                onDoubleClicked: removeCurrentTab()
                hoverEnabled: true
                ToolTip.delay: 1000
                ToolTip.timeout: 5000
                ToolTip.visible: hovered
                ToolTip.text: "<font color=\"%2\">%1</font>".arg(fileUrl).arg(Qt.color(palette.link))
            }
        }
        Component.onCompleted: Qt.callLater(tabBar.setCurrentIndex, settings.currentTabIndex)
    }
    StackLayout {
        id: stackLayout
        anchors.fill: parent
        currentIndex: tabBar.currentIndex
        Component.onDestruction: settings.currentTabIndex = currentIndex
        Repeater {
            anchors.fill: parent
            model: listModel
            delegate: Component {
                Page {
                    id: page
                    required property url fileUrl
                    required property string filePath
                    required property string fileBaseName
                    readonly property SahKdTreeViewer sahKdTreeViewerRef: sahKdTreeViewer
                    readonly property string fileUrlHash: Qt.md5(fileUrl)
                    Action {
                        id: actionResetContentOrientation
                        text: qsTr("Reset view orientation")
                        icon.name: "zoom-original-symbolic"
                        onTriggered: {
                            actionContentVisibility.checked = true
                            rotationSlider.value = 0
                            scaleSlider.value = 1
                            alphaSlider.value = 1
                            actionLayerEnabled.checked = false
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionContentVisibility
                        text: qsTr("Content visibility")
                        checkable: true
                        checked: true
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
                        text: qsTr("Decrease view scale")
                        icon.name: "zoom-out-symbolic"
                        onTriggered: {
                            scaleSlider.decrease()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionScaleInc
                        text: qsTr("Increase view scale")
                        icon.name: "zoom-in-symbolic"
                        onTriggered: {
                            scaleSlider.increase()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionAlphaDec
                        text: qsTr("Decrease view opacity")
                        icon.name: "path-combine-symbolic"
                        onTriggered: {
                            alphaSlider.decrease()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionAlphaInc
                        text: qsTr("Increase view opacity")
                        icon.name: "path-difference-symbolic"
                        onTriggered: {
                            alphaSlider.increase()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionLayerEnabled
                        text: qsTr("Layer enable/disable")
                        checkable: true
                        icon.name: "application-add-symbolic"
                        onCheckedChanged: sahKdTreeViewer.update()
                    }
                    header: ToolBar {
                        visible: visibility !== Window.FullScreen
                        RowLayout {
                            anchors.fill: parent
                            Label {
                                text: qsTr("<b>Camera:</b>")
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
                                text: qsTr("<b>View:</b>")
                            }
                            ToolButton {
                                text: qsTr("Reset")
                                action: actionResetContentOrientation
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Switch {
                                text: qsTr("Show/Hide")
                                action: actionContentVisibility
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
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
                                ToolTip.visible: pressed
                                ToolTip.text: value
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
                                ToolTip.visible: pressed
                                ToolTip.text: value
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
                                stepSize: 0.0625
                                ToolTip.visible: pressed
                                ToolTip.text: value
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionAlphaInc
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Switch {
                                text: qsTr("Layer")
                                action: actionLayerEnabled
                                ToolTip.delay: 1000
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            ToolSeparator {}
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
                    background: Image {
                        fillMode: Image.Tile
                        source: app.getQtLogoUrl()
                    }
                    Item {
                        id: content
                        anchors.fill: parent
                        visible: actionContentVisibility.checked
                        scale: scaleSlider.value
                        rotation: rotationSlider.value
                        opacity: alphaSlider.value
                        layer.enabled: actionLayerEnabled.checked
                        layer.live: true
                        Rectangle {
                            anchors.fill: parent
                            border.color: palette.accent
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
                                        contextMenu.popup()
                                        mouse.accepted = true
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
                        property alias visible: actionContentVisibility.checked
                        property alias rotation: rotationSlider.value
                        property alias scale: scaleSlider.value
                        property alias opacity: alphaSlider.value
                        property alias layerEnabled: actionLayerEnabled.checked
                    }
                }
            }
        }
    }
    Settings {
        id: settings
        property alias visibility: root.visibility
        property alias useOffscreenTexture: actionUseOffscreenTexture.checked
        property alias wireFrame: actionWireFrame.checked
        property alias folderUrl: sceneOpenDialog.folderUrl
        property int currentTabIndex
        property string jsonModel
    }
}
