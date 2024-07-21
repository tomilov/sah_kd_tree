import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Window
import QtQuick.Layouts
import QtQuick3D

import SahKdTree 1.0

pragma ComponentBehavior: Bound

ApplicationWindow {
    id: root
    objectName: Application.name
    function pprops(item) {
        console.log("PPROPS:")
        for (let p in item)
            console.log(p + ": " + item[p]);
    }
    visible: true
    x: Application.screens[0].width / 4
    y: Application.screens[0].height / 4
    width: Application.screens[0].width / 2
    height: Application.screens[0].height / 2
    title: {
        qsTr("%1 (screen refresh rate %2) - [%3]")
        .arg(Application.displayName)
        .arg(app.primaryScreen.refreshRate.toFixed(3))
        .arg(stackLayout.children[stackLayout.currentIndex]?.fileUrl || "-")
    }
    CenteredDialog {
        id: confirmationDialog
        title: qsTr("Close application")
        Text {
            anchors.fill: parent
            text: qsTr("Are you sure?")
        }
        standardButtons: Dialog.Yes | Dialog.No
        onAccepted: Qt.quit()
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
                let currentIndex = stackLayout.currentIndex
                listModel.remove(currentIndex)
                listModel.insert(currentIndex, listItem)
            }
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
        icon.name: "edit-find-replace-symbolic"
    }
    Action {
        id: actionReplaceScene
        text: qsTr("&Open in new tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.AddTab
        onTriggered: sceneOpenDialog.appendScene()
        icon.name: "document-open-symbolic"
    }
    Action {
        id: actionSaveSceneScreenshot
        text: qsTr("Screenshot (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Copy
        enabled: stackLayout.currentIndex >= 0
        onTriggered: stackLayout.children[stackLayout.currentIndex]?.makeScreenshot()
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
        onTriggered: root.close()
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
                tabBar.setCurrentIndex(tabBar.count - 1)
            } else {
                tabBar.decrementCurrentIndex()
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
        shortcut: StandardKey.HelpContents
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
                    append(items[i])
            }
        }
        Component.onDestruction: {
            let items = []
            for (let i = 0; i < count; ++i)
                items.push(get(i))
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
                text: fileBaseName
                onDoubleClicked: removeCurrentTab()
                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                ToolTip.timeout: 5000
                ToolTip.visible: hovered
                ToolTip.text: "<font color=\"%2\">%1</font>".arg(fileUrl).arg(Qt.color(palette.link))
            }
        }
        Component.onCompleted: Qt.callLater(setCurrentIndex, settings.currentTabIndex)
        Component.onDestruction: settings.currentTabIndex = currentIndex
    }
    StackLayout {
        id: stackLayout
        anchors.fill: parent
        currentIndex: tabBar.currentIndex
        Repeater {
            anchors.fill: parent
            model: listModel
            delegate: Component {
                Page {
                    id: page
                    required property url fileUrl
                    required property string filePath
                    required property string fileBaseName
                    readonly property string fileUrlHash: Qt.md5(fileUrl)
                    function makeScreenshot() {
                        sahKdTreeViewer.grabToImage(result => app.setClipboardImage(result.image))
                    }
                    Action {
                        id: actionResetContentOrientation
                        text: qsTr("Reset view orientation")
                        icon.name: "zoom-original-symbolic"
                        onTriggered: {
                            actionContentVisibility.checked = true
                            rotationSlider.value = 0
                            scaleSlider.value = 1
                            opacitySlider.value = 1
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
                        id: actionOpacityDec
                        text: qsTr("Decrease view opacity")
                        icon.name: "path-combine-symbolic"
                        onTriggered: {
                            opacitySlider.decrease()
                            sahKdTreeViewer.update()
                        }
                    }
                    Action {
                        id: actionOpacityInc
                        text: qsTr("Increase view opacity")
                        icon.name: "path-difference-symbolic"
                        onTriggered: {
                            opacitySlider.increase()
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
                            Text {
                                text: qsTr("<b>Camera:</b>")
                            }
                            ToolButton {
                                text: qsTr("Reset")
                                onClicked: sahKdTreeViewer.resetCameraView()
                            }
                            ToolButton {
                                text: qsTr("Align")
                                onClicked: sahKdTreeViewer.alignCameraOrientation()
                            }
                            ToolButton {
                                text: qsTr("Reflect")
                                onClicked: sahKdTreeViewer.reflectCameraOrientation()
                            }
                            ToolSeparator {}
                            Label {
                                text: qsTr("<b>View:</b>")
                            }
                            ToolButton {
                                text: qsTr("Reset")
                                action: actionResetContentOrientation
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Switch {
                                text: qsTr("Show/Hide")
                                action: actionContentVisibility
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionRotatePos
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
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
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionScaleDec
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
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
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionOpacityDec
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Slider {
                                id: opacitySlider
                                from: 0.0
                                value: 1.0
                                to: 1.0
                                stepSize: 0.0625
                                ToolTip.visible: pressed
                                ToolTip.text: value
                            }
                            ToolButton {
                                text: qsTr("")
                                action: actionOpacityInc
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                ToolTip.timeout: 5000
                                ToolTip.visible: hovered
                                ToolTip.text: action.text
                            }
                            Switch {
                                text: qsTr("Layer")
                                action: actionLayerEnabled
                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
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
                            Text {
                                text: "Mode:" + sahKdTreeViewer.modeDescription
                                ToolTip.visible: modeTextHoverHandler.hovered
                                ToolTip.text: sahKdTreeViewer.modeDescriptionVerbose
                                HoverHandler {
                                    id: modeTextHoverHandler
                                }
                            }
                            Item {
                                Layout.fillWidth: true
                            }
                            Text {
                                text: sahKdTreeViewer.cameraControllerDescription
                            }
                            Item {
                                Layout.fillWidth: true
                            }
                            Text {
                                text: {
                                    "View: rotation(%1) scale(%2)"
                                    .arg(content.rotation)
                                    .arg(content.scale)
                                }
                            }
                            Item {
                                Layout.fillWidth: true
                            }
                            Text {
                                text: "Camera:" + sahKdTreeViewer.cameraDescription
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
                        opacity: opacitySlider.value
                        layer.enabled: actionLayerEnabled.checked
                        layer.live: true
                        focus: true
                        Keys.onPressed: event => {
                            switch (event.key) {
                                case Qt.Key_0:
                                case Qt.Key_1:
                                case Qt.Key_2:
                                case Qt.Key_3:
                                case Qt.Key_4:
                                case Qt.Key_5:
                                case Qt.Key_6:
                                case Qt.Key_7:
                                case Qt.Key_8:
                                case Qt.Key_9: {
                                    let keyPrefix = "cameraView/%1/".arg(event.key)
                                    if ((event.modifiers & Qt.ControlModifier) == Qt.ControlModifier) {
                                        sceneSettings.setValue(keyPrefix + "cameraPosition", sahKdTreeViewer.cameraPosition)
                                        sceneSettings.setValue(keyPrefix + "cameraOrientation", sahKdTreeViewer.cameraOrientation)
                                        sceneSettings.setValue(keyPrefix + "cameraFieldOfView", sahKdTreeViewer.cameraFieldOfView)
                                        event.accepted = true
                                    } else if (event.modifiers === 0) {
                                        sahKdTreeViewer.cameraPosition = sceneSettings.value(keyPrefix + "cameraPosition", sahKdTreeViewer.cameraPosition)
                                        sahKdTreeViewer.cameraOrientation = sceneSettings.value(keyPrefix + "cameraOrientation", sahKdTreeViewer.cameraOrientation)
                                        sahKdTreeViewer.cameraFieldOfView = sceneSettings.value(keyPrefix + "cameraFieldOfView", sahKdTreeViewer.cameraFieldOfView)
                                        event.accepted = true
                                    }
                                    break
                                }
                            }
                        }
                        SahKdTreeViewer {
                            id: sahKdTreeViewer
                            anchors.fill: parent
                            anchors.margins: 4
                            engine: SahKdTreeEngine
                            sceneUrl: page.fileUrl
                            useOffscreenTexture: actionUseOffscreenTexture.checked
                            wireFrame: actionWireFrame.checked
                            worldScale: 1.5
                            speed: sceneAabbMax.minus(sceneAabbMin).length() * worldScale / 10.0  // 10 seconds to cross the whole world
                            Behavior on cameraPosition {
                                Vector3dAnimation {
                                    duration: 1000
                                    easing.type: Easing.InOutQuad
                                }
                            }
                            Behavior on cameraOrientation {
                                QuaternionAnimation {
                                    duration: 1000
                                    easing.type: Easing.InOutQuad
                                }
                            }
                            Behavior on cameraFieldOfView {
                                NumberAnimation {
                                    duration: 1000
                                    easing.type: Easing.InOutQuad
                                }
                            }
                        }
                        Rectangle {
                            anchors.fill: parent
                            border.color: palette.accent
                            border.width: sahKdTreeViewer.anchors.margins
                            color: "transparent"
                        }
                        MouseArea {
                            anchors.fill: parent
                            acceptedButtons: Qt.RightButton
                            cursorShape: sahKdTreeViewer.cursor
                            onClicked: mouse => {
                                switch (mouse.button) {
                                case Qt.RightButton: {
                                    contextMenu.popup()
                                    break
                                }
                                }
                            }
                        }
                        Settings {
                            id: sceneSettings
                            category: fileUrlHash
                            property alias cameraPosition: sahKdTreeViewer.cameraPosition
                            property alias cameraOrientation: sahKdTreeViewer.cameraOrientation
                            property alias cameraFieldOfView: sahKdTreeViewer.cameraFieldOfView
                        }
                    }
                    Settings {
                        category: fileUrlHash
                        property alias visible: actionContentVisibility.checked
                        property alias rotation: rotationSlider.value
                        property alias scale: scaleSlider.value
                        property alias opacity: opacitySlider.value
                        property alias layerEnabled: actionLayerEnabled.checked
                    }
                }
            }
        }
    }
    Settings {
        id: settings
        property int visibility: Window.AutomaticVisibility
        property alias x: root.x
        property alias y: root.y
        property alias width: root.width
        property alias height: root.height
        property alias useOffscreenTexture: actionUseOffscreenTexture.checked
        property alias wireFrame: actionWireFrame.checked
        property alias folderUrl: sceneOpenDialog.folderUrl
        property int currentTabIndex: -1
        property string jsonModel
    }
    Component.onCompleted: visibility = settings.visibility
    onClosing: close => {
        settings.visibility = visibility
        //confirmationDialog.open()
        //close.accepted = false
    }
}
