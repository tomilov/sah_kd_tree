import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Dialogs
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
                Qt.callLater(tabBar.setCurrentIndex, currentIndex)
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
        id: actionDiscardInvisible
        text: qsTr("Discard (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        shortcut: "F4"
    }
    Action {
        id: actionWireFrame
        text: qsTr("Wireframe (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        shortcut: "F6"
    }
    Action {
        id: actionShowAboutQt
        text: qsTr("About Qt")
        enabled: app.showAboutQt !== undefined
        onTriggered: Qt.callLater(app.showAboutQt)
        icon.source: app.getQtLogoUrl()
        shortcut: StandardKey.HelpContents
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
                action: actionDiscardInvisible
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
            //console.log("JSON model:", settings.jsonModel)
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
                    ColorDialog {
                        id: clearColorDialog
                        options: ColorDialog.ShowAlphaChannel | ColorDialog.DontUseNativeDialog | ColorDialog.NoButtons
                        onSelectedColorChanged: {
                            if (visible) // prevent feedback when WheelHandler used
                                Qt.callLater(clearColorComboBox.setIndexOfClosestColor, selectedColor)
                        }
                    }
                    Action {
                        id: actionResetContentOrientation
                        text: qsTr("Reset view orientation")
                        icon.name: "zoom-original-symbolic"
                        onTriggered: {
                            actionContentVisibility.checked = true
                            actionLayerEnabled.checked = false
                            rotationSlider.value = 0
                            scaleSlider.value = 1
                            opacitySlider.value = 1
                        }
                    }
                    Action {
                        id: actionContentVisibility
                        text: qsTr("Content visibility")
                        checkable: true
                        checked: true
                    }
                    Action {
                        id: actionLayerEnabled
                        text: qsTr("Layer enable/disable")
                        checkable: true
                        icon.name: "application-add-symbolic"
                    }
                    Action {
                        id: actionRotatePos
                        text: qsTr("Rotate view CCW")
                        icon.name: "object-rotate-left-symbolic"
                        onTriggered: rotationSlider.decrease()
                    }
                    Action {
                        id: actionRotateNeg
                        text: qsTr("Rotate view CW")
                        icon.name: "object-rotate-right-symbolic"
                        onTriggered: rotationSlider.increase()
                    }
                    Action {
                        id: actionScaleDec
                        text: qsTr("Decrease view scale")
                        icon.name: "zoom-out-symbolic"
                        onTriggered: scaleSlider.decrease()
                    }
                    Action {
                        id: actionScaleInc
                        text: qsTr("Increase view scale")
                        icon.name: "zoom-in-symbolic"
                        onTriggered: scaleSlider.increase()
                    }
                    Action {
                        id: actionOpacityDec
                        text: qsTr("Decrease view opacity")
                        icon.name: "path-combine-symbolic"
                        onTriggered: opacitySlider.decrease()
                    }
                    Action {
                        id: actionOpacityInc
                        text: qsTr("Increase view opacity")
                        icon.name: "path-difference-symbolic"
                        onTriggered: opacitySlider.increase()
                    }
                    Action {
                        id: actionSaveSceneScreenshot
                        text: qsTr("Screenshot")
                        onTriggered: sahKdTreeViewer.grabToImage(result => app.setClipboardImage(result.image))
                        icon.name: "edit-copy-symbolic"
                    }
                    Action {
                        id: actionSelectClearColor
                        text: qsTr("Select clearColor")
                        onTriggered: clearColorDialog.open()
                        icon.name: "color-select-symbolic"
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
                            action: actionDiscardInvisible
                        }
                        MenuItem {
                            action: actionWireFrame
                        }
                        MenuSeparator {}
                        MenuItem {
                            action: actionSelectClearColor
                        }
                    }
                    header: ToolBar {
                        visible: visibility !== Window.FullScreen
                        Flow {
                            anchors.fill: parent
                            RowLayout {
                                Text {
                                    text: qsTr("<b>Camera:</b>")
                                }
                                ToolButton {
                                    text: qsTr("Reset")
                                    onClicked: {
                                        sahKdTreeViewer.camera.orientation = undefined
                                        sahKdTreeViewer.camera.position = undefined
                                        sahKdTreeViewer.camera.filedOfView = undefined
                                    }
                                }
                                ToolButton {
                                    text: qsTr("Align")
                                    onClicked: sahKdTreeViewer.camera.alignOrientation()
                                }
                                ToolButton {
                                    text: qsTr("Reflect")
                                    onClicked: sahKdTreeViewer.camera.reflectOrientation()
                                }
                                ToolSeparator {
                                    Layout.fillHeight: true
                                }
                            }
                            RowLayout {
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
                                Switch {
                                    text: qsTr("Layer")
                                    action: actionLayerEnabled
                                    ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    ToolTip.timeout: 5000
                                    ToolTip.visible: hovered
                                    ToolTip.text: action.text
                                }
                            }
                            RowLayout {
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
                                    ToolTip.visible: hovered
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
                            }
                            RowLayout {
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
                                    ToolTip.visible: hovered
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
                            }
                            RowLayout {
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
                                    ToolTip.visible: hovered
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
                                ToolSeparator {
                                    Layout.fillHeight: true
                                }
                            }
                            RowLayout {
                                Text {
                                    text: qsTr("<b>Clear color:</b>")
                                }
                                ComboBox {
                                    id: clearColorComboBox
                                    textRole: "colorName"
                                    valueRole: "colorValue"
                                    implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                                    editable: true
                                    selectTextByMouse: true
                                    inputMethodHints: Qt.ImhLowercaseOnly
                                    validator: RegularExpressionValidator {
                                        regularExpression: new RegExp(app.colorNames.join("|"))
                                    }
                                    function setIndexOfClosestColor(selectedColor) {
                                        currentIndex = app.getIndexOfClosestNamedColor(selectedColor)
                                    }
                                    WheelHandler {
                                        onWheel: (wheel) => {
                                            if (wheel.angleDelta.y < 0) {
                                                if (clearColorComboBox.currentIndex + 1 < clearColorComboBox.count)
                                                    ++clearColorComboBox.currentIndex
                                            } else {
                                                if (clearColorComboBox.currentIndex > 0)
                                                    --clearColorComboBox.currentIndex
                                            }
                                            clearColorDialog.selectedColor = clearColorComboBox.currentValue
                                        }
                                    }
                                    model: ListModel {
                                        Component.onCompleted: {
                                            let colorNames = app.colorNames
                                            for (let i in colorNames) {
                                                let colorName = colorNames[i]
                                                let colorItem = {
                                                    colorName: colorName,
                                                    colorValue: Qt.color(colorName),
                                                }
                                                append(colorItem)
                                            }
                                        }
                                    }
                                    onAccepted: clearColorDialog.selectedColor = currentValue
                                    onActivated: clearColorDialog.selectedColor = currentValue
                                    delegate: ItemDelegate {
                                        id: delegate
                                        required property int index
                                        required property string colorName
                                        required property color colorValue
                                        highlighted: clearColorComboBox.highlightedIndex === index
                                        contentItem: Row {
                                            Rectangle {
                                                id: colorRect
                                                color: delegate.colorValue
                                                height: colorText.height
                                                width: height
                                                radius: height / 4
                                                border.width: 1
                                                border.color: "black"
                                            }
                                            Text {
                                                id: colorText
                                                text: delegate.colorName
                                                HoverHandler {
                                                    id: colorTextHoverHandler
                                                }
                                                ToolTip.visible: colorTextHoverHandler.hovered
                                                ToolTip.text: colorRect.color
                                            }
                                            spacing: colorText.height / 4
                                        }
                                    }
                                }
                                Rectangle {
                                    id: exactColorRect
                                    height: clearColorComboBox.height
                                    Layout.preferredWidth: height
                                    Layout.margins: height / 8
                                    Binding {
                                        exactColorRect.color: clearColorComboBox.currentValue
                                        when: clearColorComboBox.currentValue !== undefined
                                    }
                                    radius: height / 4
                                    border.width: 1
                                    border.color: "black"
                                }
                                ToolSeparator {
                                    Layout.fillHeight: true
                                }
                            }
                        }
                    }
                    footer: ToolBar {
                        visible: visibility !== Window.FullScreen
                        Flow {
                            anchors.fill: parent
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: "Mode: " + sahKdTreeViewer.modeDescription
                                        HoverHandler {
                                            id: modeTextHoverHandler
                                        }
                                        ToolTip.visible: modeTextHoverHandler.hovered
                                        ToolTip.text: sahKdTreeViewer.modeDescriptionVerbose
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: sahKdTreeViewer.cameraControllerDescription
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: {
                                            "View: rotation(%1) scale(%2)"
                                            .arg(content.rotation)
                                            .arg(content.scale)
                                        }
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: "Clear color: %1".arg(sahKdTreeViewer.clearColor)
                                    }
                                    Rectangle {
                                        Layout.fillHeight: true
                                        Layout.preferredWidth: height
                                        Layout.margins: height / 8
                                        color: Qt.alpha(sahKdTreeViewer.clearColor, 1.0)
                                        radius: height / 4
                                        border.width: 1
                                        border.color: "black"
                                    }
                                    HoverHandler {
                                        id: clearColorHoveredHandler
                                    }
                                    ToolTip.visible: clearColorHoveredHandler.hovered
                                    ToolTip.text: {
                                        'Is close to "<font color="%1">%1</font>" color'
                                        .arg(clearColorComboBox.currentText)
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: "Camera: " + sahKdTreeViewer.camera.description
                                    }
                                }
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
                        property int animationDuration: 0
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
                                    if ((event.modifiers & Qt.ControlModifier) == Qt.ControlModifier) {
                                        viewerSettings.saveCamera(event.key)
                                        event.accepted = true
                                    } else if (event.modifiers === 0) {
                                        viewerSettings.loadCamera(event.key)
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
                            discardInvisible: actionDiscardInvisible.checked
                            wireFrame: actionWireFrame.checked
                            worldScale: 1.5
                            speed: sceneAabbMax.minus(sceneAabbMin).length() * worldScale / 10.0  // 10 seconds to cross the whole world
                            camera {
                                Behavior on position {
                                    Vector3dAnimation {
                                        duration: content.animationDuration
                                        easing.type: Easing.InOutQuad
                                    }
                                }
                                Behavior on orientation {
                                    QuaternionAnimation {
                                        duration: content.animationDuration
                                        easing.type: Easing.InOutQuad
                                    }
                                }
                                Behavior on fieldOfView {
                                    NumberAnimation {
                                        duration: content.animationDuration
                                        easing.type: Easing.InOutQuad
                                    }
                                }
                            }
                            clearColor: clearColorDialog.selectedColor
                        }
                        Rectangle {
                            anchors.fill: parent
                            border.color: palette.accent
                            border.width: sahKdTreeViewer.anchors.margins
                            color: "transparent"
                        }
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: sahKdTreeViewer.cursor
                            acceptedButtons: Qt.RightButton
                            onClicked: contextMenu.popup()
                        }
                        Settings {
                            id: viewerSettings
                            category: fileUrlHash
                            property color clearColor
                            function getKeyPrefix(key) {
                                return "camera/%1/".arg(key)
                            }
                            function saveCamera(key) {
                                let keyPrefix = getKeyPrefix(key)
                                let camera = sahKdTreeViewer.camera
                                setValue(keyPrefix + "position", camera.position)
                                setValue(keyPrefix + "orientation", camera.orientation)
                                setValue(keyPrefix + "fieldOfView", camera.fieldOfView)
                            }
                            function loadCamera(key) {
                                let keyPrefix = getKeyPrefix(key)
                                let camera = sahKdTreeViewer.camera
                                camera.position = value(keyPrefix + "position", camera.position)
                                camera.orientation = value(keyPrefix + "orientation", camera.orientation)
                                camera.fieldOfView = value(keyPrefix + "fieldOfView", camera.fieldOfView)
                            }
                            Component.onCompleted: {
                                clearColorDialog.selectedColor = viewerSettings.clearColor
                                clearColorComboBox.setIndexOfClosestColor(viewerSettings.clearColor)
                                loadCamera(Qt.Key_0)
                                content.animationDuration = 1000
                            }
                            Component.onDestruction: {
                                viewerSettings.clearColor = clearColorDialog.selectedColor
                                saveCamera(Qt.Key_0)
                            }
                        }
                    }
                    Settings {
                        id: pageSettings
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
