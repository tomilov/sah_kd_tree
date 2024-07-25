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
        console.log("PPROPS:", (typeof item).toString())
        for (let p in item)
            console.log(p + ": " + item[p]);
    }
    function coloredText(text, color) {
        return '<font color="%1">%2</font>'.arg(color).arg(text)
    }
    x: Application.screens[0].width / 4
    y: Application.screens[0].height / 4
    width: Application.screens[0].width / 2
    height: Application.screens[0].height / 2
    readonly property int toolTipTimeout: 5000
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
        width: Math.min(512, root.width)
        height: Math.min(512, root.height)
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
                /*
                for (let prop in listItem) {
                    listModel.setProperty(currentIndex, prop, listItem[prop])
                }
                */
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
        icon.name: "tab-new-symbolic"
    }
    Action {
        id: actionReplaceScene
        text: qsTr("Open in &new tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.AddTab
        onTriggered: sceneOpenDialog.appendScene()
        icon.name: "application-add-symbolic"
    }
    Action {
        id: actionCloseAllTabs
        text: qsTr("Close &all tabs")
        enabled: listModel.count > 0
        onTriggered: listModel.clear()
        icon.name: "list-remove-all-symbolic"
    }
    Action {
        id: actionCloseScene
        text: qsTr("&Close")
        enabled: listModel.count > 0
        onTriggered: removeCurrentTab()
        icon.name: "list-remove-symbolic"
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
        icon.name: "go-next-symbolic"
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
        icon.name: "go-previous-symbolic"
    }
    Action {
        id: actionUiVisibility
        text: qsTr("Toggle UI visibility")
        checkable: true
        checked: true
        shortcut: StandardKey.Replace
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
    ActionGroup {
        id: texturingModeActionGroup
        Action {
            id: actionBarycentricColor
            text: qsTr("Barycentric")
            checkable: true
        }
        Action {
            id: actionWireFrame
            text: qsTr("Wireframe")
            checkable: true
        }
        Component.onCompleted: {
            texturingModeActionGroup.actions[settings.texturingModeIndex].checked = true
        }
        Component.onDestruction: {
            for (let i in texturingModeActionGroup.actions) {
                if (texturingModeActionGroup.actions[a].checked) {
                    settings.texturingModeIndex = i
                    break
                }
            }
        }
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
        visible: actionUiVisibility.checked
        Menu {
            title: qsTr("&File")
            MenuItem {
                action: actionOpenScene
            }
            MenuItem {
                action: actionReplaceScene
            }
            MenuItem {
                action: actionCloseAllTabs
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
            MenuSeparator {}
            MenuItem {
                action: actionBarycentricColor
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
        visible: actionUiVisibility.checked
        background: Pane {}
        Repeater {
            model: listModel
            TabButton {
                required property string fileBaseName
                required property url fileUrl
                text: fileBaseName
                onDoubleClicked: removeCurrentTab()
                ToolTip.visible: hovered
                ToolTip.text: {
                    "<font color=\"%2\">%1</font>"
                    .arg(fileUrl)
                    .arg(Qt.color(palette.link))
                }
                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                ToolTip.timeout: root.toolTipTimeout
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
                        text: qsTr("Toggle content visibility")
                        checkable: true
                        checked: true
                    }
                    Action {
                        id: actionLayerEnabled
                        text: qsTr("Layer enable/disable")
                        checkable: true
                        icon.name: "image-crop-symbolic"
                    }
                    Action {
                        id: actionRotatePos
                        text: qsTr("Rotate content CCW")
                        icon.name: "object-rotate-left-symbolic"
                        onTriggered: rotationSlider.decrease()
                    }
                    Action {
                        id: actionRotateNeg
                        text: qsTr("Rotate content CW")
                        icon.name: "object-rotate-right-symbolic"
                        onTriggered: rotationSlider.increase()
                    }
                    Action {
                        id: actionScaleDec
                        text: qsTr("Decrease content scale")
                        icon.name: "zoom-out-symbolic"
                        onTriggered: scaleSlider.decrease()
                    }
                    Action {
                        id: actionScaleInc
                        text: qsTr("Increase content scale")
                        icon.name: "zoom-in-symbolic"
                        onTriggered: scaleSlider.increase()
                    }
                    Action {
                        id: actionOpacityDec
                        text: qsTr("Decrease content opacity")
                        icon.name: "path-combine-symbolic"
                        onTriggered: opacitySlider.decrease()
                    }
                    Action {
                        id: actionOpacityInc
                        text: qsTr("Increase content opacity")
                        icon.name: "path-difference-symbolic"
                        onTriggered: opacitySlider.increase()
                    }
                    Action {
                        id: actionSaveSceneScreenshot
                        text: qsTr("Screenshot")
                        onTriggered: viewer.grabToImage(result => app.setClipboardImage(result.image))
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
                        MenuSeparator {}
                        MenuItem {
                            action: actionBarycentricColor
                        }
                        MenuItem {
                            action: actionWireFrame
                        }
                        MenuSeparator {}
                        MenuItem {
                            action: actionSelectClearColor
                        }
                        MenuSeparator {}
                        MenuItem {
                            text: qsTr("Dump item tree")
                            onTriggered: root.contentItem.dumpItemTree()
                        }
                        MenuItem {
                            text: qsTr("Make window invisible")
                            onTriggered: root.hide()
                        }
                        MenuItem {
                            text: qsTr("Renderdoc capture frame")
                            onTriggered: viewer.renderer.renderdocCaptureFrame()
                        }
                    }
                    header: ToolBar {
                        visible: actionUiVisibility.checked
                        Flow {
                            anchors.fill: parent
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    ToolButton {
                                        text: qsTr("Reset cam")
                                        onClicked: {
                                            viewer.cameraView.orientation = undefined
                                            viewer.cameraView.position = undefined
                                            viewer.cameraView.filedOfView = undefined
                                        }
                                        ToolTip.visible: hovered
                                        ToolTip.text: qsTr("Reset camera view")
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                    ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    ToolButton {
                                        text: qsTr("Align cam")
                                        onClicked: viewer.cameraView.alignOrientation()
                                        ToolTip.visible: hovered
                                        ToolTip.text: qsTr("Align camera view")
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                    ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    ToolButton {
                                        text: qsTr("Reflect cam")
                                        onClicked: viewer.cameraView.reflectOrientation()
                                        ToolTip.visible: hovered
                                        ToolTip.text: qsTr("Reflect camera view")
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    ToolButton {
                                        text: qsTr("Reset item")
                                        action: actionResetContentOrientation
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                    ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    Switch {
                                        text: qsTr("Show/Hide")
                                        action: actionContentVisibility
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                    ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    Switch {
                                        text: qsTr("Layer")
                                        action: actionLayerEnabled
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    ToolButton {
                                        text: qsTr("")
                                        action: actionRotatePos
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                    Slider {
                                        id: rotationSlider
                                        from: -180
                                        value: 0
                                        to: 180
                                        stepSize: 5
                                        snapMode: Slider.SnapAlways
                                        ToolTip.visible: pressed || hovered
                                        ToolTip.text: value
                                        WheelHandler {
                                            onWheel: (wheel) => {
                                                if (wheel.angleDelta.y < 0) {
                                                    rotationSlider.increase()
                                                } else {
                                                    rotationSlider.decrease()
                                                }
                                            }
                                        }
                                    }
                                    ToolButton {
                                        text: qsTr("")
                                        action: actionRotateNeg
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    ToolButton {
                                        text: qsTr("")
                                        action: actionScaleDec
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                    Slider {
                                        id: scaleSlider
                                        from: 0.125
                                        value: 1
                                        to: 1.25
                                        stepSize: 0.125
                                        ToolTip.visible: pressed || hovered
                                        ToolTip.text: value
                                        WheelHandler {
                                            onWheel: (wheel) => {
                                                if (wheel.angleDelta.y < 0) {
                                                    scaleSlider.increase()
                                                } else {
                                                    scaleSlider.decrease()
                                                }
                                            }
                                        }
                                    }
                                    ToolButton {
                                        text: qsTr("")
                                        action: actionScaleInc
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    ToolButton {
                                        text: qsTr("")
                                        action: actionOpacityDec
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                    Slider {
                                        id: opacitySlider
                                        from: 0.0
                                        value: 1.0
                                        to: 1.0
                                        stepSize: 0.0625
                                        ToolTip.visible: pressed || hovered
                                        ToolTip.text: value
                                        WheelHandler {
                                            onWheel: (wheel) => {
                                                if (wheel.angleDelta.y < 0) {
                                                    opacitySlider.increase()
                                                } else {
                                                    opacitySlider.decrease()
                                                }
                                            }
                                        }
                                    }
                                    ToolButton {
                                        text: qsTr("")
                                        action: actionOpacityInc
                                        ToolTip.visible: hovered
                                        ToolTip.text: action.text
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
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
                                        HoverHandler {
                                            id: clearColorComboBoxHoverHandler
                                        }
                                        ToolTip.visible: clearColorComboBoxHoverHandler.hovered
                                        ToolTip.text: currentText
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                        model: ListModel {
                                            Component.onCompleted: {
                                                let colorNames = app.colorNames
                                                for (let i in colorNames) {
                                                    let colorName = colorNames[i]
                                                    let colorItem = {
                                                        colorName: colorName,
                                                        colorValue: Qt.color(colorName).toString(),
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
                                                }
                                                spacing: colorText.height / 4
                                                HoverHandler {
                                                    id: colorRowHoverHandler
                                                }
                                                ToolTip.visible: colorRowHoverHandler.hovered
                                                ToolTip.text: colorRect.color
                                                ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                                ToolTip.timeout: root.toolTipTimeout
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
                                        HoverHandler {
                                            id: colorSquareHoverHandler
                                        }
                                        ToolTip.visible: colorSquareHoverHandler.hovered && clearColorComboBox.currentValue !== undefined
                                        ToolTip.text: clearColorComboBox.currentValue
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                        }
                    }
                    footer: ToolBar {
                        visible: actionUiVisibility.checked
                        Flow {
                            anchors.fill: parent
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: viewer.getRenderModeDescription(false)
                                        HoverHandler {
                                            id: modeTextHoverHandler
                                        }
                                        ToolTip.visible: modeTextHoverHandler.hovered
                                        ToolTip.text: viewer.getRenderModeDescription(true)
                                        ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: {
                                            "sens(%1) speed(%2)"
                                            .arg(viewer.cameraController.sensitivity.toFixed(4))
                                            .arg(viewer.cameraController.speed.toExponential(3))
                                        }
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: {
                                            "rot(%1) scale(%2)"
                                            .arg(content.rotation.toFixed(0))
                                            .arg(content.scale.toFixed(3))
                                        }
                                    }
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: "Clear color: %1".arg(viewer.renderer.clearColor)
                                    }
                                    Rectangle {
                                        Layout.fillHeight: true
                                        Layout.preferredWidth: height
                                        Layout.margins: height / 8
                                        color: Qt.alpha(viewer.renderer.clearColor, 1.0)
                                        radius: height / 4
                                        border.width: 1
                                        border.color: "black"
                                    }
                                    HoverHandler {
                                        id: clearColorHoveredHandler
                                    }
                                    ToolTip.visible: clearColorHoveredHandler.hovered
                                    ToolTip.text: {
                                        return 'Is %1 "<font color="%2">%3</font>" color'
                                            .arg(clearColorDialog.selectedColor === Qt.color(clearColorComboBox.currentText) ? "exactly" : "roughly")
                                            .arg(Qt.alpha(clearColorComboBox.currentValue, 1.0))
                                            .arg(clearColorComboBox.currentText)
                                    }
                                    ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                            Frame {
                                RowLayout {
                                    anchors.fill: parent
                                    Text {
                                        text: {
                                            let position = viewer.cameraView.position
                                            let orientation = viewer.cameraView.orientation.toEulerAngles()
                                            return "xyz(%1, %2, %3) \u03C6\u03B8\u03C8(%4, %5, %6) fov(%7)"
                                                .arg(position.x.toExponential(3)).arg(position.y.toExponential(3)).arg(position.z.toExponential(3))
                                                .arg(orientation.x.toFixed(1)).arg(orientation.y.toFixed(1)).arg(orientation.z.toFixed(1))
                                                .arg(viewer.cameraView.fov.toFixed(0))
                                        }
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
                                        viewerSettings.saveCameraView(event.key)
                                        event.accepted = true
                                    } else if (event.modifiers === 0) {
                                        viewerSettings.loadCameraView(event.key, true)
                                        event.accepted = true
                                    }
                                    break
                                }
                            }
                        }
                        Rectangle {
                            id: boundingRect
                            anchors.fill: parent
                            border {
                                color: clearColorDialog.selectedColor
                                width: 4
                            }
                            color: "transparent"
                        }
                        Viewer {
                            id: viewer
                            objectName: fileUrlHash
                            anchors.fill: boundingRect
                            anchors.margins: boundingRect.border.width
                            readonly property int animationDuration: 1000
                            function getRenderModeDescription(verbose) {
                                let description = []
                                let renderMode = viewer.renderer.renderMode
                                if (renderMode & RendererSettings.UseOffscreenTexture) {
                                    description.push(coloredText(verbose ? "Use offscreen texture" : "O", "fuchsia"))
                                }
                                if (renderMode & RendererSettings.DiscardInvisibleFragments) {
                                    description.push(coloredText(verbose ? "Discard invisible pixels" : "D", "blue"))
                                }
                                let texturingMode
                                switch (viewer.renderer.texturingMode) {
                                case RendererSettings.BarycentricColor: {
                                    texturingMode = verbose ? "Barycentric Color" : "B";
                                    break
                                }
                                case RendererSettings.WireFrame: {
                                    texturingMode = verbose ? "Wireframe" : "W";
                                    break
                                }
                                }
                                description.push(coloredText(texturingMode, "green"))
                                return "%1<b>%2</b>"
                                    .arg(verbose ? "" : "Mode: ")
                                    .arg(description.join(verbose ? " AND " : "|"))
                            }
                            engine: SahKdTreeEngine
                            scene {
                                url: page.fileUrl
                                worldScale: 1.5
                            }
                            renderer {
                                renderMode: {
                                    let value = 0
                                    if (actionUseOffscreenTexture.checked) {
                                        value |= RendererSettings.UseOffscreenTexture
                                    }
                                    if (actionDiscardInvisible.checked) {
                                        value |= RendererSettings.DiscardInvisibleFragments
                                    }
                                    return value
                                }
                                texturingMode: {
                                    if (actionBarycentricColor.checked) {
                                        return RendererSettings.BarycentricColor
                                    }
                                    if (actionWireFrame.checked) {
                                        return RendererSettings.WireFrame
                                    }
                                }
                                clearColor: clearColorDialog.selectedColor
                            }
                            cameraController {
                                speed: scene.sceneAabbMax.minus(scene.sceneAabbMin).length() * scene.worldScale / 10.0  // 10 seconds to cross the whole world
                            }
                            cameraView {
                                Behavior on position {
                                    Vector3dAnimation {
                                        duration: viewer.animationDuration
                                        easing.type: Easing.InOutQuad
                                    }
                                }
                                Behavior on orientation {
                                    QuaternionAnimation {
                                        duration: viewer.animationDuration
                                        easing.type: Easing.InOutQuad
                                    }
                                }
                                Behavior on fov {
                                    NumberAnimation {
                                        duration: viewer.animationDuration
                                        easing.type: Easing.InOutQuad
                                    }
                                }
                            }
                        }
                        MouseArea {
                            anchors.fill: parent
                            cursorShape: viewer.cursor
                            acceptedButtons: Qt.RightButton
                            onClicked: contextMenu.popup()
                        }
                        Settings {
                            id: viewerSettings
                            category: fileUrlHash
                            property color clearColor
                            function getKeyPrefix(key) {
                                return "cameraView/%1/".arg(key)
                            }
                            function saveCameraView(key) {
                                let keyPrefix = getKeyPrefix(key)
                                let cameraView = viewer.cameraView
                                setValue(keyPrefix + "position", cameraView.position)
                                setValue(keyPrefix + "orientation", cameraView.orientation)
                                setValue(keyPrefix + "fov", cameraView.fov)
                            }
                            function loadCameraView(key, animate) {
                                let keyPrefix = getKeyPrefix(key)
                                let cameraView = viewer.cameraView
                                let position = value(keyPrefix + "position", cameraView.position)
                                let orientation = value(keyPrefix + "orientation", cameraView.orientation)
                                let fov = value(keyPrefix + "fov", cameraView.fov)
                                if (animate) {
                                    cameraView.position = position
                                    cameraView.orientation = orientation
                                    cameraView.fov = fov
                                } else {
                                    cameraView.setPosition(position)
                                    cameraView.setOrientation(orientation)
                                    cameraView.setFov(fov)
                                }
                            }
                            Component.onCompleted: {
                                clearColorDialog.selectedColor = viewerSettings.clearColor
                                clearColorComboBox.setIndexOfClosestColor(viewerSettings.clearColor)
                                loadCameraView(Qt.Key_0, false)
                            }
                            Component.onDestruction: {
                                viewerSettings.clearColor = clearColorDialog.selectedColor
                                saveCameraView(Qt.Key_0)
                            }
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
        property int rootVisibility: Window.AutomaticVisibility
        property alias x: root.x
        property alias y: root.y
        property alias width: root.width
        property alias height: root.height
        property alias uiVisibility: actionUiVisibility.checked
        property alias useOffscreenTexture: actionUseOffscreenTexture.checked
        property int texturingModeIndex: 0
        property int currentTabIndex: -1
        property string jsonModel
    }
    Component.onCompleted: visibility = settings.rootVisibility
    onClosing: close => {
        settings.rootVisibility = visibility
        //confirmationDialog.open()
        //close.accepted = false
    }
    Timer {
        id: visibilityTimer
        interval: 2000
        repeat: false
        onTriggered: root.show()
    }
    onVisibleChanged: {
        if (!visible) {
            visibilityTimer.start()
        }
    }
}
