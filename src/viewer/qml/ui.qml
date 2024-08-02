import QtCore
import QtQuick
import QtQuick.Controls as QQC
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Dialogs as Dialogs
import QtQuick3D

import SahKdTree 1.0

import "utils.js" as Utils


pragma ComponentBehavior: Bound

QQC.ApplicationWindow {
    id: root
    objectName: Application.name
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
    SceneOpenDialog {
        id: sceneOpenDialog
        title: qsTr("Open scene file")
        width: Math.min(Math.max(implicitWidth, 512), parent.width)
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
    QQC.Action {
        id: actionOpenScene
        text: qsTr("&Open (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Open
        onTriggered: sceneOpenDialog.replaceScene()
        icon.name: "tab-new-symbolic"
    }
    QQC.Action {
        id: actionReplaceScene
        text: qsTr("Open in &new tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.AddTab
        onTriggered: sceneOpenDialog.appendScene()
        icon.name: "application-add-symbolic"
    }
    QQC.Action {
        id: actionCloseAllTabs
        text: qsTr("Close &all tabs")
        enabled: listModel.count > 0
        onTriggered: listModel.clear()
        icon.name: "list-remove-all-symbolic"
    }
    QQC.Action {
        id: actionCloseScene
        text: qsTr("&Close")
        enabled: listModel.count > 0
        onTriggered: removeCurrentTab()
        icon.name: "list-remove-symbolic"
    }
    QQC.Action {
        id: actionExit
        text: qsTr("&Exit (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Cancel
        onTriggered: root.close()
        icon.name: "window-close-symbolic"
    }
    QQC.Action {
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
    QQC.Action {
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
    QQC.Action {
        id: actionUiVisibility
        text: qsTr("Toggle UI visibility")
        checkable: true
        checked: true
        shortcut: StandardKey.Replace
    }
    QQC.Action {
        id: actionUseOffscreenTexture
        text: qsTr("Offscreen (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        checked: true
        shortcut: "F4"
    }
    QQC.Action {
        id: actionDiscardInvisible
        text: qsTr("Discard")
        checkable: true
    }
    QQC.Action {
        id: actionTraceSahKdTree
        text: qsTr("Trace/Rasterize (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        shortcut: "F2"
    }
    QQC.ActionGroup {
        id: texturingModeActionGroup
        QQC.Action {
            id: actionBarycentricColor
            text: qsTr("Barycentric")
            checkable: true
        }
        QQC.Action {
            id: actionWireFrame
            text: qsTr("Wireframe")
            checkable: true
        }
        Component.onCompleted: {
            texturingModeActionGroup.actions[settings.texturingModeIndex].checked = true
        }
        Component.onDestruction: {
            for (let i in texturingModeActionGroup.actions) {
                if (texturingModeActionGroup.actions[i].checked) {
                    settings.texturingModeIndex = i
                    break
                }
            }
        }
    }
    QQC.Action {
        id: actionShowAboutQt
        text: qsTr("About Qt")
        enabled: app.showAboutQt !== undefined
        onTriggered: Qt.callLater(app.showAboutQt)
        icon.source: app.getQtLogoUrl()
        shortcut: StandardKey.HelpContents
    }
    QQC.Action {
        id: actionShowTaskQueueDialog
        text: qsTr("Show task queue info")
        onTriggered: taskQueueDialog.open()
        icon.name: "view-list-symbolic"
    }
    menuBar: QQC.MenuBar {
        visible: actionUiVisibility.checked
        QQC.Menu {
            title: qsTr("&File")
            QQC.MenuItem {
                action: actionOpenScene
            }
            QQC.MenuItem {
                action: actionReplaceScene
            }
            QQC.MenuItem {
                action: actionCloseAllTabs
            }
            QQC.MenuItem {
                action: actionCloseScene
            }
            QQC.MenuSeparator {}
            QQC.MenuItem {
                action: actionExit
            }
        }
        QQC.Menu {
            title: qsTr("&Navigation")
            QQC.MenuItem {
                action: actionNextTab
            }
            QQC.MenuItem {
                action: actionPreviosTab
            }
        }
        QQC.Menu {
            title: qsTr("&Mode")
            QQC.MenuItem {
                action: actionUseOffscreenTexture
            }
            QQC.MenuItem {
                action: actionDiscardInvisible
            }
            QQC.MenuSeparator {}
            QQC.MenuItem {
                action: actionBarycentricColor
            }
            QQC.MenuItem {
                action: actionWireFrame
            }
        }
        QQC.Menu {
            title: qsTr("&Help")
            QQC.MenuItem {
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
    TaskQueue {
        id: taskQueue
    }
    CenteredDialog {
        id: taskQueueDialog
        title: "Task queue"
        width: Math.min(Math.max(implicitWidth, 512), parent.width)
        standardButtons: QQC.Dialog.Close
        contentItem: QQC.Frame {
            TableView {
                anchors.fill: parent
                onContentWidthChanged: {
                    if (implicitWidth < contentWidth) {
                        implicitWidth = contentWidth
                    }
                }
                onContentHeightChanged: {
                    if (implicitHeight < contentHeight) {
                        implicitHeight = contentHeight
                    }
                }
                clip: true
                model: taskQueue
                delegate: QQC.ItemDelegate {
                    id: tableViewDelegate
                    required property var modelData
                    contentItem: Text {
                        id: tableFiledText
                        clip: true
                        readonly property var display: tableViewDelegate.modelData.display
                        readonly property var tooltip: tableViewDelegate.modelData.tooltip
                        text: display !== undefined ? display : ""
                        HoverHandler {
                            id: tableFieldTextHoverHandler
                        }
                        QQC.ToolTip.visible: tableFieldTextHoverHandler.hovered
                        QQC.ToolTip.text: tooltip !== undefined ? tooltip : ""
                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                        QQC.ToolTip.timeout: root.toolTipTimeout
                    }
                }
                QQC.ScrollIndicator.vertical: QQC.ScrollIndicator {}
                QQC.ScrollIndicator.horizontal: QQC.ScrollIndicator {}
            }
        }
    }
    header: QQC.TabBar {
        id: tabBar
        visible: actionUiVisibility.checked
        background: QQC.Pane {}
        Repeater {
            model: listModel
            QQC.TabButton {
                required property string fileBaseName
                required property url fileUrl
                text: fileBaseName
                onDoubleClicked: removeCurrentTab()
                QQC.ToolTip.visible: hovered
                QQC.ToolTip.text: {
                    "<font color=\"%2\">%1</font>"
                    .arg(fileUrl)
                    .arg(Qt.color(palette.link))
                }
                QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                QQC.ToolTip.timeout: root.toolTipTimeout
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
                QQC.Page {
                    id: page
                    required property url fileUrl
                    required property string filePath
                    required property string fileBaseName
                    readonly property string fileUrlHash: Qt.md5(fileUrl)
                    Dialogs.ColorDialog {
                        id: clearColorDialog
                        options: Dialogs.ColorDialog.ShowAlphaChannel | Dialogs.ColorDialog.DontUseNativeDialog | Dialogs.ColorDialog.NoButtons
                        onSelectedColorChanged: {
                            if (visible) // prevent feedback when WheelHandler used
                                Qt.callLater(clearColorComboBox.setIndexOfClosestColor, selectedColor)
                        }
                    }
                    QQC.Action {
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
                    QQC.Action {
                        id: actionContentVisibility
                        text: qsTr("Toggle content visibility")
                        checkable: true
                        checked: true
                    }
                    QQC.Action {
                        id: actionLayerEnabled
                        text: qsTr("Layer enable/disable")
                        checkable: true
                        icon.name: "image-crop-symbolic"
                    }
                    QQC.Action {
                        id: actionRotatePos
                        text: qsTr("Rotate content CCW")
                        icon.name: "object-rotate-left-symbolic"
                        onTriggered: rotationSlider.decrease()
                    }
                    QQC.Action {
                        id: actionRotateNeg
                        text: qsTr("Rotate content CW")
                        icon.name: "object-rotate-right-symbolic"
                        onTriggered: rotationSlider.increase()
                    }
                    QQC.Action {
                        id: actionScaleDec
                        text: qsTr("Decrease content scale")
                        icon.name: "zoom-out-symbolic"
                        onTriggered: scaleSlider.decrease()
                    }
                    QQC.Action {
                        id: actionScaleInc
                        text: qsTr("Increase content scale")
                        icon.name: "zoom-in-symbolic"
                        onTriggered: scaleSlider.increase()
                    }
                    QQC.Action {
                        id: actionOpacityDec
                        text: qsTr("Decrease content opacity")
                        icon.name: "path-combine-symbolic"
                        onTriggered: opacitySlider.decrease()
                    }
                    QQC.Action {
                        id: actionOpacityInc
                        text: qsTr("Increase content opacity")
                        icon.name: "path-difference-symbolic"
                        onTriggered: opacitySlider.increase()
                    }
                    QQC.Action {
                        id: actionSaveSceneScreenshot
                        text: qsTr("Screenshot")
                        onTriggered: viewer.grabToImage(result => app.setClipboardImage(result.image))
                        icon.name: "edit-copy-symbolic"
                    }
                    QQC.Action {
                        id: actionSelectClearColor
                        text: qsTr("Select clearColor")
                        onTriggered: clearColorDialog.open()
                        icon.name: "color-select-symbolic"
                    }
                    QQC.Action {
                        id: actionChangeTreeBuildParams
                        text: qsTr("Change SAH kd-tree build parameters")
                        onTriggered: treeParametersDialog.open()
                        icon.name: "edit-symbolic"
                    }
                    QQC.Menu {
                        id: contextMenu
                        title: "Context menu"
                        parent: QQC.Overlay.overlay
                        QQC.MenuItem {
                            action: actionSaveSceneScreenshot
                        }
                        QQC.MenuSeparator {}
                        QQC.MenuItem {
                            action: actionTraceSahKdTree
                        }
                        QQC.MenuItem {
                            action: actionUseOffscreenTexture
                        }
                        QQC.MenuItem {
                            action: actionDiscardInvisible
                        }
                        QQC.MenuSeparator {}
                        QQC.MenuItem {
                            action: actionBarycentricColor
                        }
                        QQC.MenuItem {
                            action: actionWireFrame
                        }
                        QQC.MenuSeparator {}
                        QQC.MenuItem {
                            action: actionSelectClearColor
                        }
                        QQC.MenuItem {
                            action: actionChangeTreeBuildParams
                        }
                        QQC.MenuItem {
                            action: actionShowTaskQueueDialog
                        }
                        QQC.MenuSeparator {}
                        QQC.MenuItem {
                            text: qsTr("Dump item tree")
                            onTriggered: root.contentItem.dumpItemTree()
                        }
                        QQC.MenuItem {
                            text: qsTr("Make window invisible")
                            onTriggered: root.hide()
                        }
                        QQC.MenuItem {
                            text: qsTr("Renderdoc capture frame")
                            onTriggered: viewer.renderer.renderdocCaptureFrame()
                        }
                    }
                    header: QQC.ToolBar {
                        visible: actionUiVisibility.checked
                        Flow {
                            anchors.fill: parent
                            QQC.Frame {
                                RowLayout {
                                    QQC.ToolButton {
                                        text: qsTr("Reset cam")
                                        onClicked: {
                                            viewer.cameraView.orientation = undefined
                                            viewer.cameraView.position = undefined
                                            viewer.cameraView.filedOfView = undefined
                                        }
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: qsTr("Reset camera view")
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                    QQC.ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    QQC.ToolButton {
                                        text: qsTr("Align cam")
                                        onClicked: viewer.cameraView.alignOrientation()
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: qsTr("Align camera view")
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                    QQC.ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    QQC.ToolButton {
                                        text: qsTr("Reflect cam")
                                        onClicked: viewer.cameraView.reflectOrientation()
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: qsTr("Reflect camera view")
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    QQC.ToolButton {
                                        text: qsTr("Reset item")
                                        action: actionResetContentOrientation
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                    QQC.ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    QQC.Switch {
                                        text: qsTr("Show/Hide")
                                        action: actionContentVisibility
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                    QQC.ToolSeparator {
                                        Layout.fillHeight: true
                                    }
                                    QQC.Switch {
                                        text: qsTr("Layer")
                                        action: actionLayerEnabled
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    QQC.ToolButton {
                                        text: qsTr("")
                                        action: actionRotatePos
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                    QQC.Slider {
                                        id: rotationSlider
                                        from: -180
                                        value: 0
                                        to: 180
                                        stepSize: 5
                                        snapMode: QQC.Slider.SnapAlways
                                        QQC.ToolTip.visible: pressed || hovered
                                        QQC.ToolTip.text: value
                                        WheelHandler {
                                            onWheel: (wheel) => {
                                                if (wheel.angleDelta.y < 0) {
                                                    rotationSlider.decrease()
                                                } else {
                                                    rotationSlider.increase()
                                                }
                                            }
                                        }
                                    }
                                    QQC.ToolButton {
                                        text: qsTr("")
                                        action: actionRotateNeg
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    QQC.ToolButton {
                                        text: qsTr("")
                                        action: actionScaleDec
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                    QQC.Slider {
                                        id: scaleSlider
                                        from: 0.125
                                        value: 1
                                        to: 1.25
                                        stepSize: 0.125
                                        QQC.ToolTip.visible: pressed || hovered
                                        QQC.ToolTip.text: value
                                        WheelHandler {
                                            onWheel: (wheel) => {
                                                if (wheel.angleDelta.y < 0) {
                                                    scaleSlider.decrease()
                                                } else {
                                                    scaleSlider.increase()
                                                }
                                            }
                                        }
                                    }
                                    QQC.ToolButton {
                                        text: qsTr("")
                                        action: actionScaleInc
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    QQC.ToolButton {
                                        text: qsTr("")
                                        action: actionOpacityDec
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                    QQC.Slider {
                                        id: opacitySlider
                                        from: 0.0
                                        value: 1.0
                                        to: 1.0
                                        stepSize: 0.0625
                                        QQC.ToolTip.visible: pressed || hovered
                                        QQC.ToolTip.text: value
                                        WheelHandler {
                                            onWheel: (wheel) => {
                                                if (wheel.angleDelta.y < 0) {
                                                    opacitySlider.decrease()
                                                } else {
                                                    opacitySlider.increase()
                                                }
                                            }
                                        }
                                    }
                                    QQC.ToolButton {
                                        text: qsTr("")
                                        action: actionOpacityInc
                                        QQC.ToolTip.visible: hovered
                                        QQC.ToolTip.text: action.text
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    Text {
                                        text: qsTr("<b>Clear color:</b>")
                                    }
                                    QQC.ComboBox {
                                        id: clearColorComboBox
                                        textRole: "colorName"
                                        valueRole: "colorValue"
                                        implicitContentWidthPolicy: QQC.ComboBox.WidestTextWhenCompleted
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
                                                    if (clearColorComboBox.currentIndex > 0)
                                                        --clearColorComboBox.currentIndex
                                                } else {
                                                    if (clearColorComboBox.currentIndex + 1 < clearColorComboBox.count)
                                                        ++clearColorComboBox.currentIndex
                                                }
                                                clearColorDialog.selectedColor = clearColorComboBox.currentValue
                                            }
                                        }
                                        HoverHandler {
                                            id: clearColorComboBoxHoverHandler
                                        }
                                        QQC.ToolTip.visible: clearColorComboBoxHoverHandler.hovered
                                        QQC.ToolTip.text: currentText
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
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
                                        delegate: QQC.ItemDelegate {
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
                                                QQC.ToolTip.visible: colorRowHoverHandler.hovered
                                                QQC.ToolTip.text: colorRect.color
                                                QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                                QQC.ToolTip.timeout: root.toolTipTimeout
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
                                        QQC.ToolTip.visible: colorSquareHoverHandler.hovered && clearColorComboBox.currentValue !== undefined
                                        QQC.ToolTip.text: clearColorComboBox.currentValue
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                        }
                    }
                    footer: QQC.ToolBar {
                        visible: actionUiVisibility.checked
                        Flow {
                            anchors.fill: parent
                            QQC.Frame {
                                RowLayout {
                                    Text {
                                        text: viewer.getRenderModeDescription(false)
                                        HoverHandler {
                                            id: modeTextHoverHandler
                                        }
                                        QQC.ToolTip.visible: modeTextHoverHandler.hovered
                                        QQC.ToolTip.text: viewer.getRenderModeDescription(true)
                                        QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QQC.ToolTip.timeout: root.toolTipTimeout
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    Text {
                                        text: {
                                            qsTr("sens(%1) speed(%2)")
                                            .arg(viewer.cameraController.sensitivity.toFixed(4))
                                            .arg(viewer.cameraController.speed.toExponential(3))
                                        }
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    Text {
                                        text: {
                                            qsTr("rot(%1) scale(%2)")
                                            .arg(content.rotation.toFixed(0))
                                            .arg(content.scale.toFixed(3))
                                        }
                                    }
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    Text {
                                        text: qsTr("Clear color: %1").arg(viewer.renderer.clearColor)
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
                                    QQC.ToolTip.visible: clearColorHoveredHandler.hovered
                                    QQC.ToolTip.text: {
                                        return qsTr('Is %1 "<font color="%2">%3</font>" color')
                                            .arg(clearColorDialog.selectedColor === Qt.color(clearColorComboBox.currentText) ? "exactly" : "roughly")
                                            .arg(Qt.alpha(clearColorComboBox.currentValue, 1.0))
                                            .arg(clearColorComboBox.currentText)
                                    }
                                    QQC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QQC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                            QQC.Frame {
                                RowLayout {
                                    Text {
                                        text: {
                                            let position = viewer.cameraView.position
                                            let orientation = viewer.cameraView.orientation.toEulerAngles()
                                            return qsTr("xyz(%1, %2, %3) \u03C6\u03B8\u03C8(%4, %5, %6) fov(%7)")
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
                        Dialogs.MessageDialog {
                            id: buildFailMessageBox
                            text: qsTr("Failed to build SAH kd-tree")
                            informativeText: sceneSettings.treeStatus
                            detailedText: qsTr("Try to change SAH kd-tree build parameters")
                            buttons: Dialogs.MessageDialog.Ok
                            Connections {
                                target: sceneSettings
                                function onTreeStatusChanged() {
                                    if (sceneSettings.treeStatus)
                                        buildFailMessageBox.open()
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
                                if (renderMode & RendererSettings.TraceSahKdTree) {
                                    description.push(Utils.coloredText(verbose ? "Trace SAH kd-tree" : "T", "red"))
                                } else {
                                    description.push(Utils.coloredText(verbose ? "Rasterize" : "R", "springgreen"))
                                }
                                if (renderMode & RendererSettings.UseOffscreenTexture) {
                                    description.push(Utils.coloredText(verbose ? "Use offscreen texture" : "O", "fuchsia"))
                                }
                                if (renderMode & RendererSettings.DiscardInvisibleFragments) {
                                    description.push(Utils.coloredText(verbose ? "Discard invisible pixels" : "D", "blue"))
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
                                description.push(Utils.coloredText(texturingMode, "green"))
                                return "%1<b>%2</b>"
                                    .arg(verbose ? "" : "Mode: ")
                                    .arg(description.join(verbose ? " OR " : "|"))
                            }
                            engine: SahKdTreeEngine
                            taskQueue: taskQueue
                            scene: SceneSettings {
                                id: sceneSettings
                                url: page.fileUrl
                            }
                            CenteredDialog {
                                id: treeParametersDialog
                                title: qsTr("Tree parameters")
                                QQC.Frame {
                                    GridLayout {
                                        columns: 2
                                        Text {
                                            text: "emptinessFactor"
                                        }
                                        NumberSpinBox {
                                            id: emptinessFactorSpinBox
                                            editable: true
                                            decimals: 2
                                            from: decimalToInt(0)
                                            to: decimalToInt(1)
                                        }
                                        Text {
                                            text: "traversalCost"
                                        }
                                        NumberSpinBox {
                                            id: traversalCostSpinBox
                                            editable: true
                                            decimals: 2
                                            from: decimalToInt(0)
                                            to: decimalToInt(10)
                                        }
                                        Text {
                                            text: "intersectionCost"
                                        }
                                        NumberSpinBox {
                                            id: intersectionCostSpinBox
                                            editable: true
                                            decimals: 2
                                            from: decimalToInt(0)
                                            to: decimalToInt(10)
                                        }
                                        Text {
                                            text: "maxDepth"
                                        }
                                        QQC.SpinBox {
                                            id: maxDepthSpinBox
                                            editable: true
                                            from: 1
                                            to: 1000
                                            WheelHandler {
                                                onWheel: (wheel) => {
                                                    if (wheel.angleDelta.y < 0) {
                                                        maxDepthSpinBox.decrease()
                                                    } else {
                                                        maxDepthSpinBox.increase()
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                onOpened: {
                                    emptinessFactorSpinBox.updateValue(sceneSettings.emptinessFactor)
                                    traversalCostSpinBox.updateValue(sceneSettings.traversalCost)
                                    intersectionCostSpinBox.updateValue(sceneSettings.intersectionCost)
                                    maxDepthSpinBox.value = sceneSettings.maxDepth
                                }
                                standardButtons: QQC.Dialog.Apply | QQC.Dialog.Discard
                                onApplied: {
                                    sceneSettings.emptinessFactor = emptinessFactorSpinBox.realValue
                                    sceneSettings.traversalCost = traversalCostSpinBox.realValue
                                    sceneSettings.intersectionCost = intersectionCostSpinBox.realValue
                                    sceneSettings.maxDepth = maxDepthSpinBox.value
                                    accept()
                                }
                                onDiscarded: reject()
                            }
                            renderer {
                                renderMode: {
                                    let value = 0
                                    if (actionTraceSahKdTree.checked) {
                                        value |= RendererSettings.TraceSahKdTree
                                    }
                                    if (actionUseOffscreenTexture.checked) {
                                        value |= RendererSettings.UseOffscreenTexture
                                    }
                                    if (actionDiscardInvisible.checked) {
                                        value |= RendererSettings.DiscardInvisibleFragments
                                    }
                                    return value
                                }
                                texturingMode: {
                                    let value
                                    if (actionBarycentricColor.checked) {
                                        value = RendererSettings.BarycentricColor
                                    }
                                    if (actionWireFrame.checked) {
                                        value = RendererSettings.WireFrame
                                    }
                                    return value
                                }
                                clearColor: clearColorDialog.selectedColor
                            }
                            cameraController {
                                speed: scene.sceneAabbMax.minus(scene.sceneAabbMin).length() / 10.0  // 10 seconds to cross AABB
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
                            property alias emptinessFactor: sceneSettings.emptinessFactor
                            property alias traversalCost: sceneSettings.traversalCost
                            property alias intersectionCost: sceneSettings.intersectionCost
                            property alias maxDepth: sceneSettings.maxDepth
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
    footer: QQC.ToolBar {
        visible: actionUiVisibility.checked
        Flow {
            anchors.fill: parent
            Item {
                implicitWidth: taskQueueFrame.width
                implicitHeight: taskQueueFrame.height
                QQC.Frame {
                    id: taskQueueFrame
                    contentItem: RowLayout {
                        id: taskQueueRowLayout
                        Text {
                            text: {
                                qsTr("Task queue (%1):")
                                .arg(taskQueue.taskCount)
                            }
                        }
                        QQC.ProgressBar {
                            id: taskQueueProgressBar
                            indeterminate: taskQueue.taskCount === 0
                            value: taskQueue.progress
                            Text {
                                anchors.fill: parent
                                verticalAlignment: Text.AlignVCenter
                                horizontalAlignment: Text.AlignHCenter
                                z: 1
                                text: {
                                    qsTr("%1\%")
                                    .arg(Number(taskQueueProgressBar.value * 100).toLocaleString(locale, 'f', 0))
                                }
                            }
                        }
                    }
                }
                MouseArea {
                    id: taskQueueMouseArea
                    anchors.fill: parent
                    acceptedButtons: Qt.LeftButton
                    onClicked: actionShowTaskQueueDialog.trigger(taskQueueMouseArea)
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
        property alias sceneOpenFolderUrl: sceneOpenDialog.folderUrl
        property alias uiVisibility: actionUiVisibility.checked
        property alias traceSahKdTree: actionTraceSahKdTree.checked
        property alias useOffscreenTexture: actionUseOffscreenTexture.checked
        property alias discardInvisible: actionDiscardInvisible.checked
        property int texturingModeIndex: 0
        property int currentTabIndex: -1
        property string jsonModel
    }
    Component.onCompleted: visibility = settings.rootVisibility
    onClosing: close => settings.rootVisibility = visibility
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
