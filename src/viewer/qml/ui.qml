import QtCore
import QtQuick
import QtQuick.Controls as QC
import QtQuick.Window
import QtQuick.Layouts
import QtQuick.Dialogs as Dialogs
import QtQuick3D
import QtQml.Models

import Qt.labs.qmlmodels

import SahKdTree 1.0

import "utils.js" as Utils

pragma ComponentBehavior: Bound

QC.ApplicationWindow {
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
    QC.Action {
        id: actionOpenScene
        text: qsTr("&Open (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Open
        onTriggered: sceneOpenDialog.replaceScene()
        icon.name: "tab-new-symbolic"
    }
    QC.Action {
        id: actionReplaceScene
        text: qsTr("Open in &new tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.AddTab
        onTriggered: sceneOpenDialog.appendScene()
        icon.name: "application-add-symbolic"
    }
    QC.Action {
        id: actionCloseAllTabs
        text: qsTr("Close &all tabs")
        enabled: listModel.count > 0
        onTriggered: listModel.clear()
        icon.name: "list-remove-all-symbolic"
    }
    QC.Action {
        id: actionCloseScene
        text: qsTr("&Close")
        enabled: listModel.count > 0
        onTriggered: removeCurrentTab()
        icon.name: "list-remove-symbolic"
    }
    QC.Action {
        id: actionExit
        text: qsTr("&Exit (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.Cancel
        onTriggered: root.close()
        icon.name: "window-close-symbolic"
    }
    QC.Action {
        id: actionNextTab
        text: qsTr("Next tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.NextChild
        enabled: tabBar.count !== 0
        onTriggered: {
            if (tabBar.currentIndex + 1 === tabBar.count) {
                tabBar.setCurrentIndex(0)
            } else {
                tabBar.incrementCurrentIndex()
            }
        }
        icon.name: "go-next-symbolic"
    }
    QC.Action {
        id: actionPreviosTab
        text: qsTr("Previous tab (%1)").arg(app.keySequenceToString(shortcut))
        shortcut: StandardKey.PreviousChild
        enabled: tabBar.count !== 0
        onTriggered: {
            if (tabBar.currentIndex === 0) {
                tabBar.setCurrentIndex(tabBar.count - 1)
            } else {
                tabBar.decrementCurrentIndex()
            }
        }
        icon.name: "go-previous-symbolic"
    }
    QC.Action {
        id: actionUiVisibility
        text: qsTr("Toggle UI visibility")
        checkable: true
        checked: true
        shortcut: StandardKey.Replace
    }
    QC.Action {
        id: actionUseOffscreenTexture
        text: qsTr("Offscreen (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        checked: true
        shortcut: "F4"
    }
    QC.Action {
        id: actionDiscardInvisible
        text: qsTr("Discard")
        checkable: true
    }
    QC.Action {
        id: actionTraceSahKdTree
        text: qsTr("Trace/Rasterize (%1)").arg(app.keySequenceToString(shortcut))
        checkable: true
        shortcut: "F2"
    }
    QC.ActionGroup {
        id: texturingModeActionGroup
        QC.Action {
            id: actionBarycentricColor
            text: qsTr("Barycentric")
            checkable: true
        }
        QC.Action {
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
    QC.Action {
        id: actionShowAboutQt
        text: qsTr("About Qt")
        enabled: app.showAboutQt !== undefined
        onTriggered: Qt.callLater(app.showAboutQt)
        icon.source: app.getQtLogoUrl()
        shortcut: StandardKey.HelpContents
    }
    QC.Action {
        id: actionShowTaskQueueDialog
        text: qsTr("Show task queue info")
        onTriggered: taskQueueDialog.open()
        icon.name: "view-list-symbolic"
    }
    menuBar: QC.MenuBar {
        visible: actionUiVisibility.checked
        QC.Menu {
            title: qsTr("&File")
            QC.MenuItem {
                action: actionOpenScene
            }
            QC.MenuItem {
                action: actionReplaceScene
            }
            QC.MenuItem {
                action: actionCloseAllTabs
            }
            QC.MenuItem {
                action: actionCloseScene
            }
            QC.MenuSeparator {}
            QC.MenuItem {
                action: actionExit
            }
        }
        QC.Menu {
            title: qsTr("&Navigation")
            QC.MenuItem {
                action: actionNextTab
            }
            QC.MenuItem {
                action: actionPreviosTab
            }
        }
        QC.Menu {
            title: qsTr("&Mode")
            QC.MenuItem {
                action: actionUseOffscreenTexture
            }
            QC.MenuItem {
                action: actionDiscardInvisible
            }
            QC.MenuSeparator {}
            QC.MenuItem {
                action: actionBarycentricColor
            }
            QC.MenuItem {
                action: actionWireFrame
            }
        }
        QC.Menu {
            title: qsTr("&Help")
            QC.MenuItem {
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
    header: QC.TabBar {
        id: tabBar
        visible: actionUiVisibility.checked
        background: QC.Pane {}
        Repeater {
            model: listModel
            QC.TabButton {
                required property string fileBaseName
                required property url fileUrl
                text: fileBaseName
                onDoubleClicked: removeCurrentTab()
                QC.ToolTip.visible: hovered
                QC.ToolTip.text: {
                    "<font color=\"%2\">%1</font>"
                    .arg(fileUrl)
                    .arg(Qt.color(palette.link))
                }
                QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                QC.ToolTip.timeout: root.toolTipTimeout
            }
        }
        Component.onCompleted: Qt.callLater(setCurrentIndex, settings.currentTabIndex)
        Component.onDestruction: settings.currentTabIndex = currentIndex
    }
    TaskQueue {
        id: taskQueue
        function updateThreadPoolInfo() {
            activeThreadCountText.text = qsTr("Active thread count: %1").arg(threadPool.activeThreadCount)
            expiryTimeoutText.text = qsTr("Expiry timeout: %1ms").arg(threadPool.expiryTimeout)
            maxThreadCountText.text = qsTr("Max thread count: %1").arg(threadPool.maxThreadCount)
            stackSizeText.text = qsTr("Stack size: %1").arg(threadPool.stackSize)
            threadPriorityText.text = qsTr("Thread priority: %1").arg(threadPriorityToString(threadPool.threadPriority))
        }
    }
    Timer {
        interval: 1000
        running: true
        triggeredOnStart: true
        repeat: true
        onTriggered: taskQueue.updateThreadPoolInfo()
    }
    CenteredDialog {
        id: taskQueueDialog
        title: qsTr("Task queue")
        standardButtons: QC.Dialog.Close
        contentItem: QC.Page {
            header: QC.ToolBar {
                contentItem: RowLayout {
                    QC.ToolButton {
                        Layout.fillHeight: true
                        icon.name: "media-playback-pause-symbolic"
                        text: qsTr("Suspend all")
                        onClicked: taskQueue.suspendAll()
                    }
                    QC.ToolButton {
                        Layout.fillHeight: true
                        icon.name: "media-playback-start-symbolic"
                        text: qsTr("Resume all")
                        onClicked: taskQueue.resumeAll()
                    }
                    QC.DelayButton {
                        Layout.fillHeight: true
                        icon.name: "media-playback-stop-symbolic"
                        text: qsTr("Cancel all")
                        delay: 1000
                        onActivated: taskQueue.cancelAll()
                        onReleased: checked = false
                    }
                    QC.DelayButton {
                        Layout.fillHeight: true
                        icon.name: "media-playback-stop-symbolic"
                        text: qsTr("Cancel checked")
                        delay: 1000
                        onActivated: taskQueue.cancelChecked()
                        onReleased: checked = false
                    }
                    Item {
                        Layout.fillWidth: true
                    }
                }
            }
            footer: QC.ToolBar {
                contentItem: Flow {
                    QC.Frame {
                        CenteredText {
                            text: qsTr("Task count: %1").arg(taskQueue.taskCount)
                        }
                    }
                    QC.Frame {
                        CenteredText {
                            id: activeThreadCountText
                        }
                    }
                    QC.Frame {
                        CenteredText {
                            id: expiryTimeoutText
                        }
                    }
                    QC.Frame {
                        CenteredText {
                            id: maxThreadCountText
                        }
                    }
                    QC.Frame {
                        CenteredText {
                            id: stackSizeText
                        }
                    }
                    QC.Frame {
                        CenteredText {
                            id: threadPriorityText
                        }
                    }
                }
            }
            contentItem: GridLayout {
                columns: 2
                QC.HorizontalHeaderView {
                    Layout.column: 1
                    Layout.fillWidth: true
                    syncView: tableView
                    clip: true
                    delegate: QC.ItemDelegate {
                        id: horizontalHeaderDelegate
                        required property var modelData
                        contentItem: CenteredText {
                            text: horizontalHeaderDelegate.modelData.display
                        }
                    }
                }
                QC.VerticalHeaderView {
                    Layout.fillHeight: true
                    syncView: tableView
                    clip: true
                    delegate: QC.ItemDelegate {
                        id: verticalHeaderDelegate
                        required property var modelData
                        contentItem: CenteredText {
                            text: verticalHeaderDelegate.modelData.display
                        }
                    }
                }
                QC.ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    TableView {
                        id: tableView
                        clip: true
                        model: taskQueue
                        delegate: DelegateChooser {
                            role: "type"
                            DelegateChoice {
                                roleValue: "item"
                                delegate: QC.ItemDelegate {
                                    id: itemDelegate
                                    required property string type
                                    required property var modelData
                                    required property string toolTip
                                    contentItem: CenteredText {
                                        Binding on text {
                                            when: itemDelegate.modelData.display !== undefined
                                            value: itemDelegate.modelData.display
                                        }
                                    }
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: toolTip
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                            DelegateChoice {
                                roleValue: "progress"
                                delegate: QC.ItemDelegate {
                                    id: progressBarDelegate
                                    required property string type
                                    required property var modelData
                                    required property string toolTip
                                    background: QC.ProgressBar {
                                        indeterminate: progressBarDelegate.modelData.display === undefined
                                        Binding on value {
                                            when: progressBarDelegate.modelData.display !== undefined
                                            value: progressBarDelegate.modelData.display
                                        }
                                    }
                                    contentItem: CenteredText {
                                        text: progressBarDelegate.toolTip
                                    }
                                }
                            }
                            DelegateChoice {
                                roleValue: "switch"
                                delegate: QC.SwitchDelegate {
                                    required property string type
                                    required property int row
                                    required property int column
                                    required property string toolTip
                                    required property int checkState
                                    function setChecked(value) {
                                        let index = TableView.view.index(row, column)
                                        TableView.view.model.setData(index, value, Qt.CheckStateRole)
                                    }
                                    checked: checkState === Qt.Checked
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: toolTip
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                    onToggled: setChecked(checked ? Qt.Checked : Qt.Unchecked)
                                }
                            }
                            DelegateChoice {
                                roleValue: "delay"
                                delegate: QC.DelayButton {
                                    required property string type
                                    required property int row
                                    required property int column
                                    required property string toolTip
                                    required property int checkState
                                    function setChecked(value) {
                                        let index = TableView.view.index(row, column)
                                        TableView.view.model.setData(index, value, Qt.CheckStateRole)
                                    }
                                    checked: checkState === Qt.Checked
                                    delay: 1000
                                    text: qsTr("Cancel")
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: toolTip
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                    onActivated: setChecked(Qt.Checked)
                                }
                            }
                            DelegateChoice {
                                roleValue: "check"
                                delegate: QC.CheckBox {
                                    required property string type
                                    required property int row
                                    required property int column
                                    required property var modelData
                                    function setChecked(value) {
                                        let index = TableView.view.index(row, column)
                                        TableView.view.model.setData(index, value, Qt.CheckStateRole)
                                    }
                                    checkState: modelData.checkState
                                    onCheckStateChanged: setChecked(checkState)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    footer: QC.ToolBar {
        visible: actionUiVisibility.checked
        contentItem: Flow {
            Item {
                implicitWidth: taskQueueFrame.width
                implicitHeight: taskQueueFrame.height
                QC.Frame {
                    id: taskQueueFrame
                    RowLayout {
                        id: taskQueueRowLayout
                        CenteredText {
                            text: qsTr("Task queue (%1):").arg(taskQueue.taskCount)
                        }
                        QC.ProgressBar {
                            id: taskQueueProgressBar
                            indeterminate: taskQueue.taskCount === 0
                            value: taskQueue.progress
                            CenteredText {
                                anchors.centerIn: parent
                                visible: taskQueue.taskCount > 0
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
    StackLayout {
        id: stackLayout
        anchors.fill: parent
        currentIndex: tabBar.currentIndex
        Repeater {
            model: listModel
            delegate: QC.Page {
                id: page
                required property url fileUrl
                required property string filePath
                required property string fileBaseName
                readonly property string fileUrlHash: Qt.md5(fileUrl)
                Dialogs.ColorDialog {
                    id: clearColorDialog
                    options: Dialogs.ColorDialog.ShowAlphaChannel | Dialogs.ColorDialog.DontUseNativeDialog | Dialogs.ColorDialog.NoButtons
                    onSelectedColorChanged: {
                        if (visible) { // prevent feedback when WheelHandler used
                            Qt.callLater(clearColorComboBox.setIndexOfClosestColor, selectedColor)
                        }
                    }
                }
                QC.Action {
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
                QC.Action {
                    id: actionContentVisibility
                    text: qsTr("Toggle content visibility")
                    checkable: true
                    checked: true
                }
                QC.Action {
                    id: actionLayerEnabled
                    text: qsTr("Layer enable/disable")
                    checkable: true
                    icon.name: "image-crop-symbolic"
                }
                QC.Action {
                    id: actionRotatePos
                    text: qsTr("Rotate content CCW")
                    icon.name: "object-rotate-left-symbolic"
                    onTriggered: rotationSlider.decrease()
                }
                QC.Action {
                    id: actionRotateNeg
                    text: qsTr("Rotate content CW")
                    icon.name: "object-rotate-right-symbolic"
                    onTriggered: rotationSlider.increase()
                }
                QC.Action {
                    id: actionScaleDec
                    text: qsTr("Decrease content scale")
                    icon.name: "zoom-out-symbolic"
                    onTriggered: scaleSlider.decrease()
                }
                QC.Action {
                    id: actionScaleInc
                    text: qsTr("Increase content scale")
                    icon.name: "zoom-in-symbolic"
                    onTriggered: scaleSlider.increase()
                }
                QC.Action {
                    id: actionOpacityDec
                    text: qsTr("Decrease content opacity")
                    icon.name: "path-combine-symbolic"
                    onTriggered: opacitySlider.decrease()
                }
                QC.Action {
                    id: actionOpacityInc
                    text: qsTr("Increase content opacity")
                    icon.name: "path-difference-symbolic"
                    onTriggered: opacitySlider.increase()
                }
                QC.Action {
                    id: actionSaveSceneScreenshot
                    text: qsTr("Screenshot")
                    onTriggered: viewer.grabToImage(result => app.setClipboardImage(result.image))
                    icon.name: "edit-copy-symbolic"
                }
                QC.Action {
                    id: actionSelectClearColor
                    text: qsTr("Select clearColor")
                    onTriggered: clearColorDialog.open()
                    icon.name: "color-select-symbolic"
                }
                QC.Action {
                    id: actionChangeTreeBuildParams
                    text: qsTr("Change SAH kd-tree build parameters")
                    onTriggered: treeParametersDialog.open()
                    icon.name: "edit-symbolic"
                }
                QC.Menu {
                    id: contextMenu
                    title: "Context menu"
                    parent: QC.Overlay.overlay
                    QC.MenuItem {
                        action: actionSaveSceneScreenshot
                    }
                    QC.MenuSeparator {}
                    QC.MenuItem {
                        action: actionTraceSahKdTree
                    }
                    QC.MenuItem {
                        action: actionUseOffscreenTexture
                    }
                    QC.MenuItem {
                        action: actionDiscardInvisible
                    }
                    QC.MenuSeparator {}
                    QC.MenuItem {
                        action: actionBarycentricColor
                    }
                    QC.MenuItem {
                        action: actionWireFrame
                    }
                    QC.MenuSeparator {}
                    QC.MenuItem {
                        action: actionSelectClearColor
                    }
                    QC.MenuItem {
                        action: actionChangeTreeBuildParams
                    }
                    QC.MenuItem {
                        action: actionShowTaskQueueDialog
                    }
                    QC.MenuSeparator {}
                    QC.MenuItem {
                        text: qsTr("Dump item tree")
                        onTriggered: root.contentItem.dumpItemTree()
                    }
                    QC.MenuItem {
                        text: qsTr("Make window invisible")
                        onTriggered: root.hide()
                    }
                    QC.MenuItem {
                        text: qsTr("Renderdoc capture frame")
                        onTriggered: viewer.renderer.renderdocCaptureFrame()
                    }
                }
                header: QC.ToolBar {
                    visible: actionUiVisibility.checked
                    contentItem: Flow {
                        QC.Frame {
                            RowLayout {
                                QC.ToolButton {
                                    text: qsTr("Reset cam")
                                    onClicked: {
                                        viewer.cameraView.orientation = undefined
                                        viewer.cameraView.position = undefined
                                        viewer.cameraView.filedOfView = undefined
                                    }
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: qsTr("Reset camera view")
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                QC.ToolButton {
                                    text: qsTr("Align cam")
                                    onClicked: viewer.cameraView.alignOrientation()
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: qsTr("Align camera view")
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                QC.ToolButton {
                                    text: qsTr("Reflect cam")
                                    onClicked: viewer.cameraView.reflectOrientation()
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: qsTr("Reflect camera view")
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                QC.ToolButton {
                                    text: qsTr("Reset item")
                                    action: actionResetContentOrientation
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                QC.Switch {
                                    text: qsTr("Show/Hide")
                                    action: actionContentVisibility
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                QC.Switch {
                                    text: qsTr("Layer")
                                    action: actionLayerEnabled
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                QC.ToolButton {
                                    text: qsTr("")
                                    action: actionRotatePos
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                                QC.Slider {
                                    id: rotationSlider
                                    from: -180
                                    value: 0
                                    to: 180
                                    stepSize: 5
                                    snapMode: QC.Slider.SnapAlways
                                    QC.ToolTip.visible: pressed || hovered
                                    QC.ToolTip.text: value
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
                                QC.ToolButton {
                                    text: qsTr("")
                                    action: actionRotateNeg
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                QC.ToolButton {
                                    text: qsTr("")
                                    action: actionScaleDec
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                                QC.Slider {
                                    id: scaleSlider
                                    from: 0.125
                                    value: 1
                                    to: 1.25
                                    stepSize: 0.125
                                    QC.ToolTip.visible: pressed || hovered
                                    QC.ToolTip.text: value
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
                                QC.ToolButton {
                                    text: qsTr("")
                                    action: actionScaleInc
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                QC.ToolButton {
                                    text: qsTr("")
                                    action: actionOpacityDec
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                                QC.Slider {
                                    id: opacitySlider
                                    from: 0.0
                                    value: 1.0
                                    to: 1.0
                                    stepSize: 0.0625
                                    QC.ToolTip.visible: pressed || hovered
                                    QC.ToolTip.text: value
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
                                QC.ToolButton {
                                    text: qsTr("")
                                    action: actionOpacityInc
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: action.text
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                CenteredText {
                                    text: qsTr("<b>Clear color:</b>")
                                }
                                QC.ComboBox {
                                    id: clearColorComboBox
                                    textRole: "colorName"
                                    valueRole: "colorValue"
                                    implicitContentWidthPolicy: QC.ComboBox.WidestText
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
                                                if (clearColorComboBox.currentIndex + 1 < clearColorComboBox.count) {
                                                    ++clearColorComboBox.currentIndex
                                                }
                                            } else {
                                                if (clearColorComboBox.currentIndex > 0) {
                                                    --clearColorComboBox.currentIndex
                                                }
                                            }
                                            clearColorDialog.selectedColor = clearColorComboBox.currentValue
                                        }
                                    }
                                    QC.ToolTip.visible: hovered
                                    QC.ToolTip.text: currentText
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
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
                                    delegate: QC.ItemDelegate {
                                        id: clearColorDelegate
                                        required property int index
                                        required property string colorName
                                        required property color colorValue
                                        highlighted: clearColorComboBox.highlightedIndex === index
                                        contentItem: RowLayout {
                                            spacing: height / 8
                                            Rectangle {
                                                id: colorRect
                                                Layout.fillHeight: true
                                                color: clearColorDelegate.colorValue
                                                width: colorText.height
                                                radius: height / 4
                                                border.width: 1
                                                border.color: "black"
                                            }
                                            Text {
                                                id: colorText
                                                Layout.fillWidth: true
                                                Layout.fillHeight: true
                                                text: clearColorDelegate.colorName
                                            }
                                        }
                                        QC.ToolTip.visible: hovered
                                        QC.ToolTip.text: colorRect.color
                                        QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                        QC.ToolTip.timeout: root.toolTipTimeout
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
                                    QC.ToolTip.visible: colorSquareHoverHandler.hovered && clearColorComboBox.currentValue !== undefined
                                    QC.ToolTip.text: clearColorComboBox.currentValue
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                        }
                    }
                }
                footer: QC.ToolBar {
                    visible: actionUiVisibility.checked
                    contentItem: Flow {
                        QC.Frame {
                            RowLayout {
                                CenteredText {
                                    text: viewer.getRenderModeDescription(false)
                                    HoverHandler {
                                        id: modeTextHoverHandler
                                    }
                                    QC.ToolTip.visible: modeTextHoverHandler.hovered
                                    QC.ToolTip.text: viewer.getRenderModeDescription(true)
                                    QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                    QC.ToolTip.timeout: root.toolTipTimeout
                                }
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                CenteredText {
                                    text: qsTr("sens: %1").arg(viewer.cameraController.sensitivity.toFixed(4))
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                CenteredText {
                                    text: qsTr("speed: %1").arg(viewer.cameraController.speed.toExponential(3))
                                }
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                CenteredText {
                                    text: qsTr("rot: %1").arg(content.rotation.toFixed(0))
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                CenteredText {
                                    text: qsTr("scale: %1").arg(content.scale.toFixed(3))
                                }
                            }
                        }
                        QC.Frame {
                            Item {
                                implicitWidth: clearColorInfoRow.implicitWidth
                                implicitHeight: clearColorInfoRow.implicitHeight
                                RowLayout {
                                    id: clearColorInfoRow
                                    CenteredText {
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
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: clearColorDialog.open()
                                }
                                HoverHandler {
                                    id: clearColorHoveredHandler
                                }
                                QC.ToolTip.visible: clearColorHoveredHandler.hovered
                                QC.ToolTip.text: {
                                    qsTr('Is %1 "<font color="%2">%3</font>" color')
                                    .arg(clearColorDialog.selectedColor === Qt.color(clearColorComboBox.currentText) ? "exactly" : "roughly")
                                    .arg(Qt.alpha(clearColorComboBox.currentValue, 1.0))
                                    .arg(clearColorComboBox.currentText)
                                }
                                QC.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                QC.ToolTip.timeout: root.toolTipTimeout
                            }
                        }
                        QC.Frame {
                            RowLayout {
                                CenteredText {
                                    readonly property vector3d position: viewer.cameraView.position
                                    text: qsTr("xyz: %1 %2 %3").arg(position.x.toExponential(3)).arg(position.y.toExponential(3)).arg(position.z.toExponential(3))
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                CenteredText {
                                    readonly property vector3d orientation: viewer.cameraView.orientation.toEulerAngles()
                                    text: qsTr("\u03C6\u03B8\u03C8: %1 %2 %3").arg(orientation.x.toFixed(1)).arg(orientation.y.toFixed(1)).arg(orientation.z.toFixed(1))
                                }
                                QC.ToolSeparator {
                                    Layout.fillHeight: true
                                }
                                CenteredText {
                                    text: qsTr("fov: %1").arg(viewer.cameraView.fov.toFixed(0))
                                }
                            }
                        }
                    }
                }
                background: Image {
                    fillMode: Image.Tile
                    source: app.getQtLogoUrl()
                }
                contentItem: Item {
                    id: content
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
                            contentItem: QC.Frame {
                                GridLayout {
                                    columns: 2
                                    CenteredText {
                                        text: "emptinessFactor"
                                    }
                                    NumberSpinBox {
                                        id: emptinessFactorSpinBox
                                        editable: true
                                        decimals: 2
                                        from: decimalToInt(0)
                                        to: decimalToInt(1)
                                    }
                                    CenteredText {
                                        text: "traversalCost"
                                    }
                                    NumberSpinBox {
                                        id: traversalCostSpinBox
                                        editable: true
                                        decimals: 2
                                        from: decimalToInt(0)
                                        to: decimalToInt(10)
                                    }
                                    CenteredText {
                                        text: "intersectionCost"
                                    }
                                    NumberSpinBox {
                                        id: intersectionCostSpinBox
                                        editable: true
                                        decimals: 2
                                        from: decimalToInt(0)
                                        to: decimalToInt(10)
                                    }
                                    CenteredText {
                                        text: "maxDepth"
                                    }
                                    QC.SpinBox {
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
                            standardButtons: QC.Dialog.Apply | QC.Dialog.Discard
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
