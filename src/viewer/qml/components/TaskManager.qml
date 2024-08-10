import QtQuick
import QtQuick.Controls as C
import QtQuick.Layouts

import Qt.labs.qmlmodels as LM

import SahKdTree 1.0 as SKT

pragma ComponentBehavior: Bound

CenteredDialog {
    required property SKT.TaskQueue taskQueue
    property int toolTipTimeout: 5000
    title: qsTr("Task queue")
    standardButtons: C.Dialog.Close
    Timer {
        interval: 1000
        running: visible
        triggeredOnStart: true
        repeat: true
        onTriggered: {
            let threadPool = taskQueue.threadPool
            activeThreadCountText.text = qsTr("Active thread count: %1").arg(threadPool.activeThreadCount)
            expiryTimeoutText.text = qsTr("Expiry timeout: %1ms").arg(threadPool.expiryTimeout)
            maxThreadCountText.text = qsTr("Max thread count: %1").arg(threadPool.maxThreadCount)
            stackSizeText.text = qsTr("Stack size: %1").arg(threadPool.stackSize !== 0 ? threadPool.stackSize : "system default")
            threadPriorityText.text = qsTr("Thread priority: %1").arg(taskQueue.threadPriorityToString(threadPool.threadPriority))
        }
    }
    contentItem: C.Page {
        header: C.ToolBar {
            contentItem: RowLayout {
                C.Frame {
                    RowLayout {
                        C.ToolButton {
                            Layout.fillHeight: true
                            icon.name: "media-playback-pause-symbolic"
                            text: qsTr("Suspend all")
                            onClicked: taskQueue.suspendAll()
                        }
                        C.ToolButton {
                            Layout.fillHeight: true
                            icon.name: "media-playback-start-symbolic"
                            text: qsTr("Resume all")
                            onClicked: taskQueue.resumeAll()
                        }
                        C.DelayButton {
                            Layout.fillHeight: true
                            icon.name: "media-playback-stop-symbolic"
                            text: qsTr("Cancel all")
                            delay: 1000
                            onActivated: taskQueue.cancelAll()
                            onReleased: checked = false
                        }
                    }
                }
                C.ToolSeparator {
                    Layout.fillHeight: true
                }
                C.Frame {
                    RowLayout {
                        C.ToolButton {
                            Layout.fillHeight: true
                            icon.name: "media-playback-pause-symbolic"
                            text: qsTr("Suspend checked")
                            onClicked: taskQueue.suspendChecked()
                        }
                        C.ToolButton {
                            Layout.fillHeight: true
                            icon.name: "media-playback-start-symbolic"
                            text: qsTr("Resume checked")
                            onClicked: taskQueue.resumeChecked()
                        }
                        C.DelayButton {
                            Layout.fillHeight: true
                            icon.name: "media-playback-stop-symbolic"
                            text: qsTr("Cancel checked")
                            delay: 1000
                            onActivated: taskQueue.cancelChecked()
                            onReleased: checked = false
                        }
                    }
                }
                Item {
                    Layout.fillWidth: true
                }
            }
        }
        footer: C.ToolBar {
            contentItem: Flow {
                C.Frame {
                    CenteredText {
                        text: qsTr("Task count: %1").arg(taskQueue.taskCount)
                    }
                }
                C.Frame {
                    CenteredText {
                        id: activeThreadCountText
                    }
                }
                C.Frame {
                    CenteredText {
                        id: expiryTimeoutText
                    }
                }
                C.Frame {
                    CenteredText {
                        id: maxThreadCountText
                    }
                }
                C.Frame {
                    CenteredText {
                        id: stackSizeText
                    }
                }
                C.Frame {
                    CenteredText {
                        id: threadPriorityText
                    }
                }
            }
        }
        contentItem: GridLayout {
            columns: 2
            C.HorizontalHeaderView {
                Layout.column: 1
                Layout.fillWidth: true
                syncView: tableView
                clip: true
                delegate: C.ItemDelegate {
                    id: horizontalHeaderDelegate
                    required property var modelData
                    contentItem: CenteredText {
                        text: horizontalHeaderDelegate.modelData.display
                    }
                }
            }
            C.VerticalHeaderView {
                Layout.fillHeight: true
                syncView: tableView
                clip: true
                delegate: C.ItemDelegate {
                    id: verticalHeaderDelegate
                    required property var modelData
                    contentItem: CenteredText {
                        text: verticalHeaderDelegate.modelData.display
                    }
                }
            }
            C.ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                TableView {
                    id: tableView
                    clip: true
                    model: taskQueue
                    delegate: LM.DelegateChooser {
                        role: "type"
                        LM.DelegateChoice {
                            roleValue: "item"
                            delegate: C.ItemDelegate {
                                required property string type
                                required property var modelData
                                required property string toolTip
                                contentItem: CenteredText {
                                    Binding on text {
                                        when: modelData.display !== undefined
                                        value: modelData.display
                                    }
                                }
                                C.ToolTip.visible: hovered
                                C.ToolTip.text: toolTip
                                C.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                C.ToolTip.timeout: toolTipTimeout
                            }
                        }
                        LM.DelegateChoice {
                            roleValue: "progress"
                            delegate: C.ItemDelegate {
                                required property string type
                                required property var modelData
                                required property string toolTip
                                background: C.ProgressBar {
                                    indeterminate: modelData.display === undefined
                                    Binding on value {
                                        when: modelData.display !== undefined
                                        value: modelData.display
                                    }
                                }
                                contentItem: CenteredText {
                                    text: toolTip
                                }
                            }
                        }
                        LM.DelegateChoice {
                            roleValue: "switch"
                            delegate: C.SwitchDelegate {
                                required property string type
                                required property int row
                                required property int column
                                required property string toolTip
                                required property int checkState
                                function setChecked(value) {
                                    let index = TableView.view.index(row, column)
                                    if (!TableView.view.model.setData(index, value, Qt.CheckStateRole)) {
                                        console.log("cannot set checkStateRole for (%1, %2)".arg(row).arg(column))
                                    }
                                }
                                checked: checkState === Qt.Checked
                                C.ToolTip.visible: hovered
                                C.ToolTip.text: toolTip
                                C.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                C.ToolTip.timeout: toolTipTimeout
                                onToggled: setChecked(checked ? Qt.Checked : Qt.Unchecked)
                            }
                        }
                        LM.DelegateChoice {
                            roleValue: "delay"
                            delegate: C.DelayButton {
                                required property string type
                                required property int row
                                required property int column
                                required property string toolTip
                                required property int checkState
                                function setChecked(value) {
                                    let index = TableView.view.index(row, column)
                                    if (!TableView.view.model.setData(index, value, Qt.CheckStateRole)) {
                                        console.log("cannot set checkStateRole for (%1, %2)".arg(row).arg(column))
                                    }
                                }
                                checked: checkState === Qt.Checked
                                delay: 1000
                                text: qsTr("Cancel")
                                C.ToolTip.visible: hovered
                                C.ToolTip.text: toolTip
                                C.ToolTip.delay: Application.styleHints.mousePressAndHoldInterval
                                C.ToolTip.timeout: toolTipTimeout
                                onActivated: setChecked(Qt.Checked)
                            }
                        }
                        LM.DelegateChoice {
                            roleValue: "check"
                            delegate: C.CheckBox {
                                required property string type
                                required property int row
                                required property int column
                                required property var modelData
                                function setChecked(value) {
                                    let index = TableView.view.index(row, column)
                                    if (!TableView.view.model.setData(index, value, Qt.CheckStateRole)) {
                                        console.log("cannot set checkStateRole for (%1, %2)".arg(row).arg(column))
                                    }
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
