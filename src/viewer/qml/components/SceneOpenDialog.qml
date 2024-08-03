import QtCore
import QtQuick
import QtQuick.Controls as QC
import QtQuick.Layouts

import Qt.labs.folderlistmodel

import SahKdTree 1.0

pragma ComponentBehavior: Bound

CenteredDialog {
    id: sceneOpenDialog
    standardButtons: QC.Dialog.Close
    property url folderUrl
    function setFolderUrl(path) {
        if (path.toString() !== "") {
            if (sceneOpenDialog.folderUrl.toString() !== "") {
                page.previousFolders.push(sceneOpenDialog.folderUrl)
            }
            sceneOpenDialog.folderUrl = path
        }
    }
    readonly property alias fileAccessed: page.fileAccessed
    readonly property alias fileSize: page.fileSize
    readonly property alias fileUrl: page.fileUrl
    readonly property alias fileModified: page.fileModified
    readonly property alias fileBaseName: page.fileBaseName
    readonly property alias filePath: page.filePath
    readonly property alias fileName: page.fileName
    readonly property alias fileSuffix: page.fileSuffix
    readonly property alias fileIsDir: page.fileIsDir
    contentItem: QC.Page {
        id: page
        property list<url> previousFolders
        property date fileAccessed
        property int fileSize
        property url fileUrl
        property date fileModified
        property string fileBaseName
        property string filePath
        property string fileName
        property string fileSuffix
        property bool fileIsDir
        header: RowLayout {
            QC.ToolButton {
                icon.name: "go-up-symbolic"
                onClicked: sceneOpenDialog.setFolderUrl(folderListModel.parentFolder)
            }
            QC.ToolButton {
                icon.name: "go-previous-symbolic"
                enabled: page.previousFolders.length !== 0
                onClicked: sceneOpenDialog.folderUrl = page.previousFolders.pop()
            }
            QC.Label {
                id: currentPathLabel
                textFormat: Text.StyledText
                text: {
                    '<tt><a href="%1">%2</a></tt>'
                    .arg(sceneOpenDialog.folderUrl)
                    .arg(app.toLocalFile(sceneOpenDialog.folderUrl))
                }
                onLinkActivated: link => Qt.openUrlExternally(link)
            }
            Item {
                Layout.fillWidth: true
            }
        }
        contentItem: QC.Frame {
            ListView {
                id: listView
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
                highlightFollowsCurrentItem: true
                model: FolderListModel {
                    id: folderListModel
                    folder: sceneOpenDialog.folderUrl
                    nameFilters: SahKdTreeEngine.supportedSceneFileExtensions.map((ext) => "*." + ext)
                    sortField: FolderListModel.Size
                    showDirsFirst: true
                    showOnlyReadable: true
                }
                delegate: QC.ItemDelegate {
                    id: listElement
                    highlighted: ListView.isCurrentItem
                    implicitWidth: row.implicitWidth
                    implicitHeight: row.implicitHeight
                    width: ListView.view.width
                    required property int index
                    required property date fileAccessed
                    required property int fileSize
                    required property url fileUrl
                    required property date fileModified
                    required property string fileBaseName
                    required property string filePath
                    required property string fileName
                    required property string fileSuffix
                    required property bool fileIsDir
                    contentItem: RowLayout {
                        id: row
                        anchors.fill: parent
                        Text {
                            text: listElement.fileName + (listElement.fileIsDir ? "/" : "")
                            Layout.fillHeight: true
                        }
                        Item {
                            Layout.fillWidth: true
                        }
                        Text {
                            visible: !listElement.fileIsDir
                            text: locale.formattedDataSize(listElement.fileSize)
                            Layout.fillHeight: true
                        }
                    }
                    MouseArea {
                        id: mouseArea
                        anchors.fill: row
                        hoverEnabled: true
                        onEntered: listView.currentIndex = listElement.index
                        acceptedButtons: Qt.LeftButton
                        onClicked: {
                            if (listElement.fileIsDir) {
                                sceneOpenDialog.setFolderUrl(listElement.fileUrl)
                            } else {
                                page.fileAccessed = listElement.fileAccessed
                                page.fileSize = listElement.fileSize
                                page.fileUrl = listElement.fileUrl
                                page.fileModified = listElement.fileModified
                                page.fileBaseName = listElement.fileBaseName
                                page.filePath = listElement.filePath
                                page.fileName = listElement.fileName
                                page.fileSuffix = listElement.fileSuffix
                                page.fileIsDir = listElement.fileIsDir
                                sceneOpenDialog.accept()
                            }
                        }
                    }
                }
                QC.ScrollBar.vertical: QC.ScrollBar {
                    policy: QC.ScrollBar.AlwaysOn
                }
            }
        }
    }
}
