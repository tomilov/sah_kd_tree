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
        header: QC.ToolBar {
            contentItem: RowLayout {
                QC.ToolButton {
                    Layout.fillHeight: true
                    icon.name: "go-up-symbolic"
                    onClicked: sceneOpenDialog.setFolderUrl(folderListModel.parentFolder)
                }
                QC.ToolButton {
                    Layout.fillHeight: true
                    icon.name: "go-previous-symbolic"
                    enabled: page.previousFolders.length !== 0
                    onClicked: sceneOpenDialog.folderUrl = page.previousFolders.pop()
                }
                QC.Label {
                    Layout.fillHeight: true
                    id: currentPathLabel
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
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
        }
        contentItem: QC.Frame {
            ListView {
                id: listView
                implicitWidth: Math.max(contentItem.childrenRect.width, 128)
                implicitHeight: Math.min(contentHeight, 256)
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
                    highlighted: ListView.isCurrentItem
                    contentItem: RowLayout {
                        id: row
                        Text {
                            Layout.fillHeight: true
                            text: listElement.fileName + (listElement.fileIsDir ? "/" : "")
                        }
                        Text {
                            Layout.fillHeight: true
                            visible: !listElement.fileIsDir
                            text: locale.formattedDataSize(listElement.fileSize)
                        }
                    }
                    onHoveredChanged: if (hovered) listView.currentIndex = index
                    onClicked: {
                        if (fileIsDir) {
                            sceneOpenDialog.setFolderUrl(fileUrl)
                        } else {
                            page.fileAccessed = fileAccessed
                            page.fileSize = fileSize
                            page.fileUrl = fileUrl
                            page.fileModified = fileModified
                            page.fileBaseName = fileBaseName
                            page.filePath = filePath
                            page.fileName = fileName
                            page.fileSuffix = fileSuffix
                            page.fileIsDir = fileIsDir
                            sceneOpenDialog.accept()
                        }
                    }
                }
                QC.ScrollBar.vertical: QC.ScrollBar {}
            }
        }
    }
}
