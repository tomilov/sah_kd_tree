import QtCore
import QtQuick
import QtQuick.Controls as C
import QtQuick.Layouts

import Qt.labs.folderlistmodel as LF

pragma ComponentBehavior: Bound

CenteredDialog {
    standardButtons: C.Dialog.Close
    property url folderUrl
    required property list<string> nameFilters
    function setFolderUrl(path) {
        if (path.toString() !== "") {
            if (folderUrl.toString() !== "") {
                page.previousFolders.push(folderUrl)
            }
            folderUrl = path
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
    contentItem: C.Page {
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
        header: C.ToolBar {
            contentItem: RowLayout {
                C.ToolButton {
                    Layout.fillHeight: true
                    icon.name: "go-up-symbolic"
                    onClicked: setFolderUrl(folderListModel.parentFolder)
                }
                C.ToolButton {
                    Layout.fillHeight: true
                    icon.name: "go-previous-symbolic"
                    enabled: page.previousFolders.length !== 0
                    onClicked: folderUrl = page.previousFolders.pop()
                }
                C.Label {
                    Layout.fillHeight: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    textFormat: Text.StyledText
                    text: {
                        '<tt><a href="%1">%2</a></tt>'
                        .arg(folderUrl)
                        .arg(app.toLocalFile(folderUrl))
                    }
                    onLinkActivated: link => Qt.openUrlExternally(link)
                }
                Item {
                    Layout.fillWidth: true
                }
            }
        }
        contentItem: C.Frame {
            ListView {
                id: listView
                implicitWidth: Math.max(contentItem.childrenRect.width, 640)
                implicitHeight: 480
                clip: true
                highlightFollowsCurrentItem: true
                model: LF.FolderListModel {
                    id: folderListModel
                    folder: folderUrl
                    Binding on nameFilters {  // to squelch "Expression depends on non-NOTIFYable properties"
                        value: nameFilters
                    }
                    sortField: LF.FolderListModel.Size
                    showDirsFirst: true
                    showOnlyReadable: true
                }
                delegate: C.ItemDelegate {
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
                        CenteredText {
                            Layout.fillHeight: true
                            text: fileName + (fileIsDir ? "/" : "")
                        }
                        CenteredText {
                            Layout.fillHeight: true
                            visible: !fileIsDir
                            text: locale.formattedDataSize(fileSize)
                        }
                    }
                    onHoveredChanged: {
                        if (hovered) {
                            listView.currentIndex = index
                        }
                    }
                    onClicked: {
                        if (fileIsDir) {
                            setFolderUrl(fileUrl)
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
                            accept()
                        }
                    }
                }
                C.ScrollBar.vertical: C.ScrollBar {}
            }
        }
    }
}
