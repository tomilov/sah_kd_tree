import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Qt.labs.folderlistmodel

import SahKdTree 1.0

pragma ComponentBehavior: Bound

CenteredDialog {
    id: sceneOpenDialog
    standardButtons: Dialog.Close
    property url folderUrl
    readonly property alias fileAccessed: page.fileAccessed
    readonly property alias fileSize: page.fileSize
    readonly property alias fileUrl: page.fileUrl
    readonly property alias fileModified: page.fileModified
    readonly property alias fileBaseName: page.fileBaseName
    readonly property alias filePath: page.filePath
    readonly property alias fileName: page.fileName
    readonly property alias fileSuffix: page.fileSuffix
    readonly property alias fileIsDir: page.fileIsDir
    Page {
        id: page
        anchors.fill: parent
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
            ToolButton {
                icon.name: "go-up-symbolic"
                onClicked: sceneOpenDialog.folderUrl = folderListModel.parentFolder
            }
            Label {
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
        Frame {
            anchors.fill: parent
            ListView {
                id: listView
                anchors.fill: parent
                clip: true
                flickableDirection: Flickable.AutoFlickIfNeeded
                highlightFollowsCurrentItem: true
                highlight: Rectangle {
                    color: palette.highlight
                    radius: Math.min(height, width) / 4
                }
                model: FolderListModel {
                    id: folderListModel
                    folder: sceneOpenDialog.folderUrl
                    nameFilters: SahKdTreeEngine.supportedSceneFileExtensions.map((ext) => "*." + ext)
                    sortField: FolderListModel.Size
                    showDirsFirst: true
                    showOnlyReadable: true
                }
                delegate: Component {
                    Item {
                        id: listElement
                        width: ListView.view.width
                        height: row.implicitHeight
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
                        RowLayout {
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
                                    sceneOpenDialog.folderUrl = listElement.fileUrl
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
                }
                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AlwaysOn
                }
            }
        }
    }
}
