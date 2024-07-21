import QtCore
import QtQuick
import QtQuick.Controls

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
        header: Label {
            textFormat: Text.StyledText
            text: '<tt><a href="%1">%1</a></tt>'.arg(sceneOpenDialog.folderUrl)
            onLinkActivated: link => Qt.openUrlExternally(link)
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
                    radius: Mat.min(height, width) / 2
                }
                model: FolderListModel {
                    folder: sceneOpenDialog.folderUrl
                    nameFilters: SahKdTreeEngine.supportedSceneFileExtensions
                    showDirsFirst: true
                    showOnlyReadable: true
                    showDotAndDotDot: true
                }
                delegate: Component {
                    Text {
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
                        text: fileName + (fileIsDir ? "/" : "")
                        MouseArea {
                            id: mouseArea
                            anchors.fill: parent
                            acceptedButtons: Qt.LeftButton
                            hoverEnabled: true
                            onEntered: listView.currentIndex = index
                            onClicked: mouse => {
                                if (fileIsDir) {
                                    sceneOpenDialog.folderUrl = fileUrl
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
                    }
                }
                ScrollBar.vertical: ScrollBar {
                    policy: ScrollBar.AlwaysOn
                }
            }
        }
    }
}
