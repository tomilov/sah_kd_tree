import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Qt.labs.folderlistmodel

import SahKdTree 1.0

CenteredDialog {
    id: sceneOpenDialog
    standardButtons: Dialog.Close
    property url folderUrl
    readonly property alias fileAccessed: columnLayout.fileAccessed
    readonly property alias fileSize: columnLayout.fileSize
    readonly property alias fileUrl: columnLayout.fileUrl
    readonly property alias fileModified: columnLayout.fileModified
    readonly property alias fileBaseName: columnLayout.fileBaseName
    readonly property alias filePath: columnLayout.filePath
    readonly property alias fileName: columnLayout.fileName
    readonly property alias fileSuffix: columnLayout.fileSuffix
    readonly property alias fileIsDir: columnLayout.fileIsDir
    ColumnLayout {
        id: columnLayout
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
        Frame {
            height: contentItem.implicitHeight
            Label {
                text: sceneOpenDialog.folderUrl + "/"
            }
            Layout.fillWidth: true
        }
        ListView {
            clip: true
            Layout.fillWidth: true
            Layout.fillHeight: true
            flickableDirection: Flickable.AutoFlickIfNeeded
            model: FolderListModel {
                folder: sceneOpenDialog.folderUrl
                nameFilters: SahKdTreeEngine.supportedSceneFileExtensions
                showDirsFirst: true
                showOnlyReadable: true
                showDotAndDotDot: true
            }
            delegate: Component {
                Label {
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
                        anchors.fill: parent
                        onDoubleClicked: (mouse) => {
                            if (fileIsDir) {
                                sceneOpenDialog.folderUrl = fileUrl
                            } else {
                                columnLayout.fileAccessed = fileAccessed
                                columnLayout.fileSize = fileSize
                                columnLayout.fileUrl = fileUrl
                                columnLayout.fileModified = fileModified
                                columnLayout.fileBaseName = fileBaseName
                                columnLayout.filePath = filePath
                                columnLayout.fileName = fileName
                                columnLayout.fileSuffix = fileSuffix
                                columnLayout.fileIsDir = fileIsDir
                                sceneOpenDialog.accept()
                            }
                            mouse.accepted = true
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
