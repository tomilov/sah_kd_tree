import QtQuick.Controls as QQC
import QtQuick

QQC.Dialog {
    parent: QQC.Overlay.overlay
    anchors.centerIn: parent
    modal: true
    clip: true
}
