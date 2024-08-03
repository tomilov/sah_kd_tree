import QtQuick.Controls as QC
import QtQuick

QC.Dialog {
    parent: QC.Overlay.overlay
    anchors.centerIn: parent
    modal: true
    clip: true
}
