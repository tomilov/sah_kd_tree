import QtQuick.Controls
import QtQuick

Dialog {
    parent: Overlay.overlay
    clip: true
    modal: true
    x: (parent.width - width) / 2
    y: (parent.height - height) / 2
}
