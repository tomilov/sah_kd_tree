import QtQuick.Controls

Dialog {
    parent: ApplicationWindow.overlay

    clip: true
    modal: true

    x: (parent.width - width) / 2
    y: (parent.height - height) / 2
}
