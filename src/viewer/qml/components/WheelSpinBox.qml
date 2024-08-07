import QtQuick
import QtQuick.Controls as C

C.SpinBox {
    live: true
    WheelHandler {
        onWheel: wheel => {
            if (wheel.angleDelta.y < 0) {
                decrease()
            } else {
                increase()
            }
        }
    }
}
