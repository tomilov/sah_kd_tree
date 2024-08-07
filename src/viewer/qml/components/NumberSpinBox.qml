import QtQuick
import QtQuick.Controls as C

C.SpinBox {
    id: spinBox
    required property int decimals
    readonly property int decimalFactor: Math.pow(10, decimals)
    readonly property real realValue: value / decimalFactor
    function decimalToInt(x) {
        return Math.round(x * decimalFactor)
    }
    function updateValue(x) {
        value = decimalToInt(x)
    }
    validator: DoubleValidator {
        top:  Math.max(from, to)
        bottom: Math.min(from, to)
        decimals: spinBox.decimals
        notation: DoubleValidator.StandardNotation
        locale: spinBox.locale.toString()
    }
    textFromValue: function(x, locale) {
        return Number(x / decimalFactor).toLocaleString(locale, 'f', decimals)
    }
    valueFromText: function(text, locale) {
        return Math.round(Number.fromLocaleString(locale, text) * decimalFactor)
    }
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
