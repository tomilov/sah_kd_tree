import QtQuick
import QtQuick.Controls

SpinBox {
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
        top:  Math.max(spinBox.from, spinBox.to)
        bottom: Math.min(spinBox.from, spinBox.to)
        decimals: spinBox.decimals
        notation: DoubleValidator.StandardNotation
        locale: spinBox.locale.toString()
    }
    textFromValue: function(x, locale) {
        return Number(x / decimalFactor).toLocaleString(locale, 'f', spinBox.decimals)
    }
    valueFromText: function(text, locale) {
        return Math.round(Number.fromLocaleString(locale, text) * decimalFactor)
    }
    WheelHandler {
        onWheel: (wheel) => {
            if (wheel.angleDelta.y < 0) {
                spinBox.decrease()
            } else {
                spinBox.increase()
            }
        }
    }
}
