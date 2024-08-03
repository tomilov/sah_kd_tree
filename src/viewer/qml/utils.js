function pprops(item) {
  console.log('PPROPS:', (typeof item).toString())
  for (let p in item) console.log(p + ': ' + item[p]);
}

function coloredText(text, color) { return '<font color="%1">%2</font>'.arg(color).arg(text) }
