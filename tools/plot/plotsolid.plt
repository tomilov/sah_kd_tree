#usage: gnuplot -c plot.plt file
reset

set view equal xyz
set autoscale
unset border
unset xtics
unset ytics
unset ztics
set xrange [-6:6]
set yrange [-6:6]
set zrange [-6:6]

set table $table
    plot ARG1 binary skip=12 format='%float32' using 1 with table
unset table

set palette file "-"
0 0 1
0 1 0
1 0 0
0 1 1
1 1 0
1 0 1
e

set cbrange [0:1]
do for [i=1:|$table|:9] {
    set object (1 + i/9) polygon from $table[i],$table[i+1],$table[i+2] \
                                   to $table[i+3],$table[i+4],$table[i+5] \
                                   to $table[i+6],$table[i+7],$table[i+8] \
                                   to $table[i],$table[i+1],$table[i+2] \
                         fillstyle solid fillcolor palette frac (i-1.0)/(|$table|-9.0)
}

$origin <<EOD
0 0 0
EOD

splot '$origin' with points notitle
pause -1
