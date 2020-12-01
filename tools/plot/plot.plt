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
    plot ARG1 binary skip=12 format='%float32%float32%float32' using 1:2:3 with table
unset table

set print $data
    do for [i=1:|$table|:3] {
        print $table[i]
        print $table[i+1]
        print $table[i+2]
        print $table[i]
        print ""
        print ""
    }
set print

splot '$data' using 1:2:3:(column(-2)) notitle with lines linecolor variable
pause -1
