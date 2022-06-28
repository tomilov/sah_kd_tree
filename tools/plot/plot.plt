#usage: gnuplot -c plot.plt 'data/fuzz/artifacts/crash-1234abcd'
#from gnuplot: call 'plot.plt' 'data/fuzz/artifacts/crash-1234abcd'
reset

set table $table
    plot ARG1 binary skip=16 format='%float32%float32%float32' using 1:2:3 with table
unset table

set print $data
    do for [i=1:|$table|:3] {
        print $table[i]
        print $table[i+1]
        print $table[i+2]
        print $table[i]
        print ''
        print ''
    }
set print

print $data

unset key
set xrange [0:10]
set yrange [0:10]
set zrange [0:10]
set view equal xyz
set walls
unset autoscale
unset border
unset tics

set multiplot layout 2,2 title 'General 3D view and 2D projections of a 3D scene' font ':Bold'

set title 'general view'
set angles degrees
set view acos(1/sqrt(3)), 135, 0.97
set xyplane 0

set arrow 1 from graph -1,  0,  0 to graph 1, 0, 0 filled size graph 0.05, 15
set arrow 2 from graph  0, -1,  0 to graph 0, 1, 0 filled size graph 0.05, 15
set arrow 3 from graph  0,  0, -1 to graph 0, 0, 1 filled size graph 0.05, 15
set label 1 at graph 1.05,    0,    0 'X' center
set label 2 at graph    0, 1.05,    0 'Y' center
set label 3 at graph    0,    0, 1.05 'Z' center

splot '$data' using 1:2:3:(column(-2)) notitle with lines linecolor variable

unset for [i=1:3] arrow i
unset for [i=1:3] label i

set xlabel 'X-axis' offset 0, 0 rotate by 90
set ylabel 'Y-axis' offset 0, 0
set zlabel 'Z-axis' offset 0, 0
set xtics border mirror
set ytics border mirror
set ztics border mirror
set grid xtics ytics ztics vertical

set title 'xz projection'
set view projection xz
replot

set title 'yz projection'
set view projection yz
replot

set title 'xy projection'
set xlabel norotate
set view projection xy
replot

unset multiplot
