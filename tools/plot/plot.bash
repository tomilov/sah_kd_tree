plt="$( mktemp '/tmp/sahkdtree.XXXXXXXXX' )"
trap 'rm "$plt"' EXIT

cat <<__EOF >"$plt"
reset

\$data <<EOD
__EOF
od --skip-bytes=12 --address-radix=n -f $1 |
    sed -e 's/[[:space:]]\+/\n/g' |
    sed '/^[[:space:]]*$/d' |
    awk '
{
    if (NR != 1 && NR % 9 == 1)
        print ""
    if (NR % 3 == 0) {
        print $0
    } else {
        printf "%s ", $0
    }
    switch (NR % 9) {
    case 1 :
        x=$0
        break
    case 2 :
        y=$0
        break
    case 3 :
        z=$0
        break
    case 0 :
        print x, y, z
        print ""
        break
    }
}
' >>"$plt"
cat <<__EOF >>"$plt"
EOD

set view equal xyz
set autoscale
unset border
unset xtics
unset ytics
unset ztics
set xrange [-6:6]
set yrange [-6:6]
set zrange [-6:6]

splot '\$data' using 1:2:3:(column(-2)) notitle with lines linecolor variable
pause -1
__EOF
gnuplot -c "$plt"

