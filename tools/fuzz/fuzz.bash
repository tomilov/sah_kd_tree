nproc=$( nproc )
data="$( dirname "$0" )/../../data/fuzz/"
./Fuzz -print_final_stats=1 -artifact_prefix="$data/artifacts/" -jobs=$(( $nproc / 2 )) "$data/CORPUS/"
