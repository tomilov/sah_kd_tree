dir="$( dirname "$0" )"
data="$dir/../../data/fuzz"
"$dir/fuzzer" -print_final_stats=1 -artifact_prefix="$data/artifacts/" -jobs=$(( $( nproc ) / 2 )) "$data/CORPUS"
