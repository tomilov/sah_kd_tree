dir="$( dirname "$0" )"
data="$dir/../../data/fuzz"
"$dir/fuzzer" \
    -print_final_stats=1 \
    -print_corpus_stats=1 \
    -print_coverage=1 \
    -print_pcs=1 \
    -artifact_prefix="$data/artifacts/" \
    -fork=$(( $( nproc ) / 2 )) \
    -rss_limit_mb=64 \
    -timeout=1 \
    -report_slow_units=1 \
    -reduce_inputs=1 \
    -shrink=1 \
    "$data/CORPUS/" \
    "$data/artifacts/"
