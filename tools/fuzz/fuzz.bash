dir="$( dirname "$0" )"
data="$dir/../../data/fuzz"
"$dir/fuzzer" \
    -fork=$(( $( nproc ) / 1 )) \
    -use_value_profile=1 \
    -rss_limit_mb=64 \
    -timeout=10 \
    -report_slow_units=10 \
    -print_final_stats=1 \
    -print_corpus_stats=1 \
    -print_coverage=1 \
    -print_pcs=1 \
    -artifact_prefix="$data/artifacts/" \
    -reduce_depth=1 \
    -reduce_inputs=1 \
    -shrink=1 \
    -max_total_time=60 \
    -prefer_small=1 \
    "$data/CORPUS/" \
    "$data/artifacts/"
