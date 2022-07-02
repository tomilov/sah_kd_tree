dir="$( dirname "$0" )"
data="$dir/../../data/fuzz"
"$dir/fuzzer" \
    -fork=$(( $( nproc ) / 2 )) \
    -use_value_profile=1 \
    -rss_limit_mb=128 \
    -timeout=30 \
    -report_slow_units=30 \
    -print_final_stats=1 \
    -print_corpus_stats=1 \
    -print_pcs=1 \
    -reduce_depth=1 \
    -reduce_inputs=1 \
    -shrink=1 \
    -max_total_time=60 \
    -prefer_small=1 \
    -artifact_prefix="$data/artifacts/" \
    "$data/CORPUS/"
