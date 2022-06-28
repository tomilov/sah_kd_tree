set -e
[[ $1 ]]

dir="$( dirname "$0" )"
data="$dir/../../data/fuzz"
"$dir/fuzzer" -minimize_crash=1 -max_total_time=60 -artifact_prefix="$data/artifacts/" $1

