set -e
[[ $1 ]]

dir="$( dirname "$0" )"
data="$dir/../../data/fuzz/"
"$data/Fuzz" -minimize_crash=1 -max_total_time=10 -artifact_prefix="$data/artifacts/" $1

