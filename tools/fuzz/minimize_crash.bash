set -e
[[ $1 ]]
data="$( dirname "$0" )/../../data/fuzz/"
./Fuzz -minimize_crash=1 -max_total_time=10 -artifact_prefix="$data/artifacts/" $1

