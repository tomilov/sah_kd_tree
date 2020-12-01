set -e
[[ $1 ]]
./Fuzz -minimize_crash=1 -max_total_time=10 -artifact_prefix=artifacts/ $1

