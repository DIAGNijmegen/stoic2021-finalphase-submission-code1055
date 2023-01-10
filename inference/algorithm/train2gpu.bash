#!/usr/bin/env bash
# important to run from bash!

FAIL=0

echo "Building cache"
python3 -m submission_build_cache
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Cache failed ($retVal)"
    exit 1
fi
echo Cache is ready

python3 -m submission_train 0 forwards &
python3 -m submission_train 1 backwards &

for job in $(jobs -p)
do
    echo "Wait for job $job"
    wait $job || let "FAIL+=1"
done

echo "Failed: $FAIL"
echo "done"
