#!/usr/bin/env bash

runs=10

for ((i=1; i<=runs; i++)); do
    time -p ./advection2D >/dev/null
done 2>&1 | awk '
    $1 == "real" { sum += $2; count++ }
    END {
        print "Average runtime:", sum / count, "seconds"
    }
'
