#!/bin/sh
#
# Run this to convert ADBench data files to Futhark-compatible data
# files.  They are pretty large, which is why we don't just commit
# them (also, eventually we should get rid of this - the data files
# are massively replicated, and should be constructed on-demand).

set -e
set -x

if [ $# -ne 1 ]; then
    echo "Use: $0 path/to/ADBench"
    exit 1
fi

g++ data2fut.cpp -o data2fut -O3 -std=c++14

ADBench=$1

datadir="$ADBench/data/hand"

mkdir -p data/simple_small
for x in $datadir/simple_small/hand*.txt; do
    echo $x
    ./data2fut simple "$x" "$datadir/simple_small/model/" | futhark dataset -b > data/simple_small/$(basename -s .txt $x).in
done

mkdir -p data/simple_big
for x in $datadir/simple_big/hand*.txt; do
    echo $x
    ./data2fut simple "$x" "$datadir/simple_big/model/" | futhark dataset -b > data/simple_big/$(basename -s .txt $x).in
done
