#!/bin/sh
#
# Run this to convert ADBench data files to Futhark-compatible data
# files.  They are pretty large, which is why we don't just commit
# them (also, eventually we should get rid of this - the data files
# are massively replicated, and should be constructed on-demand).

set -e

if [ $# -ne 1 ]; then
    echo "Use: $0 path/to/ADBench"
    exit 1
fi

ADBench=$1

futhark c expand.fut
mkdir -p data

for x in $(find "$ADBench/data/ba/" -name \*.txt); do
    echo $x
    ./expand -b < $x > data/$(basename -s .txt $x).in;
done
