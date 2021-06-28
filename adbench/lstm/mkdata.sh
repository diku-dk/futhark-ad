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

datadir="$ADBench/data/lstm"



#lstm_l2_c1024.txt
#lstm_l2_c4096.txt
#lstm_l4_c1024.txt
#lstm_l4_c4096.txt

if [ -f tmp.txt ]; then
    # Remove  the file with permission
    rm   tmp.txt
fi

for x in $datadir/lstm_l*.txt; do
    echo $x
    ./data2fut "$x" "tmp.txt"
    cat tmp.txt | futhark dataset -b > data/$(basename -s .txt $x)_d14.in
    rm tmp.txt
done
