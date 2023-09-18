#!/bin/bash
ENCODER="../build/lfEncoder"
STREAMER="../build/lfStreamer"
METRICS="./imageQualityMetrics.sh"

WORKPATH=$(mktemp -d)
TESTFILE="$WORKPATH/test.lf"
INPUT=$1
# The input directory is expected to contain only one time frame 01, works with more frames too but is slower
INPUTFILES="$INPUT/inputs"
REFERENCEFILES="$INPUT/references/01"
ASPECT=$2
FOCUS=$3
GRID_WIDTH=$4
GRID_HEIGHT=$5
RESULTLOG=$6
echo -n "" > $RESULTLOG

for q in 0.5 # 0.0 0.25 0.5 0.75 1.0
do 
    echo "Quality: $q" >> $RESULTLOG
    for (( x=0; x<$GRID_WIDTH; x++ ))
    do
        for (( y=0; y<$GRID_HEIGHT; y++ ))
        do 
                $ENCODER -i "$INPUTFILES" -q $q -f H265 -o $TESTFILE -k "$x"_"$y" -a $ASPECT -s $FOCUS
                PSNR=0
                SSIM=0
                VMAF=0
                for file in $REFERENCEFILES/*
                do
                    FILENAME=$(basename -- "$file")
                    FILENAME="${FILENAME%.*}"
                    IFS='_' read -r -a COORDS <<< "$FILENAME"
                    echo ${COORDS[0]}
                    nx=$(bc <<< "scale=5;${COORDS[0]}/($GRID_WIDTH-1)")
                    ny=$(bc <<< "scale=5;${COORDS[1]}/($GRID_HEIGHT-1)")
                    $STREAMER -i $TESTFILE -m PP -t "$nx"_"$ny" -o $WORKPATH
                    OUTPUTFILES=($WORKPATH/*.ppm)
                    OUTFILE=${OUTPUTFILES[0]}
                    RESULT=$($METRICS $OUTFILE $file) 
                    RESULT=($RESULT) 
                    PSNR=$(bc <<< "scale=5;$PSNR+${RESULT[0]}")
                    SSIM=$(bc <<< "scale=5;$SSIM+${RESULT[1]}")
                    VMAF=$(bc <<< "scale=5;$VMAF+${RESULT[2]}")
                    rm $WORKPATH/*.ppm
                done
                COUNT=$( ls $REFERENCEFILES | wc -l)
                PSNR=$(bc <<< "scale=5;$PSNR/$COUNT")
                SSIM=$(bc <<< "scale=5;$SSIM/$COUNT")
                VMAF=$(bc <<< "scale=5;$VMAF/$COUNT")
                echo $x, $y, $PSNR, $SSIM, $VMAF >> $RESULTLOG
                rm $TESTFILE
        done
    done
done
rm -rf $WORKPATH
