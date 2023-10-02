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
RESULTLOGVERBOSE=$7
SCALE=5
echo -n "" > $RESULTLOG
echo -n "" > $RESULTLOGVERBOSE

for q in 0.0 0.25 0.5 0.75 1.0
do 
    echo "Quality: $q" >> $RESULTLOG
    echo "Quality: $q" >> $RESULTLOGVERBOSE
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
                    nx=$(bc <<< "scale=$SCALE;${COORDS[0]}/($GRID_WIDTH-1)")
                    ny=$(bc <<< "scale=$SCALE;${COORDS[1]}/($GRID_HEIGHT-1)")
                    $STREAMER -i $TESTFILE -m PP -t "$nx"_"$ny" -o $WORKPATH
                    OUTPUTFILES=($WORKPATH/*.ppm)
                    OUTFILE=${OUTPUTFILES[0]}
                    RESULT=$($METRICS $OUTFILE $file) 
                    RESULT=($RESULT) 
                    PSNR=$(bc <<< "scale=$SCALE;$PSNR+${RESULT[0]}")
                    SSIM=$(bc <<< "scale=$SCALE;$SSIM+${RESULT[1]}")
                    VMAF=$(bc <<< "scale=$SCALE;$VMAF+${RESULT[2]}")
                    echo "Testing reference $file ($nx $ny) against $OUTFILE, $x, $y, ${RESULT[0]}, ${RESULT[1]}, ${RESULT[2]}" >> $RESULTLOGVERBOSE
                    rm $WORKPATH/*.ppm
                done
                COUNT=$( ls $REFERENCEFILES | wc -l)
                PSNR=$(bc <<< "scale=$SCALE;$PSNR/$COUNT")
                SSIM=$(bc <<< "scale=$SCALE;$SSIM/$COUNT")
                VMAF=$(bc <<< "scale=$SCALE;$VMAF/$COUNT")
                SIZE=$(stat --printf="%s" $TESTFILE)
                echo $x, $y, $PSNR, $SSIM, $VMAF, $SIZE >> $RESULTLOG
                rm $TESTFILE
        done
    done
done
rm -rf $WORKPATH
