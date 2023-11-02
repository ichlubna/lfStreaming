#!/bin/bash
ENCODER="../build/lfEncoder"
STREAMER="../build/lfStreamer"
METRICS="./imageQualityMetrics.sh"

WORKPATH=$(mktemp -d)
TESTFILE="$WORKPATH/test.lf"
INPUT=$1
INPUTFILES="$INPUT/inputs"
REFERENCEFILES="$INPUT/references"
ASPECT=$2
FOCUS=$3
GRID_WIDTH=$4
GRID_HEIGHT=$5
RESULTLOG=$6
RESULTLOGVERBOSE=$7
SCALE=5
echo -n "" > $RESULTLOG
echo -n "" > $RESULTLOGVERBOSE

for q in 0.25 0.5 0.75 1.0
do 
    echo "Quality: $q" >> $RESULTLOG
    echo "Quality: $q" >> $RESULTLOGVERBOSE
    COUNTDIRS=$( ls $REFERENCEFILES | wc -l)
    for gop in $(seq 1 $COUNTDIRS)
    do
            $ENCODER -i "$INPUTFILES" -q $q -f H265 -o $TESTFILE -g $gop -a $ASPECT -s $FOCUS 
            PSNR=0
            SSIM=0
            VMAF=0
            COUNT=0
            DIRID=0
            for dir in $REFERENCEFILES/*
            do
                for file in $dir/*
                do
                    FILENAME=$(basename -- "$file")
                    FILENAME="${FILENAME%.*}"
                    IFS='_' read -r -a COORDS <<< "$FILENAME"
                    nx=$(bc <<< "scale=$SCALE;${COORDS[0]}/($GRID_WIDTH-1)")
                    ny=$(bc <<< "scale=$SCALE;${COORDS[1]}/($GRID_HEIGHT-1)")
                    $STREAMER -i $TESTFILE -m PP -t "$nx"_"$ny" -f $DIRID -o $WORKPATH
                    OUTPUTFILES=($WORKPATH/*.ppm)
                    OUTFILE=${OUTPUTFILES[0]}
                    RESULT=$($METRICS $OUTFILE $file) 
                    RESULT=($RESULT) 
                    PSNR=$(bc <<< "scale=$SCALE;$PSNR+${RESULT[0]}")
                    SSIM=$(bc <<< "scale=$SCALE;$SSIM+${RESULT[1]}")
                    VMAF=$(bc <<< "scale=$SCALE;$VMAF+${RESULT[2]}")
                    echo "Testing reference $file ($nx $ny) against $OUTFILE, $gop, ${RESULT[0]}, ${RESULT[1]}, ${RESULT[2]}" >> $RESULTLOGVERBOSE
                    rm $WORKPATH/*.ppm
                    COUNT=$((COUNT+1))
                done
                DIRID=$((DIRID+1))
            done
            COUNT=$( ls $REFERENCEFILES | wc -l)
            PSNR=$(bc <<< "scale=$SCALE;$PSNR/$COUNT")
            SSIM=$(bc <<< "scale=$SCALE;$SSIM/$COUNT")
            VMAF=$(bc <<< "scale=$SCALE;$VMAF/$COUNT")
            SIZE=$(stat --printf="%s" $TESTFILE)
            echo $gop, $PSNR, $SSIM, $VMAF, $SIZE >> $RESULTLOG
            rm $TESTFILE
    done
done
rm -rf $WORKPATH
