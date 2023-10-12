#!/bin/bash
IMAGEMETRICS=./imageQualityMetrics.sh
DIR1=$1
DIR2=$2

SCALE=5
PSNR=0
SSIM=0
VMAF=0
for file in $DIR1/*
do
    FILENAME=$(basename -- "$file")
    RESULT=$($IMAGEMETRICS $DIR1/$FILENAME $DIR2/$FILENAME)
    RESULT=($RESULT) 
    PSNR=$(bc <<< "scale=$SCALE;$PSNR+${RESULT[0]}")
    SSIM=$(bc <<< "scale=$SCALE;$SSIM+${RESULT[1]}")
    VMAF=$(bc <<< "scale=$SCALE;$VMAF+${RESULT[2]}")
done
COUNT=$( ls $DIR1 | wc -l)
PSNR=$(bc <<< "scale=$SCALE;$PSNR/$COUNT")
SSIM=$(bc <<< "scale=$SCALE;$SSIM/$COUNT")
VMAF=$(bc <<< "scale=$SCALE;$VMAF/$COUNT")
echo $PSNR, $SSIM, $VMAF
