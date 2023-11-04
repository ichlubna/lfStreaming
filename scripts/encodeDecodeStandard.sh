#!/bin/bash
FFMPEG=ffmpeg
DIR=$1
CRF=$2
OUTDIR=$3
GOP=$4

OUT_ORIGINAL=$OUTDIR"/original"
OUT_DECODED=$OUTDIR"/decoded"
mkdir $OUT_ORIGINAL
mkdir $OUT_DECODED
ENCODED=$OUTDIR"/encoded.mkv"

COUNTER=0
EXTENSION=""
for file in $DIR/*
do
    FILENAME=$(basename -- "$file")
    EXTENSION="${FILENAME##*.}"
    NEW_NAME=$(printf "%04d" $COUNTER)
    cp $file $OUT_ORIGINAL"/"$NEW_NAME"."$EXTENSION
    COUNTER=$((COUNTER+1))
done

$FFMPEG -i $OUT_ORIGINAL"/%04d."$EXTENSION -crf $CRF -g $GOP -c:v libaom-av1 -cpu-used 4 -row-mt 1 $ENCODED 
$FFMPEG -i $ENCODED $OUT_DECODED"/%04d."$EXTENSION

rm -rf $TEMP
