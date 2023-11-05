#!/bin/bash
FFMPEG=ffmpeg
ENCODER="../build/lfEncoder"
DIR=$1
REFERENCE=$2
CRF=$3
OUTDIR=$4

TEMP=$(mktemp -d)
INPUT=$TEMP"/input"
mkdir $INPUT
OUTPUT=$TEMP"/output"
mkdir $OUTPUT
FORMAT=libaom-av1

OUT_ORIGINAL=$OUTDIR"/original"
OUT_DECODED=$OUTDIR"/decoded"
mkdir $OUT_ORIGINAL
mkdir $OUT_DECODED

FILENAME=$(basename -- "$REFERENCE")
EXTENSION="${FILENAME##*.}"
cp $DIR"/"$REFERENCE $INPUT"/01."$EXTENSION
for file in $DIR/*
do
    cp $file $INPUT"/02."$EXTENSION
    cp $file $OUT_ORIGINAL
    $FFMPEG -i $INPUT"/%02d."$EXTENSION -pix_fmt yuv420p -crf $CRF -c:v $FORMAT -cpu-used 8 -row-mt 1 -tiles 2x2 $OUTPUT"/encoded.mkv"  
    $FFMPEG -i $OUTPUT"/encoded.mkv" $OUTPUT"/%02d."$EXTENSION
    TRIMFILE=$(basename -- "$file")
    mv $OUTPUT"/02."$EXTENSION $OUT_DECODED"/"$TRIMFILE
    rm $INPUT"/02."$EXTENSION
    rm $OUTPUT"/encoded.mkv"
    rm $OUTPUT"/01."$EXTENSION
done
$FFMPEG -i $INPUT"/%02d."$EXTENSION -pix_fmt yuv420p -crf $CRF -c:v $FORMAT -cpu-used 8 -row-mt 1 -tiles 2x2 $OUTPUT"/encoded.mkv"  
$FFMPEG -i $OUTPUT"/encoded.mkv" $OUTPUT"/%02d."$EXTENSION
TRIMFILE=$(basename -- "$REFERENCE")
mv $OUTPUT"/01."$EXTENSION $OUTDIR"/"$TRIMFILE 

rm -rf $TEMP
