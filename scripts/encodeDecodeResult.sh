#!/bin/bash
IMAGEMETRICS=./imageQualityMetrics.sh
FFMPEG=ffmpeg
DIR=$1
REFERENCE=$2
CRF=$3
OUTDIR=$4

TEMP=$(mktemp -d)
INPUT=$TEMP"/input"
mkdir $INPUT
OUTPUT=$TEMP"/output"
mkdir $OUTPUT

FILENAME=$(basename -- "$REFERENCE")
EXTENSION="${FILENAME##*.}"
cp $DIR"/"$REFERENCE $INPUT"/01."$EXTENSION
for file in $DIR/*
do
    cp $file $INPUT"/02."$EXTENSION
    $FFMPEG -i $INPUT"/%02d."$EXTENSION -crf $CRF -c:v libaom-av1 $OUTPUT"/encoded.mkv"  
    $FFMPEG -i $OUTPUT"/encoded.mkv" $OUTPUT"/%02d."$EXTENSION
    TRIMFILE=$(basename -- "$file")
    mv $OUTPUT"/02."$EXTENSION $OUTDIR"/"$TRIMFILE 
    rm $INPUT"/02."$EXTENSION
    rm $OUTPUT"/encoded.mkv"
    rm $OUTPUT"/01."$EXTENSION
done
$FFMPEG -i $INPUT"/%02d."$EXTENSION -crf $CRF -c:v libaom-av1 $OUTPUT"/encoded.mkv"  
$FFMPEG -i $OUTPUT"/encoded.mkv" $OUTPUT"/%02d."$EXTENSION
TRIMFILE=$(basename -- "$REFERENCE")
mv $OUTPUT"/01."$EXTENSION $OUTDIR"/"$TRIMFILE 

rm -rf $TEMP
