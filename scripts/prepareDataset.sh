#!/bin/bash
#https://github.com/ichlubna/lfFocusMaps/blob/main/scripts/renameLFImages.py
RENAMESCRIPT=./renameLFImages.py
WORKPATH=./workdir
mkdir $WORKPATH

wget https://merlin.fit.vutbr.cz/LightField/datasets/lfDataset/data/animBonfire.7z -O $WORKPATH/
EXTRACTED=$WORKPATH/extracted
mkdir $EXTRACTED
7za x -y animBonfire.7z -o$EXTRACTED/
INPATH=$EXTRACTED/animfire

OUTPATH=$WORKPATH/decoded
mkdir $OUTPATH
for FILE in $INPATH/*; do
    FILENAME=$(basename -- "$FILE")
    FILENAME="${FILENAME%.*}"
    mkdir $OUTPATH/$FILENAME
    ffmpeg -i $FILE -vf scale=1920x1080 $OUTPATH/$FILENAME/%04d.png
done

TEMP=$WORKPATH/temp
mkdir $TEMP
OUTPATHFINAL=finalResult
mkdir $OUTPATHFINAL
INPUTS=$OUTPATHFINAL/inputs
mkdir $INPUTS
REFS=$OUTPATHFINAL/references
mkdir $REFS
for FILE in $OUTPATH/*; do
    FILENAME=$(basename -- "$FILE")
    FILENAME="${FILENAME%.*}"
    python $RENAMESCRIPT $FILE $TEMP 0 0 15 15 0 1 1
    mv $TEMP/data $INPUTS/$FILENAME
    mv $TEMP/reference $REFS/$FILENAME
done
