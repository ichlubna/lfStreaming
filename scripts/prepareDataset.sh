#!/bin/bash
#https://github.com/ichlubna/lfFocusMaps/blob/main/scripts/renameLFImages.py
RENAMESCRIPT=./renameLFImages.py
WORKPATH=./workdir
mkdir $WORKPATH

NAME=animBonfire

#Comment for static scenes:
wget https://merlin.fit.vutbr.cz/LightField/datasets/lfDataset/data/$NAME.7z -P $WORKPATH/

#Use for static scenes:
#wget https://merlin.fit.vutbr.cz/LightField/datasets/lfDataset/data/$NAME.mkv -P $WORKPATH/

EXTRACTED=$WORKPATH/extracted
mkdir $EXTRACTED

#Comment for static scenes:
7z x -y $WORKPATH/$NAME.7z -o$EXTRACTED/

INPATH=$EXTRACTED/$NAME

#Use for static scenes:
#mkdir $INPATH
#mv $WORKPATH/$NAME.mkv $INPATH

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
