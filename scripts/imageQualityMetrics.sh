#!/bin/bash
FFMPEG=ffmpeg
INPUT=$1
REFERENCE=$2

RESULT=$($FFMPEG -i $INPUT -i $REFERENCE -filter_complex "psnr" -f null /dev/null 2>&1)
PSNR=$(echo "$RESULT" | grep -oP '(?<=average:).*?(?= min)')
RESULT=$($FFMPEG -i $INPUT -i $REFERENCE -filter_complex "ssim" -f null /dev/null 2>&1)
SSIM=$(echo "$RESULT" | grep -oP '(?<=All:).*?(?= )')
RESULT=$($FFMPEG -i $INPUT -i $REFERENCE -lavfi libvmaf -f null /dev/null 2>&1)
VMAF=$(echo "$RESULT" | grep -oP '(?<=VMAF score: ).*')
echo "$PSNR $SSIM $VMAF"
