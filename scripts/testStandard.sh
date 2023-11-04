ENCODE_DECODE=./encodeDecodeStandard.sh
ENCODE_DECODE_PROP=./encodeDecodeResult.sh
COMPARE=./compareDirs.sh
TEMP=$(mktemp -d)
DATA=../build/oneTimeFrame
ENCODED=$TEMP"/encoded.mkv"
ENCODED_PROP=$TEMP"/encoded.lf"
ORIGINAL=$TEMP"/original"
DECODED=$TEMP"/decoded"
FFMPEG=ffmpeg
REPORT=standardReport.csv
STREAMER="../build/lfStreamer"
ENCODER="../build/lfEncoder"
echo -n "" > $REPORT

for crf in {0..63..7}
do  
    for scene in animBonfire animKey animStreet animTimelapse
    do
        INPUT_DATA=$DATA"/"$scene"/inputs"
        for gop in 1 64 8
        do  
                $ENCODE_DECODE $INPUT_DATA/01  $crf $TEMP $gop
                METRICS=$($COMPARE $ORIGINAL $ENCODED)
                SIZE=$(stat --printf="%s" $ENCODED)
                TIME_STRING=$($FFMPEG -hwaccel nvdec -c:v av1 -i $ENCODED -benchmark -f null - 2>&1)
                TIME=$(echo "$TIME_STRING" | grep -oP '(?<=rtime=).*?(?=s)')
                echo $crf, $gop, $scene, $SIZE, $TIME, $METRICS >> $REPORT
                rm -rf $TEMP"/*"
        done
        $ENCODE_DECODE_PROP $INPUT_DATA/01 04_04.png $crf $TEMP $gop
        METRICS=$($COMPARE $ORIGINAL $DECODED)
        NQ=$(bc <<< "scale=5;$crf/63")
        $ENCODER -i "$INPUT_DATA" -q $NQ -f AV1 -o $ENCODED_PROP -k 4_4 -a 1 -s 1
        SIZE=$(stat --printf="%s" $ENCODED_PROP)
        TIME_STRING=$($STREAMER -i $ENCODED_PROP -m PP -t 0.5_0.5 -o $TEMP)
        TIME=$(echo "$TIME_STRING" | grep -oP '(?<=GPU elapsed time: ).*?(?= ms)' | head -1)
        TIME=$(bc <<< "scale=5;$TIME/4")
        echo $crf", proposed,"$scene, $SIZE, $TIME, $METRICS >> $REPORT
        rm -rf $TEMP"/*"
    done

rm -rf $TEMP
