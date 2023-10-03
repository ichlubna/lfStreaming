#!/bin/bash

DATAPATH=./oneTimeFrame
./testKeyFramePosition.sh $DATAPATH/animBonfire 2.276 0.06_0.3 8 8 bonfire.csv bonfireV.csv 
./testKeyFramePosition.sh $DATAPATH/animStreet 1.816 0.16_0.29 8 8 street.csv streetV.csv 
./testKeyFramePosition.sh $DATAPATH/animeTimelapse 2.046 0.0_0.29 8 8 timelapse.csv timelapseV.csv 
./testKeyFramePosition.sh $DATAPATH/animKey 1.909 0.19_0.53 8 8 key.csv keyV.csv 

DATAPATH=./videos
./testGoPSize.sh $DATAPATH/animBonfire 2.276 0.06_0.3 8 8 bonfireG.csv bonfireVG.csv 
./testGoPSize.sh $DATAPATH/animStreet 1.816 0.16_0.29 8 8 streetG.csv streetVG.csv 
./testGoPSize.sh $DATAPATH/animeTimelapse 2.046 0.0_0.29 8 8 timelapseG.csv timelapseVG.csv 
./testGoPSize.sh $DATAPATH/animKey 1.909 0.19_0.53 8 8 keyG.csv keyVG.csv 
