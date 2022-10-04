#!/bin/bash

# set path to data and scripts
export MISS_STEREO=/tsi/tp/TP/MissStereo

# set path to binaries
export MISS_STEREO_PATH=/tsi/tp/TP/MissStereo/build/bin

# Exit in case of error
set -e

# Necessary utilities
DIRNAME=/usr/bin/dirname

IMTIFF1=${1}1.tif
IMTIFF2=${1}2.tif
IMTIFF3=${1}G.tif
IMTIFF4=${1}D.tif


IMG1="im1.pgm"
IMG2="im2.pgm"

convert $IMTIFF1 $IMG1
convert $IMTIFF2 $IMG2

sift <  $IMG1 > im1.key 
sift <  $IMG2 > im2.key 

match-sift -im1 im1.pgm -k1 im1.key -im2 im2.pgm -k2 im2.key > out.pgm
convert out.pgm sift.tif 

gimp sift.tif &

mv matches.txt pairs.txt


IMG1="im1.png"
IMG2="im2.png"

convert $IMTIFF1 $IMG1
convert $IMTIFF2 $IMG2


DIM="`$MISS_STEREO_PATH/size $IMG1`"

echo $DIM

[ "$DIM" = "`$MISS_STEREO_PATH/size $IMG2`" ] || { echo Images are not of same size. Unable to proceed; false; }



$MISS_STEREO_PATH/orsa $DIM pairs.txt pairs_good.txt 500 1 0 2 0

$MISS_STEREO_PATH/rectify pairs_good.txt $DIM homography1.txt  homography2.txt 

$MISS_STEREO_PATH/homography $IMG1  homography1.txt testG.png 

$MISS_STEREO_PATH/homography $IMG2  homography2.txt testD.png 

convert testG.png testG.tif 
convert testD.png testD.tif 

echo ${1}

rvb2ima testG.tif toto.tif 

mv totoR.tif ${1}RG.tif 
mv totoV.tif ${1}VG.tif 
mv totoB.tif ${1}BG.tif 

rvb2ima testD.tif toto.tif 

mv totoR.tif ${1}RD.tif 
mv totoV.tif ${1}VD.tif 
mv totoB.tif ${1}BD.tif 


rm totoR.tif 
rm totoV.tif 
rm totoB.tif 

