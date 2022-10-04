#!/bin/bash

# set path to data and scripts
export MISS_STEREO=/tsi/tp/TP/MissStereo

# set path to binaries
export MISS_STEREO_PATH=./bin/

# Exit in case of error
set -e

# Necessary utilities
DIRNAME=/usr/bin/dirname

IMTIFF1=${1}1.tif
IMTIFF2=${1}2.tif
IMTIFF3=${1}G.tif
IMTIFF4=${1}D.tif

IMG1="im1.png"
IMG2="im2.png"

convert $IMTIFF1 $IMG1
convert $IMTIFF2 $IMG2

$MISS_STEREO_PATH/asift $IMG1 $IMG2 pairs.txt sift.png 

convert sift.png -compress Lossless  sift.tif

xv sift.tif &


DIM="`$MISS_STEREO_PATH/size $IMG1`"

echo $DIM

[ "$DIM" = "`$MISS_STEREO_PATH/size $IMG2`" ] || { echo Images are not of same size. Unable to proceed; false; }


$MISS_STEREO_PATH/orsa $DIM pairs.txt pairs_good.txt 500 1 0 2 0

$MISS_STEREO_PATH/rectify pairs_good.txt $DIM homography1.txt  homography2.txt 

$MISS_STEREO_PATH/homography $IMG1  homography1.txt testG.png 

$MISS_STEREO_PATH/homography $IMG2  homography2.txt testD.png 

convert testG.png -compress Lossless -colorspace Gray $IMTIFF3
convert testD.png -compress Lossless -colorspace Gray $IMTIFF4

