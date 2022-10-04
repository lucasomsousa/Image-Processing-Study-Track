#! /bin/ksh

#########################################################
#       Run stereo matching (3 views)                   #
#                                                       #
#                                                       #
#       MAXIM FRADKIN      1-6-99                       #
#                                                       #
#########################################################
if [ $# != 6 ] ; then
echo " Usage : $0 <image originale> <image epipolaire> <start x> <start y> <largeur> <hauteur>"
exit 1
fi


DISK="."
LE=$1
R1=$2
xstart=$3
ystart=$4
np=$5
nl=$6

env LD_LIBRARY_PATH=/tsi/tp/lib img2epip-ign ${LE}.tif ${R1}.tif ${LE}.ori ${LE}.mat $xstart $ystart 0 0 $np $nl





