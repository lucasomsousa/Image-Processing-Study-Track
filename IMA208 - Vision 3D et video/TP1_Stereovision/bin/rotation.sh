#! /bin/ksh

#########################################################
#       Run stereo matching (3 views)                   #
#                                                       #
#                                                       #
#       MAXIM FRADKIN      1-6-99                       #
#                                                       #
#########################################################
if [ $# != 2 ] ; then
echo " Usage : $0 <image gauche> <image droite>"
exit 1
fi

BIN="./bin" 

DISK="."
LE=$1
R1=$2

ext2rel-ign ${LE}.ori ${R1}.ori ${LE}.mat ${R1}.mat






