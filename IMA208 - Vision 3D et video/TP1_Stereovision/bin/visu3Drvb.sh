if [ $# != 4 ] ; then
echo " Usage : $0 <image de profondeur> <image de texture> <coef en z(1)> <saut(2)>"
exit 1
fi


PROFONDEUR=$1
TEXTURE=$2
COEF=$3
SAUT=$4

median  ${PROFONDEUR} tmpmedian.tif -0 

ima2imw  tmpmedian.tif tmpmedian.tif 

disp2bin tmpmedian.tif  ${TEXTURE} visu.pn -kz ${COEF} -kz2 0 -tri -s ${SAUT} -ply visu.ply -rvb

pmini visu.pn &

rm -f tmpmedian.tif 


