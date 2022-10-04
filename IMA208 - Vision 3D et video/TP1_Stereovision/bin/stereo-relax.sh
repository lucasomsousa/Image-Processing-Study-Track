#! /bin/ksh
#########################################################
#       Construction d'une carte de disparite           #
#                                                       #
#       Algrithme de Ugur Murat Leloglu                 #
#                                                       #
#       Image:  Michel ROUX      19-2-99                #
#                                                       #
#########################################################
if [ $# != 3 ] ; then
echo " Usage : $0 <scene> <dispa-min> <dispa-max>"
exit 1
fi

SCENE=$1
BAS=$2
HAUT=$3

if [ -z "$IMAGES_DIR" ] ; then
	IMAGE="."                             	# CNET
fi

if [ -z "$SCRATCH_DIR" ] ; then
	SCRATCH="scratch"                        		# CNET
fi

\rm -r $SCRATCH
mkdir $SCRATCH
env LD_LIBRARY_PATH=/tsi/tp/lib 


subsampling ${IMAGE}/${SCENE}G.tif ${SCRATCH}/${SCENE}Gs.tif -c 2 -moy -v 
subsampling ${SCRATCH}/${SCENE}Gs.tif ${SCRATCH}/${SCENE}Gss.tif -c 2  -moy -v

subsampling ${IMAGE}/${SCENE}D.tif ${SCRATCH}/${SCENE}Ds.tif -c 2   -moy -v 
subsampling ${SCRATCH}/${SCENE}Ds.tif ${SCRATCH}/${SCENE}Dss.tif -c 2   -moy -v 


factor=4
let HAUT=HAUT/factor
let BAS=BAS/factor

echo $HAUT $BAS

env LD_LIBRARY_PATH=/tsi/tp/lib correlation_bw ${SCRATCH}/${SCENE}Gss.tif ${SCRATCH}/${SCENE}Dss.tif \
	${BAS} ${HAUT} 0.0 \
	${SCRATCH}/disN1ss.tif ${SCRATCH}/disN2ss.tif \
	${SCRATCH}/corN1ss.tif ${SCRATCH}/corN2ss.tif


env LD_LIBRARY_PATH=/tsi/tp/lib relaxation ${SCRATCH}/disN1ss.tif ${SCRATCH}/disN2ss.tif \
	${SCRATCH}/corN1ss.tif ${SCRATCH}/corN2ss.tif \
	${SCRATCH}/disNss.tif ${BAS} ${HAUT}


let HAUT=HAUT*2
let BAS=BAS*2

echo $HAUT $BAS

env LD_LIBRARY_PATH=/tsi/tp/lib correlation_bw_h ${SCRATCH}/${SCENE}Gs.tif ${SCRATCH}/${SCENE}Ds.tif \
	${BAS} ${HAUT} 0.0 \
	${SCRATCH}/disN1s.tif ${SCRATCH}/disN2s.tif \
	${SCRATCH}/corN1s.tif ${SCRATCH}/corN2s.tif \
	${SCRATCH}/disNss.tif

env LD_LIBRARY_PATH=/tsi/tp/lib relaxation ${SCRATCH}/disN1s.tif ${SCRATCH}/disN2s.tif \
	${SCRATCH}/corN1s.tif ${SCRATCH}/corN2s.tif \
	${SCRATCH}/disNs.tif ${BAS} ${HAUT}


let HAUT=HAUT*2
let BAS=BAS*2

echo $HAUT $BAS

env LD_LIBRARY_PATH=/tsi/tp/lib correlation_bw_h ${IMAGE}/${SCENE}G.tif ${IMAGE}/${SCENE}D.tif \
	${BAS} ${HAUT} 0.0 \
	${SCRATCH}/disN1.tif ${SCRATCH}/disN2.tif \
	${SCRATCH}/corN1.tif ${SCRATCH}/corN2.tif \
	${SCRATCH}/disNs.tif

env LD_LIBRARY_PATH=/tsi/tp/lib relaxation ${SCRATCH}/disN1.tif ${SCRATCH}/disN2.tif \
	${SCRATCH}/corN1.tif ${SCRATCH}/corN2.tif \
	${SCRATCH}/disN.tif ${BAS} ${HAUT}

noir2blanc ${SCRATCH}/disN.tif ${SCRATCH}/disN.tif

inversion ${SCRATCH}/disN.tif ${IMAGE}/dispa-${1}.tif

echo "Image de disparité :  ${IMAGE}/dispa-${1}.tif"
