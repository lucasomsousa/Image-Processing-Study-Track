#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:54:13 2022

@author: mroux
"""

import numpy as np


R=np.array([[-0.944829, 0.084028, 0.316604],
[0.327323, 0.204999, 0.922407],                                                                                                                                                                                                                                                     
[0.012604, 0.975149, -0.221193]])

Ri=np.matrix.transpose(R)

t=np.array([600357.608941406, 2427659.984507813, 53.202734375])


C=np.array([1152.00,1728.000000])

f=2724.568661



K=np.array([[ f, 0.0, C[0]],
            [ 0.0, f, C[1]],
            [0.0, 0.0, 1.0]])
#
#try:
#    file = open('selection2.dat')
#    print(file)
#    file.close()
#except FileNotFoundError:
#    print('Fichier introuvable.')
#except IOError:
#    print('Erreur d\'ouverture.') 


ITER=10
distance=2
test=1
pts=[]
p=[0.00,0.0,0.0]
i=0



with open("b.dat", "r") as filin, open('c.dat', 'w') as dst:
    ligne = filin.readline()
    while ligne != "":
        s = ligne.strip ("\n\r\t")       # on enlève les caractères de fin de ligne
        l = s.split(" ")

        i=i+1
   
        p[0]=float(l[0])
        p[1]=float(l[1])
        p[2]=float(l[2])
    
        pts.append(np.array([p[0],p[1],p[2]]))
        
        
        ligne = filin.readline()
 
filin.close()
dst.close()   

s=[]
tirage=[]
for iter in range(ITER):
  
    a=np.int32(np.random.uniform(0,len(pts),3))
    p0=pts[a[0]]
    p1=pts[a[1]]
    p2=pts[a[2]]

    tirage.append([p0,p1,p2])

    p=(p0+p1+p2)/3
    p=p/np.linalg.norm(p)
    
    dist=np.abs(np.dot(pts,p))
    
    s.append(np.sum(dist<distance))
    
indice=np.argmax(s)

print(indice,s[indice])

p0=tirage[indice][0]
p1=tirage[indice][1]
p2=tirage[indice][2]
  

p=(p0+p1+p2)
p=p/np.linalg.norm(p)

    
dist=np.abs(np.dot(pts,p))

inliers=(dist<=distance) 
outliers1=(np.dot(pts,p)+d<-distance)
outliers2=(np.dot(pts,p)+d>distance)
fichier = open("data.asc", "w")
    
for i in range(len(pts)):
    aa=str(pts[i][0])
    fichier.write(aa)
    fichier.write('\t')
    aa=str(pts[i][1])
    fichier.write(aa)
    fichier.write('\t')
    aa=str(pts[i][2])
    fichier.write(aa)
    fichier.write('\t')
    if inliers[i]==True:
        couleur="255 255 255"
    if outliers1[i]==True:
        couleur="0 255 0"
    if outliers2[i]==True:
        couleur="255 0 0"
    fichier.write(couleur)
    fichier.write("\n")
fichier.close()
