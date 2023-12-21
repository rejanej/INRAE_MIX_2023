# Grégoire Doat
# Raphaël Barateau
# Malik Masri

import os
import numpy as np
from skimage import io,filters
import matplotlib
import matplotlib.pyplot as plt
from CalculMask import *
matplotlib.use("TkAgg")

#Création des 3 masques de bathymétries

    # Récupération du masque fait à la main
path2mask='Figures'
mask=plt.imread(path2save+'/bathyB&N.jpg')

    # Lissage du masque pour nettoyer les petits changement de valeur
bathy=np.zeros_like(mask)

bathy[30<mask]=80
bathy[100<mask]=150
bathy[160<mask]=255

    # Création des 3 masques pour les 3 niveaux de profondeur
bathyBord=np.zeros_like(mask)
bathyBord[bathy==80]=1

bathyMid=np.zeros_like(mask)
bathyMid[bathy==150]=1

bathyInter=np.zeros_like(mask)
bathyInter[bathy==255]=1

    # Plot des masques obtenues
fig, ax=plt.subplots(1,3)

ax[0].imshow(bathyBord)
ax[0].set_title('Masque pour le bord')

ax[1].imshow(bathyMid)
ax[1].set_title("Masque pour l'entre-deux")

ax[2].imshow(bathyInter)
ax[2].set_title("Masque pour l'interieur")
plt.show()

    # Sauvegarde des images masque au cas où
path2save='Figures/Masques/Bathy'
#np.save(path2save+'/BarthBordPetit',bathyBord)
#np.save(path2save+'/BarthMidPetit',bathyMid)
#np.save(path2save+'/BarthDepthPetit',bathyInter)


# Mise à la taille 100x100

    # Récuperation d'une image sans nuage
img=plt.imread('depot/donnee-S2A_red/Data-avec-0%--nuage/2015/02-08/T31TDL_20150802T104026_B8A_20m.jp2')

    # Calcul du shift et d'un masque simple 100x100
shift,mask,_=NettoyageImg(img, mask, fullinfo=True)

    # Placement des 3 masque bathymetriques

n,p=bathy.shape

MaskBord=np.zeros_like(img)
MaskBord[shift[0]:shift[0]+n,shift[1]:shift[1]+p]=bathyBord

MaskMid=np.zeros_like(img)
MaskMid[shift[0]:shift[0]+n,shift[1]:shift[1]+p]=bathyMid

MaskInter=np.zeros_like(img)
MaskInter[shift[0]:shift[0]+n,shift[1]:shift[1]+p]=bathyInter

    # Affichage des masques
fig, ax=plt.subplots(1,3)

ax[0].imshow(MaskBord)
ax[0].set_title('Masque pour le bord')

ax[1].imshow(MaskMid)
ax[1].set_title("Masque pour l'entre-deux")

ax[2].imshow(MaskInter)
ax[2].set_title("Masque pour l'interieur")
plt.show()

    # Sauvegarde des masques
path2save='C:/Users/gregd/Desktop/Geo/Boss(z)e/M1/MathEnt/Figures/Masques/Bathy'

plt.imsave(path2save+'/MasqueBathyBord.gif',MaskBord)
np.save(path2save+'/MasqueBathyBord.npy',MaskBord)

plt.imsave(path2save+'/MasqueBathyMid.gif',MaskMid)
np.save(path2save+'/MasqueBathyMid.npy',MaskMid)

plt.imsave(path2save+'/MasqueBathyDepth.gif',MaskInter)
np.save(path2save+'/MasqueBathyDepth.npy',MaskInter)
