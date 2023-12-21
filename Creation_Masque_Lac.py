# Grégoire Doat
# Raphaël Barateau
# Malik Masri

import numpy as np
from skimage import io, filters
import skimage.morphology as mm
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")

path2img="..\\depot\\donnee-S2A_red\\Data-avec-0%--nuage\\2015\\02-08\\T31TDL_20150802T104026_B8A_20m.jp2"
B8A = io.imread(path2img)

plt.imshow(B8A)
plt.show()


    # Séparation en deux couleurs par méthode OTSU

thresh = filters.threshold_otsu(B8A)
mask = B8A > thresh

plt.imshow(mask)
plt.show()


    # Nettoyage des artéfactes avec de la morpho math

selem=mm.square(2)

mask=mm.closing(mask,selem)
mask=mm.opening(mask,selem)

plt.imshow(mask)
plt.show()


    # Tronquage de l'image

mask=mask[30:86,23:73]
mask=1-mask.astype(int)

plt.imshow(mask)
plt.show()

    # Suppression des derniers pixels génants

mask[10:22,0:12]=0
print(mask.shape)
plt.imshow(mask)
plt.show()

    #Sauvegarde
path2save=""
plt.imsave(path2save+"mask0.gif",mask)
#np.save(path2save+"maskss.npy",mask)