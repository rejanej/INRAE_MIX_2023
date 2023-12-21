# Océanne Bousquet

# Ce code permet de visualiser les différentes concentrations et absorption
# en fonction des différents indices de réflectances (MPH/NDCI/B4/OC2V4)

import matplotlib
import matplotlib.colors as mcolors
from CalculMask import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from skimage import transform
import skimage.morphology as mm
matplotlib.use('QtAgg')

# Chargement des bandes de l'image
dictionnaire = process_files('donnee-S2A_red/Data-avec-0%--nuage/2018/20-10/')
for clé,image in dictionnaire.items():
    if 'B8A' in clé:
        image8A = image
        nom = clé[7:11] +'-'+ clé[11:13] +'-'+ clé[13:15]
    if 'B05' in clé:
        image5 = image
    if 'B04' in clé:
        image4 = image

# Récupération du masque définissant le lac
# Masque que l'on a créé
mask = np.load("C:\\Users\\ocean\\Documents\\M1_MIX\\Maths-Entrep\\mask100x100.npy")

# Masque de Deguene
'''mask = np.load("C:\\Users\\ocean\\Documents\\M1_MIX\\Maths-Entrep\\Code-Python\\Maskaydat_maskedarray.npy")
mask = np.where(mask, False, 1)
selem = mm.disk(4)
mask_open = mm.erosion(mask, selem)
mask = transform.resize(mask_open, (image8A.shape[0], image8A.shape[1]), anti_aliasing=True)'''

# Calcul de la réflectance
B8A = image8A * 1 / 10000.0
B5 = image5 * 1 / 10000.0
B4 = image4 * 1 / 10000.0

# Appliquer le masque délimitant le lac Aydat aux images
B8A = np.where(mask, B8A, np.nan)
B5 = np.where(mask, B5, np.nan)
B4 = np.where(mask, B4, np.nan)

# Définition de longueurs d'onde lambda pour les bandes
lambda_b4 = 0.665
lambda_b5 = 0.705
lambda_b8A = 0.865

# Calcul de l'indice MPH (Maximum peak-height)
MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)

# Calcul de Chl-a (Chlorophylle-a concentration) en fonction de MPH
#chla_mph = 2223.18 * MPH + 24.03
chla_mph = 1726.50 * MPH + 18.29

# Calcul de l'indice NDCI (Normalized Difference Chlorophyll Index)
NDCI = (B5 - B4)/(B5 + B4)

# Calcul de Chl-a en fonction de NDCI_Yannis
chla_ndci_Y = 22.747 + 86.157 * NDCI + 205.681 * NDCI**2

# Calcul de l'indice d'absorption de la bande B4 (rouge:665 nm)
B_4 = (1/ B4) * 0.005

# Calcul de Chl-a en fonction de la bande B4
chla_B4 = np.exp((np.log(0.017)-np.log(B4))/0.089)

# Calcul de l'indice OC2V4 (Ocean Chlorophyll 2-band)
OC2V4 = np.log(B5 / B4)

# Calcul de Chl-a en fonction de l'indice OC2V4
chla_oc2v4 = -34.815 * OC2V4**2 + 79.363 * OC2V4 + 20.066

# Afficher les index et les concentrations/absorption de Chl-a
fig, ax = plt.subplots(2,4,figsize=(12, 8))

fig.suptitle('Comparaison des différents indices et de leurs concentrations/absorption de chlorophylle associée : ' +nom,fontsize='x-large')

norm = mcolors.Normalize(vmin=0, vmax=100)
norm1 = mcolors.Normalize(vmin=0, vmax=0.6)
norm2 = mcolors.Normalize(vmin=0, vmax=0.8)
norm3 = mcolors.Normalize(vmin=0, vmax=0.08)

ax[0,0].set_title('Indice_MPH')
im = ax[0,0].imshow(MPH,cmap='jet')
plt.colorbar(im, extend='max', shrink=0.5)
ax[0,0].axis('off')

ax[0,1].set_title("Indice_NDCI")
im1 = ax[0,1].imshow(NDCI,cmap='jet',norm=norm1)
plt.colorbar(im1, extend='max', shrink=0.5)
ax[0,1].axis('off')

ax[0,2].set_title("Indice_B4")
im2 = ax[0,2].imshow(B_4, cmap='jet',norm=norm1)
plt.colorbar(im2, extend='max', shrink=0.5)
ax[0,2].axis('off')

ax[0,3].set_title('Indice_OC2V4')
im6 = ax[0,3].imshow(OC2V4,cmap='jet',norm=norm2)
plt.colorbar(im6, extend='max', shrink=0.5)
ax[0,3].axis('off')

ax[1,0].set_title("Concentration Chl-A_MPH")
im3 = ax[1,0].imshow(chla_mph,cmap='jet',norm=norm)
plt.colorbar(im3, extend='max', shrink=0.5)
ax[1,0].axis('off')

ax[1,1].set_title("Concentration Chl-A_NDCI")
im4 = ax[1,1].imshow(chla_ndci_Y,cmap='jet',norm=norm)
plt.colorbar(im4, extend='max', shrink=0.5)
ax[1,1].axis('off')

ax[1,2].set_title("Absorption Chl-A_B4")
im5 = ax[1,2].imshow(chla_B4,cmap='jet',norm=norm3)
plt.colorbar(im5, extend='max', shrink=0.5)
ax[1,2].axis('off')

ax[1,3].set_title('Concentration Chl-A_OC2V4')
im6 = ax[1,3].imshow(chla_oc2v4,cmap='jet',norm=norm)
plt.colorbar(im6, extend='max', shrink=0.5)
ax[1,3].axis('off')

plt.show()

#%% Calcul de la nouvelle concentration de Chlorophylle à partir de l'indice OC2V4

# Régression, on veut trouver une formule Chla_mph = a * OC2V4**2 + b * OC2V4 + c

# Supprimer les valeurs NaN
nan_indices = np.isnan(chla_mph)
Chla_mph = np.ravel(chla_mph[~nan_indices]) # aplatir l'array

# Supprimer les valeurs NaN
nan_indices = np.isnan(OC2V4)
OC2V4 = np.ravel(OC2V4[~nan_indices]) # aplatir l'array

# Régression quadratique
X = np.empty((len(OC2V4), 3))
X[:, 0] = np.ones((len(OC2V4)))
X[:, 1] = OC2V4[:]
X[:, 2] = OC2V4[:]**2

# Détermination des coefficients polynomiaux
b = np.linalg.pinv(X) @ Chla_mph

# Affichage de la régression (nuages de points)
OC2V4_range = np.linspace(np.min(OC2V4), np.max(OC2V4), 100)
Chla_predite = b[0] + b[1] * OC2V4_range + b[2] * OC2V4_range**2

plt.figure()
plt.scatter(OC2V4, Chla_mph, label='Data Points', color='blue')
plt.plot(OC2V4_range, Chla_predite, label='Régression quadratique', color='red')
plt.xlabel('OC2V4')
plt.ylabel('Chla prédite')
plt.title('Régression quadratique')
plt.legend()
plt.show()

# Affichage des coefficients polynomiaux
print(f'La formule est : Chla_OC2V4 = {b[2]} OC2V4**2 + {b[1]} OC2V4 + {b[0]}')

MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)
Chla_mph = 1726.50 * MPH + 18.29
OC2V4 = np.log(B5 / B4)

# Redéfiniton car arrays aplatis précédemment
Chla_oc2v4_predite = b[0] + b[1] * OC2V4 + b[2] * OC2V4**2

# Affichage du résultat (comparaison Chla_mph et Chla_oc2v4)
cmap = plt.get_cmap('jet')
norm = mcolors.Normalize(vmin=0, vmax=100)
cmap.set_over('fuchsia')

fig, ax = plt.subplots(1, 2)
im1 = ax[0].imshow(Chla_mph, cmap=cmap, norm=norm)
cbar = plt.colorbar(im1, extend='max')
ax[0].set_title('Chla_MPH')
im2 = ax[1].imshow(Chla_oc2v4_predite, cmap=cmap, norm=norm)
cbar = plt.colorbar(im2, extend='max')
ax[1].set_title('Chla_OC2V4_predite')
plt.show()