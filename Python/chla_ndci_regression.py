# Yannis LEBRUN

# Import des bibliothèques
import matplotlib
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
import skimage.morphology as mm
import matplotlib.colors as mcolors

# Utilisation d'un backend pour une meilleure visualisation
matplotlib.use('TkAgg')

# Ce code cherche à établir une nouvelle formule de régression à partir du NDCI pour retrouver un résultat proche du MPH calculé par Deguene.
# On prend une image bien nuancée en chlorophylle pour favoriser une bonne régression.

# Chargement des données
B8A = 'donnee-S2A_red/Data-avec-0%--nuage/2018/30-06/T31TDL_20180630T105031_B8A_20m.jp2'
image8 = io.imread(B8A)

B04 = 'donnee-S2A_red/Data-avec-0%--nuage/2018/30-06/T31TDL_20180630T105031_B04_20m.jp2'
image4 = io.imread(B04)

B05 = 'donnee-S2A_red/Data-avec-0%--nuage/2018/30-06/T31TDL_20180630T105031_B05_20m.jp2'
image5 = io.imread(B05)

# Calcul des réflectances
B8A = image8 * 1 / 10000.0
B5 = image5 * 1 / 10000.0
B4 = image4 * 1 / 10000.0

# Chargement du masque crée par Deguene
mask = np.load("Code-Python/Maskaydat_maskedarray.npy")
# Erosion du masque pour supprimer les défauts dans la zone background
mask = np.where(mask, False, 1)
selem = mm.disk(4)
mask_open = mm.erosion(mask, selem)

# Redimmension diu masque pour pouvoir l'appliquer à notre image
mask_redim = transform.resize(mask_open, (image8.shape[0], image8.shape[1]), anti_aliasing=True)

# Masquage des bandes
B8A = np.where(mask_redim, B8A, np.nan)
B5 = np.where(mask_redim, B5, np.nan)
B4 = np.where(mask_redim, B4, np.nan)

# Définition de longueurs d'onde lambda pour les bandes
lambda_b4 = 0.665
lambda_b5 = 0.705
lambda_b8A = 0.865

# Calcul avec le MPH
MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)
Chla_mph = 1726.50 * MPH + 18.29

# Calcul du NDCI
NDCI = (B5 - B4)/(B5 + B4)
# Formule de régression de la concentration de chlorophylle-a obtenue en ligne
Chla_ndci = 14.039 + 86.115 * NDCI + 194.325 * NDCI**2

# Affichage des concentrations calculées selon les deux indices pour comparer
# On oberve une différence, et on aimerait s'approcher du MPH
cmap = plt.get_cmap('jet')
norm = mcolors.Normalize(vmin=0, vmax=100)
cmap.set_over('fuchsia')

fig, ax = plt.subplots(1, 2)
im1 = ax[0].imshow(Chla_mph, cmap=cmap, norm=norm)
cbar = plt.colorbar(im1, extend='max')
ax[0].set_title('Chla MPH')
im2 = ax[1].imshow(Chla_ndci, cmap=cmap, norm=norm)
cbar = plt.colorbar(im2, extend='max')
ax[1].set_title('Chla NDCI')
plt.show()

# Régression, on veut trouver une formule Chla_mph = a * NDCI**2 + b * NDCI + c

# Supprimer les valeurs NaN
nan_indices = np.isnan(Chla_mph)
Chla_mph = np.ravel(Chla_mph[~nan_indices]) # aplatir l'array

# Supprimer les valeurs NaN
nan_indices = np.isnan(NDCI)
NDCI = np.ravel(NDCI[~nan_indices]) # aplatir l'array

# Régression quadratique
X = np.empty((len(NDCI), 3))
X[:, 0] = np.ones((len(NDCI)))
X[:, 1] = NDCI[:]
X[:, 2] = NDCI[:]**2

# Détermination des coefficients polynomiaux
b = np.linalg.pinv(X) @ Chla_mph

# Affichage de la régression (nuages de points)
NDCI_range = np.linspace(np.min(NDCI), np.max(NDCI), 100)
Chla_predite = b[0] + b[1] * NDCI_range + b[2] * NDCI_range**2

plt.figure()
plt.scatter(NDCI, Chla_mph, label='Data Points', color='blue')
plt.plot(NDCI_range, Chla_predite, label='Régression quadratique', color='red')
plt.xlabel('NDCI')
plt.ylabel('Chla prédite')
plt.title('Régression quadratique')
plt.legend()
plt.show()
#%% Calcul de la nouvelle concentration de Chlorophylle

# Redéfiniton car arrays aplatis précédemment
NDCI = (B5 - B4)/(B5 + B4)
Chla_ndci_predite = b[0] + b[1] * NDCI + b[2] * NDCI**2

MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)
Chla_mph = 1726.50 * MPH + 18.29

# Affichage du résultat (comparaison Chla_mph et Chla_ndci)
cmap = plt.get_cmap('jet')
norm = mcolors.Normalize(vmin=0, vmax=100)
cmap.set_over('fuchsia')

fig, ax = plt.subplots(1, 2)
im1 = ax[0].imshow(Chla_mph, cmap=cmap, norm=norm)
cbar = plt.colorbar(im1, extend='max')
ax[0].set_title('Chla MPH')
im2 = ax[1].imshow(Chla_ndci_predite, cmap=cmap, norm=norm)
cbar = plt.colorbar(im2, extend='max')
ax[1].set_title('Chla NDCI predite')
plt.show()

# Affichage des coefficients polynomiaux
print(f'La formule est : Chla_ndci = {b[2]} NDCI**2 + {b[1]} NDCI + {b[0]}')