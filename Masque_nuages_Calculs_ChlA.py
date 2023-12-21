# Océanne Bousquet
# Réjane Joyard
# Yannis Lebrun

# Ce code utilise le masque que l'on a créé pour délimiter les contours du lac Aydat
# On crée une fonction pour pouvoir obtenir un masque sans nuages pour chaque image choisie
# On calcule ensuite la concentration de chlorophylle à l'aide de différents indices (MPH/NDCI)

import matplotlib
import numpy as np
from skimage import transform
import matplotlib.colors as mcolors
from skimage.filters import threshold_otsu
from CalculMask import *
import matplotlib.pyplot as plt
import skimage.morphology as mm
from skimage import exposure
matplotlib.use('QtAgg')

# Récupérer le seuil de l'histogramme à appliquer, pour réduire les nuages
def seuil(image):
    """
    :param
            image (ndarray): image correspondant à la somme des bandes B04, B05, B8A
    :return
            seuil (int): seuillage pour le troncage de l'histogramme
            mask_sans_nuage (ndarray): masque de l'image sans les nuages
    """
    # Calculer l'histogramme
    hist, bins_center = exposure.histogram(image)
    # Enlever la première valeur (0) de l'histogramme qui correspond au background
    hist = hist[1:]
    bins_center = bins_center[1:]

    # Disjonction de cas
    sup = len(hist)
    if sup > 10000:
        # Présence de nuages
        max = np.argmax(hist)
        if max > sup / 2:
            # Seulement des nuages
            seuil = np.nonzero(hist)[0][0]  # Première valeur non-nulle
            print(f'Seulement des nuages : {seuil}')
        else:
            # Nuage et lac : histogramme bimodal
            seuil = threshold_otsu(image) # Application de la méthode d'Otsu (seuillage automatique)
            if seuil <10:
                # Cas où les nuages sont trop fins pour être détectés par Otsu qui seuille à zéro
                seuil = sup # Garder tous les nuages car ils sont négligeables
            print(f'Nuages et lac visibles : {seuil}')
    else:
        # Seulement le lac Aydat
        seuil = sup # Garder toute l'image : pas de troncage
        print(f'Uniquement lac : {seuil}')

    plt.figure(figsize=(15,8))
    # Afficher l'histogramme tronqué de l'image masquée
    plt.subplot(1, 2, 1)
    plt.plot(bins_center, hist, lw=2)
    plt.title('Histogramme de image masquée')
    plt.xlabel('Intensité des pixels')
    plt.ylabel('Frequences')
    plt.axvline(seuil, color='r') # Ligne représentant le seuillage

    # Appliquer le masque sans nuages à notre image : garder l'image sans le backgroung intersectée avec l'image tronquée
    image_ss_nuage = np.where((image > 0) & (image <= seuil + 1),image,0)
    # Afficher l'image sans nuage
    plt.subplot(1, 2, 2)
    plt.imshow(image_ss_nuage)
    plt.title('Image sans nuages : 25/08/2023')
    plt.colorbar()
    plt.show()

    # Faire un masque avec l'image sans nuages
    mask_sans_nuage = np.where(image_ss_nuage == 0, 0, 1)

    return seuil, mask_sans_nuage

if __name__ == "__main__":

    # Chargement des bandes de l'image
    dictionnaire = process_files('donnee-S2A_red/data100%-nuage/2023/25-08/')
    for clé,image in dictionnaire.items():
        if 'B8A' in clé:
            image8A = image
            nom = clé[7:11] +'-'+ clé[11:13] +'-'+ clé[13:15]
        if 'B05' in clé:
            image5 = image
        if 'B04' in clé:
            image4 = image

    # Afficher les bandes B04,B05,B8A
    fig, ax = plt.subplots(1, 3,figsize=(15,8))
    fig.suptitle('Image du ' + nom)
    ax[0].imshow(image8A)
    ax[0].set_title('B8A')
    ax[1].imshow(image5)
    ax[1].set_title('B05')
    ax[2].imshow(image4)
    ax[2].set_title('B04')
    plt.show()

    # Récupération du masque définissant le lac
    # Masque que l'on a créé
    mask = np.load("C:\\Users\\ocean\\Documents\\M1_MIX\\Maths-Entrep\\mask100x100.npy")

    # Masque créé par Deguene
    '''mask = np.load("C:\\Users\\ocean\\Documents\\M1_MIX\\Maths-Entrep\\Code-Python\\Maskaydat_maskedarray.npy")
    # Érosion du masque pour supprimer les défauts dans la zone background
    mask = np.where(mask, False, 1)
    selem = mm.disk(4)
    mask_open = mm.erosion(mask, selem)
    # Redimensionner le masque aux images
    mask = transform.resize(mask_open, (image8A.shape[0], image8A.shape[1]), anti_aliasing=True)'''

    # Somme des bandes pour prendre en compte les 3 bandes dans la suppression des nuages
    image_sum = image8A + image5 + image4

    # Appliquer le masque délimitant le lac aux bandes sommées
    image_mask = np.where(mask, image_sum, 0)

    # Récupérer masque sans nuages
    seuil, mask_sans_nuage = seuil(image_mask)

    # Calcul de la réflectance
    B8A = image8A * 1 / 10000.0
    B5 = image5 * 1 / 10000.0
    B4 = image4 * 1 / 10000.0

    # Appliquer le masque délimitant le lac aux bandes
    B8A = np.where(mask, B8A, np.nan)
    B5 = np.where(mask, B5, np.nan)
    B4 = np.where(mask, B4, np.nan)

    # Définition de longueurs d'onde lambda pour les bandes
    lambda_b4 = 0.665
    lambda_b5 = 0.705
    lambda_b8A = 0.865

    # Calcul de MPH (Maximum peak-height)
    MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)

    # Calcul du NDCI (Normalized Difference Chlorophyll Index)
    NDCI = (B5 - B4)/(B5 + B4)

    # Calcul de la somme des index
    Somme = (MPH + NDCI)/2

    # Afficher les index
    fig, ax = plt.subplots(1,3,figsize=(15,8))
    fig.suptitle('Image du ' + nom)
    ax[0].set_title("Index_MPH")
    ax[0].imshow(MPH)
    ax[1].set_title("Index_NDCI")
    ax[1].imshow(NDCI)
    ax[2].set_title("Somme_index")
    ax[2].imshow(Somme)

    # Calcul de Chl-a (concentration Chlorophylle-a) en fonction de MPH
    #chla_mph = 2223.18 * MPH + 24.03
    chla_mph = 1726.50 * MPH + 18.29

    # Calcul de Chl-a en fonction du NDCI
    chla_ndci = 14.039 + 86.115 * NDCI + 194.325 * NDCI**2

    # Calcul de Chl-a par régression quadratique en fonction du NDCI et de la concentration de Chl-a à partir du MPH
    chla_ndci_Yannis = 22.747 + 86.157 * NDCI + 205.681 * NDCI**2

    # Appliquer le masque sans nuages
    chla_mph_mask = np.where(mask_sans_nuage, chla_mph, np.nan)
    chla_ndci_mask = np.where(mask_sans_nuage, chla_ndci, np.nan)
    chla_ndci_Y_mask = np.where(mask_sans_nuage, chla_ndci_Yannis, np.nan)

    # Affichage des concentrations de chlorophylle avec les différents indices
    fig = plt.figure(figsize=(15, 8))
    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap.set_over('fuchsia')
    fig.suptitle('Image du ' + nom)

    # MPH
    plt.subplot(1, 3, 1)
    plt.title("Concentration de la chlorophylle (MPH)")
    im = plt.imshow(chla_mph_mask, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, extend='max', shrink=0.5)
    cbar.set_label('Chla (μg/L)')

    # NDCI
    plt.subplot(1, 3, 2)
    plt.title('Concentration de la chlorophylle (NDCI)')
    im2 = plt.imshow(chla_ndci_mask, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im2, extend='max', shrink=0.5)
    cbar.set_label('Chla (μg/L)')

    # NDCI établi par régression
    plt.subplot(1, 3, 3)
    plt.title('Concentration de la chlorophylle (NDCI_Yannis)')
    im2 = plt.imshow(chla_ndci_Y_mask, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im2, extend='max', shrink=0.5)
    cbar.set_label('Chla (μg/L)')

    plt.tight_layout()
    plt.show()

    #Afficher valeurs de concentrations de la Chl-a supérieures ou inférieures à 30 μg/L
    colors = ['deepskyblue', 'tomato']  # définit les couleurs pour la colormap.
    cmap = mcolors.ListedColormap(colors)  # crée une colormap à partir des couleurs définies.

    # Définir les limites des couleurs
    bounds = [0, 30]  # Inférieur à 30 sera en bleu, supérieur à 30 sera en rouge
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Créer le graphique
    plt.figure(figsize=(10,8))
    plt.imshow(chla_mph_mask, cmap=cmap, norm=norm)
    plt.title('Chlorophylle supérieure à 30 le ' + nom)

    # Ajouter une légende
    legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', markersize=10, label='>30'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='deepskyblue', markersize=10, label='<30')
            ]
    plt.legend(handles=legend_handles, loc='upper right')

    plt.show()

