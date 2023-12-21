# Yannis LEBRUN

# Import des bibliothèques
import matplotlib
import numpy as np
from CalculMask import *
import Masque_nuages_Calculs_ChlA

# Utilisation d'un backend pour une meilleure visualisation
matplotlib.use('TkAgg')

# Ce code parcourt toutes les images fournies (jp2) pour supprimer les nuages
# puis calculer la concentration de chlorophylle à l'aide de l'index NDCI établi par régression
# On se servira de ces images pour appliquer la bathymétrie par-dessus dans le code Bathymetrie.py

dictionnaire = process_files('donnee-S2A_red/')
compt = 0

for clé,image in dictionnaire.items():
    # Le compteur sert à parcourir les images trois par trois
    compt += 1
    if 'B8A' in clé:
        image8A = image
        nom = clé[7:11] +'-'+ clé[11:13] +'-'+ clé[13:15] # Récupération du nom de l'image
    if 'B05' in clé:
        image5 = image
    if 'B04' in clé:
        image4 = image
    if compt == 3:
        # Somme des bandes pour prendre en compte les 3 bandes
        image_sum = image8A + image5 + image4

        # Chargement du masque que l'on a créé
        mask = np.load("Code-Python/mask100x100.npy")
        # Appliquer le masque à l'image
        image_mask = np.where(mask, image_sum, 0)

        # Récupération du seuillage
        seuil, mask_sans_nuage = Masque_nuages_Calculs_ChlA.seuil(image_mask)

        # Calcul de la réflectance
        B8A = image8A * 1 / 10000.0
        B5 = image5 * 1 / 10000.0
        B4 = image4 * 1 / 10000.0

        # Définition de longueurs d'onde lambda pour les bandes
        lambda_b4 = 0.665
        lambda_b5 = 0.705
        lambda_b8A = 0.865

        # Calcul de l'incice du Maximum Peak Height (MPH)
        MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)

        # Calcul du Normalized Difference Chlorophyll Index (NDCI)
        NDCI = (B5 - B4) / (B5 + B4)

        # Calcul de la concentration de Chlorophylle-a avec le NDCI obtenu par régression
        Chla_ndci_predite = 205.6982 * NDCI ** 2 + 86.1573 * NDCI + 22.7467
        # Appliquer le masque sans nuage
        Chla_ndci_predite = np.where(mask_sans_nuage, Chla_ndci_predite, 0)

        # Enregistrement des images dans le répertoire "bathymetrie_npy"
        # Enregistrement au format npy pour éviter une compression
        np.save(f'bathymetrie_npy/{nom}.npy', Chla_ndci_predite)

        # Remise à zéro du compteur pour la boucle suivante
        compt = 0