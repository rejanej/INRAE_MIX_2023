"""
Réjane Joyard - Semaine Math-Entreprise
Récupération des données de Chl-A à l'aide des images donnee-2A_red dans un fichier csv
La première partie applique le code sur une seule image et la deuxième partie peut être appliquer à un dossier
Ce code est adaptable selon les données qu'on veut récupérer et insérer dans le fichier csv
"""

# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage import io
import csv
from CalculMask import *

def calculate_chla_stats(image_dict):
    """
    Fonction permettant de récupérer la valeur max, la valeur min et la moyenne du taux
    de Chl-A sur la zone sélectionnée (zone de la bouée)

    :param image_dict: dictionnaire des images
    :return: liste des données à récupérer
    """

    # Création de la liste qui va contenir les données à enregistrer
    chla_stats_list = []
    compt = 0
    for file, image in image_dict.items():
        compt += 1
        # Boucles pour attribuer chaque image du dico à la bande qui correspond
        if 'B8A' in file:
            image8A = image
        if 'B04' in file :
            image4 = image
        if 'B05' in file:
            image5 = image
        if compt == 3:
            # Calcul de la réflectance
            B8A = image8A * 1 / 10000.0
            B5 = image5 * 1 / 10000.0
            B4 = image4 * 1 / 10000.0

            # Définition de longueurs d'onde lambda pour les bandes
            lambda_b4 = 0.665
            lambda_b5 = 0.705
            lambda_b8A = 0.865

            # Calcul du MPH
            MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)
            chla = 2223.18 * MPH + 24.03
            MPH_mask = np.where(image_mask, chla, 0)

            # Délimitation de la zone
            zone = MPH_mask[65:70, 40:47]

            # Calcul des données souhaitées
            valeur_max = np.max(zone)
            valeur_min = np.min(zone)
            moyenne = np.mean(zone)

            #Ajout à la liste les données
            chla_stats_list.append({
                'File': file,
                'Concentration max en Chl-A' : valeur_max,
                'Concentration min en Chl-A' : valeur_min,
                'Moyenne de la concentration en Chla' : moyenne
            })
            compt = 0
    return chla_stats_list

def save_to_csv(data, csv_filename='chl_a_stats.csv'):
    """
    Fonction permettant d'enregistrer les données récupérer dans la fonction précédente dans un fichier csv

    :param data: dictionnaire récupérer
    :param csv_filename: nom du fichier enregistré
    """
    with open(csv_filename, 'w', newline='') as csvfile:
        # Création des différentes colonnes que l'on veut ajouter
        fieldnames = ['File', 'Concentration max en Chl-A', 'Concentration min en Chl-A', 'Moyenne de la concentration en Chla']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    # Partie 1 : application de l'algorithme pour une seule image
    # Lecture des 3 bandes
    B8A = 'donnee-S2A_red/Data-minimum-30%-nuage/2018/27-07/T31TDL_20180727T104021_B8A_20m.jp2'
    image8A = io.imread(B8A)
    B04 = 'donnee-S2A_red/Data-minimum-30%-nuage/2018/27-07/T31TDL_20180727T104021_B04_20m.jp2'
    image4 = io.imread(B04)
    B05 = 'donnee-S2A_red/Data-minimum-30%-nuage/2018/27-07/T31TDL_20180727T104021_B05_20m.jp2'
    image5 = io.imread(B05)

    # Somme des bandes pour prendre en compte les 3 bandes
    image_sum = image8A + image5 + image4
    plt.imshow(image_sum)
    plt.show()

    # Récupération du masque définissant le lac
    mask = np.load("./mask100x100.npy")

    # Application du masque à l'image
    image_mask = np.where(mask, image_sum, 0)

    # Calcul de la réflectance
    B8A = image8A * 1 / 10000.0
    B5 = image5 * 1 / 10000.0
    B4 = image4 * 1 / 10000.0

    # Définition de longueurs d'onde lambda pour les bandes
    lambda_b4 = 0.665
    lambda_b5 = 0.705
    lambda_b8A = 0.865

    # Calcul du MPH, de la Chl-A en fonction de ce MPH puis application du taux récupérer sur l'image masquée
    MPH = B5 - B4 - ((B8A - B4) * (lambda_b5 - lambda_b4)) / (lambda_b8A - lambda_b4)
    Chla = 2223.18 * MPH + 24.03
    MPH_mask = np.where(image_mask, Chla, 0)

    # Calcul OC2V4 pour obtenir la luminosité
    OC2V4 = np.log(B5 / B4) * 244 + 201.33
    OC2V4_mask = np.where(image_mask, OC2V4, 0)

    # Affichage des concentrations
    plt.figure(figsize=(11, 5))
    cmap = plt.get_cmap('jet')
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap.set_over('fuchsia')

    # Afficher la zone qui contient la bouée
    zone = MPH_mask[65:70, 40:47]
    plt.title("Chlorophylle masquée (avec MPH)")
    im = plt.imshow(zone, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, extend='max', shrink=0.5)
    cbar.set_label('Chla (μg/L)')
    plt.tight_layout()
    plt.show()

    # Partie 2 : application de l'algorithme pour toutes les images
    # Application sur notre jeu de données
    image_dict = process_files('./donnee-S2A_red/')
    chla_stats_list = calculate_chla_stats(image_dict)
    save_to_csv(chla_stats_list)
