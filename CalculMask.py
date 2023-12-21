# Grégoire Doat
# Raphaël Barateau
# Malik Masri

import os
from skimage import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


def process_files(directory, extension='.jp2'):
    """
    Permet de parcourir tout les dossiers et de les mettres dans un dictionnaire, associer avec leur nom.

    :param directory: Le repertoire des images (Parcours automatique des dossiers)
    :param extension: Le format des images (par defaut jp2)
    :return: Dictionnaire des images associer au noms de ceux ci
    """
        # Création d'un dictionnaire vide pour stocker les données d'image
    image_dict = {}

        # Parcours récursif des fichiers et répertoires dans le répertoire spécifié
    for root, dirs, files in os.walk(directory):
        for file in files:
            if extension != 'npy':
                if file.endswith(extension):
                    image_dict[file] = (np.load(root + "\\" + file))
            else:
                if file.endswith(extension):
                    image_dict[file] = (io.imread(root + "\\" + file))
    return image_dict  # Dictionnaire des images


def NettoyageImg(img, mask, fullinfo=False):
    """

    :param img: L'image que l'ont veut traiter
    :param mask: Masque lu, d'une taille independante (plus petite)
    :param fullinfo: Si True retourner tout les informations le décalage, le masque translatés, et l'image masquée
    :return: L'image masquée (si fullinfo=False)
    """

    # Mise a la taille du filtre

        # recuperation des tailles
    h, w = img.shape
    n, p = mask.shape

        # ajustement du masque a l'image
    mask = np.concatenate((mask, np.zeros((h - n, p))), axis=0) # (h,p)
    mask = np.concatenate((mask, np.zeros((h, w - p))), axis=1) # (h,w)

    # Obtention de la translation par Fourier

        # passage dans Fourier
    imgFT = np.fft.fft2(img)
    maskFT = np.fft.fft2(mask)

        # Cross Power Spectrum et Fourier inverse
    Cross = (imgFT * maskFT.conjugate()) / np.abs(imgFT * maskFT.conjugate())
    invCross = np.fft.ifft2(Cross)

    # Recuperation du shift

        # translation (shift) pour placer le masque au bonne endroit
    shift = np.unravel_index(np.argmax(np.abs(invCross)), invCross.shape)
        # masque après translation
    translated = np.roll(mask, shift=shift, axis=(0, 1))
        # application du masque a l'image
    clear = img * translated

    # Returns en fonction des besoins

    if fullinfo == False:
        return clear
    else:
        return shift, translated, clear


def GetFiltre(directory,mask,path=None):
    """
    :param directory: Le repertoire des images
    :param mask: Masque lu d'une taille independante (plus petite)
    :param path: chemein ou enregister les images si besoin
    :return: Retourne un dictionnaire avec toutes les images masquees

    /!\ La sauvegarde des images est specifique au nom des images fournis,
       si le format est différent le code fonctionne mais donnera pas le resultat attendu
    """

    Dicimage = process_files(directory)                          # recupere le dictionnaire avec toutes les images
    Dic_mask = {}                                                # Dictionnaire final
    for nom_fichier,img in Dicimage.items():                     # iteration sur les cles et les images associees aux cles
        if 'B8A' in nom_fichier:                                 # check si dans le nom il y a B8A
            _, trans, _ = NettoyageImg(img, mask, fullinfo=True) # calcule de decalage du masque et recuperation le masque decale
            Dic_mask[nom_fichier[:22]]=trans


    # Si le chemain path est precise, sauvegarde des images en gif/npy dans le path si indique,
    #   Nom du fichier cree : "nom_fichier.gif/.npy"

    if path != None:
        for nom_fichier,img in Dic_mask.items(): # Parcours du dictionnaire des images masquées

        # ajoute de l'annee au chemin de sauvegarde
            savepath=path+'\\'+nom_fichier[7:11]

        # creation du dossier annee si besoin
            try:
                os.mkdir(savepath)
                print("dir made")
            except :
                print("deja existant")

        # ajoute de la date mois/jour au chemin de sauvegarde
            savepath+="\\"+nom_fichier[11:13]+"-"+nom_fichier[13:15]

        # creation du sous-dossier mois-jour si besoin
            try:
                os.mkdir(savepath)
            except:
                print("deja existant")

            plt.imsave( savepath + "\\" + nom_fichier + ".gif",img ) #sauvegarde en gif
            #np.save( savepath + "\\" + nom_fichier + ".npy",img ) #sauvegarde en npy si besoin
    return Dic_mask


def ComparImg(Images, mask, subtitles=None, title=None):
    """
    Fonction qui permet de comparer les resultats des calculs de masques pour les images Images

    :param Images: Liste des images (supposees tous de meme tailles)
    :param mask: Masque lu
    :param subtitles: Liste de titres associe a chaques images de Images
    :param title: Pour avoir le titre générale
    :return: Affiche le plot des images seuls, avec le masques en transparence dessus et avec le masques applique
    """

    # Masque et Creation des plots

    h, w = Images[0].shape      # on suppose que toutes les images ont la même taille
    n, p = mask.shape           # taille du masque

    # Creation d'un masque

    mask = np.concatenate((mask, np.zeros((h - n, p))), axis=0) #on redimensionne le masque de la taille de notre image
    mask = np.concatenate((mask, np.zeros((h, w - p))), axis=1) #Idem
    mask = np.roll(mask, (n // 2, p // 2), axis=(0, 1))

    Pinky = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8) #??
    Pinky[mask == 1] = [255, 105, 180, 255] #??

    fig, ax = plt.subplots(3, len(Images))

        # Boucle sur les images

    for i, img in enumerate(Images):

            # recuperation du masque et de l'image masque
        _, mask, cleaned = NettoyageImg(img, mask, fullinfo=True)

            # creation d'un masque rose en transparance
        Pinky = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        Pinky[mask == 1] = [255, 105, 180, 255]

        # Affichage

            # image de base
        ax[0, i].imshow(img)

            # image avec masque par dessus
        ax[1, i].imshow(img)
        ax[1, i].imshow(Pinky, alpha=0.5)

            # image masquee
        ax[2, i].imshow(cleaned)

        # Les titres, si besoin

        if subtitles != None:
            ax[0, i].set_title(subtitles[i])
    if title != None:
        plt.suptitle(title)
    plt.show()



    # Application des fonctions :


if __name__ == "__main__":

        # Recuperation du masque

    path2mask = ""
    mask = np.load(path2mask+"maskss.npy")

        # Recuperation d'images pour la demo

    path2imgs = 'C:/Users/mberthie/Documents/Math_entreprise/donnee-S2A_red'
    imgNames=['/Data-avec-0%--nuage/2015/03-12/T31TDL_20151203T105422_B8A_20m.jp2',
              '/Data-minimum-30%-nuage/2023/11-07/T31TDL_20230711T103631_B8A_20m.jp2',
              '/data100%-nuage/2023/25-08/T31TDL_20230825T103629_B8A_20m.jp2',
              '/data100%-nuage/2023/28-08/T31TDL_20230828T104629_B8A_20m.jp2']
    Images=[plt.imread(path2imgs+name) for name in imgNames]

        # Plots des images avec leur filtre

    subts=['Decouvert', 'Tres peu couvert','Peu couvert', 'Couvert']
    ComparImg(Images, mask, subtitles=subts, title=None)

    path2save='Figures'

        # Telechargement de toutes les images
    # GetFiltre(directory, mask, saveas='masque', path='Figures\\Masques')