#Raphaël

from matplotlib import pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
from CalculMask import *
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
import cv2
import tkinter as tk
from tkinter import *
matplotlib.use('TkAgg')



def application_bathy(image,mask,masque1,masque2,masque3,path_bathy,tout=True,bord=False,Mid=False,depth=False):
    """
    Application de la bathymétrie aux images masquées

    :param image: l'image traitée
    :param mask: le masque de base appliqué sur l'image
    :param masque1: le masque de la zone du milieu de la bathymétrie
    :param masque2: le masque de la zone du bord de la bathymétrie
    :param masque3: le masque de la zone de centre de la bathymétrie
    :param path_bathy: le chemin de l'image de la bathymétrie
    :param tout: True pour retourner l'image masquée avec la bathymétrie (par défaut True)
    :param bord: True pour retourner l'image masquée avec le masque bord (par défaut False)
    :param Mid:  True pour retourner l'image masquée avec le masque milieu (par défaut False)
    :param depth: True pour retourner l'image masquée avec le masque centre (par défaut False)
    :return: retourne l'image masquée correspondant au True !
    """


    #lecture de l'image bathy
    img_bathy = cv2.imread(path_bathy) #utilisation de CV2 uniquement pour la convertir en binaire

    #mise en binaire de la bathymetrie
    _ , img_bathy = cv2.threshold(img_bathy,127,255,cv2.THRESH_BINARY)
    img_bathy = img_bathy[:,:,0]/np.max(img_bathy[:,:,0]) # normalisation

#################################################################################################################################################################

    #recadrage de la bathymetrie ( fait à la mains et adaptée au filtre, à adapter si le filtre change de forme en fonction des images )
    img_bathy = img_bathy[0:789,9:805]
    #on recupere la taille de la bathy initiale
    h1, w1 = img_bathy.shape

    #calibrage manuelle
    img_bathy = np.concatenate((img_bathy, np.ones((10, w1))), axis=0)
    img_bathy = np.roll(img_bathy, 17, axis=0)
    img_bathy = np.roll(img_bathy,5,axis=1)


#################################################################################################################################################################

    #image en niveau de gris (Optionnelle)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Calcul du décalage et de l'image masquée
    shift2, _, cleaned2 = NettoyageImg(image, mask, fullinfo=True)  # img[-1] -> B8A

#################################################################################################################################################################

    #On coupe notre image 100 par 100, pour avoir uniquement une image de la taille du masque
    n,p = mask.shape     #on recupere la taille du masque
    cleanedPetit2 = cleaned2[shift2[0]:n + shift2[0], shift2[1]:p + shift2[1]]  # même taille que mask

#################################################################################################################################################################

    #On recupère la nouvelle taille de la bathymetrie
    h1, w1 = img_bathy.shape

    #On remet les images satellites masquées à la taille de la bathymetrie
    image_resized = resize(cleanedPetit2, (h1, w1), anti_aliasing=True, preserve_range=True)
    #On remet les masques à la taille de la bathymetrie
    masque1 = resize(masque1, (h1, w1), anti_aliasing=True, preserve_range=True)
    masque2 = resize(masque2, (h1, w1), anti_aliasing=True, preserve_range=True)
    masque3 = resize(masque3, (h1, w1), anti_aliasing=True, preserve_range=True)

    #Implémentation de la bathymetrie à l'image
    image_f=image_resized*img_bathy

    if tout == True:
        return image_f
    if Mid == True: #application du masque milieu sur la bathymetrie
        return image_f*masque1
    if bord == True: #application du masque bord sur la bathymetrie
        return image_f*masque2
    if depth == True: #application du masque centre sur la bathymetrie
        return image_f*masque3


#Affichage de l'interface graphique
class interface_graphique:
    def __init__(self,Liste_image,Liste_image_Depth,Liste_image_Bord,Liste_image_Mid,key):
        """
        :param Liste_image:Liste des images masquées avec la bathymetrie
        :param Liste_image_Depth: Liste des images masquées centre avec la bathymetrie
        :param Liste_image_Bord: Liste des images masquées bord la bathymetrie
        :param Liste_image_Mid: Liste des images masquées milieu la bathymetrie
        :param key: Liste des noms des images
        """
        self.fenetre = Tk()
        self.fenetre.geometry("500x800")
        self.indice = 0
        self.var = tk.BooleanVar()
        self.bouton_afficher_image = Button(self.fenetre, text="Afficher image "+"📷",bg='#96c0eb',activebackground='white',
                                       command=self.plot_image)

        self.bouton_afficher_image.pack(fill=X, ipady=10, padx=10,pady=10)

        self.bouton_afficher_image = Button(self.fenetre, text="Afficher la Bathymetrie Bord" + "📷", bg='#96c0eb',
                                            activebackground='white',
                                            command=self.plot_image_Bord)

        self.bouton_afficher_image.pack(fill=X, ipady=10, padx=10, pady=10)

        self.bouton_afficher_image = Button(self.fenetre, text="Afficher la Bathymetrie centrale" + "📷", bg='#96c0eb',
                                            activebackground='white',
                                            command=self.plot_image_Mid)

        self.bouton_afficher_image.pack(fill=X, ipady=10, padx=10, pady=10)

        self.bouton_afficher_image = Button(self.fenetre, text="Afficher Bathymetrie profondeur" + "📷", bg='#96c0eb',
                                            activebackground='white',
                                            command=self.plot_image_Depth)

        self.bouton_afficher_image.pack(fill=X, ipady=10, padx=10, pady=10)

        self.check_button = tk.Checkbutton(self.fenetre, text="sauvegarder "+"💾",variable=self.var,bg='yellow',activebackground='white' ,command=self.depliage_boutton)
        self.check_button.pack(fill=X, ipady=10, padx=10,pady=10)

        self.Nom_du_fichier = Label(self.fenetre, text="Enregistrer le fichier sous le nom :")
        self.Nom_du_fichier_entry = Entry(self.fenetre)

        self.bouton_sauvegarde = Button(self.fenetre,bg='#00FF00',activebackground='white', text="valider "+"✅",command=self.sauvegarde)

        self.bouton_No_sauvegarde = Button(self.fenetre,bg='red',activebackground='white', text="Ne pas sauvegarder "+ "🗑️",command=self.No_sauvegarde)
        self.bouton_No_sauvegarde.pack(fill=X, ipady=10, padx=10,pady=10)

        self.bouton_retour = Button(self.fenetre,bg='black',fg='white',activebackground='white', text="retour "+"↩", command=self.retour)
        self.bouton_retour.pack(fill=X, ipady=10, padx=10, pady=10)

        self.key = key
        self.Liste_image = Liste_image
        self.Liste_image_Bord = Liste_image_Bord
        self.Liste_image_Depth = Liste_image_Depth
        self.Liste_image_Mid = Liste_image_Mid
        self.fenetre.mainloop()
    #plot de la liste des images masquées avec la bathymetrie
    def plot_image(self):

        try:
            plt.close() #fermeture de toutes les fenetres (pour regler le probleme de la color bar)

            cmap = plt.get_cmap('jet') #choix de la cmap
            norm = mcolors.Normalize(vmin=0, vmax=100) #Choix de la color map
            cmap.set_over('fuchsia') # Si les valeurs prises dépasses la color bar alors elles sont affichées 'fushia'

            plt.imshow(self.Liste_image[self.indice],cmap = cmap, norm = norm )

            plt.title(self.key[self.indice])
            plt.colorbar()
            plt.axis("off")
            plt.show()
        except Exception as e:
            tk.messagebox.showerror("Erreur", f"Il n'y a plus d'image dans la liste :{str(e)}") #message d'erreur


    #même fonction pour le centre
    def plot_image_Depth(self):

        try:
            plt.close()

            cmap = plt.get_cmap('jet')
            norm = mcolors.Normalize(vmin=0, vmax=100)
            cmap.set_over('fuchsia')

            plt.imshow(self.Liste_image_Depth[self.indice],cmap = cmap, norm = norm )

            plt.title(self.key[self.indice])
            plt.colorbar()
            plt.axis("off")
            plt.show()
        except Exception as e:
            tk.messagebox.showerror("Erreur", f"Il n'y a plus d'image dans la liste :{str(e)}")

    # meme fonction pour le bord
    def plot_image_Bord(self):

        try:
            plt.close()

            cmap = plt.get_cmap('jet')
            norm = mcolors.Normalize(vmin=0, vmax=100)
            cmap.set_over('fuchsia')

            plt.imshow(self.Liste_image_Bord[self.indice],cmap = cmap, norm = norm )

            plt.title(self.key[self.indice])
            plt.colorbar()
            plt.axis("off")
            plt.show()
        except Exception as e:
            tk.messagebox.showerror("Erreur", f"Il n'y a plus d'image dans la liste :{str(e)}")

    # meme fonction pour le milieu
    def plot_image_Mid(self):
        try:
            plt.close()

            cmap = plt.get_cmap('jet')
            norm = mcolors.Normalize(vmin=0, vmax=100)
            cmap.set_over('fuchsia')

            plt.imshow(self.Liste_image_Mid[self.indice], cmap=cmap, norm=norm)

            plt.title(self.key[self.indice])
            plt.colorbar()
            plt.axis("off")
            plt.show()
        except Exception as e:
            tk.messagebox.showerror("Erreur", f"Il n'y a plus d'image dans la liste :{str(e)}")

    def sauvegarde(self,type=".png"):
        if self.Nom_du_fichier_entry.get() == None:
            plt.imsave(self.key + type, self.Liste_image[self.indice])
        else:
            plt.imsave(self.Nom_du_fichier_entry.get() + type, self.Liste_image[self.indice])
        self.indice += 1
        self.Nom_du_fichier.pack_forget()
        self.Nom_du_fichier_entry.pack_forget()
        self.bouton_sauvegarde.pack_forget()
        self.var.set(not self.var.get())
        plt.close()
    def depliage_boutton(self):
        if self.var.get():
            self.Nom_du_fichier.pack(fill=X, ipady=10, padx=10,pady=20)
            self.Nom_du_fichier_entry.pack(fill=X, ipady=20, padx=10,pady=20)
            self.bouton_sauvegarde.pack(fill=X, ipady=30, padx=10,pady=10)
        else:
            self.Nom_du_fichier.pack_forget()
            self.Nom_du_fichier_entry.pack_forget()
            self.bouton_sauvegarde.pack_forget()
    def No_sauvegarde(self):
        self.indice += 1
        plt.close()
    def retour(self):
        self.indice += -1
        plt.close()



if __name__ == '__main__':

    #Exemple d'application
    #import de la fonction process_files pour avoir un dictionnaire de toute les images
    #Attention les images doivent etre en niveau de gris (il faudra modifier la fonction application_bathy pour lire les images RGB)
    dico = process_files("C:\\Users\\mberthie\\PycharmProjects\\Master\\test\\bathymetrie_npy\\",extension=".npy")


    #Chargement des differents masques
    #Attention ils doivent tous avoir la meme taille
    masque = np.load("maskss.npy")
    masque1 = np.load("MasqueBarthMid.npy")
    masque2 = np.load("MasqueBarthBord.npy")
    masque3 = np.load("MasqueBarthDepth.npy")


    #Initialisation des listes
    L_image=[]
    L_image_Mid=[]
    L_image_Bord=[]
    L_image_Depth=[]
    key_img = []

    for key,img in dico.items():
        #Calculs des images
        L_image.append(application_bathy(img,masque,masque1,masque2,masque3,"bathy.png"))
        L_image_Bord.append(application_bathy(img,masque,masque1,masque2,masque3,"bathy.png",tout=False,bord=True))
        L_image_Mid.append(application_bathy(img, masque, masque1, masque2, masque3, "bathy.png", tout=False, Mid=True))
        L_image_Depth.append(application_bathy(img, masque, masque1, masque2, masque3, "bathy.png", tout=False, depth=True))
        key_img.append(key)

    #Affichage de l'interface
    interface_graphique=interface_graphique(L_image,L_image_Depth,L_image_Bord,L_image_Mid,key_img)








