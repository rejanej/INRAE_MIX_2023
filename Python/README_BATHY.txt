#######################

Ce code possède une fonction qui permet d'appliquer la bathymetrie avec 4 masques, 1 masque bathymetrie simple, 1 masque bathymetrie des zones les plus profondes,
1 masque bathymetrie des zones au bord et 1 masque bathymetrie de la zone entre le bord et les zones les plus profondes (milieu).


Ce code possède une interface graphique qui permet d'enregistrer les images ou la bathymetrie a été appliquée sur vos images filtrées.


#######################


Voici comment utiliser ce code :

Tout d'abord, il vous faut un dictionnaire avec les images en valeurs et les noms des images en clés.

Pour utiliser la fonction application bathy de vos images, vous aurez besoin des 4 masques, du masque total, du masque de la zone du milieu, du masque de la zone du bord, 
et du masque de la zone la plus profonde. Enfin, il faudra le path de la bathymétrie (pas de path dans l'exemple donc "bathy.png")

Les masques doivent être lus et être en binaire (dans l'exemple ils sont en ".npy")

Pour utiliser l'interface il faut créer des listes qui contiennent chacune les images filtrées par les masques.

Pour se faire, parcourez le dictionnaire comme ceci :
'''
for key,img in dictionnaire.items():
'''
et ajouter les valeurs au fur et à mesure dans la liste. Attention il faut préciser ce que l'on veut resortir avec les paramètres (tout, bord, inter, depth), ⚠ par defaut tout prend la valeur True
'''
Liste_image_filtre_bathy.append(application_bathy(img,masque,masque1,etc..., tout = True)
'''
de même avec les autres:
'''

Liste_image_filtre_bathy_bord.append(application_bathy(img,masque,masque1,etc..., tout = False, bord = True)
 
'''
etc etc

⚠ il faut le faire pour tous les masques !

Rajouter également une liste pour sauvegarder les clés lorsque l'on parcourt le dico comme ceci !

'''

key_img.append(key)

'''
##################################################################################

Une fois que vous avez vos listes, il suffit d'appeler l'interface comme ceci:
"""
interface_graphique = interface_graphique(Liste_image_filtre_bathy,Liste_image_filtre_bathy_Depth,Liste_image_filtre_bathy_Bord,Liste_image_filtre_bathy_Mid,key_img)
"""
En premier : liste des images avec le masque de base.
En deuxième : liste des images avec le masque uniquement des profondeurs de la bathymétrie
En troisième : liste des images avec le masque uniquement du bord de la bathymétrie
En quatrième : liste des images avec le masque uniquement des valeurs milieu de la bathymétrie
En cinquième : liste avec les noms des images