# -*- coding: utf-8 -*-
"""
    Dans cette librairie, on trouvera les fonctions de préparation
    de jeux de test, et des fonctions d'affichages'
"""
import numpy as np
import matplotlib.pyplot as plt

import cv2

from keras.preprocessing.image import ImageDataGenerator



def image_data_generator(data, directory, 
                         filename, classes, 
                         image_width, image_height, 
                         canal, batch_size, 
                         shuffle = True, validationSplit = 0.2,
                         seed = None):
    """
    Définition: Cette fonction procède à la génération
    d'images déformées supplémentaires pour l'entrainement 
    du modèle.

    Parameters
    ----------
    data : DataFrame
        Fichier contenant les informations sur les images et
        les classes contenues dans l'image.
    directory : string
        Nom du répertoire contenant les images.
    filename : string
        Nom de la colonne contenant le nom des images.
    classes : list
        Liste contenant le nom des classes.
    image_width : int
        Largeur des images.
    image_height : int
        Hauteur des images.
    canal : int
        Permet de générer des images en couleur ou en niveau de gris.
    batch_size : int
        Taille des batchs à générer.
    shuffle : boolean, optional
        Mélange aléatoire des données. The default is True.
    validationSplit : float, optional
        Portion des données réservées pour la validation. Compris entre 0 et 
        1 exclu. The default is 0.2.
    seed : int, optional
        Permet de reproduire un mélange pour les tests. The default is None.

    Returns
    -------
    train_generator : tensor
        Tenseur pour l'entrainement du modèle.
    validation_generator : tensor
        Tenseur pour la validation du modèle.

    """
       
    if canal == 1:
        mode_color = "grayscale"
    else:
        mode_color = "rgb"

    datagen = ImageDataGenerator(rescale = 1./255, 
                                 horizontal_flip = True,
                                 vertical_flip = True,
                                 rotation_range = 20,
                                 fill_mode = 'constant',
                                 width_shift_range = 0.1,
                                 height_shift_range = 0.1,
                                 zoom_range = .1,                             
                                 validation_split = validationSplit)

    train_generator = datagen.flow_from_dataframe(
        dataframe = data,
        directory = directory,
        x_col = filename,
        y_col = classes,
        subset = 'training',
        class_mode = 'raw',
        shuffle = shuffle,                    ## On melange tout
        color_mode = mode_color,          ## On passe en nuance de gris
        target_size = (image_width, image_height), ## On divise la taille par 10
        batch_size = batch_size, 
        seed = seed) ## On fixe le seed pour comparer sur les même mélanges

    validation_generator = datagen.flow_from_dataframe(
        dataframe = data,
        directory = directory,
        x_col = filename,
        y_col = classes,
        subset = 'validation',
        class_mode = 'raw',
        shuffle = shuffle,
        color_mode = mode_color,
        target_size = (image_width, image_height),
        batch_size = batch_size, 
        seed = seed)
    
    return train_generator, validation_generator


def plotImages(images_arr, canal):
    """
    Fonction qui permet d'afficher les déformations subies par les images.
    La fonction va donc afficher la première fonction du jeu de données
    et toutes ses déformations.
        @images_arr : jeux de données contenant les images
        @canal : permet de spécifier en niveau de gris ou en couleur
   
    Parameters
    ----------
    images_arr : tensor
        tenseur contenant les jeux de donées transformés.
    canal : int
        Permet de spécifier le mode de couleur, niveau de gris ou couleur.

    """
    
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    
    axes = axes.flatten()
    
    for img, ax in zip( images_arr, axes):

        ## Affichage en fonction du mode d'image
        if canal == 1:
            
            ## Les images physiques sont en rgb mais en grayscale 
            ## dans le generateur, Il faut repasser de 1 a 3 canaux
            stacked_img = np.squeeze(np.stack((img,) * 3, -1)) 
            
            ax.imshow(stacked_img, cmap='gray')
            
        else:
            
            ## Ici, les images sont dans le bon format
            img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_color)
    
    plt.tight_layout()
    
    plt.show()
    
    
def convert_image(data, image_width, image_height, canaux, repPath):
    """
    Fonction qui produit un tableau d'images pour la prédiction

    Parameters
    ----------
    data : dataframe
        dataframe contenant le nom des images.
    image_width : int
        Largeur des images.
    image_height : int
        Hauteur des images.
    canaux : int
        Permet de définir le nombre de canaux (1 ou 3).
    repPath : string
        Nom du répertoire contenant les images.

    Returns
    -------
    array
        Un tableau d'images pour les prédictions.

    """

    
    X_img=[]
    
    for image in data:
        
        path = repPath + image
        
        ## Load image
        if canaux == 3:
            img = cv2.imread(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        ## Resize image
        img = cv2.resize(img,(image_width,image_height))
        
        ## for the black and white image
        img = img.reshape([image_width,image_height,canaux])
            
        ## cv2 load the image BGR sequence color (not RGB)
        X_img.append(img[...,::-1])
        
    X_img = np.array(X_img) / 255
    
    return X_img


def affiche_resultat(history, epoch, classes):
    """
    La fonction affiche un graphe pour la fonction Accuracy 
    et un autre pour la fonction Loss

    Parameters
    ----------
    history : tensor
        Tenseur contenant les résultat de l'entrainement
    epoch : int
        Le nombre de générations d'apprentissage utilisées par le modèle.
    classes : int
        Le nombre de classes sur lesquelles portent l'apprentissage.

    Returns
    -------
    None.

    """
          
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize = (14,5))
    plt.suptitle(f"Résultats sur {classes} classes\n et {epoch} générations")
    
    epoques = np.arange(1, epoch+1, 1)
    
    ## Labels des axes
    plt.subplot(121) 
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    ## Courbe de la précision sur l'échantillon d'entrainement
    plt.plot(epoques,
             train_accuracy,
             label = 'Training Accuracy',
             color = 'blue')

    ## Courbe de la précision sur l'échantillon de validation
    plt.plot(epoques,
             val_accuracy,
             label = 'Validation Accuracy',
             color = 'red')

    ## Affichage de la légende
    plt.legend()

    plt.subplot(122) 
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    ## Courbe de la précision sur l'échantillon d'entrainement
    plt.plot(epoques,
             train_loss,
             label = 'Training loss',
             color = 'blue')

    ## Courbe de la précision sur l'échantillon de validation
    plt.plot(epoques,
             val_loss,
             label = 'Validation loss',
             color = 'red')

    ## Affichage de la légende
    plt.legend()
    
    ## Affichage de la figure
    plt.show()