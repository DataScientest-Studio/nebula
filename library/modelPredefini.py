#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Dans cette librairie, on trouvera différents modèles de CNN, une fonction
    permettant de créer un modèle libre et des fonctions pour créer 
    des callback

@author: joca
"""
from keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import GlobalAveragePooling2D

from tensorflow.keras import callbacks


def model_3couches(image_width, image_height,
                  nb_classes, 
                  canaux, 
                  activation = 'relu', 
                  activation_sortie = 'sigmoid',
                  affichage = False):
    """
    Ce modèle de base contient :
        - trois couches de convolution:
            2 couches avec 64 filtres et un noyau de 5 par 5
            1 couche avec 64 filtres et un noyau de 3 par 3
        - trois couches de maxpooling noyau 2 par 2
        - tois couches dense:
            1 couche de 128 neuronnes
            1 couche de 254 neuronnes
            1 couche de sortie avec le nombre de neuronnes correspondants
            au nombre de classes fixé en paramètre
        2 couches dropout avec un seuil de 0.2

    Parameters
    ----------
    image_width : int
        Largeur des images.
    image_height : int
        Hauteur des images.
    canaux : int
        Permet de générer des images en couleur ou en niveau de gris.
    nb_classes : int
        Permet de fixer le nombre de neuronnes de la couche de sortie.
    activation : str, optional
        Permet de fixer la fonction d'activation des différentes couches
        du réseau à part la couche de sortie. The default is 'relu'.
    activation_sortie : str, optional
        Permet de fixer la fonction d'activation de la couche de sortie. 
        The default is 'sigmoid'.
    affichage : boolean, optional
        Permet d'afficher le descritif du réseau. The default is False.

    Returns
    -------
    model : sequential
        Retourne le réseau de convolution.

    """
    model = Sequential()

    model.add(Conv2D(64, (5,5), 
                     padding = 'valid', 
                     input_shape = (image_width, image_height,canaux), 
                     activation = activation))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (5,5), 
                     padding = 'valid', 
                     activation = activation))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, (3,3), 
                     padding = 'valid', 
                     activation = activation))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())

    model.add(Dense(128, 
                    activation = activation))
    model.add(Dropout(0.2))

    model.add(Dense(254, 
                    activation = activation))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes, 
                    activation = activation_sortie))

    if affichage == True:
        model.summary()
        
    return model

def model_4couches(image_width, image_height,
                  nb_classes, 
                  canaux, 
                  activation = 'relu', 
                  activation_sortie = 'sigmoid',
                  affichage = False):
    """
    Ce modèle de base contient :
        - quatres couches de convolution:
            2 couches avec 64 filtres et un noyau de 5 par 5
            2 couches avec 64 filtres et un noyau de 3 par 3
        - quatre couches de maxpooling noyau 2 par 2
        - quatre couches dense:
            2 couches de 128 neuronnes
            1 couche de 254 neuronnes
            1 couche de sortie avec le nombre de neuronnes correspondants
            au nombre de classes fixé en paramètre
        1 couche dropout avec un seuil de 0.2

    Parameters
    ----------
    image_width : int
        Largeur des images.
    image_height : int
        Hauteur des images.
    canaux : int
        Permet de générer des images en couleur ou en niveau de gris.
    nb_classes : int
        Permet de fixer le nombre de neuronnes de la couche de sortie.
    activation : str, optional
        Permet de fixer la fonction d'activation des différentes couches
        du réseau à part la couche de sortie. The default is 'relu'.
    activation_sortie : str, optional
        Permet de fixer la fonction d'activation de la couche de sortie. 
        The default is 'sigmoid'.
    affichage : boolean, optional
        Permet d'afficher le descritif du réseau. The default is False.

    Returns
    -------
    model : sequential
        Retourne le réseau de convolution.

    """
    model = Sequential()

    model.add(Conv2D(64, (5,5), 
                     padding='valid', 
                     input_shape=(image_width, image_height,canaux), 
                     activation = activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5,5), 
                     padding='valid', 
                     activation = activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3,3), 
                     padding='valid', 
                     activation = activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3,3), 
                     padding='valid', 
                     activation = activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, 
                    activation = activation))

    model.add(Dense(128, 
                    activation = activation))

    model.add(Dense(254, 
                    activation = activation))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes, 
                    activation = activation_sortie))

    if affichage == True:
        model.summary()
        
    return model

def model_5couches(image_width, image_height,
                  nb_classes, 
                  canaux, 
                  activation = 'relu', 
                  activation_sortie = 'sigmoid',
                  affichage = False):
    """
    Ce modèle de base contient :
        - cinq couches de convolution:
            1 couche avec 192 filtres et un noyau de 7 par 7
            1 couche avec 256 filtres et un noyau de 3 par 3
            1 couche avec 512 filtres et un noyau de 3 par 3
            2 couches avec 1024 filtres et un noyau de 3 par 3
        - couches couches de maxpooling noyau 2 par 2
        - trois couches dense:
            1 couche de 1024 neuronnes
            1 couche de 4096 neuronnes
            1 couche de sortie avec le nombre de neuronnes correspondants
            au nombre de classes fixé en paramètre
        3 couches dropout avec un seuil de 0.2

    Parameters
    ----------
    image_width : int
        Largeur des images.
    image_height : int
        Hauteur des images.
    canaux : int
        Permet de générer des images en couleur ou en niveau de gris.
    nb_classes : int
        Permet de fixer le nombre de neuronnes de la couche de sortie.
    activation : str, optional
        Permet de fixer la fonction d'activation des différentes couches
        du réseau à part la couche de sortie. The default is 'relu'.
    activation_sortie : str, optional
        Permet de fixer la fonction d'activation de la couche de sortie. 
        The default is 'sigmoid'.
    affichage : boolean, optional
        Permet d'afficher le descritif du réseau. The default is False.

    Returns
    -------
    model : sequential
        Retourne le réseau de convolution.

    """
    model = Sequential()

    model.add(Conv2D(filters = 192,
                       kernel_size = (7,7),
                       input_shape=(image_width, image_height,canaux),
                       padding = 'valid',
                       activation = activation))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(filters = 256,
                     kernel_size = (3,3),
                     padding = 'valid',
                     activation = activation))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(filters = 512,
                     kernel_size = (3,3),
                     padding = 'valid',
                     activation = activation))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(filters = 1024,
                     kernel_size = (3,3),
                     padding = 'valid',
                     activation = activation))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = 1024,
                     kernel_size = (3,3),
                     padding = 'valid',
                     activation = activation))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024,
                    activation = activation))

    model.add(Dense(4096,
                    activation = activation))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes, 
                    activation = activation_sortie))

    if affichage == True:
        model.summary()
        
    return model

def model_libre(image_width, image_height,
                  nb_classes, 
                  canaux, 
                  coucheConv2D,
                  coucheDense,
                  GlobalAveragePooling2D_layer = False,
                  nb_filters_start = 64,
                  kernel_size_start = (3,3),
                  activation_start = 'relu', 
                  pool_size_start = (2,2),
                  padding_start = 'valid',
                  nb_filters = 32,
                  kernel_size = (3,3),
                  pool_size = (2,2),
                  padding = 'valid',
                  nb_units = 128,
                  activation = 'relu', 
                  coeff_drop = 0.2,
                  activation_sortie = 'sigmoid',
                  affichage = False):
    """
    Cette fonction permet de créer un réseaux de neuronnes simples 
    personnalisé où on peut définir les paramètres de la couche de
    convolution en entrée et les paramètre s de la couche de maxpooloing
    associée. 
    - On peut définir un nombre de couple de couches Con2D, 
    maxpooling avec les paramètres associés.
    - On peut définir si on utilise la couche flatten ou GlobalAveragePooling2D.
    - On peut aussi définir le nombre de couche Dense avec les paramètres 
    associés.
    - Il es ausi possible de définir le coefficient des deux couches Dropout.
    L'une se trouve juste avant la couche d'applatissement et l'autre, 
    juste avant la couche de sortie.
    - On peut également fixer le nombre de classes qui correspondra au 
    nombre de neuronnes de la couche Dense de sortie dans laquelle on peut 
    également définir la fonction d'activation qui par défaut est à 'sigmoïd'

    Parameters
    ----------
    image_width : int
        Largeur des images.
    image_height : int
        Hauteur des images.
    nb_classes : int
        Permet de fixer le nombre de neuronnes de la couche de sortie.
    canaux : int
        Permet de générer des images en couleur ou en niveau de gris.
    coucheConv2D : int
        Permet de fixer le nombre de couches de convolution suplémentaires
    CoucheDense : int
        Permet de fixer le nombre de couches dense
    GlobalAveragePooling2D_layer : boolean, optional
        Permet de positionner une couche GlobalAveragePooling2D. par défaut,
        la valeur est 'False' et donc on aura une couche de Flatten
    nb_filters_start : int, optional
        Permet de fixer le nombre de filtres de la couche de convolution 
        en entrée. The default is 64.
    kernel_size_start : tuple , optional
        Permet de fixer la taille du noyau de la couche de convolution
        en entrée. The default is (3,3).
    activation_start : str, optional
        Permet de fixer la fonction d'activation de la couche de convolution 
        en entrée. The default is 'relu'.
    pool_size_start : tuple , optional
        Permet de fixer la taille du pool de la couche de maxpooling
        en entrée. The default is (2,2).
    padding_start : str, optional
        Permet de fixer la valeur du padding de couche de maxpooling
        en entrée.The default is 'valid'.
    nb_filters : int, optional
        Permet de fixer le nombre de filtres des couches de convolution 
        supplémentaires. The default is 32.
    kernel_size : tuple , optional
        Permet de fixer la taille du noyau des couches de convolution
        supplémentaires. The default is (3,3).
    pool_size : tuple , optional
        Permet de fixer la taille du pool des couches de maxpooling
        supplémentaires. The default is (2,2).
    padding : str, optional
        Permet de fixer la valeur du padding des couches de maxpooling
        supplémentaires.The default is 'valid'.
    nb_units : int, optional
        Permet de fixer le nombre de neuronnes des couches Dense.
        The default is 128.
    activation : str, optional
        Permet de fixer la fonction d'activation des différentes couches
        du réseau à part la couche de sortie et la couche d'entrée. 
        The default is 'relu'.
    coeff_drop : float ]0,1[, optional
        Permet de fixer le coefficient des couches Dropout.
        The default is 0.2.
    activation_sortie : str, optional
        Permet de fixer la fonction d'activation de la couche de sortie. 
        The default is 'sigmoid'.
    affichage : boolean, optional
        Permet d'afficher le descritif du réseau. The default is False.

    Returns
    -------
    model : sequential
        Retourne le réseau de convolution.

    """
    model = Sequential()

    model.add(Conv2D(filters = nb_filters_start,
                       kernel_size = kernel_size_start,
                       input_shape=(image_width, image_height,canaux),
                       padding = padding_start,
                       activation = activation_start))
    model.add(MaxPooling2D(pool_size = (pool_size_start)))

    for nbConv in range(coucheConv2D):
        model.add(Conv2D(filters = nb_filters,
                         kernel_size = kernel_size,
                         padding = padding,
                         activation = activation))
        model.add(MaxPooling2D(pool_size = pool_size))

    model.add(Dropout(coeff_drop))

    if GlobalAveragePooling2D_layer == False:
        model.add(Flatten())
    else:
        model.add(GlobalAveragePooling2D())
    
    for nbConv in range(coucheDense):
        model.add(Dense(nb_units,
                        activation = activation))

    model.add(Dropout(coeff_drop))

    model.add(Dense(nb_classes, 
                    activation = activation_sortie))

    if affichage == True:
        model.summary()
        
    return model


def model_EfficientNet(image_width, image_height,
                  nb_classes, 
                  model, 
                  activation = 'relu', 
                  activation_sortie = 'sigmoid',
                  affichage = False):
    """
    Ce modèle permet de tester un des model EfficientNet B0 ou B1 avec deux couches 
    dense avec respectivement 1024 et 512 neuronnes et une couche de sortie 
    avec les neuronnes correspondants au nombre de classes

    Parameters
    ----------
    image_width : int
        Largeur des images.
    image_height : int
        Hauteur des images.
    nb_classes : int
        Permet de fixer le nombre de neuronnes de la couche de sortie.
    model : int
        Vaut 0 ou 1 pour choisir le model efficientNet.
    activation : str, optional
        Permet de fixer la fonction d'activation des différentes couches
        du réseau à part la couche de sortie. The default is 'relu'.
    activation_sortie : str, optional
        Permet de fixer la fonction d'activation de la couche de sortie. 
        The default is 'sigmoid'.
    affichage : boolean, optional
        Permet d'afficher le descritif du réseau. The default is False.

    Returns
    -------
    model : sequential
        Retourne le réseau de convolution.
    """
    if model == 0:
        efficientNet = EfficientNetB0(include_top=False,
                                      input_shape=(image_width, 
                                                   image_height,3))
        for layer in efficientNet.layers:
            layer.trainable = False
    else:
        efficientNet = EfficientNetB1(include_top=False,
                                      input_shape=(image_width, 
                                                   image_height,3))
        for layer in efficientNet.layers:
            layer.trainable = False
    
    model = Sequential()
    
    model.add(efficientNet)  
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(units = 1024, 
                    activation = activation))
    model.add(Dropout(0.2))
    
    model.add(Dense(units = 512, 
                    activation = activation))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes, 
                    activation = activation_sortie))

    if affichage == True:
        model.summary()
        
    return model

## On crée une fonction qui permet de sauvegarder les poids
## du modèle, voir le modèle en enier
def checkpoint_function(filepath, save_best_only = True, save_weights_only = False):
    """
    Fonction qui permet de créer une sauvegarde du modèle en entier, ou simplement
    des meilleurs poids.
    
    Parameters
    ----------
    filepath : string
        Permet de donner le répertoire de sauvegarde ou le nom du fichier 
        où on souhaite sauvegarder les poids.("Chemin relatif complet")
    save_best_only : boolean, optional
        Booléain permettant de sauvegarder tous les poids, ou simplement 
        les meilleurs (True). The Default is True
    save_weights_only : boolean, optional
        Booléain permettant de sauvegarder le modèle entier (True). 
        The Default is False
    """
    
    checkpoint = callbacks.ModelCheckpoint(filepath,
                                           monitor='val_loss',
                                           save_best_only = save_best_only,
                                           save_weights_only = save_weights_only,
                                           mode = 'min',
                                           save_freq = 'epoch')
    return checkpoint


## On crée une fonction qui permet de créer un callback
## pour gérer le learning rate si le CNN n'évolue plus
def reduceLR_function(monitor = 'val_loss', patience = 5, mode = 'min'):
    """
    Fonction qui permet de créer un callback de mise à jour du learning rate.
    
    Parameters
    ----------
    monitor : string, optional
        Permet de définir la fonction à surveiller. La fonction de perte, ou la fonction
        d'évaluation. The Default is 'val_loss'
    patience : int, optional
        Permet de définir au bout de combien de génération on modifie le learning rate
        si la fonction surveillée n'évolue plus. The Default is 5
    mode : string, optional
        Permet de définir si on cherche à minimiser ou à maximiser la fonction
        monitorée. The Default is 'min'
    """
    
    lr_plateau = callbacks.ReduceLROnPlateau(monitor = monitor,
                                         patience = patience,
                                         verbose = 2,
                                         factor=0.1,
                                         mode = mode)
    
    return lr_plateau


## On crée une fonction qui permet de créer un callback
## pour arrêter l'apprentissage si le CNN n'évolue plus
def estopping_function(monitor = 'val_loss', patience = 10, mode = 'min'):
    """
    Fonction qui permet de créer un callback d'arrêt d'apprentissage.
    
    Parameters
    ----------
    monitor : string, optional
        Permet de définir la fonction à surveiller. La fonction de perte, ou la fonction
        d'évaluation. The Default is 'val_loss'
    patience : int, optional
        Permet de définir au bout de combien de génération arrête l'apprentissage
        si le CNN n'évolue plus. The Default is 10
    mode : string, optional
        Permet de définir si on cherche à minimiser ou à maximiser la fonction
        monitorée. The Default is 'min'
    """
    
    e_stopping = callbacks.EarlyStopping(monitor = monitor,
                                     patience = patience,
                                     mode = mode,
                                     restore_best_weights = True)
    
    return e_stopping