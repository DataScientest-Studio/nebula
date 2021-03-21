import numpy as np
import os
import cv2
import PIL
from PIL import Image


#-----------------------------------------------------------
# PROJET NEBULA
# Datascientest DS continu oct. 2020
# Librairie : GRAPHICS
#             Traitement des images et masques RLE
# Auteurs : Cathy Baynaud Samson
#           José Castro
#           Yann Bernery
#           Ludovic Changeon
#-----------------------------------------------------------


def rleToMask(rle: str, shape: tuple  =(1400, 2100)) -> np.ndarray:
    """
    Conversion d'un codage RLE en masque

     Paramètre
     ----------
     rle   : encodage RLE a convertir 
     shape : format du masque

     Retour
     ----------
     np.array : masque
    """

    width, height = shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T



def surfaceFromRle(rle: str) -> int:
    """Determination de la surface (en pixels) converte par le masque correspondage 
       a l'encodage RLE specifie en parametre

     Paramètre
     ----------
     rle : encodage RLE a analyser 

     Retour
     ----------
     int : surface couverte en pixels
    """
    
    
    #Initialisation de la surface
    surface = 0
    
    rleNumbers = [int(numstring) for numstring in rle.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)

    #Pour chaque pairs, on incrémente la surface par la nouvelle
    #longueur de chaine
    for index,length in rlePairs:
        surface += length
    
    return surface


def imshowSuperimposed(nomImage: str, rle: str = '', classe: str = 'Fish', repImage: str = '.', rleSize: tuple = (1400,2100))-> dict:
    """Superposition d'une image RGB et de sa segmentation RLE, colorée par classe
    
     Paramètre
     ----------
     nomImage : nom de l'image à afficher
     repImage : nom du répertoire vers les images ('.' par défaut)
     rle : encodage RLE du masque à superposer
     classe : nom de la classe du masque, parmi {'Fish', 'Flower', 'Gravel', 'Sugar'}
     rleSize : taille du masque RLE en pixels
     
     Retour
     ----------
     dict : {'Superimposed': image ndarray ('uint8') RGB avec sa segmentation,
             'Label': nom de l'image / classe de segmentation,
             'Image': image originale ndarray ('uint8') RGB,
             'Segmentation': image ndarray ('uint8') RGB du masque de segmentation}
    """
    
    #Initialisation du dictionnaire des couleurs (sur base de RGB)
    couleurs = {'Fish':  2, #blue
                'Flower':0, #red
                'Gravel':1, #green
                'Sugar':slice(0,2,1)} #yellow

    #Chargement de l'image demandée
    path = os.path.join(repImage, nomImage)
    img = Image.open(path)

    #Si la taille de l'image est différente de celle du masque,
    #on met l'image à l'échelle du masque
    img = img.resize(size = rleSize[::-1], resample = Image.BILINEAR ).convert('RGB')

    #Chargement du masque à partir du codage RLE
    mask = np.zeros(rleSize)
    if rle != '' : mask = rleToMask(rle, shape = rleSize)

    #Coloration du masque
    couleur = couleurs[classe]                            # on récupère la couleur de la classe
    imgMask = np.zeros(rleSize+(3,))                      # initialisation du masque pour 3 channels
    imgMask[:,:,couleur] =  np.dstack([mask,mask,mask])[:,:,couleur] * 255  # modification du masque vers trois channels
    imgMask = Image.fromarray(imgMask.astype('uint8'))

    #Superposition de l'image et de son masque
    #Images converties en numpy.array pour utiliser la fonction addWeighted d'OpenCV
    #Cette fonction est la plus pratique pour conserver l'alpha de l'image originale à 1
    
    imgOverlay = cv2.addWeighted(np.array(img), 1, np.array(imgMask), 0.5, 0.0)
    
    #Si le code rle fourni était vide, on efface la classe par défaut
    if rle == '': classe = ''
    
    dico = {'Superimposed' : imgOverlay,
            'Label' : nomImage + ' / ' + classe,
            'Image' : np.array(img),
            'Segmentation': np.array(imgMask)}
    
    return dico



