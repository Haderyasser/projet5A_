import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from cv2 import cv2


HEIGHT_PATCH, WIDTH_PATCH = 400, 400


def load_image_into_numpy_array(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img

def get_annots_from_xml(path):
    """
    Récupère les annotations contenues dans un fichier .xml au format Pascal VOC.
    Retourne un array utilisable en entrée des modèles Tensorflow et des fonctions d'affichage.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    
    size_x = int(root.findall('size')[0][0].text)
    size_y = int(root.findall('size')[0][1].text)
    objects_annot = root.findall('object')
    arr_box_annot = np.zeros((len(objects_annot), 4), dtype=np.float32)

    for i, box_annot in enumerate(root.findall('object')):
        arr_box_annot[i, :] = [int(box_annot[4][1].text) / size_y,
                               int(box_annot[4][0].text) / size_x,
                               int(box_annot[4][3].text) / size_y,
                               int(box_annot[4][2].text) / size_x]
    return arr_box_annot


def create_xml_from_detection(path_xml, detection_boxes, detection_scores, detection_classes,
                              seuil_detection, name_image, path_image, width, height, depth):
    """
    A rédiger
    Génère un XML au format Pascal VOC contenant les annotations détectées avec une confiance supérieure au seuil.
    """
    # Création de la racine du document et des infos générales
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "patchs"
    ET.SubElement(root, "filename").text = str(name_image)
    ET.SubElement(root, "path").text = str(path_image)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Detection"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmented").text = "0"
    
    # Nombre de bounding boxes détectées avec assez de confiance
    nb_boxes = np.sum(detection_scores >= seuil_detection)
    
    # Création de chaque bounding box
    # Le modèle renvoie les bounding boxes par ordre décroissant de confiance
    for i in range(nb_boxes):
        arr_boxes = np.array(detection_boxes)[i, :]
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "chardon"

        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(np.round(arr_boxes[1] * width)))
        ET.SubElement(bndbox, "ymin").text = str(int(np.round(arr_boxes[0] * height)))
        ET.SubElement(bndbox, "xmax").text = str(int(np.round(arr_boxes[3] * width)))
        ET.SubElement(bndbox, "ymax").text = str(int(np.round(arr_boxes[2] * height)))
                                                    
    # Assemblage de l'arbre
    tree = ET.ElementTree(root)
    
    # Ecriture du fichier
    tree.write(path_xml)


def cut(img, bboxes, folder_output, name_img):
    height, width, depth = img.shape
    height_patch, width_patch = HEIGHT_PATCH, WIDTH_PATCH
    
    for i in range(height // height_patch):
        for j in range(width // width_patch):

            bboxes_patch = []
            x0_patch, y0_patch, x1_patch, y1_patch = i*height_patch, j*width_patch, (i+1)*height_patch, (j+1)*width_patch
            for bbox in bboxes: # fait le tour de toutes les box dans l'image d'origne
                x_min, y_min, x_max, y_max = bbox
                
                if x0_patch < x_min * height and x_max * height < x1_patch and y0_patch < y_min * width and y_max * width < y1_patch:
                    x_min = (x_min*height - x0_patch) / height_patch
                    y_min = (y_min*width - y0_patch) / width_patch  
                    x_max = (x_max*height - x0_patch) / height_patch
                    y_max = (y_max*width - y0_patch) / width_patch
                    bbox = x_min, y_min, x_max, y_max
                    bboxes_patch.append(bbox) # bboxes contenus dans lel patch en cours
                    print("in")
            
            # Commenter ce bloc pour récupérer tous les patchs
            #if len(bboxes_patch) == 0:
            #    continue
            new_img = img[x0_patch : x1_patch, y0_patch : y1_patch, :]
            name_new_img = name_img.replace('.png', '_{}_{}.png'.format(i, j))
            name_xml = name_new_img.replace('.png', '.xml')
            cv2.imwrite(os.path.join(folder_output, name_new_img), cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
            create_xml_from_detection(os.path.join(folder_output, name_xml),
                                  np.array(bboxes_patch), 
                                  np.ones(len(bboxes_patch)),
                                  np.zeros(len(bboxes_patch)),
                                  0,
                                  name_new_img,
                                  os.path.join(folder_output, name_new_img),
                                  width_patch, height_patch, 3)
            
        

if __name__ == '__main__':
    _, path_folder_input, path_folder_output = sys.argv

    list_xml = glob.glob(os.path.join(path_folder_input, '*.xml'))
    list_img = [s.replace('.xml', '.png') for s in list_xml]

    if not os.path.isdir(path_folder_output):
        os.mkdir(path_folder_output)

    for path_xml, path_img in zip(list_xml, list_img):
        img = load_image_into_numpy_array(path_img)
        annots = get_annots_from_xml(path_xml)
        cut(img, annots, path_folder_output, path_img.split(os.path.sep)[-1])
        
