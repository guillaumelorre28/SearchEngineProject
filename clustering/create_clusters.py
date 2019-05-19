from clustering import Quantification
import os
import json
import numpy as np


def load_descriptors(filename):
    data = json.load(open(filename))
    descriptors = [x[2] for x in data]
    descriptors = [np.array(x) for x in descriptors]
    image_names = [x[0] for x in data]

    return descriptors, image_names


def load_descriptors_list(folder_name):
    descriptors_list = []
    names_list = []
    for file in os.listdir(folder_name):
        if file.endswith('.json'):
            descriptors, names = load_descriptors(os.path.join(folder_name, file))
            descriptors_list.append(descriptors)
            names_list.append(names)

    return descriptors_list, names_list


def dataset_handling(folder_name, saving_path):
    """
    folder_name: folder with all the json files containing the descriptors
    (dataset_imagesall_souls.json, dataset_imagesashmolean.json, ...)
    saving_path: Path where the encoder and kmeans object are stored in pickle
    Returns a dictionary:
        -key: image names
        -values: list of visual words
    """
    descriptors_list, names_list = load_descriptors_list(folder_name)
    quantifier = Quantification(Ks=256, k=1000, num_subdim=4)
    result = quantifier.clustering(descriptors_list, names_list)
    quantifier.save_kmeans(saving_path)

    return result

if __name__ == '__main__':

    folder_name = 'image_dataset_descriptors'
    saving_path = 'saved_kmeans'

    result = dataset_handling(folder_name, saving_path)

    print(result.keys())
    print(len(result[result.keys()[0]]))
    # print(result[result.keys()[0]])
