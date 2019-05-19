from clustering import Quantification
import os
import numpy as np
import json


def request_handling(folder, saved_path, dataset_name, image_name):
    """
    folder: folder with all the json files containing the descriptors
    (dataset_imagesall_souls.json, dataset_imagesashmolean.json, ...)
    saved_path: path where encoder and kmeans objects are saved_path
    dataset_name: json file in which the request image is
    image_name: name of the request image
    Return: List of visual words of the image
    """
    data_request = json.load(open(name_dataset_file))
    request = data_request[[x[0] for x in data_request].index(image_name)][2]
    request = np.array(request)

    quantifier = Quantification(Ks=256, k=1000, num_subdim=4)
    quantifier.load_kmeans(saving_filename)

    quantified_request = quantifier.predict_cluster(request)

    return quantified_request


if __name__ == '__main__':

    folder_name = 'image_dataset_descriptors'
    saving_filename = 'saved_kmeans'

    name_dataset_file = 'dataset_imagesall_souls.json'
    image_name = 'all_souls_000091'

    result = request_handling(folder_name, saving_filename, name_dataset_file,
                                                    image_name)
    print(result.shape)
