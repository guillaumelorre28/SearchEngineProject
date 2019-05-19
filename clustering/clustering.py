import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pqkmeans
import os
import pickle


class Quantification:

    def __init__(self, num_subdim, Ks, k):

        self.Ks = Ks
        self.k = k
        self.num_subdim = num_subdim
        self.encoder = None
        self.kmeans = None

    def clustering(self, descriptors_list, names_list):

        flattened_descriptors_list = [y for x in descriptors_list for y in x]
        flattened_names_list = [y for x in names_list for y in x]

        descriptors_concat = np.concatenate(flattened_descriptors_list, axis=0)
        descriptors_concat_shuffle = shuffle(descriptors_concat)

        self.encoder = pqkmeans.encoder.PQEncoder(num_subdim=self.num_subdim,
                                        Ks=self.Ks)
        self.encoder.fit(descriptors_concat_shuffle[:1000])

        descriptors_pqcode = self.encoder.transform(descriptors_concat)

        self.kmeans = pqkmeans.clustering.PQKMeans(encoder=self.encoder, k=self.k)
        clusters = self.kmeans.fit_predict(descriptors_pqcode)

        length_list = [len(x) for x in flattened_descriptors_list]
        length_list_cumsum = np.cumsum(length_list, axis=0)
        length_list_cumsum = np.concatenate([np.array([0]), length_list_cumsum])

        result = {}
        # print(length_list_cumsum)
        for i in range(len(length_list_cumsum)-1):
            begin = length_list_cumsum[i]
            end = length_list_cumsum[i+1]
            result[flattened_names_list[i]] = clusters[begin:end]

        return result

    def save_kmeans(self, filename):

        pickle.dump(self.encoder, open(os.path.join(filename, 'encoder.pkl'), 'wb'))
        pickle.dump(self.kmeans, open(os.path.join(filename,'kmeans.pkl'), 'wb'))

    def load_kmeans(self, filename):
        
        self.encoder = pickle.load(open(os.path.join(filename,'encoder.pkl'), 'rb'))
        self.kmeans = pickle.load(open(os.path.join(filename,'kmeans.pkl'), 'rb'))

    def predict_cluster(self, descriptors):

        descriptors_pqcode = self.encoder.transform(descriptors)
        result = self.kmeans.predict(descriptors_pqcode)
        return result
