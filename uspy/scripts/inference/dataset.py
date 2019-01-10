import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from nmapy.execution import execute_feat_vecs

#scikit learn dataset object for SVMs and Decision Trees
class Neighborhoods():

    def __init__(self):
        self.root_dir = None
        self.image_filenames = None
        self.data = None
        self.labels = None
        self.label_names = None
        self.feature_names = None
        self.feature_hyperparams = None
        self.txt_feature_hyperparams = None

    def set_label_names(self, label_names):
        self.label_names = label_names

    def set_labels(self, labels):
        self.labels = labels

    def set_data(self, data):
        self.data = data

    def set_image_filenames(self, filenames):
        self.image_filenames = filenames

    def set_feature_hyperparams(self, feature_hyperparams):
        self.feature_hyperparams = feature_hyperparams

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def set_txt_feature_hyperparams(self, txt_feature_hyperparams):
        self.txt_feature_hyperparams = txt_feature_hyperparams

    def save_feature_hyperparams(self, filename):
        with open(filename, "wb") as output:
            pickle.dump(self.feature_hyperparams, output, -1)

    def load_dataset(self, image_dir):
        """
        sets the self.image_filenames
            the full path to the images
        sets the self.labels
            (an array of shape=(n_samples, 1) of integer values corresponding to the class)
        sets the self.label_names 
            (an array of shape(n_classes) of strings. The index value each string is located at
            corresponds to the integer class value in self.labels)
        """
        self.root_dir = image_dir
        if os.path.exists(image_dir) == False:
            print("error: image directory doesn't exist")
            return
        image_filenames = []
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if f[-4:] == ".tif" and "SIFT" not in f and "gabor" not in f:
                    image_filenames.append(os.path.join(root, f))
        self.set_image_filenames(np.array(image_filenames))
        labels = []
        for n in self.image_filenames:
            # take the second to last item in the split of the filename as the label
            label = n.split("_")[-1][:-4]
            labels.append(label)

            # take the filename as the label
            # labels.append(n.split("_")[-1][:-4])
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        self.set_label_names(le.classes_)
        int_labels = le.transform(labels)
        self.set_labels(int_labels.reshape(-1,1))

    def strat_split_train_test(self, test_size):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_index, test_index in sss.split(self.image_filenames, self.labels):
            train_image_filenames = self.image_filenames[train_index]
            train_labels = self.labels[train_index]
            test_image_filenames = self.image_filenames[test_index]
            test_labels =  self.labels[test_index]

        train = Training()
        train.set_image_filenames(train_image_filenames)
        train.set_labels(train_labels)
        train.set_label_names(self.label_names)
        train.root_dir = self.root_dir

        test = Test()
        test.set_image_filenames(test_image_filenames)
        test.set_labels(test_labels)
        test.set_label_names(self.label_names)
        test.root_dir = self.root_dir
        return train, test

    def split_train_test(self, test_size, shuffle):
        train_image_filenames, test_image_filenames, train_labels, test_labels = train_test_split(self.image_filenames, self.labels, test_size=test_size, random_state=42, shuffle=shuffle)
        
        train = Training()
        train.set_image_filenames(train_image_filenames)
        train.set_labels(train_labels)
        train.set_label_names(self.label_names)
        train.root_dir = self.root_dir

        test = Test()
        test.set_image_filenames(test_image_filenames)
        test.set_labels(test_labels)
        test.set_label_names(self.label_names)
        test.root_dir = self.root_dir
        return train, test

    def drop_classes(self, class_names):
        """
        removes data and labels corresponding to each class in 
        class_names

        class_names: list
            a list of strings where each string indicates a class that will
            be dropped from the dataset.
        """
        new_label_names = []
        for l in self.label_names:
            if l not in class_names:
                new_label_names.append(l)
        self.set_label_names(new_label_names)

        class_idx = []
        for n in class_names:
            class_idx.append(np.where(self.label_names==n)[0])
        new_labels = []
        new_data = []
        count = 0
        while count < len(self.labels):
            if self.labels[count] not in class_idx:
                new_labels.append(self.labels[count])
                new_data.append(self.data[count])
            count+=1
        self.set_labels(new_labels)
        self.set_data(new_data)


    def merge_classes(self, class_map):
        """
        class_map: dict
            a dictionary where keys are new label_names and values are the label_names that will be merged together to
            form a single class
        """
        return NotImplemented


    def compute_features(self, feature_params, n_jobs=1):
        """
        params specifies the feature that will be calculated, along with
        the parameters that the feature will be calculated with (scale, property, stat, etc.)

        the dataset parameter should be a string indicating either to compute features for the test, or train images

        parameters in the form...
        params = [{"feature":"glcm",
                         "params":{"scales":[10,20,30],
                          "prop":"dissimilarity",
                          "stat":None}},
                    {"feature":"w_hog",
                       "params":{"scales":[10,20,30]}}]
        """
        assert(type(feature_params) == list)
        self.set_feature_hyperparams(feature_params)
        params = self.build_params(feature_params)

        # feature_vectors is a n_sample x n_feature ndarray, where feature_vectors[-1] is the label
        feature_vectors, tot_secs = execute_feat_vecs.execute(params, n_jobs)

        # a sloppy check to make sure things are good. to avoid setting a sequence as an array
        # feature_vectors = []
        # for n in feature_vecs:
        #     if type(n) == list or type(n) == np.ndarray:
        #         for m in n:
        #             feature_vectors.append(m)
        #     else:
        #         feature_vectors.append(n)
        try:
            feature_vectors = np.array(feature_vectors)
        except ValueError as e:
            print(e)
            print()
            print(feature_params)
            print()
            print()
            print(feature_vectors[0])

        self.set_labels(feature_vectors[:,-1].astype(int))
        self.set_data(feature_vectors[:,:-1].astype(float))
        return tot_secs

    def build_params(self, feature_params):
        """
        builds parameter list for training on image chips.

        Returns:
        --------
        params_list: list
            a list holding the parameters for features to be computed.
            has the form:
            [ [{},{}], [{},{}] ]
            where len(params_list) is the number of images in the dataset
                  len(params_list[0]) is the number of features to be computed for
                  the first image
        """
        feature_hyperparams = []
        feature_names = []
        params_list = []
        n = 0
        while n < len(self.image_filenames):
            feature_list = []
            i = 0
            while i < len(feature_params):
                if n == 0: # only append the desc once for each image
                    if feature_params[i]["feature"].upper() not in feature_names:
                        feature_names.append(feature_params[i]["feature"].upper())
                    keys = feature_params[i]["params"].keys()
                    for s in feature_params[i]["params"]["scales"]:
                        params_string = feature_params[i]["feature"].upper()+"_SC"+str(s)+"_"
                        for k in keys:
                            if k is not "scales":
                                params_string+=k.upper()+str(feature_params[i]["params"][k])+"_"
                        feature_hyperparams.append(params_string[:-1]) # avoid appending the last '_' 
                feature_list.append({"input":self.image_filenames[n],
                                     "label":self.labels[n][0],
                                     "feature":feature_params[i]["feature"],
                                     "params":feature_params[i]["params"]})
                i+=1
            params_list.append(feature_list)
            n+=1
        self.set_txt_feature_hyperparams(feature_hyperparams)
        self.set_feature_names(feature_names)
        return params_list

    def plot_class_histogram(self):
        num_classes = len(self.label_names)
        class_counts = np.histogram(self.labels, bins=num_classes)[0]
        plt.bar(range(len(self.label_names)), class_counts)
        plt.xticks(range(len(self.label_names)), self.label_names, rotation='vertical')
        plt.ylabel("number of samples")
        plt.xlabel("class")
        plt.show()

class Training(Neighborhoods):

    def __init__(self):
        super().__init__()

class Test(Neighborhoods):

    def __init__(self):
        super().__init__()