import os
import pickle

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.externals import joblib

from nmapy.classification import *

def main():
    #n = dataset.Neighborhoods()
    working_dir = "/mnt/GATES/UserDirs/4ja/data/image_chips_IGARSS_svms/dar_es_salaam"
    #n.load_dataset(working_dir)

    #train, test = n.strat_split_train_test(0.20)

    run_desc = "singlescale"

    train = dataset.Training()
    test = dataset.Test()
    train.load_dataset(working_dir + "/train")
    test.load_dataset(working_dir + "/test")


    print("computing features")

    params = [{'feature': 'glcm',
   'params': {'scales': ['120 meters'],
    'prop': 'energy',
    'distances': [1, 2],
    'angles': [0.0,
     0.5235987755982988,
     0.7853981633974483,
     1.0471975511965976,
     1.5707963267948966,
     2.0943951023931953,
     2.356194490192345,
     2.6179938779914944],
    'smooth_factor': None,
    'levels': None,
    'stat': None}},
  {'feature': 'gabor',
   'params': {'scales': ['120 meters'],
    'thetas': [0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345],
    'sigmas': [1, 3, 7],
    'frequencies': [0.9],
    'n_clusters': 32,
    'mean_var_method': False}},
  {'feature': 'glcm',
   'params': {'scales': ['120 meters'],
    'prop': 'ASM',
    'distances': [2, 4, 6, 8, 10],
    'angles': [0.0,
     0.5235987755982988,
     0.7853981633974483,
     1.0471975511965976,
     1.5707963267948966,
     2.0943951023931953,
     2.356194490192345,
     2.6179938779914944],
    'smooth_factor': None,
    'levels': 200,
    'stat': None}},
  {'feature': 'w_hog', 'params': {'scales': ['120 meters']}}]
    #train.set_feature_hyperparams(params)
    #test.set_feature_hyperparams(params)
    
    train.compute_features(params, n_jobs=16)
    test.compute_features(params, n_jobs=16)
    print("done computing features")

    scaler = StandardScaler(copy=False)
    scaler.fit(train.data)

    scaler.transform(train.data, copy=False)
    scaler.transform(test.data, copy=False)

    # grid search SVM
    # print("starting grid search")
    # svm_params = {'kernel':['linear','rbf','poly'], 'C':[2**-4, 2**-3, 2**-2, 2**-1, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7], 'gamma':[2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7], 'degree':[3]}
    # svmsvc = SVC()
    # clf = GridSearchCV(estimator=svmsvc, param_grid=svm_params)
    # clf.fit(train.data, train.labels)
    # print("done with grid search")
    # print(clf.best_estimator_)
    # print(clf.best_score_)

    # print("making predictions")
    # test_pred = clf.predict(test.data)
    # print(clf.score(test.data, test.labels))
    # cnf_matrix = confusion_matrix(test.labels, test_pred)
    # print(cnf_matrix)
    # clf = clf.best_estimator

    # with setting params explicitly
    clf = SVC(C=12.041044961603584, kernel='linear', gamma=4.958572644482876)
    clf.fit(train.data, train.labels)
    print("training accuracy")
    print(clf.score(train.data, train.labels))

    print("making predictions")
    test_pred = clf.predict(test.data)
    print(clf.score(test.data, test.labels))
    cnf_matrix = confusion_matrix(test.labels, test_pred)
    print(cnf_matrix)

    # save the confusion matrix, the classifier, the feature parameters, 
    # and write human readable metadata about the training and testing datasets
    np.save(os.path.join(working_dir, run_desc + "_conf_matrix.npy"), cnf_matrix)
    with open(os.path.join(working_dir, run_desc + "_svm_model.pkl"), "wb") as clf_output:
        pickle.dump(clf, clf_output, -1)
    #joblib.dump(clf, "C:/Users/4ja/data/neighborhood_mapping/image_chips/johannesburg/svm_model.pkl")
    train.save_feature_hyperparams(os.path.join(working_dir, run_desc + "_feature_params.pkl"))
    with open(os.path.join(working_dir, run_desc + "_data_scaler.pkl"), "wb") as output:
        pickle.dump(scaler, output, -1)
    write_metadata(working_dir, run_desc, train, test, clf, scaler)

def write_metadata(working_dir, train, test, clf, scaler):
    metadata_file = open(os.path.join(working_dir, run_desc + "_metadata.txt"), "w")
    
    labelnames = str(train.label_names[0])
    for n in train.label_names[1:]:
        labelnames += ","+str(n)
    metadata_file.write("CLASS NAMES           : " + labelnames + "\n")

    trainclasses = len(train.label_names)
    traincounts = np.histogram(train.labels, bins=trainclasses)[0]
    trainclasscount = str(traincounts[0])
    for n in traincounts[1:]:
        trainclasscount+=","+str(n)
    metadata_file.write("TRAIN SAMPLES         : " + trainclasscount + "\n")
    
    testclasses = len(test.label_names)
    testcounts = np.histogram(test.labels, bins=testclasses)[0]
    testclasscount = str(testcounts[0])
    for n in testcounts[1:]:
        testclasscount+=","+str(n)
    metadata_file.write("TEST SAMPLES          : " + testclasscount + "\n")
    metadata_file.write("-------------------------------------- Features ------------------------------------\n")

    metadata_file.write("FEATURE VECTOR LENGTH : " + str(len(train.data[0])) + "\n")
    feature_means = str(scaler.mean_[0])
    feature_vars = str(scaler.var_[0])
    for n in range(1,len(scaler.mean_[:])):
        feature_means += "," + str(scaler.mean_[n])
        feature_vars += "," + str(scaler.var_[n])
    metadata_file.write("FEATURE MEANS         : " + feature_means + "\n")
    metadata_file.write("FEATURE VARIANCES     : " + feature_vars + "\n")

    featurenames = train.feature_names[0]
    for n in train.feature_names[1:]:
        featurenames += "," + n
    metadata_file.write("PRIMARY FEATURES      : " + featurenames + "\n")
    metadata_file.write("--------------------------------- Feature Parameters ----------------------------------\n")
    for n in train.txt_feature_hyperparams:
        metadata_file.write(str(n) + "\n")

    metadata_file.write("--------------------------------- Class Means -----------------------------------------\n")
    preds = clf.predict(train.data)
    p = 0
    mean_dict = {}
    while p < len(preds):
        if train.label_names[preds[p]] not in mean_dict:
            mean_dict[train.label_names[preds[p]]] = [[train.data[p]]]
        else:
            mean_dict[train.label_names[preds[p]]].append([train.data[p]])
        p+=1
    for k in mean_dict:
        mean_dict[k] = np.mean(np.array(mean_dict[k]), axis=0)
        values_string = k + "    :    " + str(mean_dict[k][0][0])
        for val in mean_dict[k][0][1:]:
            values_string+= "," + str(val)
        metadata_file.write(values_string + "\n")
    

    metadata_file.write("-------------------------------------------------------------------\n")
    metadata_file.write("SVM PARAMS            : " + str(clf.get_params()) + "\n")
    # np.save("C:/Users/4ja/data/neighborhood_mapping/image_chips/johannesburg/svm_support.npy",
    #         np.array(clf.best_estimator_.support_))
    # np.save("C:/Users/4ja/data/neighborhood_mapping/image_chips/johannesburg/svm_support_vectors.npy",
    #         np.array(clf.best_estimator_.support_vectors_))
    # np.save("C:/Users/4ja/data/neighborhood_mapping/image_chips/johannesburg/svm_n_support.npy",
    #         np.array(clf.best_estimator_.n_support_))
    # np.save("C:/Users/4ja/data/neighborhood_mapping/image_chips/johannesburg/svm_dual_coef.npy",
    #         np.array(clf.best_estimator_.dual_coef_))
    # np.save("C:/Users/4ja/data/neighborhood_mapping/image_chips/johannesburg/svm_intercept.npy",
    #         np.array(clf.best_estimator_.intercept_))
    # if clf.best_estimator_.get_params()["kernel"] == "linear":
    #     np.save("C:/Users/4ja/data/neighborhood_mapping/image_chips/johannesburg/svm_coef.npy",
    #         np.array(clf.best_estimator_.coef_))

    # metadata_file.write("support_           : " + np.array(clf.best_estimator_.support_) + "\n")
    # metadata_file.write("support_vectors_   : " + np.array(clf.best_estimator_.support_vectors_) + "\n")
    # metadata_file.write("n_support_         : " + np.array(clf.best_estimator_.n_support_) + "\n")
    # metadata_file.write("dual_coef_         : " + np.array(clf.best_estimator_.dual_coef_) + "\n")
    # metadata_file.write("intercept_         : " + np.array(clf.best_estimator_.intercept_) + "\n")
    # if clf.best_estimator_.get_params()["kernel"] == "linear":
    #     metadata_file.write("coef_              : " + np.array(clf.best_estimator_.coef_) + "\n")
    metadata_file.close()


if __name__ == "__main__":
    main()