import os
import json
import argparse
import time
import math
from multiprocessing import Pool

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import *

from nmapy.classification import *
from nmapy.features import sift, gabor

N_JOBS = 16

def __parse_args():
    parser = argparse.ArgumentParser(description="""Script for doing a random search for feature parameters 
                                                    """)

    parser.add_argument('-w',
                        '--working_directory',
                        dest='working_dir',
                        required=True,
                        type=str)
    parser.add_argument('-t',
                        '--trials',
                        dest='num_trials',
                        default=10,
                        type=int)
    parser.add_argument('-fpt',
                        '--features per trial',
                        dest='feats_per_trial',
                        default=3,
                        type=int)
    parser.add_argument('-f',
                        '--features',
                        dest='feats',
                        type=str,
                        nargs="+")
    parser.add_argument('-s',
                        '--scales',
                        dest='scales',
                        type=int,
                        nargs="+")
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=True)
    args = parser.parse_args()

    if args.verbose:
        print("---------------- Input ----------------")
        print('working directory:\t' + args.working_dir)
        print('number of trials:\t' + str(args.num_trials))
        print('features per trial:\t' + str(args.feats_per_trial))
        print('features:\t' + str(args.feats))
        print('scales:\t' + str(args.scales))
        print("---------------------------------------")
        print()

    return args

def __check_args(args):
    return NotImplemented

def rand_feature_param_search(working_dir, results_filename, num_trials, max_num_feats, feats=["glcm", "w_hog", "lbp", "lac", "pantex", "sift", "gabor"], scales=["50 meters", "90 meters", "120 meters"], verbose=True):
    """
    runs training and testing on random features

    Parameters:
    -----------


    Returns:
    --------
    out_results: list
        a list of results from the random search. The list has length equal to
        the number of trials specified in the input parameters to this function.

        out_results has the following form:

        [{"val_acc": <the accuracy of the best estimator on the test dataset>,
          "train_acc": <mean cross-validated score of the best estimator>,
          "svm": <the svm parameters>,
          "features": <the feature parameters>,
          "compute_time": <the total time it took to compute the features on the training dataset>,
          "feature_vector_length": <the length of the feature vector used for classification}, 

          {"val_acc": <the accuracy of the best estimator>,
          "train_acc": <mean cross-validated score of the best estimator>,
          "svm": <the svm parameters>,
          "features": <the feature parameters>,
          "compute_time": <the total time it took to compute the features on the training dataset>,
          "feature_vector_length": <the length of the feature vector used for classification},

          ...

          ]
    """
    rfp = random_feats.Random_Feat_Params(feature_names=feats, scales=scales) # a random feature parameter generator object
    out_results = [] # a list of results from the random search
    count = 0

    # change flags when SIFT or gabor features have not been made
    # because the sift and gabor features follow the bag of visual words model, it
    # is usually just easier to compute them prior to the random feature search. Unless
    # the parameters for the bovw model or gabor filters would like to be explored
    sift_already_calculated = True 
    gabor_already_calculated = True


    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))
    #print("trial".ljust(10) + " | " + "train_acc" + " | " + "JHB val_acc" + " | " + "DES val_acc" + " | "+ "features" +  "                       | " + "elapsed_time (min:sec)")
    print("trial".ljust(10) + " | " + "train_acc" + " | " + "val_acc" + " | " + "features" +  "                       | " + "elapsed_time (min:sec)")
    print("------------------------------------------------------------------------------------------")
    tot_time_start = time.time()
    while len(out_results) < num_trials:
        feats_per_trial = random.randint(1, max_num_feats)
        if feats_per_trial > len(feats):
            dups_per_trial = random.randint(feats_per_trial - len(feats) + 1, feats_per_trial)
        else:
            dups_per_trial = random.randint(1, feats_per_trial)
        # print(feats_per_trial, dups_per_trial)
        params = rfp.get_n_rand_feats(feats_per_trial, dups_per_trial) # get 3 random features with a maximum of 2 of the same feature
        param_string = params[0]["feature"]
        for p in params[1:]:
            param_string += " " + p["feature"]
        # do a check to see if the same params have been previously computed and assessed
        # print(param_string)
        already_computed = False
        for i in out_results:
            if params == i["features"]:
                already_computed = True
        if not already_computed:
            # print("start already_computed loop")
            #n = dataset.Neighborhoods()
            #n.load_dataset(working_dir)

            #train, test = n.split_train_test(0.20, True) # splits the neighborhood() class into train and test

            train = dataset.Training()
            test = dataset.Test()
            #test1 = dataset.Test()
            #test2 = dataset.Test()
            train.load_dataset(working_dir + "/train")
            test.load_dataset(working_dir + "/test")
            
            #test1.load_dataset(working_dir + "/test/jhb")
            #test2.load_dataset(working_dir + "/test/des")


            train.set_feature_hyperparams(params)
            test.set_feature_hyperparams(params)
            
            #test1.set_feature_hyperparams(params)
            #test2.set_feature_hyperparams(params)
            # print(" ... computing features")

            # the preprocessing for SIFT and textons (gabor) needs
            # to happen a little differently. the preprocessing for these
            # is more involved and so we need to catch it
            for p in params:
                if p['feature'] == "sift" and sift_already_calculated == False:
                    # run sift on all images
                    sift.write_sift_desc(n.image_filenames)

                    # create the sift codebook using only the training images
                    train_sift_dat_files = [f[:-4] + ".siftdat" for f in train.image_filenames if os.path.exists(f[:-4] + ".siftdat")]
                    all_sift_dat_files = [f[:-4] + ".siftdat" for f in n.image_filenames if os.path.exists(f[:-4] + ".siftdat")]
                    sift.create_sift_codebook(train_sift_dat_files, train.root_dir, n_clusters=32, rand_samp_num=10000)

                    # assign codewords to all sift keypoints using the codebook that was computed on the training data
                    sift.assign_codeword(all_sift_dat_files, os.path.join(n.root_dir, "sift_kmeans_codebook.dat"))

                    # create the codeword images
                    for i in n.image_filenames:
                        sift.create_codeword_id_image(i, i[:-4] + ".siftdat")
                    sift_already_calculated = True
                if p['feature'] == 'gabor' and gabor_already_calculated == False:
                    # get the filterbank. could randomize this in the future.
                    # return a random filterbank
                    bank = gabor.get_default_filter_bank()

                    files_and_filterbanks = []
                    for i in n.image_filenames:
                        files_and_filterbanks.append([i, bank])
                        #gabor.compute_filter_responses(i, bank, mean_var=False)

                    p = Pool(N_JOBS)
                    p.map(gabor.compute_filter_responses_p, files_and_filterbanks)
                    p.close()
                    p.join()

                    train_gabor_files = [f[:-4] + "_gabor_responses.tif" for f in train.image_filenames]
                    all_gabor_files = [f[:-4] + "_gabor_responses.tif" for f in n.image_filenames]
                    # create the codebook using only the training images
                    gabor.create_gabor_codebook(train_gabor_files, train.root_dir, n_clusters=32, rand_samp_num=10000)

                    # create the codeword id images using the kmeans codebook and the gabor filter response images
                    gabor.assign_codeword(all_gabor_files, os.path.join(n.root_dir, "gabor_kmeans_codebook.dat"))
                    gabor_already_calculated = True
            compute_time = train.compute_features(params, n_jobs=N_JOBS)
            test.compute_features(params, n_jobs=N_JOBS)
            
            #test1.compute_features(params, n_jobs=N_JOBS)
            #test2.compute_features(params, n_jobs=N_JOBS)
            # print(" ... done computing features")

            try:

                scaler = StandardScaler(copy=False)
                scaler.fit(train.data)

                scaler.transform(train.data, copy=False)
                scaler.transform(test.data, copy=False)
                #scaler.transform(test1.data, copy=False)
                #scaler.transform(test2.data, copy=False)

                # random search SVM
                #svm_params = {'kernel':['linear','rbf','poly'], 'C':[2**-4, 2**-3, 2**-2, 2**-1, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7], 'gamma':[2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7], 'degree':[3]}
                svm_params = {'kernel':['linear','rbf'], 'C':uniform(loc=0.001, scale=128), 'gamma':uniform(loc=0.001, scale=5), 'degree':[3]}

                svmsvc = SVC()
                #clf = GridSearchCV(estimator=svmsvc, param_grid=svm_params)
                # print(" ... start grid search")
                clf = RandomizedSearchCV(estimator=svmsvc, param_distributions=svm_params, n_iter=30, cv=5, iid=True, n_jobs=16)
                clf.fit(train.data, train.labels)
                # print(" ... done with grid search")

                ############### these are for combined training
                #test_pred1 = clf.predict(test1.data)
                #test_score1 = clf.score(test1.data, test1.labels)
                #cnf_matrix1 = confusion_matrix(test1.labels, test_pred1)
                #test_pred2 = clf.predict(test2.data)
                #test_score2 = clf.score(test2.data, test2.labels)
                #cnf_matrix2 = confusion_matrix(test2.labels, test_pred2)

                ##################### these are for single city
                test_pred = clf.predict(test.data)
                test_score = clf.score(test.data, test.labels)
                cnf_matrix = confusion_matrix(test.labels, test_pred)

                tot_sec_elapsed = time.time() - tot_time_start
                minutes = int(tot_sec_elapsed // 60)
                sec = tot_sec_elapsed % 60
                
                ################ for single training
                print((str(count+1)+"/"+str(num_trials)).ljust(10) + " | " + ("%.3f" %clf.best_score_).ljust(9) + " | " + ("%.3f" %test_score).ljust(7) + " | " + param_string.ljust(30)  + " | " + str(minutes) + ":" + ("%.2f" %(sec)))
                
                ################### for combined-training
                #print((str(count+1)+"/"+str(num_trials)).ljust(10) + " | " + ("%.3f" %clf.best_score_).ljust(9) + " | " + ("%.3f" %test_score).ljust(7) + " | " + ("%.3f" %test_score2).ljust(7) + " | " + param_string.ljust(30)  + " | " + str(minutes) + ":" + ("%.2f" %(sec)))


                out_results.append({"val_acc":test_score,
                                    "train_acc": clf.best_score_,
                                    "svm": clf.best_estimator_.get_params(),
                                    "features": params,
                                    "compute_time":compute_time,
                                    "feature_vector_length":len(train.data[0])})
                # rewrite the json each time to save trials and avoid losing trials if it fails.
                with open(os.path.join(working_dir, results_filename), "w") as outfile:
                    json.dump(out_results, outfile)
                count+=1

            except Exception as e:
                print(e)
                print()
                print(params)

            
    return out_results

def grid_search(working_dir, grid):
    """
    grid search on a grid of feature parameters
    """
    out_results = []
    count = 0
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))
    print("trial".ljust(10) + " | " + "train_acc" + " | " + "val_acc" + " | " + "elapsed_time (min:sec)")
    tot_time_start = time.time()
    for p in grid:
        #print("------------ Begin Trial " + str(count+1)+"/"+str(len(lbp_param_grid)) + "----------------")
        start_time = time.time()
        #print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))
        n = dataset.Neighborhoods()
        n.load_dataset(working_dir)

        train, test = n.split_train_test(0.20, True)

        n.set_feature_hyperparams(p)
        #print("computing features")
        compute_time = train.compute_features(p, n_jobs=N_JOBS)
        test.compute_features(p, n_jobs=N_JOBS)
        #print("done computing features")

        scaler = StandardScaler(copy=False)
        scaler.fit(train.data)

        scaler.transform(train.data, copy=False)
        scaler.transform(test.data, copy=False)

        # grid search SVM
        #print("starting grid search")
        # lbp
        # svm_params = {'kernel':['linear','rbf','poly'], 'C':[2**-4, 2**-3, 2**-2, 2**-1, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7], 'gamma':[2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7], 'degree':[3]}
        # glcm
        svm_params = {'kernel':['linear','rbf'], 'C':[2**-4, 2**-3, 2**-2, 2**-1, 2., 2**2., 2**3., 2**4., 2**5., 2**6., 2**7.], 'gamma':[2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2., 2**2., 2**3., 2**4., 2**5., 2**6., 2**7.], 'degree':[3]}

        svmsvc = SVC()
        clf = GridSearchCV(estimator=svmsvc, param_grid=svm_params, n_jobs=1)
        #clf = SVC()
        clf.fit(train.data, train.labels)
        #print("done with grid search")
        #print(clf.best_estimator_)
        #print(clf.best_score_)

        #print("making predictions")
        test_pred = clf.predict(test.data)
        test_score = clf.score(test.data, test.labels)
        #print(test_score)
        cnf_matrix = confusion_matrix(test.labels, test_pred)
        #print(cnf_matrix)
        out_results.append({"val_acc": test_score,
                            "train_acc": clf.best_score_,
                            "svm": clf.best_estimator_.get_params(),
                            "features": p,
                            "compute_time":compute_time,
                            "feature_vector_length":len(train.data[0])})

        
        tot_sec_elapsed = time.time() - tot_time_start
        minutes = int(tot_sec_elapsed // 60)
        sec = tot_sec_elapsed % 60
        print((str(count+1)+"/"+str(len(grid))).ljust(10) + " | " + ("%.3f" %clf.best_score_).ljust(9) + " | " + ("%.3f" %test_score).ljust(7) + " | " + str(minutes) + ":" + ("%.2f" %(sec)))
        count+=1
    with open(os.path.join(working_dir, "grid_results_"+grid[0][0]['feature'] + ".json"), "w") as outfile:
        json.dump(out_results, outfile)
    return out_results
    

def glcm_grid(scales=["50 meters", "90 meters", "120 meters"]):
    props = ["contrast","dissimilarity","ASM","energy","homogeneity"]
    distances = [[1,2], [1,2,3,4,5,6,7,8,9,10], [2,4,6,8,10], [1,5,10], [1,3,5,7,11,13,17]]
    angles = [[0.,math.pi/6.,math.pi/4.,math.pi/3.,math.pi/2.,(2.*math.pi)/3.,(3.*math.pi)/4.,(5.*math.pi)/6.],
              [0.,math.pi/4.,math.pi/2.,(3.*math.pi)/4.],
              [0.,math.pi/2.]]
    smooth_factor = [None, 1.5]
    levels = [16, 32, 64, 128, 200, None]
    glcm_grid = {"scales":[scales],
                 "prop":props,
                 "distances":distances,
                 "angles":angles,
                 "stat":[None],
                 "smooth_factor":smooth_factor,
                 "levels":levels}
    glcm_grid = ParameterGrid([glcm_grid])
    glcm_param_grid = []
    for n in glcm_grid:
        glcm_param_grid.append([{"feature":"glcm",
                           "params":n}])
    return glcm_param_grid


def lbp_grid(scales=["50 meters", "90 meters", "120 meters"]):
    radii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    points = [4, 8, 12, 16, 20, 24, 32]
    smooth_factor = [None, 1.5]
    levels = [16, 32, 64, 128, 200, None]
    methods = ["uniform", "default"]
    lbp_grid = {"scales":[scales],
                           "radius":radii,
                           "n_points":points,
                           "method":methods,
                           "hist":[True],
                           "stat":[None],
                           "smooth_factor":smooth_factor,
                           "levels":levels}
    lbp_grid = ParameterGrid([lbp_grid])
    lbp_param_grid = []
    for n in lbp_grid:
        lbp_param_grid.append([{"feature":"lbp",
                           "params":n}])
    return lbp_param_grid

def lac_grid(scales=["50 meters", "90 meters", "120 meters"]):
    box_size = [5, 10, 30]
    slide_style = [-1, 5, 10]
    lac_type = ["grayscale"]
    smooth_factor = [None, 1.5]
    levels = [16, 32, 64, 128, 200, None]
    lac_grid = {"scales":[scales],
                "box_size":box_size,
                 "slide_style":slide_style,
                 "lac_type":lac_type,
                 "smooth_factor":smooth_factor,
                 "levels":levels}
    lac_grid = ParameterGrid([lac_grid])
    lac_param_grid = []
    for n in lac_grid:
        lac_param_grid.append([{"feature":"lac",
                               "params":n}])
    return lac_param_grid


def plot_feature_search_results(results_filename, interpolate_plot=True):
    with open(results_filename) as infile:
        data = json.load(infile)
    data_points = [(n['val_acc'],n['features'][0]['params']['radius'],n['features'][0]['params']['n_points']) for n in data]
    radii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    points = [4, 8, 12, 16, 20, 24, 32]

    X, Y = np.meshgrid(radii, points)
    Z = np.zeros(shape=(len(points),(len(radii))))
    xs = []
    ys = []
    zs = []
    for n in data_points:
        Z[points.index(n[2]), radii.index(n[1])] = n[0]
        zs.append(n[0])
        xs.append(n[1])
        ys.append(n[2])

    if interpolate_plot:
        # Create grid values first.
        x = np.linspace(radii[0], radii[-1], 100)
        y = np.linspace(points[0], points[-1], 100)

        # Perform linear interpolation of the data (x,y)
        # on a grid defined by (xi,yi)
        triang = tri.Triangulation(xs, ys)
        interpolator = tri.LinearTriInterpolator(triang, zs)
        Xi, Yi = np.meshgrid(x, y)
        Zi = interpolator(Xi, Yi)
        plt.contour(Xi,Yi,Zi,14,linewidths=0.5, colors='k')
        plt.contourf(Xi,Yi,Zi,14,cmap='Reds')
    else:
        plt.contour(X,Y,Z, 14, cmap="Reds")
    plt.xlabel("radius")
    plt.ylabel("number of points")
    plt.colorbar()
    plt.show()
    

def main():
    start_time = time.time()
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    args = __parse_args()
    __check_args(args)
    rand_feature_param_search(args.working_dir,
                      args.num_trials,
                      args.feats_per_trial,
                      args.feats,
                      args.scales,
                      args.verbose)

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    print('\nEnd data & time -- (%s)\nTotal processing time -- (%d min %f sec)\n' %
        (time.asctime(time.localtime(time.time())), minutes, sec))


if __name__ == "__main__":
    #grid = lac_grid(scales=[60,90,120])
    #grid_search("C:/Users/4ja/data/neighborhood_mapping/image_chips/johan_sub", grid)

    #dirs = ["nairobi", "dakar", "addis_ababa", "dar_es_salaam"]
    #dirs = ["johannesburg", "dar_es_salaam", "nairobi", "dakar", "addis_ababa"]
    dirs = ["dar_es_salaam"]
    scales = ["50 meters", "90 meters", "120 meters"]
    #scales = ["120 meters"]
    print("working on directories: ")
    print(dirs)
    for d in dirs:
        print("START RANDOM SEARCH: " + d)
        working_dir = "/mnt/GATES/UserDirs/4ja/data/image_chips_IGARSS_svms/" + d
        num_trials = 50
        results_filename = "multiscale_randsearch_50t5m_allf.json"
        max_num_feats = 5
        rand_feature_param_search(working_dir, results_filename, num_trials, max_num_feats, feats=["sift", "lbp", "gabor", "w_hog", "glcm", "lac", "pantex"], scales=scales, verbose=True)