import time
import os
import pprint
from multiprocessing import Pool

from ..features import *

def execute(execution_parameters, n_jobs=1):
    start_time = time.time()
    #print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))
    #print()
    #print("processing " + str(len(execution_parameters)) + " images\n")
    #print("------------------------------------------------------------------")
    #print()

    if n_jobs == -1:
        output = __sequential_process(execution_parameters)
    else: # data parralelism
        p = Pool(n_jobs)
        output = p.map(__process, execution_parameters)
        p.close()
        p.join()

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    #print('\nEnd data & time -- (%s)\nTotal processing time -- (%d min %f sec)\n' %
    #    (time.asctime(time.localtime(time.time())), minutes, sec))
    #print("----------------------- End ------------------------------")
    return output, tot_sec

def __sequential_process(execution_parameters_list):
    pp = pprint.PrettyPrinter(indent=4,width=5)
    feature_vectors = []
    for execution_parameters in execution_parameters_list:
        label = None
        feature_vector = []
        for feature_param in execution_parameters:
            #print("---------------------- Computing Feature ------------------------")
            #pp.pprint(feature_param)
            #print()
            if label == None:
                label = feature_param["label"]
            input_im = feature_param["input"]
            # label = execution_parameters["label"]
            feature = feature_param["feature"]
            params = feature_param["params"]
            orig_scales = params["scales"]

            # scales can be specified in pixels or meters. 
            # depending on input, convert to pixels.
            # the scales that are used as input to the feature
            # functions must always be specified in pixels
            scales = []
            for s in orig_scales:
                if "meters" in s:
                    if "wv3" in input_im:
                        cell_width = 0.31
                    if "wv2" in input_im or "ge1" in input_im:
                        cell_width = 0.46
                    else:
                        cell_width = 0.46
                    scales.append(int(int(s.split(" ")[0])/cell_width))
                else: 
                    scales.append(int(s.split(" ")[0]))

            print("scales: ", scales)
            if feature == "w_hog":
                feats = hog.w_hog_feat_vec(input_im,
                                   scales)

            elif feature == "glcm":
                feats = glcm.glcm_feat_vec(input_im,
                                   scales,
                                   prop=params["prop"],
                                   distances=params["distances"],
                                   angles=params["angles"],
                                   stat=params["stat"],
                                   smooth_factor=params["smooth_factor"],
                                   levels=params["levels"])

            elif feature == "pantex":
                feats = glcm.pantex_feat_vec(input_im,
                                     scales)

            elif feature == "lbp":
                feats = lbp.lbp_feat_vec(input_im,
                                 scales,
                                 method=params["method"],
                                 radius=params["radius"],
                                 n_points=params["n_points"],
                                 hist=params["hist"],
                                 stat=params["stat"],
                                 smooth_factor=params["smooth_factor"],
                                 levels=params["levels"])

            elif feature == "lac":
                feats = lac.lac_feat_vec(input_im,
                                 scales,
                                 box_size=params["box_size"],
                                 slide_style=params["slide_style"],
                                 lac_type=params["lac_type"],
                                 smooth_factor=params["smooth_factor"],
                                 levels=params["levels"])
            elif feature == "sift":
                pass

            elif feature == "gabor":
                pass

            feature_vector.extend(feats)
        feature_vector.append(label)
        feature_vectors.append(feature_vector)
    return feature_vectors


def __process(execution_parameters_list):
    pp = pprint.PrettyPrinter(indent=4,width=5)
    feature_vector = []
    label = None
    for execution_parameters in execution_parameters_list:
        #print("---------------------- Computing Feature ------------------------")
        #pp.pprint(execution_parameters)
        #print()
        if label == None:
            label = execution_parameters["label"]
        input_im = execution_parameters["input"]
        # label = execution_parameters["label"]
        feature = execution_parameters["feature"]
        params = execution_parameters["params"]
        orig_scales = params["scales"]

        # scales can be specified in pixels or meters. 
        # depending on input, convert to pixels.
        # the scales that are used as input to the feature
        # functions must always be specified in pixels
        scales = []
        for s in orig_scales:
            if "meters" in s:
                if "wv3" in input_im:
                    cell_width = 0.31
                if "wv2" in input_im or "ge1" in input_im:
                    cell_width = 0.46
                else:
                    cell_width = 0.46
                scales.append(int(int(s.split(" ")[0])/cell_width))
            else: 
                scales.append(int(s.split(" ")[0]))

        if feature == "w_hog":
            feats = hog.w_hog_feat_vec(input_im,
                               scales)

        elif feature == "glcm":
            feats = glcm.glcm_feat_vec(input_im,
                                   scales,
                                   prop=params["prop"],
                                   distances=params["distances"],
                                   angles=params["angles"],
                                   stat=params["stat"],
                                   smooth_factor=params["smooth_factor"],
                                   levels=params["levels"])

        elif feature == "pantex":
            feats = glcm.pantex_feat_vec(input_im,
                                 scales)

        elif feature == "lbp":
            feats = lbp.lbp_feat_vec(input_im,
                                 scales,
                                 method=params["method"],
                                 radius=params["radius"],
                                 n_points=params["n_points"],
                                 hist=params["hist"],
                                 stat=params["stat"],
                                 smooth_factor=params["smooth_factor"],
                                 levels=params["levels"])

        elif feature == "lac":
            feats = lac.lac_feat_vec(input_im,
                             scales,
                             box_size=params["box_size"],
                             slide_style=params["slide_style"],
                             lac_type=params["lac_type"],
                             smooth_factor=params["smooth_factor"],
                             levels=params["levels"])
        elif feature == "sift":
            feats = hist.hist_feat_vec(input_im[:-4] + "_SIFT_codewords.tif", 
                                       scales,
                                       output=None,
                                       numbins=32)
        elif feature == "gabor":
            feats = hist.hist_feat_vec(input_im[:-4] + "_gabor_codeword_ids.tif",
                                       scales,
                                       output=None,
                                       numbins=32)

        feature_vector.extend(feats)
    feature_vector.append(label)
    return feature_vector
