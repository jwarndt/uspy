import os

from nmapy.classification import *

if __name__ == "__main__":
    
    #in_image = "C:/Users/4ja/data/neighborhood_mapping/imagery_orig/dakar/dakar_e_wv2_05272018.tif"
    #in_image = "C:/Users/4ja/data/neighborhood_mapping/imagery_orig/johannesburg/johannesburg_cw_wv2_tiles/johannesburg_cw_wv2_tiles/johannesburg_cw_wv2_000006000_000072000_00038.tif"
    in_image = "/mnt/GATES/FSEG/ethiopia/addis_ababa/imagery/addis_ababa_c_wv3_12142017.tif"
    #out = "C:/Users/4ja/data/neighborhood_mapping/imagery_orig/dakar/classification/classified_dakar_e_wv2.tif"
    out = "/mnt/GATES/UserDirs/4ja/data/results/addis_ababa/classified_addis_ababa_c_wv3.tif"
    block = "30 meters"
    tile_size = 5720
    #tile_size = None

    #model_dir = "C:/Users/4ja/data/neighborhood_mapping/image_chips_final/dakar"
    model_dir = "/mnt/GATES/UserDirs/4ja/data/image_chips_final/dar_es_salaam"
    classifier_file = os.path.join(model_dir, "svm_model.pkl")
    scaler_file = os.path.join(model_dir, "data_scaler.pkl")
    feature_file = os.path.join(model_dir, "feature_params.pkl")
    sift_codebook_file = os.path.join(model_dir, "sift_kmeans_codebook.dat")
    gabor_codebook_file = os.path.join(model_dir, "gabor_kmeans_codebook.dat")

    in_images = ["/mnt/GATES/FSEG/tanzania/dar_es_salaam/imagery/dar_es_salaam_e_ge1_06212017.tif", "/mnt/GATES/FSEG//tanzania/dar_es_salaam/imagery/dar_es_salaam_w_ge1_06212017.tif"]
    outs = ["/mnt/GATES/UserDirs/4ja/data/results/dar_es_salaam/classified_dar_es_salaam_e_ge1.tif", "/mnt/GATES/UserDirs/4ja/data/results/dar_es_salaam/classified_dar_es_salaam_w_ge1.tif"]
    i = 0
    while i < len(in_images):
        in_image = in_images[i]
        out = outs[i]
        classify.classify_pixels(in_image,
                                 out,
                                 block,
                                 classifier_file,
                                 scaler_file,
                                 feature_file,
                                 sift_codebook_file,
                                 gabor_codebook_file,
                                 tile_size=tile_size,
                                 n_data_chunks=4,
                                 n_jobs=4)
        i+=1