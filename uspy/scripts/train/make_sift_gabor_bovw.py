from nmapy.classification import dataset
from nmapy.features import *

from multiprocessing import Pool

N_JOBS = 16


def run_gabor(dataset_dir):
    #n = dataset.Neighborhoods()
    #n.load_dataset(dataset_dir)
    #train, test = n.split_train_test(0.20, True)

    print(" ---- GABOR ----")
    train = dataset.Training()
    test = dataset.Test()
    other_test = dataset.Test() # for combined training
    train.load_dataset(dataset_dir + "/train")
    test.load_dataset(dataset_dir + "/test/jhb")
    other_test.load_dataset(dataset_dir + "/test/des")

    #print(" ... getting filter bank")   
    #bank = gabor.get_default_filter_bank()

    # get the filenames of all images, make a nested list for parallel processing of gabor filters
    # get the filenames of all the gabor response images
    all_files_and_filterbanks = []
    all_gabor_files = []
    for i in [train, test, other_test]:
        for single_file in i.image_filenames:
            #all_files_and_filterbanks.append([single_file, bank])
            all_gabor_files.append(single_file[:-4] + "_gabor_responses.tif")
    # collect the training gabor response filenames to generate the codebook
    train_gabor_files = [f[:-4] + "_gabor_responses.tif" for f in train.image_filenames]

    #print(" ... computing filter responses")
    #p = Pool(N_JOBS)
    #p.map(gabor.compute_filter_responses_p, all_files_and_filterbanks)
    #p.close()
    #p.join()
    #print(" ... done computing filter responses")

    # create the codebook using only the training images
    print(" ... creating codebook")
    gabor.create_gabor_codebook(train_gabor_files, train.root_dir, n_clusters=32, rand_samp_num=10000)

    print(" ... assigning codewords")
    # create the codeword id images using the kmeans codebook and the gabor filter response images
    gabor.assign_codeword(all_gabor_files, os.path.join(train.root_dir, "gabor_kmeans_codebook.dat"))

def run_sift(dataset_dir):
    #n = dataset.Neighborhoods()
    #n.load_dataset(dataset_dir)
    #train, test = n.split_train_test(0.20, True)
    print("---- SIFT ---- ")
    train = dataset.Training()
    test = dataset.Test()
    other_test = dataset.Test() # for combined training
    train.load_dataset(dataset_dir + "/train")
    test.load_dataset(dataset_dir + "/test/jhb")
    other_test.load_dataset(dataset_dir + "/test/des")

    all_image_names = []
    for f in [train, test, other_test]:
        for single_file in f.image_filenames:
            all_image_names.append(single_file)
    
    print(" ... writing sift keypoint descriptions")
    # run sift on all images
    sift.write_sift_desc(all_image_names)

    print(" ... creating sift codebook")
    # create the sift codebook using only the training images
    train_sift_dat_files = [f[:-4] + ".siftdat" for f in train.image_filenames if os.path.exists(f[:-4] + ".siftdat")]
    all_sift_dat_files = [f[:-4] + ".siftdat" for f in all_image_names if os.path.exists(f[:-4] + ".siftdat")]
    sift.create_sift_codebook(train_sift_dat_files, train.root_dir, n_clusters=32, rand_samp_num=10000)

    print(" ... assigning codewords")
    # assign codewords to all sift keypoints using the codebook that was computed on the training data
    sift.assign_codeword(all_sift_dat_files, os.path.join(train.root_dir, "sift_kmeans_codebook.dat"))

    print(" ... creating codeword id images")
    # create the codeword images
    for i in all_image_names:
        sift.create_codeword_id_image(i, i[:-4] + ".siftdat")

if __name__ == "__main__":
    dataset_directory = "/mnt/GATES/UserDirs/4ja/data/image_chips_IGARSS_svms/jhb_des_combined"
    #run_gabor(dataset_directory)
    run_sift(dataset_directory)
    #dataset_directory = "/mnt/GATES/UserDirs/4ja/data/image_chips_IGARSS_svms/addis_ababa"
    #run_gabor(dataset_directory)
    #run_sift(dataset_directory)