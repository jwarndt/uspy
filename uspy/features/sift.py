import os
import random
import math

import numpy as np
from sklearn.cluster import KMeans
import skimage
from skimage.color import rgb2gray
from skimage._shared._warnings import expected_warnings
from osgeo import gdal
from osgeo import osr
import cv2

from ..utilities.stats import *
from ..utilities.io import *
from .hist import *


def __image_coord_to_geo_coord(keypoints, geotran):
    x_coor = geotran[0] + keypoints[0] * geotran[1]
    y_coor = geotran[3] + keypoints[1] * geotran[5]
    return (x_coor, y_coor)

def get_sift_keypoint_desc(image, geotran):
    """
    This is for inference. it's basically the same
    as the write_sift_keypoint_desc except that it doesn't write
    an output file and that it expects an array as a parameter instead of
    an image filename
    """
    image_chunk_size = 2500
    all_sift_des = [] # aggregated SIFT des from all the chunks of the image.
    i = 0
    while i < image.shape[0]:
        if i+image_chunk_size > image.shape[0]:
            endrow = image.shape[0]
        else:
            endrow = i+image_chunk_size
        j = 0
        while j < image.shape[1]:
            if j+image_chunk_size > image.shape[1]:
                endcol = image.shape[1]
            else:
                endcol = j+image_chunk_size
            image_subset = image[i:endrow, j:endcol]
            # find keypoints and compute descriptions
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image_subset, None)
            # image coordinates here are set relative to the whole image, not the chunk
            kp_image_coords = [(kp[n].pt[0] + j, kp[n].pt[1] + i)  for n in range(len(kp))]
            if len(kp) > 0:
                kp_geo_coords = np.apply_along_axis(__image_coord_to_geo_coord, 1, kp_image_coords, geotran)
                kp_coords = np.concatenate((kp_image_coords, kp_geo_coords), axis=1)
                sift_des = np.concatenate((kp_coords, des), axis=1)
                
                if len(all_sift_des) == 0:
                    all_sift_des = sift_des
                else:
                    all_sift_des = np.concatenate((all_sift_des, sift_des))

            
            j+=image_chunk_size
        i+=image_chunk_size
    if len(all_sift_des) != 0:
        #out_dat_file = os.path.join(image_name[:-4] + ".siftdat")
        #all_sift_des.tofile(out_dat_file)
        return all_sift_des

def write_sift_keypoint_desc(image_name):
    """
    Writes sift keypoint descriptions to .siftdat binary files
    Each of these files contains sift feature vectors of length 128. 
    Four additional columns are prepended onto the sift feature vectors.
    These additional columns are image and geographic coordinates of the
    sift feature vectors calculate from the image.
    
    Parameters:
    -----------
    
    Returns:
    --------
    None
    """
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]

    ds = None
    image = np.moveaxis(image, 0, -1)
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    
    image_chunk_size = 2500
    all_sift_des = [] # aggregated SIFT des from all the chunks of the image.
    i = 0
    while i < image.shape[0]:
        if i+image_chunk_size > image.shape[0]:
            endrow = image.shape[0]
        else:
            endrow = i+image_chunk_size
        j = 0
        while j < image.shape[1]:
            if j+image_chunk_size > image.shape[1]:
                endcol = image.shape[1]
            else:
                endcol = j+image_chunk_size
            image_subset = image[i:endrow, j:endcol]
            # find keypoints and compute descriptions
            sift = cv2.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(image_subset, None)
            # image coordinates here are set relative to the whole image, not the chunk
            kp_image_coords = [(kp[n].pt[0] + j, kp[n].pt[1] + i)  for n in range(len(kp))]
            if len(kp) > 0:
                kp_geo_coords = np.apply_along_axis(__image_coord_to_geo_coord, 1, kp_image_coords, geotran)
                kp_coords = np.concatenate((kp_image_coords, kp_geo_coords), axis=1)
                sift_des = np.concatenate((kp_coords, des), axis=1)
                
                if len(all_sift_des) == 0:
                    all_sift_des = sift_des
                else:
                    all_sift_des = np.concatenate((all_sift_des, sift_des))

            
            j+=image_chunk_size
        i+=image_chunk_size
    if len(all_sift_des) != 0:
        out_dat_file = os.path.join(image_name[:-4] + ".siftdat")
        all_sift_des.tofile(out_dat_file)
        return all_sift_des

def get_rand_sift_feats(siftdat_files, sample_num=100000):
    keypoints = []
    rand_file_idx = [random.randint(0, len(siftdat_files)-1) for n in range(sample_num)]
    rand_file_idx.sort()
    count = 0
    cur_file = None
    while len(keypoints) < sample_num:
        if cur_file != siftdat_files[rand_file_idx[count]]:
            cur_file = siftdat_files[rand_file_idx[count]]
            siftdat = np.fromfile(cur_file) # retrieve and open a random file
            siftdat = siftdat.reshape(-1,132) # reshape into a 2D array of (n_samples, n_features). 128 features plus and x and y coord for a total of 130 features. 
        rand_kp = random.randint(0, len(siftdat)-1)
        keypoints.append(siftdat[rand_kp,4:]) # read a random sift feature from the array and append only its description to keypoints
        count+=1
    return keypoints

def create_sift_codebook(siftdat_files, out_dir, n_clusters=32, rand_samp_num=100000):
    """
    creates a codebook out of siftdat files (i.e. the sift keypoint description files)

    Returns:
    --------
    codebook: sklearn kmeans class
        The cluster centers obtained by running kmeans on randomly
        sampled sift keypoint descriptions.
        Each cluster center is a vector of 128 features
    """
    out_codebook_file = os.path.join(out_dir, 'sift_kmeans_codebook.dat')
    sift_feats = get_rand_sift_feats(siftdat_files, rand_samp_num)
    codebook = KMeans(n_clusters=n_clusters, random_state=42).fit(sift_feats)
    codebook.cluster_centers_.tofile(out_codebook_file)
    return codebook

def write_sift_desc(image_names):
    """
    Parameters:
    -----------
    image_dirs: list
        a list of directories holding the original images that sift keypoints will be
        computed for
    out_dir: string
        the output directory where the sift keypoint description files (.siftdat) will be saved
    
    Returns:
    --------
    None
    """
    # image_names = []
    # for im_dir in image_dirs:
    #     image_names.extend([os.path.join(im_dir,n) for n in os.listdir(im_dir) if n[-4:] == ".tif"])
    for n in image_names:
        write_sift_keypoint_desc(n)

def restore_codebook(codebook_filename):
    """
    reads the cluster_centers from the codebook file
    and restores a kmeans model to use for prediction
    """
    cluster_centers = np.fromfile(codebook_filename).reshape(-1,128)
    n_clusters = len(cluster_centers)
    codebook = KMeans(n_clusters=n_clusters, random_state=42)
    codebook.cluster_centers_ = cluster_centers
    return codebook

def apply_codeword(siftdat, codebook_file):
    """
    This is the same as assign_codeword but it's for inference
    """
    codebook = restore_codebook(codebook_file) # get the cluster centers from kmeans. (an ndarray)
    #siftdat = np.fromfile(n).reshape(-1, 132)
    coords = siftdat[:,:4]
    feats = siftdat[:,4:]
    pred = codebook.predict(feats) # assign codeword ids to each sift feature vector
    pred = pred.reshape(-1, 1)
    pred = np.concatenate((pred, siftdat), axis=1) # concatenate the codeword id onto the original sift feature vector
    return pred

def assign_codeword(siftdat_files, codebook_file):
    """
    Restores a previous codebook calculated using K-means on the sift
    feature vectors. The codebook consists of cluster centers. Each
    sift feature vector is assigned a codeword-id corresponding to
    the closest cluster center in the codebook. The codeword-id is
    prepended to the sift feature vectors for an output of shape
    (n_samples, 133). Length of 133 is composed of a 1)codeword-id,
    2) image_col, 3) image_row, 4) geo_x, 5) geo_y, and 128 features that
    make up the SIFT keypoint description.
    
    Parameters:
    -----------
    siftdat_dir: string
        the directory name where the .siftdat files are located
    codebook_file: string
        the file name corresponding to the codebook of k-means cluster centers
    
    Returns:
    --------
    all_predictions: ndarray with shape=(n_samples, 133)
    """
    #siftdat_files = [os.path.join(siftdat_dir, n) for n in os.listdir(siftdat_dir) if n[-8:] == ".siftdat"]
    codebook = restore_codebook(codebook_file) # get the cluster centers from kmeans. (an ndarray)
    all_predictions = []
    orig_im_basename = ""
    for n in siftdat_files:
        orig_im_basename = n[:-20]
        siftdat = np.fromfile(n).reshape(-1, 132)
        coords = siftdat[:,:4]
        feats = siftdat[:,4:]
        pred = codebook.predict(feats) # assign codeword ids to each sift feature vector
        pred = pred.reshape(-1, 1)
        pred = np.concatenate((pred, siftdat), axis=1) # concatenate the codeword id onto the original sift feature vector
        pred.tofile(n)
    return pred

def get_codeword_array(orig_im, sift_keypoint_desc):
    """
    the same as create_codeword_id_image() but for inference
    """
    out_image = np.ones(shape=(orig_im.shape[0], orig_im.shape[1]), dtype=np.uint8) * 255
    for sample in sift_keypoint_desc:
        codeword_id = sample[0]
        row = math.floor(sample[2])
        col = math.floor(sample[1])
        out_image[row,col] = int(codeword_id)
    return out_image
    
def create_codeword_id_image(image_name, codeword_id_file=None):
    """
    creates a one band tif image where each pixel has an integer value
    that is the codeword-id. valid codeword-ids range from 0 to 31 if there are
    32 clusters.
    A codeword-id of 255 indicates a pixel location where SIFT did
    not assign as a keypoint.
    
    Parameters:
    -----------
    
    Returns:
    --------
    None
    """
    # get the relevant image dimension information for writing output
    ds = gdal.Open(image_name)
    geotran = ds.GetGeoTransform()
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    ds = None
    
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    
    out_image = np.ones(shape=(2, rows, cols), dtype=np.uint8) * 255
    if os.path.exists(codeword_id_file):
        siftdat = np.fromfile(codeword_id_file).reshape(-1, 133)
        for sample in siftdat:
            codeword_id = sample[0]
            row = math.floor(sample[2])
            col = math.floor(sample[1])
            out_image[0,row,col] = int(codeword_id)
            out_image[1,row,col] = 1
    # out_image[1] = np.where(out_image[1] != 1, 0, 1)
    full_out_im_name = image_name[:-4] + "_SIFT_codewords.tif"

    # out_im_name1 = os.path.join(outdir, "sift_keypoint_locs.tif")
    write_geotiff(full_out_im_name, out_image[0], geotran, out_srs_wkt)
    # write_geotiff(out_im_name1, out_image[1], geotran, out_srs_wkt)
    
def create_sift_codeword_images(image_names, out_dir, n_clusters=32, rand_samp_num=100000):
    # get all the images in the image_dir and write sift keypoint description files
    write_sift_desc(image_names)

    # now convert to the .siftdat image names
    siftdats = []
    for f in image_names:
        siftdats.append(f[:-4] + ".siftdat")
    
    # use k means clustering to create the codebook. create the codebook
    # from the siftdat files that were computed in the write_sift_desc() function
    # By default, the name of the codebook is "sift_kmeans_codebook.dat"
    create_sift_codebook(siftdats, out_dir, n_clusters=n_clusters, rand_samp_num=rand_samp_num)
    
    # use the codebook to assign codeword-ids to the individual chunked up sift feature vector files.
    assign_codeword(siftdats, os.path.join(out_dir, "sift_kmeans_codebook.dat"))
    
    # use the aggregated sift feature vector file (it now has codeword ids associated with every feature vector)
    # and create images with codeword-id pixels. write the output codeword images to the out_sift_dir. The
    # output sift images have a basename of the input image plus "_SIFT_codewords.tif"
    for i in image_names:
        create_codeword_id_image(i, i[:-4] + ".siftdat")

def sift_feature(image_name, block, scale, output=None):
    # BOVW uses computes a histogram of the codewords in the give
    # scale. The codewords correspond to the id of the closest
    # cluster center in the codebook.
    return hist_feature(image_name, block, scale, output=None)

def sift_feat_vec(image_name, scales, output=None, numbins=32):
    return hist_feat_vec(image_name, scales, output, numbins)