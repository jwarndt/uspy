import time
import math
import os
import random

from sklearn.cluster import KMeans
import skimage
from scipy import ndimage as ndi
import numpy as np
from osgeo import gdal
from osgeo import osr
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
from skimage._shared._warnings import expected_warnings

from ..utilities.io import *
from .hist import *

def convolve_filters(image_arr, filterbank, mean_var=True):
    """
    Parameters:
    ------------
    image_arr: ndarray (2D)
        the input image array
    filterbank: list
        a list of filters to convolve over the image
    mean_var: boolean
        if true, then the output of each filter convolution is 
        the mean and variance of the responses for the entire array.
        if false, then the output is of each filter is a 3D array that
        contains the response of each filter, at each pixel.
    """
    image_arr = image_arr.astype(np.float32)
    if mean_var:
        feats = np.zeros(shape=(len(filterbank*2),), dtype=np.float32)
    else:
        feats = np.zeros(shape=(len(filterbank), image_arr.shape[0], image_arr.shape[1]), dtype=np.int16)
    count = 0
    for kernel in filterbank:
        filtered = ndi.convolve(image_arr, kernel, mode='wrap').astype(np.int16) # the convolution is done with float32s but cast the responses to int16 afterwards (for memory/filesize reasons)
        if mean_var:
            feats[count] = filtered.mean()
            count+=1
            feats[count] = filtered.var()
            count+=1
        else:
            feats[count] = filtered
            count+=1
    return feats

def convolve_filters_p(in_data):
    image_arr = in_data[0].astype(np.float32)
    feats = np.zeros(shape=(image_arr.shape[0], image_arr.shape[1]), dtype=np.int16)
    filtered = ndi.convolve(image_arr, in_data[1], mode='wrap').astype(np.int16)
    return [in_data[2], filtered]

def get_default_filter_bank():
    filterbank = []
    filterbank.extend(create_filter_bank(thetas=[0, math.pi/3., math.pi/6., math.pi/2., (2*math.pi)/3., (5*math.pi)/6.], sigmas=[2, 3.5, 7], frequencies=[0.1]))
    filterbank.extend(create_filter_bank(thetas=[0],  sigmas=[2, 7], frequencies=[1]))
    filterbank.append(filterbank[-2]*-1)
    filterbank.append(filterbank[-2]*-1)
    filterbank.extend(create_filter_bank(thetas=[math.pi/3., math.pi/6.], sigmas=[1.5], frequencies=[0.9]))
    return filterbank

def create_filter_bank(thetas=[0, math.pi/3., math.pi/6., math.pi/2., (2*math.pi)/3., (5*math.pi)/6.], sigmas=[1, 3], frequencies=[0.1, 0.5]):
    filterbank = []
    for theta in thetas:
        #theta = theta / float(len(thetas)) * math.pi
        for sigma in sigmas:
            for frequency in frequencies:
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                filterbank.append(kernel)
    return filterbank

def compute_filter_responses(image_name, filterbank, mean_var=False):
    # for BOVW feature descriptors
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ds = None
    
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    
    image = np.moveaxis(image, 0, -1)
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    
    out_image = convolve_filters(image, filterbank, mean_var)
    output = image_name[:-4] + "_gabor_responses.tif"
    write_geotiff(output, out_image, geotran, out_srs_wkt)
    return out_image

def compute_filter_responses_p(files_and_filterbanks):
    ds = gdal.Open(files_and_filterbanks[0])
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ds = None
    
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    
    image = np.moveaxis(image, 0, -1)
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    
    out_image = convolve_filters(image, files_and_filterbanks[1], mean_var=False)
    output = files_and_filterbanks[0][:-4] + "_gabor_responses.tif"
    write_geotiff(output, out_image, geotran, out_srs_wkt)


def get_rand_gabor_feats(image_files, sample_num=100000):
    gabor_feats = []
    rand_file_idx = [random.randint(0, len(image_files)-1) for n in range(sample_num)]
    rand_file_idx.sort()
    count = 0
    cur_file = None
    while len(gabor_feats) < sample_num:
        if cur_file != image_files[rand_file_idx[count]]:
            ds = None
            gabor_im = None
            cur_file = image_files[rand_file_idx[count]]
            ds = gdal.Open(cur_file)
            gabor_im = ds.ReadAsArray()
        rand_gabor_feat = get_rand_image_feat(gabor_im)
        gabor_feats.append(rand_gabor_feat) # read a random sift feature from the array and append only its description to keypoints
        count+=1
    return gabor_feats


def get_rand_image_feat(image_array):
    if len(image_array.shape) == 2:
        rand_row = random.randint(0, image_array.shape[0]-1)
        rand_col = random.randint(0, image_array.shape[1]-1)
        rand_feat = image_array[rand_row, rand_col]
    if len(image_array.shape) == 3:
        rand_row = random.randint(0, image_array.shape[1]-1)
        rand_col = random.randint(0, image_array.shape[2]-1)
        rand_feat = image_array[:,rand_row,rand_col]
    return rand_feat


def create_gabor_codebook(gabor_images, out_dir, n_clusters=32, rand_samp_num=100000):
    out_codebook_file = os.path.join(out_dir, "gabor_kmeans_codebook.dat")
    gabor_feats = get_rand_gabor_feats(gabor_images, rand_samp_num)
    codebook = KMeans(n_clusters=n_clusters, random_state=42).fit(gabor_feats)
    feature_vector_len = np.array([[len(gabor_feats[0]) for n in range(len(gabor_feats[0]))]]) # prepend a row of values. each value is the length of the feature vector. (for easy reading and reshaping later on)
    cluster_cents = np.concatenate((feature_vector_len, codebook.cluster_centers_))
    cluster_cents.tofile(out_codebook_file)
    return codebook


def restore_codebook(codebook_filename):
    """
    reads the cluster_centers from the codebook file
    and restores a kmeans model to use for prediction
    """
    cluster_centers = np.fromfile(codebook_filename)
    cluster_centers = cluster_centers.reshape(-1, int(cluster_centers[0])) # do this because the first value of the first row for the codebook was filled with values corresponding to the length of the feature vectors.
    cluster_centers = cluster_centers[1:,:] # now remove the row with the feature vector length values
    n_clusters = len(cluster_centers)
    codebook = KMeans(n_clusters=n_clusters, random_state=42)
    codebook.cluster_centers_ = cluster_centers
    return codebook

def assign_codeword(gabor_feat_files, codebook_file):
    """
    Restores a previous codebook calculated using K-means on the feature 
    vectors. The codebook consists of cluster centers. Each
    feature vector is assigned a codeword-id corresponding to
    the closest cluster center in the codebook. The codeword-id is
    prepended to the feature vectors for an output of shape
    (n_samples, n_features). n_features has length corresponding to the number
    of gabor filters and is composed of a 1)codeword-id,
    2) image_col, 3) image_row, 4) geo_x, 5) geo_y, and the rest of features that
    make up the gabor response description.
    
    Parameters:
    -----------
    siftdat_dir: string
        the directory name where the .siftdat files are located
    codebook_file: string
        the file name corresponding to the codebook of k-means cluster centers
    
    Returns:
    --------
    None
    """
    # gabor_feat_files = [os.path.join(gabor_feats_image_dir,n) for n in os.listdir(gabor_feats_image_dir) if n[-19:] == "gabor_responses.tif"]
    codebook = restore_codebook(codebook_file) # get the cluster centers from kmeans. (an ndarray)
    for n in gabor_feat_files:

        orig_im_basename = n[:-20]
        
        ds = gdal.Open(n)
        image = ds.ReadAsArray()
        geotran = ds.GetGeoTransform()
        ds = None

        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(4326)
        out_srs_wkt = out_srs.ExportToWkt()
        
        feats = image.reshape(image.shape[0],-1)
        feats = feats.transpose()
        pred = codebook.predict(feats)
        out = pred.reshape(image.shape[1],image.shape[2])

        out_im_name = orig_im_basename + "_gabor_codeword_ids.tif"
        write_geotiff(out_im_name, out, geotran, out_srs_wkt)

def create_codeword_im(gabor_response_im, codebook_file):
    """
    this is used for inference with gabor feature
    """
    codebook = restore_codebook(codebook_file)
    feats = gabor_response_im.reshape(gabor_response_im.shape[0],-1)
    feats = feats.transpose()
    pred = codebook.predict(feats)
    out = pred.reshape(gabor_response_im.shape[1],gabor_response_im.shape[2])
    return out
    
def create_gabor_codeword_images(image_dirs, out_gabor_dir, n_clusters=32, rand_samp_num=100000):
    bank = create_filter_bank([0, math.pi/3., math.pi/6., math.pi/2., (2*math.pi)/3., (5*math.pi)/6.], [1, 3], [0.1, 0.5])
    
    image_names = []
    for i in image_dirs:
        image_names.extend([os.path.join(i, n) for n in os.listdir(i) if n[-4:] == ".tif"])
    
    # easily parallelized
    for im_filename in image_names:
        print("processing: ", im_filename)
        compute_filter_responses(im_filename, out_gabor_dir, bank, mean_var=False)
    
    # don't parallelize
    create_gabor_codebook(out_gabor_dir, n_clusters=n_clusters, rand_samp_num=rand_samp_num)
    
    # can parallelize
    assign_codeword(out_gabor_dir, os.path.join(out_gabor_dir, "gabor_kmeans_codebook.dat"))

# to be used on gabor codeword ID images
def gabor_bovw_feature(image_name, block, scale, output=None):
    return hist_feature(image_name, block, scale, output=None)
    
# to be used on regular RGB images
def gabor_feature(image_name, block, scale, output=None, thetas=[0, math.pi/3., math.pi/6., math.pi/2., (2*math.pi)/3., (5*math.pi)/6.], sigmas=[1, 3], frequencies=[0.1, 0.5]):
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]
    
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    out_cell_width = block * cell_width
    out_cell_height = block * cell_height
    
    ds = None
    
    filterbank = create_filter_bank(thetas, sigmas, frequencies)

    out_image = []
    for i in range(0, image.shape[0], block):
        outrow = []
        for j in range(0, image.shape[1], block):
            block_arr = image[i:i+block,j:j+block]
            center_i = int(i+block/2)
            center_j = int(j+block/2)
            if center_i-int(scale/2) < 0:
                top = 0
            else:
                top = center_i-int(scale/2)
            if center_i+int(scale/2) > image.shape[0]:
                bot = image.shape[0]
            else:
                bot = center_i+int(scale/2)
            if center_j-int(scale/2) < 0:
                left = 0
            else:
                left = center_j-int(scale/2)
            if center_j+int(scale/2) > image.shape[1]:
                right = image.shape[1]
            else:
                right = center_j+int(scale/2)
            if block%2 != 0 and scale%2 == 0: # make sure the scale window is the correct size for the block
                scale_arr = image[top:bot,left:right]
            else:
                scale_arr = image[top:bot+1,left:right+1]

            # this convolve function returns mean and variance of gabor filter
            # responses over the entire scale array
            out = convolve_filters(scale_arr, filterbank) # could do interesting things with density=True
            outrow.append(out)
        out_image.append(outrow)
    out_image = np.array(out_image)
    out_image = np.moveaxis(out_image, -1, 0)
    if output:
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, out_image, out_geotran, out_srs_wkt)
    return np.array(out_image)

def gabor_feat_vec(image_name,
                   scales,
                   mean_var=False,
                   output=None,
                   numbins=32,
                   thetas=[0, math.pi/3., math.pi/6., math.pi/2., (2*math.pi)/3., (5*math.pi)/6.],
                   sigmas=[1, 3],
                   frequencies=[0.1, 0.5]):
    if mean_var: # assumes regular RGB image
        ds = gdal.Open(image_name)
        image = ds.ReadAsArray()
        geotran = ds.GetGeoTransform()
        ulx = geotran[0]
        uly = geotran[3]
        cell_width = geotran[1]
        cell_height = geotran[5]
        ds = None

        filterbank = create_filter_bank(thetas, sigmas, frequencies)

        image = np.moveaxis(image, 0, -1)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(rgb2gray(image))
    
        # center pixel location
        center_i = int(image.shape[0] / 2.)
        center_j = int(image.shape[1] / 2.)

        out = []
        for s in scales:
            # convert meters to pixel counts
            n_pixels = s # number of pixels for the scale
            if center_i-int(n_pixels/2) < 0:
                top = 0
            else:
                top = center_i-int(n_pixels/2.)
            if center_i+int(n_pixels/2.) > image.shape[0]:
                bot = image.shape[0]
            else:
                bot = center_i+int(n_pixels/2.)
            if center_j-int(n_pixels/2.) < 0:
                left = 0
            else:
                left = center_j-int(n_pixels/2.)
            if center_j+int(n_pixels/2.) > image.shape[1]:
                right = image.shape[1]
            else:
                right = center_j+int(n_pixels/2.)
            
            scale_arr = image[top:bot+1,left:right+1]
            feat_vec = convolve_filters(scale_arr, filterbank, mean_var)

            out.append(feat_vec)
        return np.array(feat_vec).flatten()
    else: # assumes input image is a codeword image
        return hist_feat_vec(image_name, scales, output, numbins)