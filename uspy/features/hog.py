import skimage
from skimage.feature import hog as skimagehog
from skimage import filters
from skimage.color import rgb2gray
from scipy.signal import savgol_filter, find_peaks
from skimage._shared._warnings import expected_warnings
from scipy.stats import entropy
import cv2
import numpy as np
from osgeo import gdal
from osgeo import osr

from ..utilities.stats import *
from ..utilities.io import *

def hog_feature(image_name, block, scale, output=None, stat=None):
    """
    Parameters:
    ----------
    image_name: str
    block: int
    scale: int
    
    Returns:
    --------
    out_image: 3D ndarray
    """
    assert(type(block) != None and type(scale) != None)
    assert(type(stat) == list or stat == None)


    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    in_cell_width = geotran[1]
    in_cell_height = geotran[5]
    ds = None

    # block and scale parameters are in meters
    # convert meters to image space (number of pixels)
    # the conversion is very crude at the moment, should really
    # be using projected data
    if "wv2" in image_name:
        cell_width = 0.46
    if "wv3" in image_name:
        cell_width = 0.31
    # in number of pixels relative to the input data GSD
    block = int(block / cell_width)
    scale = int(scale / cell_width)

    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    out_cell_width = block * in_cell_width
    out_cell_height = block * in_cell_height

    image = np.moveaxis(image, 0, -1) # expects an image in rows, columns, channels

    out_image = []
    for i in range(0, image.shape[0], block):
        outrow = []
        for j in range(0, image.shape[1], block):
            center_i = int(i + block/2.)
            center_j = int(j + block/2.)
            if center_i-int(scale/2.) < 0:
                top = 0
            else:
                top = center_i-int(scale/2.)
            if center_i+int(scale/2.) > image.shape[0]:
                bot = image.shape[0]
            else:
                bot = center_i+int(scale/2.)
            if center_j-int(scale/2.) < 0:
                left = 0
            else:
                left = center_j-int(scale/2.)
            if center_j+int(scale/2.) > image.shape[1]:
                right = image.shape[1]
            else:
                right = center_j+int(scale/2.)

            scale_arr = image[top:bot+1,left:right+1] 

            fd = skimagehog(scale_arr, orientations=8, pixels_per_cell=(scale_arr.shape[0], scale_arr.shape[1]), cells_per_block=(1, 1), multichannel=True, feature_vector=False, block_norm='L2-Hys')
            outrow.append(fd.flatten())
        out_image.append(outrow)
    out_arr = np.moveaxis(out_image, -1, 0)
    if output:
        if stat:
            out_arr = calc_stats(out_arr, stat, 0)
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, out_arr, out_geotran, out_srs_wkt)
    else:
        if stat:
            out_arr = calc_stats(out_arr, stat, 0)
    return np.array(out_arr)

def w_hog_feature(image_name, block, scale, output=None):
    assert(type(block) != None and type(scale) != None)

    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    in_cell_width = geotran[1]
    in_cell_height = geotran[5]
    ds = None

    # block and scale parameters are in meters
    # convert meters to image space (number of pixels)
    # the conversion is very crude at the moment, should really
    # be using projected data
    if "wv2" in image_name:
        cell_width = 0.46
    if "wv3" in image_name:
        cell_width = 0.31
    # in number of pixels relative to the input data GSD
    block = int(block / cell_width)
    scale = int(scale / cell_width)

    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    out_cell_width = block * in_cell_width
    out_cell_height = block * in_cell_height

    image = np.moveaxis(image, 0, -1) # expects an image in rows, columns, channels
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    mag, ang = __calc_mag_ang(image)
    mag = mag / 1000. # scale the magnitudes back
    ang = ang % 180 # move orientations to between 0 and 180

    out_image = []
    for i in range(0, image.shape[0], block):
        outrow = []
        for j in range(0, image.shape[1], block):
            center_i = int(i + block/2.)
            center_j = int(j + block/2.)
            if center_i-int(scale/2.) < 0:
                top = 0
            else:
                top = center_i-int(scale/2.)
            if center_i+int(scale/2.) > image.shape[0]:
                bot = image.shape[0]
            else:
                bot = center_i+int(scale/2.)
            if center_j-int(scale/2.) < 0:
                left = 0
            else:
                left = center_j-int(scale/2.)
            if center_j+int(scale/2.) > image.shape[1]:
                right = image.shape[1]
            else:
                right = center_j+int(scale/2.)

            feat_vec = __weighted_hist_feature(mag[top:bot+1,left:right+1], ang[top:bot+1,left:right+1])
            outrow.append(feat_vec)
        out_image.append(outrow)
    out_image = np.moveaxis(out_image, -1, 0)
    if output:
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, out_image, out_geotran, out_srs_wkt)
    return np.array(out_image)


def hog_feat_vec(image_name, scales, output=None, stat=None):
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]
    ds = None
    
    image = np.moveaxis(image, 0, -1) # expects an image in rows, columns, channels

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
            top = center_i-int(n_pixels/2)
        if center_i+int(n_pixels/2) > image.shape[0]:
            bot = image.shape[0]
        else:
            bot = center_i+int(n_pixels/2)
        if center_j-int(n_pixels/2) < 0:
            left = 0
        else:
            left = center_j-int(n_pixels/2)
        if center_j+int(n_pixels/2) > image.shape[1]:
            right = image.shape[1]
        else:
            right = center_j+int(n_pixels/2)
        
        scale_arr = image[top:bot+1,left:right+1]
        feat_vec = skimagehog(scale_arr, orientations=8, pixels_per_cell=(scale_arr.shape[0], scale_arr.shape[1]), cells_per_block=(1, 1), multichannel=True, feature_vector=True, block_norm='L2-Hys')
        if stat:
            feat_vec = calc_stats(feat_vec, stat, 0)
        out.append(feat_vec)
    return np.array(out).flatten()

# [1] S. Kumar, and M. Herbert, Discriminative Random Fields
def w_hog_feat_vec(image_name, scales):
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]
    ds = None

    image = np.moveaxis(image, 0, -1) # expects an image in rows, columns, channels
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    mag, ang = __calc_mag_ang(image)
    mag = mag / 1000. # scale the magnitudes back
    ang = ang % 180 # move orientations to between 0 and 180
    
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
            top = center_i-int(n_pixels/2)
        if center_i+int(n_pixels/2) > image.shape[0]:
            bot = image.shape[0]
        else:
            bot = center_i+int(n_pixels/2)
        if center_j-int(n_pixels/2) < 0:
            left = 0
        else:
            left = center_j-int(n_pixels/2)
        if center_j+int(n_pixels/2) > image.shape[1]:
            right = image.shape[1]
        else:
            right = center_j+int(n_pixels/2)
        
        feat_vec = __weighted_hist_feature(mag[top:bot+1,left:right+1], ang[top:bot+1,left:right+1])
        out.append(feat_vec)
    return np.array(out).flatten()

def __weighted_hist_feature(magnitude, orientation, orders=[1,2], peak_nums=2):
    """
    Returns:
    --------
    feat_vec: ndarray
        feat_vec[0]: mean of the weigted magnitude histogram
        feat_vec[1]: heaved central moment 1
        feat_vec[2]: heaved central moment 2
        feat_vec[3]: the orientation of the peak histogram magnitude
        feat_vec[4]: the magnitude of the highest peak in the orientation magnitude histogram
        feat_vec[5]: absolute sin difference between the two peaks in the weighted magnitude histogram
        feat_vec[6]: variance of the histogram
        feat_vec[7]: minimum of the histogram
        feat_vec[8]: entropy of the histogram
    """
    mag = magnitude.flatten()
    ang = orientation.flatten()
    feat_vec = []

    # bin the magnitudes based on orientation
    hist = np.zeros(shape=(50,))
    bins = np.linspace(0,180,51)
    binrep = [] # an average for the bins
    b = 0
    while b < len(bins)-1:
        out = mag[np.where((ang >= bins[b]) & (ang < bins[b+1]))]
        hist[b] = np.sum(out)
        binrep.append((bins[b] + bins[b+1]) / 2)
        b+=1
    binrep = np.array(binrep)
    # smooth the histogram with Savitzky-Golay filter
    hist_smooth = savgol_filter(hist, 3, 1, mode='wrap')

    mu = np.mean(hist_smooth)
    feat_vec.append(mu)

    # heaved central shift moments
    zero_hist = False # this is to catch the case where the the ang/mag images do not show edges
    for p in orders:
        sumnum = 0
        sumden = 0
        for i in hist_smooth:
            step_func = 1 if i - mu > 0 else 0
            sumnum+=(i-mu)**(p+1) * step_func
            sumden+=(i-mu) * step_func
        if sumden == 0:
            print("hog division: " + str(sumnum) + " / " + str(sumden))
            print("div not possible, appending 0 to feature vector")
            #print(hist_smooth)
            feat_vec.append(0)
            zero_hist = True
        else:
            feat_vec.append(sumnum/sumden)

    if zero_hist:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        # the two highest peaks
        peak_idx, _ = find_peaks(hist_smooth)
        peak_vals = hist_smooth[peak_idx]
        peak_os = binrep[peak_idx]
        peak_vals_idx = np.argsort(peak_vals)

        # the orientations of the two highest peaks
        try:
            peak1_o = peak_os[peak_vals_idx[-1]]
            feat_vec.append(peak1_o)
            feat_vec.append(peak_vals[peak_vals_idx[-1]])
        except IndexError:
            feat_vec.append(0)
            feat_vec.append(0)
        try:
            peak2_o = peak_os[peak_vals_idx[-2]]
            # absolute sin difference of the two highest peak orientations
            peak_o_sin_diff = abs(np.sin(peak1_o - peak2_o))
            feat_vec.append(peak_o_sin_diff)
        except IndexError:
            feat_vec.append(0)
        feat_vec.append(np.var(hist_smooth))
        feat_vec.append(np.min(hist_smooth))
        feat_vec.append(entropy(hist_smooth))
        return np.array(feat_vec)

def __calc_mag_ang(im):
    dx = cv2.Sobel(np.float32(im), cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(np.float32(im), cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=1)
    return mag, ang