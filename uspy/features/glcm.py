import math

import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage._shared.utils import assert_nD
from skimage._shared._warnings import expected_warnings
import numpy as np
from osgeo import gdal
from osgeo import osr

from ..utilities.stats import *
from ..utilities.io import *

def glcm_feature(image_name,
                 block,
                 scale,
                 output=None,
                 prop=None,
                 distances=[1,2],
                 angles=[0., math.pi/6., math.pi/4., math.pi/3., math.pi/2., (2.*math.pi)/3., (3.*math.pi)/4., (5.*math.pi)/6.],
                 stat=None,
                 smooth_factor=None,
                 levels=None):

    assert(type(block) != None and type(scale) != None)
    assert(type(stat) == list or stat == None)
    if prop == None and stat == None:
        print('dimensionality too large. consider adjusting parameters')
        return None

    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    # FIXME: need a better fix for this.
    if len(image) >= 3:
        image = image[:3]
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
    
    image = np.moveaxis(image, 0, -1)
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    if smooth_factor:
        image = gaussian(image, smooth_factor)
        image = np.where(image > 1, 1, image)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(image)
            # image = skimage.img_as_ubyte(gaussian(image, smooth_factor))
    if levels:
        scale_factor = (levels-1)/255.
        image = image.astype(float)
        image *= scale_factor 
        image = np.round(image).astype(int)

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
            out = compute_glcm_feature(scale_arr, prop, distances, angles, stat, levels)

            outrow.append(out)
        out_image.append(outrow)
    out_image = np.array(out_image)
    out_image = np.moveaxis(out_image, 0, -1)
    out_image = np.moveaxis(out_image, 0, -1)
    if output:
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, out_image, out_geotran, out_srs_wkt)
    return np.array(out_image)
    
def pantex_feature(image_name, block, scale, output=None):
    return glcm_feature(image_name, block, scale, output=output, prop="contrast", stat=["min"])

def pantex_feat_vec(image_name, scales, output=None):
    return glcm_feat_vec(image_name, scales, output=output, prop="contrast", stat=["min"])

def compute_glcm_feature(in_array,
                         prop=None,
                         distances=[1,2],
                         angles=[0., math.pi/6., math.pi/4., math.pi/3., math.pi/2., (2.*math.pi)/3., (3.*math.pi)/4., (5.*math.pi)/6.],
                         stat=None,
                         levels=None):
    feat_vec = greycomatrix(in_array, distances, angles, levels)
    if prop:
        feat_vec = greycoprops(feat_vec, prop) # results 3d array [d, a] is the property at the d'th distance and a'th angle
        feat_vec = np.mean(feat_vec, axis=1) # average the property over all angles for each distance
        if stat:
            feat_vec = calc_stats(feat_vec, stat, None)
    else:
        if stat:
            feat_vec = calc_stats(feat_vec, stat, (0, 1))
        else:
            return feat_vec
    return feat_vec

# to output a single feature vector for an image chip
def glcm_feat_vec(image_name,
                  scales,
                  output=None,
                  prop=None,
                  distances=[1,2],
                  angles=[0., math.pi/6., math.pi/4., math.pi/3., math.pi/2., (2.*math.pi)/3., (3.*math.pi)/4., (5.*math.pi)/6.],
                  stat=None,
                  smooth_factor=None,
                  levels=None):

    assert(type(scales) == list), "ERROR: parameter scales must be a list"

    # scales are in meters
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    # FIXME: need a better fix for this.
    if len(image) >= 3:
        image = image[:3]
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]
    ds = None

    image = np.moveaxis(image, 0, -1)
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    if smooth_factor:
        image = gaussian(image, smooth_factor)
        image = np.where(image > 1, 1, image)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(image)
            # image = skimage.img_as_ubyte(gaussian(image, smooth_factor))
    if levels:
        scale_factor = (levels-1)/255. # need levels-1 so it rescales to number of unique gray levels equal to 'levels'
        image = image.astype(float)
        image *= scale_factor 
        image = np.round(image).astype(int)

    # center pixel location
    center_i = int(image.shape[0] / 2.)
    center_j = int(image.shape[1] / 2.)

    out = []
    for s in scales:
        # convert meters to pixel counts
        n_pixels = s # number of pixels for the scale
        if center_i-int(n_pixels/2.) < 0:
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
            right = center_j+int(n_pixels/2)
        scale_arr = image[top:bot+1,left:right+1]
        
        feat_vec = compute_glcm_feature(scale_arr, prop, distances, angles, stat, levels)

        out.append(feat_vec)
    return np.array(out).flatten()