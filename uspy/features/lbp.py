# local binary pattern
import numpy as np
import skimage
from skimage.feature import local_binary_pattern
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage._shared._warnings import expected_warnings
from osgeo import gdal
from osgeo import osr

from ..utilities.stats import *
from ..utilities.io import *

# min and max are not useful statistics for this method
def lbp_feature(image_name,
                block,
                scale,
                output=None,
                method='uniform',
                radius=1,
                n_points=4,
                hist=True,
                stat=None,
                smooth_factor=None,
                levels=None):
    assert(type(block) != None and type(scale) != None) # assert that block and scale are the same type
    assert(type(radius) == int and type(n_points) == int) # assert the radius and n_points are both integers

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

    image = np.moveaxis(image, 0, -1)
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image)) # lbp takes a gray level image
    if smooth_factor:
        image = gaussian(image, smooth_factor)
        image = np.where(image > 1, 1, image)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(image)
            # image = skimage.img_as_ubyte(gaussian(image, smooth_factor))
    if levels:
        scale_factor = levels/255.
        image = image.astype(float)
        image *= scale_factor 
        image = np.round(image).astype(int)

    lbp = local_binary_pattern(image, n_points, radius, method) # there are n_points+2 possible values for uniform
    if hist == False and stat == None:
        if output:
            write_geotiff(output, lbp, geotran, out_srs_wkt)
        return lbp

    if hist:
        if method == "uniform":
            bins = [n for n in range(n_points+2+1)] # bins for numpy histogram. there are n+1 possible patterns, plus another one because of numpy histogram functionality
        elif method == "default":
            if n_points > 5:
                bins = 32
            else:
                bins = 2**n_points

    image = lbp # image is 2D lbp (rows, cols)  
    out_image = []
    for i in range(0, image.shape[0], block):
        outrow = []
        for j in range(0, image.shape[1], block):
            center_i = int(i+block/2.)
            center_j = int(j+block/2.)
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
            if hist:
                out = np.histogram(scale_arr, bins)[0] # could do interesting things with density=True
                if stat:
                    out = calc_stats(out, stat)
            else:
                if stat:
                    out = calc_stats(scale_arr, stat)
            outrow.append(out)
        out_image.append(outrow)
    out_image = np.array(out_image)
    out_image = np.moveaxis(out_image, -1, 0)
    if output:
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, out_image, out_geotran, out_srs_wkt)
    return np.array(out_image)

def compute_lbp_feature(lbp_arr, method, n_points, hist, stat):
    if hist:
        if method == "uniform":
            bins = [n for n in range(n_points+2+1)] # bins for numpy histogram. there are n+1 possible patterns, plus another one because of numpy histogram functionality
        elif method == "default":
            if n_points > 5:
                bins = 32
            else:
                bins = 2**n_points
    if hist:
        feat_vec = np.histogram(lbp_arr, bins)[0] # could do interesting things with density=True
        if stat:
            feat_vec = calc_stats(feat_vec, stat)
    else:
        if stat:
            feat_vec = calc_stats(lbp_arr, stat)
    return feat_vec

def lbp_feat_vec(image_name,
                 scales,
                 output=None,
                 method='uniform',
                 radius=1,
                 n_points=4,
                 hist=True,
                 stat=None,
                 smooth_factor=None,
                 levels=None):

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
        image = skimage.img_as_ubyte(rgb2gray(image)) # lbp takes a gray level image
    if smooth_factor:
        image = gaussian(image, smooth_factor)
        image = np.where(image > 1, 1, image)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(image)
            # image = skimage.img_as_ubyte(gaussian(image, smooth_factor))
    if levels:
        scale_factor = levels/255.
        image = image.astype(float)
        image *= scale_factor 
        image = np.round(image).astype(int)

    lbp = local_binary_pattern(image, n_points, radius, method) # there are n_points+2 possible values for uniform
    if hist == False and stat == None:
        return lbp

    if hist:
        if method == "uniform":
            bins = [n for n in range(n_points+2+1)] # bins for numpy histogram. there are n+1 possible patterns, plus another one because of numpy histogram functionality
        elif method == "default":
            if n_points > 5:
                bins = 32
            else:
                bins = 2**n_points

    # center pixel location
    center_i = int(image.shape[0] / 2.)
    center_j = int(image.shape[1] / 2.)

    out = []
    for s in scales:
        # convert meters to pixel counts
        # n_pixels = int(s / cell_width) # number of pixels for the scale
        n_pixels = s
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
            right = center_j+int(n_pixels/2.)
        
        scale_arr = lbp[top:bot+1,left:right+1]
        if hist:
            feat_vec = np.histogram(scale_arr, bins)[0] # could do interesting things with density=True
            if stat:
                feat_vec = calc_stats(feat_vec, stat)
        else:
            if stat:
                feat_vec = calc_stats(scale_arr, stat)
        out.append(feat_vec)
    return np.array(out).flatten()

def clbp():
    return NotImplemented