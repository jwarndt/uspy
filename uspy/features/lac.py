import math

from osgeo import gdal
from osgeo import osr
import skimage
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage._shared._warnings import expected_warnings
import numpy as np

from ..utilities.stats import *
from ..utilities.io import *

def lac_feature(image_name, block, scale, box_size, output=None, slide_style=0, lac_type='grayscale', smooth_factor=None, levels=None):
    """
    differential box-counting algorithm for computing lacunarity

    Parameters:
    -----------
    image_name: str
        the input image
    block: int
        the size of the block in pixels
    scale: int
        the window size in pixels for computing lacunarity (w x w). window and scale are synonomous
    box_size: int
        the size of the cube (r x r x r)
    slide_style: int
        how the boxes slide across the window
        for glide: specify a slide_style of 0
        for block: specify a slide_style of -1
        for skip: specify the number of pixels to skip (i.e. a positive integer)
    lac_type: str
        two options are available: grayscale or binary
        lacunarity calculations are slightly different for these

    Returns:
    --------
    out: ndarray
        the lacunarity image
    """
    assert(type(block) == int and type(scale) == int)
    assert(box_size < scale)
    assert(scale % box_size == 0)

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

    #restrict bands
    image = image[0:3]

    if lac_type == "grayscale":
        image = np.moveaxis(image, 0, -1)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(rgb2gray(image))

    # could be useful to smooth...
    if smooth_factor:
        image = gaussian(image, smooth_factor)
        image = np.where(image > 1, 1, image)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(image)
            # image = skimage.img_as_ubyte(gaussian(image, smooth_factor))
    # could be useful to rescale to fewer gray levels...
    if levels:
        scale_factor = (levels-1)/255.
        image = image.astype(float)
        image *= scale_factor 
        image = np.round(image).astype(int)

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
            lac = __box_counting(scale_arr, box_size, slide_style) 
            outrow.append(lac)
        out_image.append(outrow)
    if output:
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, np.array(out_image), out_geotran, out_srs_wkt)
    return np.array(out_image)

def lac_feat_vec(image_name,
                 scales,
                 box_size,
                 output=None,
                 slide_style=0,
                 lac_type='grayscale',
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

    #restrict bands
    image = image[0:3]

    if lac_type == "grayscale":
        image = np.moveaxis(image, 0, -1)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(rgb2gray(image))

    # could be useful to smooth...
    # could be useful to smooth...
    if smooth_factor:
        image = gaussian(image, smooth_factor)
        image = np.where(image > 1, 1, image)
        with expected_warnings(['precision']):
            image = skimage.img_as_ubyte(image)
            # image = skimage.img_as_ubyte(gaussian(image, smooth_factor))
            
    # could be useful to rescale to fewer gray levels...
    if levels:
        scale_factor = (levels-1)/255.
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

        lac = __box_counting(scale_arr, box_size, slide_style)
        out.append(lac)
    return np.array(out)

def __box_counting(scale_arr, box_size, slide_style):
    # now slide the box over the window
    n_mr = {} # the number of gliding boxes of size r and mass m. a histogram
    # m = 0 # the mass of the grayscale image, the sum of the relative height of columns (n_ij)
    masses = []
    total_window_mass = 0
    total_boxes_in_window = 0
    ii = 0
    while ii + box_size <= len(scale_arr):
        jj = 0
        while jj + box_size <= len(scale_arr[0]):
            total_boxes_in_window += 1
            box = scale_arr[ii:ii+box_size,jj:jj+box_size]
            max_val = np.amax(box)
            min_val = np.amin(box)
            u = math.ceil(float(min_val) / box_size) # box with minimum pixel value
            v = math.ceil(float(max_val) / box_size) # box with maximum pixel value
            n_ij = int(v - u + 1) # relative height of column at ii and jj
            
            masses.append(n_ij)
            total_window_mass += n_ij

            # so n_mr is the number of boxes of size r and mass m
            # use a dictionary and count the number of boxes in this image
            if n_ij not in n_mr:
                n_mr[n_ij] = 1
            else:
                n_mr[n_ij] += 1
            # move the box based on the glide_style
            if slide_style == 0: # glide
                jj+=1
            elif slide_style == -1: # block
                jj+=box_size
            else: # skip
                jj+=box_size+slide_style
        if slide_style == 0: # glide
            ii+=1
        elif slide_style == -1: # block
            ii+=box_size
        else: # skip
            ii+=box_size+slide_style
    num = 0
    denom = 0.000001
    for m in masses:
        # the probability function which is the number of boxes
        # of size r and mass m divided by the total number of boxes
        q_mr = n_mr[m] / total_boxes_in_window 
        num += (m*m) * q_mr
        denom += m * q_mr
    denom = denom**2
    lac = num / denom
    return lac