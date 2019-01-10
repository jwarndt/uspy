import numpy as np
from osgeo import gdal
from osgeo import osr

from ..utilities.io import write_geotiff

def hist_feature(image_name, block, scale, output=None, numbins=32):
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
    
    bins = [n for n in range(numbins+1)] # bins for numpy histogram.

    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    out_cell_width = block * in_cell_width
    out_cell_height = block * in_cell_height
    
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
            out = np.histogram(scale_arr, bins) # could do interesting things with density=True
            outrow.append(out[0])
        out_image.append(outrow)
    out_image = np.array(out_image)
    out_image = np.moveaxis(out_image, -1, 0)
    if output:
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, out_image, out_geotran, out_srs_wkt)
    return np.array(out_image)

def hist_feat_vec(image_name, scales, output=None, numbins=32):
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]
    ds = None

    bins = [n for n in range(numbins+1)] # bins for numpy histogram.
    
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
            right = center_j+int(n_pixels/2.)
        scale_arr = image[top:bot+1,left:right+1]
        feat_vec = np.histogram(scale_arr, bins)[0] # could do interesting things with density=True
        out.append(feat_vec)
    return np.array(out).flatten()