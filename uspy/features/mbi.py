
import numpy as np
from osgeo import gdal
from osgeo import osr
from skimage.filters import gaussian
from skimage.morphology import erosion, dilation, opening, closing, white_tophat

from ..utilities.stats import *
from ..utilities.io import *

def get_se_set(sizes):
    """
    Parameters:
    -----------
    sizes: list
    
    Returns:
    --------
    se_set: ndarray (4D)
        se_set[0] gives the linear directional kernels for size
            at index zero
        se_set[1] gives the linear direction kernels for size at
            index 1
    """
    se_set = []
    for se_size in sizes:
        assert(se_size%2!=0)
        # create a structural element for the direction and size
        # directions are hardcoded to 4 for now. it generates 4
        # kernels with directions of 0, 45, 90, and 135
        se0 = np.zeros(shape=(se_size,se_size))
        se0[se_size//2,:] = 1
        se45 = np.diagflat(np.ones(shape=(se_size)))[::-1]
        se90 = np.zeros(shape=(se_size,se_size))
        se90[:,se_size//2] = 1
        se135 = np.diagflat(np.ones(shape=(se_size)))
        se_set.append([se0, se45, se90, se135])
    return se_set

def mbi_feature(image_name, output=None, postprocess=True, smooth_factor=None):
    MBI_THRESHOLD = 5.5

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
    # out_cell_width = block * cell_width
    # out_cell_height = block * cell_height

    ds = None
    image = np.moveaxis(image, 0, -1) # rows, columns, channels
    # calculate brightness as a local max
    brightness = calc_stats(image, ["max"], 2)

    # could be useful to smooth...
    if smooth_factor:
        brightness = gaussian(brightness, smooth_factor)
    # a set of linear structural elements
    # for the white tophat transformation
    # dirs = [45, 90, 135, 180]
    se_sizes = [5, 9, 13, 19, 23, 27]
    se_set = get_se_set(se_sizes)
    # 'white' top-hat transformation
    # in this case, white top-hat is the brightness image minus morphological opening
    mean_w_tophats = []
    for s in se_set: # for each size in the structural element set
        w_tophats = []
        for k in s: # for each direction kernel in the structural element set for this size
            # directional top hat transformation using linear SE
            w_tophats.append(white_tophat(brightness, k))
        mean_w_tophat = calc_stats(w_tophats, ['mean'], 0)
        mean_w_tophats.append(mean_w_tophat)
    
    th_dmp = []
    th_idx = 0
    # calculate the differential morphological profile
    while th_idx + 1 < len(mean_w_tophats):
        th_dmp.append(np.absolute(mean_w_tophats[th_idx + 1] - mean_w_tophats[th_idx]))
        th_idx+=1
    mbi = calc_stats(np.array(th_dmp), ['mean'], 0)
     
    if postprocess:
        mbi = np.where(mbi >= MBI_THRESHOLD, 1, 0)
    if output:
        # out_geotran = (out_ulx, out_cell_width, 0, out_uly, 0, out_cell_height)
        write_geotiff(output, mbi, geotran, out_srs_wkt)
    return np.array(mbi)