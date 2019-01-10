import os

import numpy as np
from osgeo import gdal
from osgeo import osr

from ..utilities.io import *

def get_sorted_tiles(image_dir):
    image_names = [os.path.join(image_dir, n) for n in os.listdir(image_dir)]
    feature_types = ["_LBP_", "_MBI_", "_GLCM_", "_PANTEX_", "_HOG_", "_GAB_", "_LAC_", "_SIFT_"]
    # sort the tiles that need to stacked into their own lists
    sorted_files = {}
    for filename in image_names:
        for ftype in feature_types:
            if ftype in filename:
                tilename = filename[:filename.index(ftype)]
                if tilename not in sorted_files:
                    sorted_files[tilename] = [filename]
                else:
                    sorted_files[tilename].append(filename)
    return sorted_files

def stack_image_tiles(image_dir, outdir, remove_old=False):
    files = get_sorted_tiles(image_dir)
    for f in files:
        out_name = os.path.join(outdir, os.path.basename(f)) + "_features.tif"
        arr = []
        out_geotran = None
        out_srs_wkt = None
        for image in files[f]:
            ds = gdal.Open(image)
            if out_geotran == None:
                out_geotran = ds.GetGeoTransform()
            if out_srs_wkt == None:
                out_srs = osr.SpatialReference()
                out_srs.ImportFromEPSG(4326)
                out_srs_wkt = out_srs.ExportToWkt()
            im = ds.ReadAsArray()
            if len(im.shape) == 3:
                for b in im:
                    arr.append(b)
            else:
                arr.append(im)
            ds = None
        write_geotiff(out_name, np.array(arr), out_geotran, out_srs_wkt)