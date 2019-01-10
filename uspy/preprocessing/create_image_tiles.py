#! /usr/bin/env python3
"""--------------------------------------*---*-----------------------------------------"""
"""!
INFO:
    * \file: tile.py
    * \author: Jacob Arndt
    * \date: July 2, 2018
    * \brief: tiles an image
    * \note: adapted from Jiangye Yuan's tiling script (tiling_imgsm.py) 
"""
import argparse
import os
import glob
import subprocess
import json
import sys
import string
import numpy as np
import random
from osgeo import gdal, gdalconst
from osgeo import ogr
from osgeo import osr
import time

SUBTILE_WIDTH  = 6000
SUBTILE_HEIGHT = 6000
buffsz = 200

def __parse_args():

    usage = "usage: python %prog [options]"
    parser = argparse.ArgumentParser(usage)

    parser.add_argument("-q",
                        action="store_false",
                        dest="verbose",
                        help="Quiet",
                        default=False)

    parser.add_argument("-v",
                        action  = "store_true",
                        dest    = "verbose",
                        help    = "Verbose",
                        default = True)

    parser.add_argument("-i",
                        action  = "store",
                        dest    = "image_name",
                        help    = "input image to be tiled",
                        type    = str,
                        required = True)

    parser.add_argument("-m",
                        action  = "store",
                        dest    = "mask_shp",
                        help    = "shapefile to use as a mask",
                        type    = str,
                        required = False)

    parser.add_argument("-sql",
                        action = "store",
                        dest = "sql_query",
                        help = "sql query to specify features in the shapefile to use in the mask",
                        type = str,
                        required = False)
    # CNTRY_NAME = 'Senegal'

    parser.add_argument("-o",
                        action  = "store",
                        dest    = "output_dir",
                        help    = "output Dir where tiles are to be stored",
                        type    = str,
                        required = True)

    args = parser.parse_args()

    if args.verbose:
        print("---------------- Input ----------------")
        print("image name:\t" + args.image_name)
        print("output dir:\t" + args.output_dir)
        print()
 
    return args


def tile_image(args):
    print(args.image_name)
    outdir = args.output_dir + "/" + os.path.basename(args.image_name)[:-4] + "_tiles"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    width, height, bands, dataType, imgpara = getImageInfo(args.image_name)
    print(width,height,bands,dataType)

    np.savetxt(args.output_dir + '/ImagePara.txt', imgpara, fmt='%d')

    driver = gdal.GetDriverByName( "GTiff" )

    src_ds = gdal.Open(args.image_name)
    if src_ds is None:
        print("error : cannot open file")
        sys.exit(1)

    trans = src_ds.GetGeoTransform()
    proj = src_ds.GetProjection()

    vector_driver = ogr.GetDriverByName("ESRI Shapefile")
    vector_ds = vector_driver.CreateDataSource(os.path.join(outdir, os.path.basename(args.image_name)[:-4] + "_footprint.shp"))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = vector_ds.CreateLayer(os.path.basename(args.image_name)[:-4] + "_footprint", srs, ogr.wkbPolygon)
    field_name = ogr.FieldDefn("tile", ogr.OFTString)
    field_name.SetWidth(100)
    layer.CreateField(field_name)


    # if a mask shapefile was given, check if this tile intersects the mask
    # if the tile intersects the mask, write the tile to disk, otherwise skip
    if args.mask_shp:
        mask_ds = ogr.Open(args.mask_shp)
        mask_layer = mask_ds.GetLayer()
        if args.sql_query: # check if need to mask with a subset of the shapefile
            mask_layer.SetAttributeFilter(args.sql_query)
    count  = 0
    for originX0 in range(0, width, SUBTILE_WIDTH):
        for originY0 in range(0, height, SUBTILE_WIDTH):
            count= count + 1

            originX = originX0 - buffsz if originX0 - buffsz >= 0 else 0
            originY = originY0 - buffsz if originY0 - buffsz >= 0 else 0

            endX = originX0 + SUBTILE_WIDTH + buffsz if originX0 + SUBTILE_WIDTH + buffsz < width else width
            endY = originY0 + SUBTILE_HEIGHT + buffsz if originY0 + SUBTILE_HEIGHT + buffsz < height else height

            x = endX - originX
            y = endY - originY

            transN = (trans[0] + trans[1] * originX, trans[1], trans[2], trans[3]
              + trans[5] * originY, trans[4], trans[5])

            lx = transN[0] + (transN[1]*(buffsz)) if originX != 0 else transN[0]
            rx = transN[0] + (x*transN[1]) - (transN[1]*(buffsz)) if endX < width else transN[0] + (x*transN[1])
            uy = transN[3] + (transN[5]*(buffsz)) if originY != 0 else transN[3]
            ly = transN[3] + (y*transN[5]) - (transN[5]*(buffsz)) if endY < height else transN[3] + (y*transN[5])
            wkt = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" % (lx, uy,
                                                                     lx, ly,
                                                                     rx, ly,
                                                                     rx, uy,
                                                                     lx, uy)
            poly = ogr.CreateGeometryFromWkt(wkt)
            
            
            if args.mask_shp:
                mask_layer.GetFeatureCount() #strange gdal business... not sure why you need this?
                for mask_feature in mask_layer:
                    mask_geom = mask_feature.GetGeometryRef()
                    if mask_geom.Intersects(poly):
                        valid_tile = True
                    else:
                        valid_tile = False
            else:
                valid_tile = True

            if valid_tile:

                subTileName  = outdir + "/" + os.path.basename(args.image_name[0:-4]) \
                               + "_" + str(originX0).zfill(9) + "_" + str(originY0).zfill(9) \
                               + "_" + str(count).zfill(5) + ".tif"

                print('Tile origin: ' + str(originX0) + ', ' + str(originY0))

                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField("tile", subTileName)
                feature.SetGeometry(poly)
                layer.CreateFeature(feature)
                feature = None

                dst_ds = driver.Create(subTileName, x, y, bands, dataType)
                if dst_ds is None:
                    print('Unable to open', subTileName, 'for writing')
                    sys.exit(1)

                for band in range( bands ):
                    band += 1
                    srcband = src_ds.GetRasterBand(band)
                    band_data = srcband.ReadAsArray(originX, originY, x, y).astype(np.int16)
                    dst_ds.GetRasterBand(band).WriteArray(band_data)

                dst_ds.SetGeoTransform(transN)
                dst_ds.SetProjection(proj)

            dst_ds = None

    vector_ds = None
    src_ds = None
    
    print("Done")


def getImageInfo(image_name):
    src_ds = gdal.Open(image_name)
    if src_ds is None:
        print("error : cannot open file")
        sys.exit(1)
    
    cols    = src_ds.RasterXSize
    rows    = src_ds.RasterYSize
    bands   = src_ds.RasterCount


    band = src_ds.GetRasterBand(1)
    dataType = gdal.GetDataTypeName(band.DataType)

    imgpara = np.zeros((bands+2, 2))
    imgpara[0, :] = rows, cols
    imgpara[1, :] = SUBTILE_HEIGHT, SUBTILE_WIDTH

    for i in range(bands):
        i += 1
        band = src_ds.GetRasterBand(i)
        stats = band.GetStatistics(0,1)
        BandHist = band.GetDefaultHistogram()
        tmp = np.asarray(BandHist[3])
        tmp = np.cumsum(tmp).astype(np.float) / np.sum(tmp).astype(np.float)
        lbd = np.where(tmp >= .05)[0][0] / np.float(tmp.size) * stats[1]
        ubd = np.where(tmp >= .98)[0][0] / np.float(tmp.size) * stats[1]
        imgpara[i+1, :] = np.round(lbd), np.round(ubd)

    return cols, rows, bands, band.DataType, imgpara


## begin --> fnCheckArgs() ###############################################################
def __check_args(args):
    
    #    check if the image  exists

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except OSError as exception:
            if exception.errno!=17:
                print (stderr, "Exception: %s" % str(exception))
                raise

    if not os.path.isfile(args.image_name):
        print("image doesn't exist ")
        sys.exit(1)

def main():
    start_time = time.time()
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    args = __parse_args()
    __check_args(args)
    tile_image(args)

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    print('\nEnd date & time -- (%s)\nTotal processing time -- (%d min %f sec)\n' % (time.asctime(time.localtime(time.time())), minutes, sec)

if __name__ == "__main__":
    main()