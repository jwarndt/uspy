import random
import os
import argparse
import time

import numpy as np
from osgeo import ogr
from osgeo import osr
from osgeo import gdal

def __parse_args():
    parser = argparse.ArgumentParser(description="""Script for generating square training boxes that fall within
    												the boundaries of human generated training polygons. 
                                                    """)

    parser.add_argument('-i',
                        '--input',
                        dest='training_shapefile',
                        required=True,
                        type=str)
    parser.add_argument('-o',
                        '--output',
                        dest='output_shapefile',
                        required=True,
                        type=str)
    parser.add_argument('-ppf',
                        '--plots_per_feature',
                        dest='plots_per_feature',
                        default=1,
                        type=int)
    parser.add_argument('-d',
                        '--dimension',
                        dest='dim',
                        default=100,
                        type=int)
    parser.add_argument('-t',
                        '--trials',
                        dest='num_trials',
                        default=20,
                        type=int)
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=True)
    args = parser.parse_args()


    if args.verbose:
        print("---------------- Input ----------------")
        print('input shapefile:\t' + args.training_shapefile)
        print('output shapefile:\t' + args.output_shapefile)
        print('plots per feature:\t' + str(args.plots_per_feature))
        print('box dimension (meters):\t' + str(args.dim))
        print('number of trials:\t' + str(args.num_trials))
        print("---------------------------------------")
        print()
    
    return args

def __check_args(args):
    return NotImplemented

def __create_random_box(training_geometry, box_dim, num_trials):
    """
    creates a random dim x dim rectangular polygon within the given
    training geometry.
    
    Parameters:
    ------------
    training_geometry: ogr.geometry
        input geometry that the box will be created inside
    geotransform: gdal Dataset Geotransform
    box_dim: int
        the dimension in meters. i.e. size of the plot (xdim, ydim)
        (default are set in the main() function)
    num_trials: int
        the number of times to try and find a random box that fits within the given
        training_geometry
        (defaults are set in the main() function)
    
    Returns:
    ---------
    training_box_geom: ogr.Geometry
        the geometry of the training box
    """
    mbr = training_geometry.GetEnvelope()
    minx = mbr[0]
    maxx = mbr[1]
    miny = mbr[2]
    maxy = mbr[3]
    
    trial_num = 0
    while trial_num < num_trials: 
        rand_lx = random.uniform(minx, maxx) # left x
        rand_uy = random.uniform(miny, maxy) # upper y
        rx = rand_lx + box_dim # right x
        ly = rand_uy - box_dim # lower y 
        wkt_box = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" % (rand_lx, rand_uy, rand_lx, ly, rx, ly, rx, rand_uy, rand_lx, rand_uy)
        training_box_geom = ogr.CreateGeometryFromWkt(wkt_box)
        if training_geometry.Contains(training_box_geom):
            return training_box_geom
        trial_num += 1
    return None
        
def create_training_plots(training_shapefile, output_shapefile, plots_per_feature=1, dim=100, num_trials=20):
    """
    creates square training image chips given a shapefile of polygons. 
    The polygons in the shapefile identify training classes. 
    The shapefile must have the following fields: FID, class_type, city, image_name. 
    The image_dir must contain the images that are identified in the image_name field 
    for each feature in the shapefile.
    
    Parameters:
    ------------
    training_shapefile: string
        file path to the shapefile that training chips will be made for
    output_shapefile: string
        the file path of the output shapefile
    plots_per_feature: int
        the maximum number of training plots possible for each feature in the shapefile
    dim: int
        the x and y dimension in meters for the training plots
    num_trials: int
        the number of attempts a random box will be created for each feature until
        moving onto the next. A random box must be Within() the training feature in
        order for it to be valid.
        
    Returns:
    ---------
    None
    """
    in_driver = ogr.GetDriverByName('ESRI Shapefile')
    in_datasource = in_driver.Open(training_shapefile, 0) # 0 means read-only. 1 means writeable.
    in_layer = in_datasource.GetLayer()
    in_srs = in_layer.GetSpatialRef()
    
    # set up the shapefile driver
    out_driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the data source
    out_datasource = out_driver.CreateDataSource(output_shapefile)
    # create the spatial reference, WGS84
    #out_srs = osr.SpatialReference()
    #srs.ImportFromEPSG(4326)
    # create the layer
    out_layer = out_datasource.CreateLayer(output_shapefile[:-4], in_srs, ogr.wkbPolygon)
    # create fields
    out_layer.CreateField(ogr.FieldDefn("poly_id", ogr.OFTInteger))
    c_type = ogr.FieldDefn("class_type", ogr.OFTString)
    c_type.SetWidth(50)
    out_layer.CreateField(c_type)
    i_name = ogr.FieldDefn("image_name", ogr.OFTString)
    i_name.SetWidth(50)
    out_layer.CreateField(i_name)
    city_name = ogr.FieldDefn("city", ogr.OFTString)
    city_name.SetWidth(50)
    out_layer.CreateField(city_name)
    
    for feature in in_layer:
        class_type = feature.GetField("class_type")
        fid = str(feature.GetFID())
        city = feature.GetField("city")
        im_name = feature.GetField("image_name")
        geom = feature.GetGeometryRef()
        count = 0
        while count < plots_per_feature:
            #city_fid_class_string = city + "_" + fid + "_" + str(count) + "_" + class_type
            training_box_geom = __create_random_box(geom, dim, num_trials)
            if training_box_geom is None: 
                print("shapefile: " + training_shapefile)
                print("INFO: NO BOX GEOM FOUND --> FID: " + fid + ", Chip Number: " + str(count))
            else:
                # create the feature
                out_feature = ogr.Feature(out_layer.GetLayerDefn())
                # Set the attributes using the values from the delimited text file
                out_feature.SetField("poly_id", int(fid))
                out_feature.SetField("class_type", class_type)
                out_feature.SetField("image_name", im_name)
                out_feature.SetField("city", city)
                # Set the feature geometry using the polygon
                out_feature.SetGeometry(training_box_geom)
                # Create the feature in the layer (shapefile)
                out_layer.CreateFeature(out_feature)
                # Dereference the feature
                out_feature = None
            count += 1
    out_datasource = None

def main():
    start_time = time.time()
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    args = __parse_args()
    __check_args(args)
    create_training_plots(args.training_shapefile,
                          args.output_shapefile,
                          args.plots_per_feature,
                          args.dim,
                          args.num_trials)

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    print('\nEnd date & time -- (%s)\nTotal processing time -- (%d min %f sec)\n' %
        (time.asctime(time.localtime(time.time())), minutes, sec))

if __name__ == '__main__':
    main()