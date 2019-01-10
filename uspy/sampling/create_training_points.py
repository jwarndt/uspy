import random
import os
import argparse
import time

import numpy as np
from osgeo import ogr
from osgeo import osr
from osgeo import gdal

def __parse_args():
    parser = argparse.ArgumentParser(description="""Script for generating random training points that fall within
                                                    the boundaries of training polygons. 
                                                    """)

    parser.add_argument('-i',
                        '--input',
                        dest='input_shapefile',
                        required=True,
                        type=str)
    parser.add_argument('-o',
                        '--output',
                        dest='output_shapefile',
                        required=True,
                        type=str)
    parser.add_argument('-ppc',
                        '--points_per_class',
                        dest='points_per_class',
                        default=25,
                        type=int)
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=True)

    args = parser.parse_args()

    if args.verbose:
        print("---------------- Input ----------------")
        print('input shapefile:\t' + args.input_shapefile)
        print('output shapefile:\t' + args.output_shapefile)
        print('points per class:\t' + str(args.points_per_class))
        print("---------------------------------------")
        print()

    return args

def __check_args(args):
    return NotImplemented

def __append_random_points(in_layer, out_layer, num_points):
    """
    randomly selects a feature from the in_layer, places a random
    point Within() the feature, appends the random point onto the out_layer
    
    Parameters:
    -------------
    in_layer: ogr.Layer
        the input layer from which features will be randomly sampled
    out_layer: ogr.Layer
        the output layer that random points will be added to
    num_points: int
        the number of random points that will be added to the out_layer
    
    Returns:
    --------
    None
    """
    fids = []
    for f in in_layer:
        fids.append(f.GetFID())
    point_count = 0
    while point_count < num_points:
        # randomly select feature in the layer
        rand_fid_idx = random.randint(0, len(fids)-1)
        selected_feature = in_layer.GetFeature(fids[rand_fid_idx])
        geom = selected_feature.GetGeometryRef()
        mbr = geom.GetEnvelope()
        minx = mbr[0]
        maxx = mbr[1]
        miny = mbr[2]
        maxy = mbr[3]
        rand_x = random.uniform(minx, maxx) # left x
        rand_y = random.uniform(miny, maxy) # upper y
        wkt_pnt = "POINT (%f %f)" % (rand_x, rand_y)
        rand_pnt = ogr.CreateGeometryFromWkt(wkt_pnt)
        if geom.Contains(rand_pnt):
            # create the feature
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            # Set the attributes using the values from the delimited text file
            out_feature.SetField("poly_id", selected_feature.GetField('poly_id'))
            out_feature.SetField("class_type", selected_feature.GetField('class_type'))
            out_feature.SetField("image_name", selected_feature.GetField('image_name'))
            out_feature.SetField("city", selected_feature.GetField('city'))
            # Set the feature geometry using the polygon
            out_feature.SetGeometry(rand_pnt)
            # Create the feature in the layer (shapefile)
            out_layer.CreateFeature(out_feature)
            # Dereference the feature
            out_feature = None
            point_count += 1
            
def create_random_points(input_shapefile, output_shapefile, points_per_class):
    """
    creates a points shapefile that is made up of random points that 
    fall within the boudaries of the features in the input shapefile.
    each 'class_type' in the input_shapefile will have the same number
    of points which is defined by the points_per_class parameter.
    
    Parameters:
    -----------
    inshapefile: string
        the filepath of the input shapefile. the input shapefile must have a class_type 
        attribute so that a uniform number of points can be distributed randomly into each class
    outshapfile: string
        the filepath of the output shapefile.
    points per class: int
        the number of random points to be distributed into each class
    """
    in_driver = ogr.GetDriverByName('ESRI Shapefile')
    in_datasource = in_driver.Open(input_shapefile, 0) # 0 means read-only. 1 means writeable.
    in_layer = in_datasource.GetLayer()
    in_srs = in_layer.GetSpatialRef()
    
    # need to open a second dataset to repeated attribute filtering
    ds2 = in_driver.Open(input_shapefile, 0)
    filter_layer = ds2.GetLayer()
    
    # set up the shapefile driver
    out_driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the data source
    out_datasource = out_driver.CreateDataSource(output_shapefile)
    # create the layer
    out_layer = out_datasource.CreateLayer(output_shapefile[:-4], in_srs, ogr.wkbPoint)
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
    
    #initialize the sampling histogram
    sampling_classes = []
    for f in in_layer:
        if f.GetField('class_type') not in sampling_classes:
            sampling_classes.append(f.GetField('class_type'))
            filter_layer.SetAttributeFilter("class_type = " + "'" + f.GetField('class_type') + "'")
            __append_random_points(filter_layer, out_layer, points_per_class)
            filter_layer = ds2.GetLayer() # point back to the original full datasource and get its layer
    out_datasource = None


def main():
    start_time = time.time()
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    args = __parse_args()
    __check_args(args)
    create_random_points(args.input_shapefile,
                         args.output_shapefile,
                         args.points_per_class)

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    print('\nEnd data & time -- (%s)\nTotal processing time -- (%d min %f sec)\n' %
        (time.asctime(time.localtime(time.time())), minutes, sec))

if __name__ == "__main__":
    main()