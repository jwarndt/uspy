import random
import os
import argparse
import time

import numpy as np
from osgeo import ogr
from osgeo import osr
from osgeo import gdal

"""
Script for generating image chips centered on random points that fall within the input shapefile's
features. In this case, the image chips do not neccessarily fall completely within the boundaries
of the shapefile's input features. However, the random points do.
"""

def __parse_args():
    parser = argparse.ArgumentParser(description="""Script for generating square image chips from random points
                                                    that fall within the boundaries of human generated training polygons. 
                                                    """)

    parser.add_argument('-inshp',
                        '--input_shapefile',
                        dest='training_shapefile',
                        required=True,
                        type=str)
    parser.add_argument('-vec_type',
                        '--vector_type',
                        dest='vector_type',
                        required=True,
                        type=str)
    parser.add_argument('-imdir',
                        '--image_directory',
                        dest='image_dir',
                        required=True,
                        type=str)
    parser.add_argument('-o',
                        '--output_directory',
                        dest='out_dir',
                        required=True,
                        type=str)
    parser.add_argument('-ppc',
                        '--points_per_class',
                        dest='points_per_class',
                        default=50,
                        type=int)
    parser.add_argument('-d',
                        '--box_dimension',
                        dest='box_dim',
                        default="256 pixels",
                        type=str)
    parser.add_argument('-t',
                        '--trials',
                        dest='num_trials',
                        default=50,
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
        print('input vector type:\t' + args.vector_type)
        print('image directory:\t' + args.image_dir)
        print('output directory:\t' + args.out_dir)
        print('points per class:\t' + str(args.points_per_class))
        print('box dimension (pixels or meters):\t' + str(args.box_dim))
        print('number of trials:\t' + str(args.num_trials))
        print("---------------------------------------")
        print()

    return args

def __check_args(args):
    return NotImplemented
            
def __extract_chip_and_write(im, outname, box_geom):
    """
    extracts a chip from the image given the box_geom that was found by the
    create_random_box() function.
    
    Parameters:
    ------------
    im: np.ndarray
        the image array
    out_dir: string
        output directory for the image chips
    box_geom: list
        box_info[0] is a list that contains the origin coordinates of the random box. (upper left x, upper left y)
        box_info[1] is the box_dim
    city_fid_class_string: string
        the name of the image chip. fid, class, and city are derived
        from the shapefile feature's attributes
        
    Returns:
    ---------
    None
    """
    # im_basename = os.path.basename(im)[:-4]
    # outname = os.path.join(out_dir, im_basename + "_" + city_fid_class_string + ".tif")
    
    # read band to get the data type
    b1 = im.GetRasterBand(1)
    im_datatype = b1.DataType
    b1 = None
    
    geotran = im.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]
    box_x = box_geom[0][0]
    box_y = box_geom[0][1]
    box_dim = box_geom[1]
    # get the origin coordinates of the box in image space
    # convert from geo coordinates to image coordinates
    pixel_coord_x = int((box_x - ulx) / cell_width)
    pixel_coord_y = int((box_y - uly) / cell_height)
    
    data = im.ReadAsArray(pixel_coord_x, pixel_coord_y, box_dim, box_dim)
    
    cols = data.shape[2]
    rows = data.shape[1]
    bands = data.shape[0]
    
    driver = gdal.GetDriverByName('GTiff')
    out_im = driver.Create(outname, cols, rows, bands, im_datatype)
    out_im.SetGeoTransform((box_x, cell_width, 0, box_y, 0, cell_height))
    for n in range(bands):
        outband = out_im.GetRasterBand(n+1)
        outband.WriteArray(data[n])
        out_im_srs = osr.SpatialReference()
        # FIXME: get the srs from the input image rather than just set it
        # to WGS84
        out_im_srs.ImportFromEPSG(4326)
        out_im.SetProjection(out_im_srs.ExportToWkt())
        outband.FlushCache()

def __chip_image(in_layer, image_dir, out_dir, out_layer, points_per_class, box_dim):
    # a dictionary of form:
    # {poly_id1: [[fid1, fid2, fid3], count], poly_id2: [fid4, fid5], ... }
    # so this associates each point fid with the polygon it came from, the count is for counting the number
    # of chips that have been made from each poly_id
    id_dict = {}
    dict_keys = []
    total_features = 0
    for f in in_layer:
        if f.GetField('ORIG_FID') not in id_dict:
            id_dict[f.GetField('ORIG_FID')] = [[], 0]
            dict_keys.append(f.GetField('ORIG_FID'))
        id_dict[f.GetField('ORIG_FID')][0].append(f.GetFID())
        total_features+=1
    if points_per_class > total_features:
        print("WARNING: total number of features in class is less than the points per class speciefied")
        print(" ... dynamically changing points_per_class to total number of features in this class")
        points_per_class = total_features
    chip_count = 0
    point_idx = 0
    while chip_count < points_per_class:
        # get the feature using the FID at index 'count' in the in_dict
        selected_feature = in_layer.GetFeature(id_dict[dict_keys[point_idx]][0][id_dict[dict_keys[point_idx]][1]])
        id_dict[dict_keys[point_idx]][1]+=1
        geom = selected_feature.GetGeometryRef()
        pnt = geom.GetPoint()
        x = pnt[0]
        y = pnt[1]

        image_name = selected_feature.GetField('image_name')
        class_type = selected_feature.GetField('class_type')
        poly_id = selected_feature.GetField('ORIG_FID')
        city = selected_feature.GetField('city')
        fid = str(selected_feature.GetFID())

        image = gdal.Open(os.path.join(image_dir, image_name))
        geotransform = image.GetGeoTransform()
        cell_width = geotransform[1]
        cell_height = geotransform[5]

        # box_dim can be specified in either pixels or meters.
        # if it is specified in meters, it must be converted to
        # number of pixels based on input image GSD
        pixel_box_dim = get_pixel_box_dim(image_name, box_dim)

        """
        Write the box to the shapefile
        """
        lx = x - ((pixel_box_dim/2)*cell_width)
        uy = y - ((pixel_box_dim/2)*cell_height)
        rx = lx + (pixel_box_dim * cell_width)
        ly = uy + (pixel_box_dim * cell_height)
        box_info = [[lx, uy], pixel_box_dim]
        wkt_box = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" % (lx, uy, lx, ly, rx, ly, rx, uy, lx, uy)
        training_box_geom = ogr.CreateGeometryFromWkt(wkt_box)
        
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        # Set the attributes using the values from the delimited text file
        out_feature.SetField("poly_id", poly_id)
        out_feature.SetField("class_type", class_type)
        out_feature.SetField("image_name", image_name)
        out_feature.SetField("city", city)
        # Set the feature geometry using the polygon
        out_feature.SetGeometry(training_box_geom)
        # Create the feature in the layer (shapefile)
        out_layer.CreateFeature(out_feature)
        out_feature = None


        city_fid_class_string = city + "_" + str(poly_id) + "_" + fid  + "_" + str(class_type)
        out_im_name = os.path.join(out_dir, image_name[:-4] + "_" + city_fid_class_string + ".tif")
        __extract_chip_and_write(image, out_im_name, box_info)

        # remove the poly_id if all points from it have been used.
        if id_dict[dict_keys[point_idx]][1] > len(id_dict[dict_keys[point_idx]][0])-1:
            del id_dict[dict_keys[point_idx]]
            del dict_keys[point_idx]
            point_idx-=1
        # start from the first poly_id if it has reached the end
        point_idx+=1
        if point_idx == len(dict_keys):
            point_idx = 0
        chip_count+=1



def __write_random_boxes(in_layer, image_dir, out_dir, out_layer, num_points, box_dim, num_trials):
    """
    randomly selects a feature from the in_layer, places a random
    point Within() the feature, appends the random point onto the out_layer
    
    Parameters:
    -------------
    in_layer: ogr.Layer
        the input layer from which features will be randomly sampled
    image_dir: str
        the image directory
    out_layer: ogr.Layer
        the output layer that boxes generated from random points will be added to
    num_points: int
        the number of random points that will be added to the out_layer
    box_dim: str
        the dimension (in pixels or meters) that the box should be
    
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

        point_not_found = True
        box_info = None
        count = 0
        while point_not_found and count < num_trials:
            rand_x = random.uniform(minx, maxx) # left x
            rand_y = random.uniform(miny, maxy) # upper y
            wkt_point = "POINT (%f %f)" %(rand_x, rand_y)
            training_point_geom = ogr.CreateGeometryFromWkt(wkt_point)
            if geom.Contains(training_point_geom):
                image_name = selected_feature.GetField('image_name')
                class_type = selected_feature.GetField('class_type')
                #poly_id = selected_feature.GetField('poly_id')
                city = selected_feature.GetField('city')
                fid = str(selected_feature.GetFID())

                image = gdal.Open(os.path.join(image_dir, image_name))
                geotransform = image.GetGeoTransform()
                cell_width = geotransform[1]
                cell_height = geotransform[5]

                # box_dim can be specified in either pixels or meters.
                # if it is specified in meters, it must be converted to
                # number of pixels based on input image GSD
                pixel_box_dim = get_pixel_box_dim(image_name, box_dim)

                """
                Write the box to the shapefile
                """
                lx = rand_x - ((pixel_box_dim/2)*cell_width)
                uy = rand_y - ((pixel_box_dim/2)*cell_height)
                rx = lx + (pixel_box_dim * cell_width)
                ly = uy + (pixel_box_dim * cell_height)
                box_info = [[lx, uy], pixel_box_dim]
                wkt_box = "POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))" % (lx, uy, lx, ly, rx, ly, rx, uy, lx, uy)
                training_box_geom = ogr.CreateGeometryFromWkt(wkt_box)
                
                out_feature = ogr.Feature(out_layer.GetLayerDefn())
                # Set the attributes using the values from the delimited text file
                out_feature.SetField("poly_id", fid)
                out_feature.SetField("class_type", class_type)
                out_feature.SetField("image_name", image_name)
                out_feature.SetField("city", city)
                # Set the feature geometry using the polygon
                out_feature.SetGeometry(training_box_geom)
                # Create the feature in the layer (shapefile)
                out_layer.CreateFeature(out_feature)
                out_feature = None


                city_fid_class_string = city + "_" + fid + "_" + str(point_count) + "_" + str(class_type)
                out_im_name = os.path.join(out_dir, image_name[:-4] + "_" + city_fid_class_string + ".tif")
                __extract_chip_and_write(image, out_im_name, box_info)

                point_not_found = False
                point_count+=1
            else:
                pass
            count+=1

def get_pixel_box_dim(image_name, box_dim):
    if "pixels" in box_dim:
        out_box_dim = int(box_dim.split(" ")[0])
    elif "meters" in box_dim:
        out_box_dim = int(box_dim.split(" ")[0])
        if "wv2" in image_name or "ge1" in image_name:
            out_box_dim = int(box_dim / 0.46)
        elif "wv3" in image_name:
            out_box_dim = int(box_dim / 0.31)
    return out_box_dim

def create_training_chips(training_shapefile, vector_type, image_dir, out_dir, points_per_class, box_dim="375 pixels", num_trials=50, verbose=False):
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
    image_dir: string
        the path to the directory that holds the images that training chips
        will be extracted from
    out_dir: string
        the path to the directory that the image chips will be saved to
    box_dim: int
        the x and y dimension in number of pixels for the training chips
    num_trials: int
        the number of attempts a random box will be created for each feature until
        moving onto the next. A random box must be Within() the training feature in
        order for it to be valid.
        
    Returns:
    ---------
    None
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(training_shapefile, 0) # 0 means read-only. 1 means writeable.
    
    layer = dataset.GetLayer()
    in_srs = layer.GetSpatialRef()

    # need to open a second dataset to repeated attribute filtering
    ds2 = driver.Open(training_shapefile, 0)
    filter_layer = ds2.GetLayer()

    out_chip_shp = os.path.join(out_dir, os.path.basename(training_shapefile)[:-4] + "_chips.shp")
    # set up the output shapefile 
    out_driver = ogr.GetDriverByName("ESRI Shapefile")
    # create the data source
    out_datasource = out_driver.CreateDataSource(out_chip_shp)
    # create the layer
    out_layer = out_datasource.CreateLayer(out_chip_shp[:-4], in_srs, ogr.wkbPolygon)
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

    sampling_classes = []
    for f in layer:
        class_label = f.GetField('class_type')
        if class_label not in sampling_classes:
            if verbose:
                print("generating chips for class: " + str(class_label))
            sampling_classes.append(f.GetField('class_type'))
            filter_layer.SetAttributeFilter("class_type = " + "'" + str(f.GetField('class_type')) + "'")
            if vector_type == "polygon":
                __write_random_boxes(filter_layer, image_dir, out_dir, out_layer, points_per_class, box_dim, num_trials)
                
            elif vector_type == "point":
                __chip_image(filter_layer, image_dir, out_dir, out_layer, points_per_class, box_dim)
            else:
                print("vector_type: '" + vector_type + "' not allowed. Only 'point' or 'polygon' is allowed")
            filter_layer = ds2.GetLayer()
    out_datasource = None

def main():
    start_time = time.time()
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    args = __parse_args()
    __check_args(args)
    create_training_chips(args.training_shapefile,
                          args.vector_type,
                          args.image_dir,
                          args.out_dir,
                          args.points_per_class,
                          args.box_dim,
                          args.num_trials,
                          args.verbose)

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    print('\nEnd data & time -- (%s)\nTotal processing time -- (%d min %f sec)\n' %
        (time.asctime(time.localtime(time.time())), minutes, sec))

if __name__ == "__main__":
    main()