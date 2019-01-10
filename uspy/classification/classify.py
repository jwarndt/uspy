from multiprocessing import Pool
import pickle
import math
import time
import os

import numpy as np
from osgeo import gdal
from osgeo import osr
import skimage
from skimage.feature import local_binary_pattern
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage._shared._warnings import expected_warnings

from nmapy.features import *
from nmapy.utilities import parallel
from nmapy.utilities import io

"""
1. load a previously trained classifier from the .pkl file
2. load the feature hyperparameter information file
3. classify_pixels(...)
"""
def preprocess(data_array, feature_hyperparams, geotran, sift_codebook=None, gabor_codebook=None):
    """
    returns a single dictionary with preprocessed data along with hyperparameters
    """
    out_data = {}
    image = np.moveaxis(data_array, 0, -1) # move channels to last index
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    out_data["grayscale"] = image
    for f in feature_hyperparams:
        if f["feature"] == "w_hog":
            mag, ang = hog.__calc_mag_ang(image)
            mag = mag / 1000. # scale the magnitudes back
            ang = ang % 180 # move orientations to between 0 and 180
            out_data["mag"] = mag
            out_data["ang"] = ang
        if f["feature"] == "lbp":
            if str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])+"_"+str(f["params"]["n_points"])+"_"+str(f["params"]["radius"])+"_"+str(f["params"]["method"]) not in out_data:
                lbp_image = image.copy()
                if f["params"]["smooth_factor"]:
                    lbp_image = gaussian(lbp_image, f["params"]["smooth_factor"])
                if f["params"]["levels"]:
                    scale_factor = f["params"]["levels"]/lbp_image.max()
                    lbp_image = lbp_image.astype(float)
                    lbp_image *= scale_factor 
                    lbp_image = lbp_image.astype(int)
                lbp_image = local_binary_pattern(lbp_image, f["params"]["n_points"], f["params"]["radius"], f["params"]["method"])
                out_data[str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])+"_"+str(f["params"]["n_points"])+"_"+str(f["params"]["radius"])+"_"+str(f["params"]["method"])] = lbp_image
        if f["feature"] == "lac":
            if str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"]) not in out_data:
                lac_image = image.copy()
                if f["params"]["smooth_factor"] != None and f["params"]["levels"] != None:
                    if f["params"]["smooth_factor"]:
                        lac_image = gaussian(lac_image, f["params"]["smooth_factor"])
                    if f["params"]["levels"]:
                        scale_factor = f["params"]["levels"]/lac_image.max()
                        lac_image = lac_image.astype(float)
                        lac_image *= scale_factor 
                        lac_image = lac_image.astype(int)
                out_data[str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])] = lac_image
        if f['feature'] == "glcm":
            if str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"]) not in out_data:
                glcm_image = image.copy()
                if f["params"]["smooth_factor"]:
                    glcm_image = gaussian(glcm_image, f["params"]["smooth_factor"])
                    glcm_image = np.where(glcm_image > 1, 1, glcm_image)
                    with expected_warnings(['precision']):
                        glcm_image = skimage.img_as_ubyte(glcm_image)
                if f["params"]["levels"]:
                    #scale_factor = f["params"]["levels"]/glcm_image.max()
                    scale_factor = (f["params"]["levels"]-1)/255.
                    glcm_image = glcm_image.astype(float)
                    glcm_image *= scale_factor 
                    #glcm_image = glcm_image.astype(int)
                    glcm_image = np.round(glcm_image).astype(int)
                out_data[str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])] = glcm_image
        if f["feature"] == "sift":
            gray_im = image.copy()
            sift_keys_and_desc = sift.get_sift_keypoint_desc(gray_im, geotran)
            preds = sift.apply_codeword(sift_keys_and_desc, sift_codebook)
            sift_im = sift.get_codeword_array(gray_im, preds)
            out_data['sift'] = sift_im
        if f["feature"] == "gabor":
            gray_im = image.copy()
            bank = gabor.get_default_filter_bank() # change this if training was done with a different filterbank... need to fix this to make it nicer
            gabor_input = []
            count = 0
            for f in bank:
                gabor_input.append([gray_im, f, count])
                count+=1
            #gabor_responses = gabor.convolve_filters(gray_im, bank, mean_var=False)
            print("convolving in parallel")
            p = Pool(8)
            gabor_responses = p.map(gabor.convolve_filters_p, gabor_input) 
            gabor_responses.sort()
            gab_im = []
            for i in gabor_responses:
                gab_im.append(i[1])
            p.close()
            p.join()
            gabor_im = gabor.create_codeword_im(np.array(gab_im), gabor_codebook)
            out_data['gabor'] = gabor_im
    return out_data


def classify_pixels(image_name,
                    output,
                    block,
                    classifier,
                    scaler,
                    feature_hyperparameters,
                    sift_codebook=None,
                    gabor_codebook=None,
                    tile_size=None,
                    n_data_chunks=0,
                    n_jobs=-1):
    """
    classifies an image using an already trained classifier

    Parameters:
    -----------
    image_name: str
        the filename of the input image
    output: str
        the output filename of the classified image
    block: int
        the size of the block for processing.
        this will also be the output resolution of the
        output classified image
    classifier: str
        the filename of the saved classifier (.pkl file)
    scaler: str
        the filename of the data scaler (.pkl file)
    feature_hyperparameters: str
        the filename of the feature hyperparameter file (.pkl)
    tile_size: None or int
        if None, then the input image will not be tiled
        if int, the input image will be tiled into tiles of 
        size tile_size. In this case, tile_size must be divisible by
        block for the mosaicing to work properly
    n_data_chunks: int
        this pararameter is only used if n_jobs is greater than 1
        It specifies the number of chunks to split the data into 
        for parallel processing
    n_jobs: int
        the number of parallel jobs to intitiate for classification
        if this is less or equal to 1, then parallel processing
        will not be used

    Returns:
    --------
    None
    """
    
    # open the classifier, scaler, and feature parameter files
    with open(classifier, "rb") as clf_file:
        classifier = pickle.load(clf_file)
    with open(scaler, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open(feature_hyperparameters, "rb") as param_file:
        feature_hyperparameters = pickle.load(param_file)

    # now convert all input scales to pixels.
    # remember, the scales can be specified as either meters or pixels, but the feature
    # functions only accept scales in number of pixels
    max_scale = -999999
    for f in feature_hyperparameters:
        scales = []
        for s in f["params"]["scales"]:
            if "meters" in s:
                if "wv3" in image_name:
                    cell_width = 0.31
                if "wv2" in image_name or "ge1" in image_name:
                    cell_width = 0.46
                else:
                    cell_width = 0.46
                scale_in_pixels = int(int(s.split(" ")[0])/cell_width)
                if scale_in_pixels > max_scale:
                    max_scale = scale_in_pixels
                scales.append(scale_in_pixels)
            else:
                scale_in_pixels = int(s.split(" ")[0])
                if scale_in_pixels > max_scale:
                    max_scale = scale_in_pixels
                scales.append(scale_in_pixels)
        f["params"]["scales"] = scales

    # convert block to pixels. block can also be specified in either meters or pixels
    if "meters" in block:
        if "wv3" in image_name:
            cell_width = 0.31
        if "wv2" in image_name or "ge1" in image_name:
            cell_width = 0.46
        else:
            cell_width = 0.46
        block = int(int(block.split(" ")[0])/cell_width)
    else:
        block = int(block.split(" ")[0])
    
    print("scales: ", feature_hyperparameters[0]["params"]["scales"])
    print("block: ", + block)
    # by the start of this condition, there are a few things to note.
    # 1. max scale is defined as the maximum scale (in pixels) found for all parameters
    # 2. block and scales are both specified in number of pixels. They have been converted
    # from meters if that was how they were first described in the input
    if tile_size == None:
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
        # if "wv2" in image_name:
        #     cell_width = 0.46
        #     cell_height = 0.46
        # if "wv3" in image_name:
        #     cell_width = 0.31
        #     cell_height = 0.31

        # convert to pixels
        # max_scale = int(max_scale / cell_height)    
        # in number of pixels relative to the input data GSD
        # block = int(block / cell_width)

        out_ulx = ulx + (max_scale*in_cell_width)
        out_uly = uly + (max_scale*in_cell_height)

        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(4326)
        out_srs_wkt = out_srs.ExportToWkt()
        out_cell_width = block * in_cell_width
        out_cell_height = block * in_cell_height
        out_array, tot_time = __classify(image,
                               block,
                               feature_hyperparameters,
                               classifier,
                               scaler,
                               sift_codebook,
                               gabor_codebook,
                               cell_width,
                               max_scale,
                               n_data_chunks,
                               n_jobs,
                               geotran)
        if output:
            out_geotran = (out_ulx, out_cell_width, 0, out_uly, 0, out_cell_height)
            io.write_geotiff(output, out_array, out_geotran, out_srs_wkt)
        return out_array, tot_time
    else: # in this case, tile the large image strip into chunks
        tile_count = 0
        assert((tile_size-(max_scale*2))%block == 0), "tile_size must be divisible by block in order for mosaicing to work properly"
        ds = gdal.Open(image_name)
        geotran = ds.GetGeoTransform()
        strip_width = ds.RasterXSize
        strip_height = ds.RasterYSize
        i = 0
        while i < strip_height and strip_height-i > max_scale*2+block:
            if i + tile_size > strip_height:
                y_size = strip_height - i
            else:
                y_size = tile_size
            j = 0
            while j < strip_width and strip_width-j > max_scale*2+block: # to catch when tile width is too small to process
                if j + tile_size > strip_width:
                    x_size = strip_width - j
                else:
                    x_size = tile_size
                # create a vrt for each tile. just for metadata purposes and sanity checks
                # vrt_ops = gdal.BuildVRTOptions()
                # gdal.BuildVRT(destName=out_vrt, srcDSOrSrcDSTab=ds)

                # image is a tile of the entire image strip. The tile includes buffered regions
                image = ds.ReadAsArray(xoff=j, yoff=i, xsize=x_size, ysize=y_size)

                # set the image tile geotran to the the origin of the
                # this just requires scaling the x and y origin to the
                # right position using i and j
                image_array_geotran = (geotran[0]+geotran[1]*j,
                                       geotran[1],
                                       geotran[2],
                                       geotran[3]+geotran[5]*i,
                                       geotran[4],
                                       geotran[5])

                ulx = image_array_geotran[0]
                uly = image_array_geotran[3]
                in_cell_width = image_array_geotran[1]
                in_cell_height = image_array_geotran[5]
                out_ulx = ulx + (max_scale*in_cell_width)
                out_uly = uly + (max_scale*in_cell_height)
                out_srs = osr.SpatialReference()
                out_srs.ImportFromEPSG(4326)
                out_srs_wkt = out_srs.ExportToWkt()
                out_cell_width = block * in_cell_width
                out_cell_height = block * in_cell_height
                # the output geotran for classified data
                # this is the exact same as the image_array_geotran except
                # that it takes into account the buffer and the block resolution
                out_geotran = (out_ulx, out_cell_width, 0, out_uly, 0, out_cell_height)
                out_array, tot_time = __classify(image,
                                       block,
                                       feature_hyperparameters,
                                       classifier,
                                       scaler,
                                       sift_codebook,
                                       gabor_codebook,
                                       cell_width,
                                       max_scale,
                                       n_data_chunks,
                                       n_jobs,
                                       image_array_geotran)
                if output:
                    out_name = output[:-4] + "_" + str(i) + "_" + str(j) + ".tif"
                    tile_count += 1
                    io.write_geotiff(out_name, out_array, out_geotran, out_srs_wkt)
                    print("-------------- done with tile: " + str(tile_count) + "-----------------")
                j+=tile_size-(2*max_scale)
            i+=tile_size-(2*max_scale)
        return None, None

def __classify(image,
               block,
               feature_hyperparameters,
               classifier,
               scaler,
               sift_codebook,
               gabor_codebook,
               cell_width,
               max_scale,
               n_data_chunks,
               n_jobs,
               geotran):
    """
    image: ndarray
    """
    image_shape = image.shape
    print("begin preprocessing")
    images = preprocess(image, feature_hyperparameters, geotran, sift_codebook, gabor_codebook)
    print("end preprocessing")
    start_time = time.time()
    if n_jobs != -1: # multiprocessing
        p = Pool(n_jobs)



        # decompose input array into chunks. data chunks is a nested list [[data_id, data_array], [data_id, data_array], [data_id, data_array]]
        data_chunks = parallel.row_deco_2(image, n_data_chunks, max_scale, block)
        
        # add all the relevant parameters to do classification on the datachunks
        for i in data_chunks:
            i.append(image.shape)
            i.append(block)
            i.append(feature_hyperparameters)
            i.append(classifier)
            i.append(scaler)
            i.append(cell_width)
            i.append(max_scale)
            i.append(images)
        print("image shape: ", image.shape)
        # data chunks is now of the form:
        # [[data_id, start_idx, end_idx, image_shape, block, feature_hyperparams, classifier, scaler, cell_width, max_scale, images],
        # [data_id, start_idx, end_idx, image_shape, block, feature_hyperparams, classifier, scaler, cell_width, max_scale, images]]
        out_chunks = p.map(__classify_p, data_chunks)
        p.close()
        p.join()

        # now need to reassemble the pieces into a single out_array
        out_array = parallel.mosaic_chunks(out_chunks)

    else:
        data_chunks = parallel.row_deco_2(image, 1, max_scale, block)
        out_array = __classify_s(image_shape, block, feature_hyperparameters, classifier, scaler, cell_width, max_scale, data_chunks, images)    
    tot_time = time.time() - start_time
    print("total processing time: " + str(tot_time))
    return out_array, tot_time

def __classify_s(image_shape, block, feature_hyperparams, classifier, scaler, cell_width, max_scale, index_info, images): # classify function for sequential processing
    start_row = index_info[0][1]
    end_row = index_info[0][2]
    start_col = max_scale
    end_col = image_shape[2] - max_scale

    return feature_extraction_and_classification_loop(block,
                                                      image_shape,
                                                      start_row, end_row,
                                                      start_col, end_col,
                                                      cell_width,
                                                      feature_hyperparams,
                                                      classifier,
                                                      scaler,
                                                      images)


def __classify_p(data): # classify function for multiprocessing
    """
    classify data in parallel

    Parameters:
    -----------
    data: list
        a nested list of the form:
        [[data_id, start_idx, end_idx, image_shape, block, feature_hyperparams, classifier, scaler, cell_width, max_scale, images],
         [data_id, start_idx, end_idx, image_shape, block, feature_hyperparams, classifier, scaler, cell_width, max_scale, images], ...]
        
        data_id: an id of the row where 0 indicates the top row and -1 indicates the bottom row
        start_row_idx: the first row of the data chunk (includes buffer)
        end_row_idx: the last row of the data chunk (includes buffer)
        image_shape: a tuple (bands, rows, cols) describing the shape of the input large image
        feature_parameters: parameter list/dictionary for computing features
        max_scale: used to know the buffer size for processing

    Returns:
    ----------
    classified_data_ob: list
        a list of similar things...
    """
    data_id = data[0]
    tot_rows = data[3][1] # the number of rows in the input image (the large whole image, NOT the data chunk)
    tot_cols = data[3][2] # the number of cols in the input image (the large whole image, NOT the data chunk)
    block = data[4]
    feature_hyperparams = data[5]
    classifier = data[6]
    scaler = data[7]
    cell_width = data[8]
    max_scale = data[9]
    images = data[10]

    #start_row_idx = data[1] + max_scale
    #end_row_idx = data[2] - max_scale
    start_row_idx = data[1]
    end_row_idx = data[2]
    start_col_idx = max_scale
    end_col_idx = tot_cols - max_scale

    print("starting inference on chunk: " + str(data_id))
    out_image = feature_extraction_and_classification_loop(block,
                                                           data[3],
                                                           start_row_idx, end_row_idx,
                                                           start_col_idx, end_col_idx,
                                                           cell_width,
                                                           feature_hyperparams,
                                                           classifier,
                                                           scaler,
                                                           images)
    return [data_id, out_image]



# input params account for buffer
def feature_extraction_and_classification_loop(block,
                                               image_shape,
                                               start_row, end_row,
                                               start_col, end_col,
                                               cell_width,
                                               feature_hyperparams,
                                               classifier,
                                               scaler,
                                               images):
    samples_per_pred = 250
    number_of_output_rows = math.ceil((end_row - start_row) / block)
    number_of_output_cols = math.ceil((end_col - start_col) / block)

    print(number_of_output_rows, number_of_output_cols)
    print("begin data chunk: origin(" + str(start_row) + ", " + str(start_col) + ")  end(" + str(end_row) + "," + str(end_col) + ")")

    out_image = np.zeros(shape=(number_of_output_rows,number_of_output_cols), dtype=np.uint8)
    
    #print("output raster size: " + str(out_image.shape))

    # print("begin prediction")
    # print("classifier params: ", classifier.best_estimator_)
    count = 0
    feature_vector_group = []
    out_i = 0 # index for placing values into the correct spot of out_image
    for i in range(start_row, end_row, block):
        out_row = []
        out_j = 0 # col index for placing values into the correct spot of out_image
        for j in range(start_col, end_col, block):
            center_i = int(i + block/2.)
            center_j = int(j + block/2.)
            feature_vector = np.zeros(shape=(len(scaler.mean_)))
            place = 0 # index into feature_vector for placing the features
            for f in feature_hyperparams:
                for s in f["params"]["scales"]:
                    #scale = int(s / cell_width) # convert the scale from meters to pixels based on the GSD of input image (use cell width param) 
                    scale = s

                    top = center_i - int(scale/2.)
                    bot = center_i + int(scale/2.)
                    left = center_j - int(scale/2.)
                    right = center_j + int(scale/2.)

                    if f["feature"] == "w_hog":
                        feat_vec = hog.__weighted_hist_feature(images["mag"][top:bot+1,left:right+1],
                                                               images["ang"][top:bot+1,left:right+1])
                    elif f["feature"] == "glcm":
                        feat_vec = glcm.compute_glcm_feature(images[str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])][top:bot+1,left:right+1],
                                                             prop=f["params"]["prop"],
                                                             distances=f['params']['distances'],
                                                             angles=f['params']['angles'],
                                                             stat=f["params"]["stat"],
                                                             levels=f['params']['levels'])
                    elif f["feature"] == "pantex":
                        # FIXME TO WORK BETTER WITH TRAINING OUTPUT. NEED TO REDO the paramaters
                        # feat_vec = np.array([glcm.compute_glcm_feature(images[str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])][top:bot+1,left:right+1],
                        #                                          prop="contrast",
                        #                                          stat=["min"],
                        #                                          distances=f['params']['distances'],
                        #                                          angles=f['params']['angles'],
                        #                                          levels=f['params']['levels'])])
                        
                        feat_vec = np.array([glcm.compute_glcm_feature(images['grayscale'][top:bot+1,left:right+1],
                                                                 prop="contrast",
                                                                 stat=["min"])])

                    elif f["feature"] == "lbp":
                        feat_vec = lbp.compute_lbp_feature(images[str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])+"_"+str(f["params"]["n_points"])+"_"+str(f["params"]["radius"])+"_"+str(f["params"]["method"])][top:bot+1,left:right+1],
                                                        f["params"]["method"],
                                                        f["params"]["n_points"],
                                                        f["params"]["hist"],
                                                        f["params"]["stat"])
                    elif f["feature"] == "lac":
                        feat_vec = np.array([lac.__box_counting(images[str(f["params"]["smooth_factor"]) + "_"+ str(f["params"]["levels"])][top:bot+1,left:right+1],
                                                                f["params"]["box_size"],
                                                                f["params"]["slide_style"])])
                    elif f["feature"] == "sift":
                        bins = [n for n in range(32+1)]
                        feat_vec = np.histogram(images['sift'][top:bot+1,left:right+1], bins)[0] # could do interesting things with density=True

                    elif f["feature"] == "gabor":
                        bins = [n for n in range(32+1)]
                        feat_vec = np.histogram(images['gabor'][top:bot+1,left:right+1], bins)[0]

                    feature_vector[place:place+len(feat_vec)] = feat_vec
                    place+=len(feat_vec)
            feature_vector_group.append(feature_vector)
            count+=1
            # if count == samples_per_pred or j+block >= int(end_col):
            #     feature_vector_group = np.array(feature_vector_group)
            #     # scale the feature vector, make prediction with the classifier
            #     print("predicting row: " + str(out_i+1))
            #     scaler.transform(feature_vector_group, copy=False)
            #     preds = classifier.predict(feature_vector_group)
            #     count = 0
            #     feature_vector_group = []
            #     print(out_image)
            #     out_image[out_i][out_j:out_j+len(preds)] = preds
            #     out_j+=len(preds)

        feature_vector_group = np.array(feature_vector_group)
        # scale the feature vector, make prediction with the classifier
        # print("predicting row: " + str(out_i+1))
        scaler.transform(feature_vector_group, copy=False)
        try:
            preds = classifier.predict(feature_vector_group)
        except:
            print(np.where(np.isnan(feature_vector_group)))
            feature_vector_group = np.where(np.isnan(feature_vector_group), 0, feature_vector_group)
            preds = classifier.predict(feature_vector_group)
        count = 0
        feature_vector_group = []
        out_image[out_i][out_j:out_j+len(preds)] = preds
        out_j+=len(preds)

        out_i+=1
        #print("------------ done with row: " + str(out_i) + " ---------------")
    return out_image