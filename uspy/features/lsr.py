import math
import os

import cv2
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import skimage
from skimage.color import rgb2gray
from skimage._shared._warnings import expected_warnings
from scipy.stats import entropy
import numpy as np

from nmapy.utilities.io import *

class LSR_data_object:
    
    def __init__(self, orientation_image, region_support, labeled_regions, label_id, orientation_threshold):
        self.orientation_image = orientation_image
        self.region_support = region_support
        self.labeled_regions = labeled_regions
        self.label_id = label_id
        self.orientation_threshold = orientation_threshold 
        
def __calc_mag_ang(im):
    dx = cv2.Sobel(np.float32(im), cv2.CV_32F, 1, 0, ksize=7)
    dy = cv2.Sobel(np.float32(im), cv2.CV_32F, 0, 1, ksize=7)
    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=1)
    return mag, ang, dx, dy
        
def write_lsr_shapefile(lsf_arr, output, geotran):
    ulx = geotran[0]
    uly = geotran[3]
    cell_width = geotran[1]
    cell_height = geotran[5]
    
    vector_driver = ogr.GetDriverByName("ESRI Shapefile")
    vector_ds = vector_driver.CreateDataSource(output)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = vector_ds.CreateLayer(output[:-4], srs, ogr.wkbLineString)
    line_length = ogr.FieldDefn("len", ogr.OFTReal)
    layer.CreateField(line_length)
    line_ori = ogr.FieldDefn("ori", ogr.OFTReal)
    layer.CreateField(line_ori)
    line_con = ogr.FieldDefn("con", ogr.OFTReal)
    layer.CreateField(line_con)
    for n in lsf_arr:
        out_feature = ogr.Feature(layer.GetLayerDefn())
        out_feature.SetField("len", n[0])
        out_feature.SetField("ori", np.rad2deg(n[3])+180)
        out_feature.SetField("con", n[4])
        
        dx = n[0]/(np.sqrt(4*(1+np.tan(n[3])**2)));
        dy = np.tan(n[3])*dx;
        x1 = ulx + (n[1]+dx) * cell_width
        y1 = uly + (n[2]+dy) * cell_height
        x2 = ulx + (n[1]-dx) * cell_width
        y2 = uly + (n[2]-dy) * cell_height
        line = ogr.Geometry(ogr.wkbLineString)
        line.AddPoint(x1, y1)
        line.AddPoint(x2, y2)
        #wkt_geom = line.ExportToWkt()
        out_feature.SetGeometry(line)
        layer.CreateFeature(out_feature)
        out_feature = None
    vector_ds = None
    
def lsr_feature(image_name, output, block, scale, mag_threshold=20, lsr_threshold=20, distance_threshold=8, orientation_threshold=22.5):
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

    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    out_cell_width = block * in_cell_width
    out_cell_height = block * in_cell_height

    image = np.moveaxis(image, 0, -1) # expects an image in rows, columns, channels
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    
    lsr_im = line_support_regions(image)
    
    for i in range(0, lsr_im.shape[0], block):
        outrow = []
        for j in range(0, lsr_im.shape[1], block):
            center_i = int(i+block/2.)
            center_j = int(j+block/2.)
            if center_i-int(scale/2.) < 0:
                top = 0
            else:
                top = center_i-int(scale/2.)
            if center_i+int(scale/2.) > lsr_im.shape[0]:
                bot = lsr_im.shape[0]
            else:
                bot = center_i+int(scale/2.)
            if center_j-int(scale/2.) < 0:
                left = 0
            else:
                left = center_j-int(scale/2.)
            if center_j+int(scale/2.) > lsr_im.shape[1]:
                right = lsr_im.shape[1]
            else:
                right = center_j+int(scale/2.)

            scale_arr = lsr_im[top:bot+1,left:right+1]
            feat_vec = __lsr_hist_feature(scale_arr)
            outrow.append(feat_vec)
        out_image.append(outrow)
    out_image = np.moveaxis(out_image, -1, 0)
    if output:
        out_geotran = (ulx, out_cell_width, 0, uly, 0, out_cell_height)
        write_geotiff(output, out_image, out_geotran, out_srs_wkt)
    return np.array(out_image)

def lsr_feat_vec(image_name, scales):
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    in_cell_width = geotran[1]
    in_cell_height = geotran[5]
    ds = None
    
    image = np.moveaxis(image, 0, -1) # expects an image in rows, columns, channels
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))
    
    lsr_im = line_support_regions(image)
    
    # center pixel location
    center_i = int(image.shape[0] / 2.)
    center_j = int(image.shape[1] / 2.)

    if "wv2" in image_name:
        cell_width = 0.46
    if "wv3" in image_name:
        cell_width = 0.31

    out = []
    for s in scales:
        # convert meters to pixel counts
        n_pixels = int(s / cell_width) # number of pixels for the scale
        if center_i-int(n_pixels/2) < 0:
            top = 0
        else:
            top = center_i-int(n_pixels/2)
        if center_i+int(n_pixels/2) > lsr_im.shape[0]:
            bot = image.shape[0]
        else:
            bot = center_i+int(n_pixels/2)
        if center_j-int(n_pixels/2) < 0:
            left = 0
        else:
            left = center_j-int(n_pixels/2)
        if center_j+int(n_pixels/2) > lsr_im.shape[1]:
            right = image.shape[1]
        else:
            right = center_j+int(n_pixels/2)
        
        feat_vec = __lsr_hist_feature(lsr_im[top:bot+1,left:right+1])
        out.append(feat_vec)
    return np.array(out).flatten()

def __lsr_hist_feature(lsr_im, orders=[1,2], peak_nums=2):
    """
    1. number of lines
    2. line length mean
    3. line length variance
    4. line orientation variance
    5. line contrast mean
    6. line orientation entropy
    7. line length entropy
    8. line contrast entropy
    """
    feat_vec = []
    
    orientations = lsr_im[1].flatten()
    orientations = orientations[np.where(orientations != -1)]
    lengths = lsr_im[0].flatten()
    lengths = lengths[np.where(lengths != -1)]
    contrasts = lsr_im[2].flatten()
    contrasts = contrasts[np.where(contrasts != -1)]

    feat_vec.append(len(lengths))
    feat_vec.append(np.mean(lengths))
    feat_vec.append(np.var(lengths))
    feat_vec.append(np.var(orientations))
    feat_vec.append(np.mean(contrasts))
    or_bins = np.linspace(90, 270, 51)
    len_bins = np.linspace(0, 200, 51)
    con_bins = np.linspace(0, 100, 51)

    or_hist = np.histogram(orientations, or_bins, density=True)
    len_hist = np.histogram(lengths, len_bins, density=True)
    con_hist = np.histogram(contrasts, con_bins, density=True)
    
    or_ent = entropy(or_hist[0])
    len_ent = entropy(len_hist[0])
    con_ent = entropy(con_hist[0])
    feat_vec.append(or_ent)
    feat_vec.append(len_ent)
    feat_vec.append(con_ent)
    return np.array(feat_vec)
    
def line_support_regions(array, mag_threshold=20, lsr_threshold=20, distance_threshold=8, orientation_threshold=22.5):
    """
    input is a gray scale image
    """
    # calculate gradient orientation and magnitude
    mag, ang, dx, dy = __calc_mag_ang(array)
    mag *= 0.001
    
    # tmp(edmim<magThreshold)=-1;
    temp = np.where(mag < mag_threshold, -1, ang) # set pixels to -1 if magnitude is below the mag_threshold
    
    data_ob = __label_regions(temp, distance_threshold, orientation_threshold)
    lsr_m = data_ob.labeled_regions
    line_idx = np.unique(lsr_m)
    
    # lsfarr = zeros(max(lsrM(:)),5);
    lsf_arr = np.zeros(shape=(np.max(lsr_m), 9))
    
    count = 0
    
    l = 1
    for l in range(1, np.max(line_idx)):
        # idx=find(lsrM==l);
        idx = np.argwhere(lsr_m.ravel() == l) # returns an array of indices

        # eim = zeros(size(im));
        eim = np.zeros(shape=temp.shape)

        # eim(idx) = 1;
        eim = np.where(lsr_m == l, 1, eim)

        # if (sum(eim(:)) <= lsrThreshold)
        if np.sum(eim) <= lsr_threshold: # ignore small line support region
            continue

        # ix_wi = dx(idx)
        # iy_wi = dy(idx)
        Ix_wi = dx.ravel()[idx] # extract elements in dx at index locations where lsr_m == l
        Iy_wi = dy.ravel()[idx]
        grd_wi = mag.ravel()[idx]
        # find major orientation
        ST = [[np.sum(Ix_wi**2), np.sum(Ix_wi*Iy_wi)],
              [np.sum(Ix_wi*Iy_wi), np.sum(Iy_wi**2)]]
        # V, D = eig(ST)
        # matlab returns returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D.
        D, V = np.linalg.eig(ST) # python's return of D is a 1D array, 
        D = np.diag(D) # make D on the diagonal to conform to matlab's procedure
        # if D(1,1)<D(2,2)
        #     lorn=atan(V(2,1)/V(1,1));
        # else
        #     lorn=atan(V(2,2)/V(1,2));
        # end 
        if D[0][0] < D[1][1]:
            # lorn=atan(V(2,1)/V(1,1));
            lorn = np.arctan(V[1][0]/V[0][0])
        else:
            # lorn=atan(V(2,2)/V(1,2));
            lorn = np.arctan(V[1][1]/V[0][1])

        # vote for r
        # [Ytmp,Xtmp]=ind2sub(size(im),idx);
        Ytmp, Xtmp = np.unravel_index(idx, temp.shape)
        Ytmp+=1 # indices need += 1 for some indexing weirdness...
        Xtmp+=1
        # Raccm=round(Xtmp.*cos(lorn-pi/2)+Ytmp.*sin(lorn-pi/2));
        Raccm=np.round(Xtmp*math.cos(lorn-(math.pi/2))+Ytmp*math.sin(lorn-(math.pi/2)))
        rng=np.arange(Raccm.min(),Raccm.max()+1)
        accm=np.zeros(shape=(len(rng)))
        for k in range(len(idx)):
            rc = np.round(Xtmp[k]*math.cos(lorn-math.pi/2)+Ytmp[k]*math.sin(lorn-math.pi/2))
            accm[np.where(rng==rc)] = accm[np.where(rng==rc)] + grd_wi[k]

        mxid = np.argmax(accm)
        Xmx=max(Xtmp[np.where(Raccm==rng[mxid])])
        Xmn=min(Xtmp[np.where(Raccm==rng[mxid])])
        Ymx=max(Ytmp[np.where(Raccm==rng[mxid])])
        Ymn=min(Ytmp[np.where(Raccm==rng[mxid])])

        lmx = ((Xmx+Xmn)/2) - 1
        lmy = ((Ymx+Ymn)/2) - 1
        llen = math.sqrt((Xmx-Xmn)**2+(Ymx-Ymn)**2)
        lsf_arr[count][0] = llen
        lsf_arr[count][1] = lmx
        lsf_arr[count][2] = lmy
        lsf_arr[count][3] = lorn
        lcon=np.mean(grd_wi[(np.where(Raccm==rng[mxid]))])
        lsf_arr[count][4] = lcon
        lsf_raster[0][int(lmy)][int(lmx)] = llen
        lsf_raster[1][int(lmy)][int(lmx)] = np.rad2deg(lorn)+180
        lsf_raster[2][int(lmy)][int(lmx)] = lcon
        count+=1
    lsf_arr = lsf_arr[0:count,:]
    return lsf_raster

def calc_line_support_regions(image_name, mag_threshold=20, lsr_threshold=20, distance_threshold=8, orientation_threshold=22.5, output_lsr_shapefile=False):
    """
    Parameters:
    ------------
    image_name: str
        the image filename
    mag_threshold: int or float
        pixels with magnitude above mag_threshold are considered for line support regions
    lsr_threshold: int or float
        threshold for the smallest line support region
    distance_threshold: int or float
        the size of the kernel used for counting the number of pixels contributing to
        a line support region
    orientation_threshold: int or float
        the number of degrees (+ or -) that is allowed when determining if pixels are in the same
        line support region
    
    Returns:
    --------
    lsf_raster: ndarray (3 bands, n rows, m cols)
        lsf_raster[0] is line length values (in pixels)
        lsf_raster[1] is line orientation values (in degrees)
        lsf_raster[2] is line contrast values
    geotran: gdal geotransform
    """
    ds = gdal.Open(image_name)
    image = ds.ReadAsArray()
    geotran = ds.GetGeoTransform()
    ulx = geotran[0]
    uly = geotran[3]
    in_cell_width = geotran[1]
    in_cell_height = geotran[5]
    ds = None
    
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)
    out_srs_wkt = out_srs.ExportToWkt()
    lsf_raster = np.ones(shape=(image.shape))*-1 # output lsf_raster is 3 band.
    
    image = np.moveaxis(image, 0, -1) # expects an image in rows, columns, channels
    with expected_warnings(['precision']):
        image = skimage.img_as_ubyte(rgb2gray(image))

    # calculate gradient orientation and magnitude
    mag, ang, dx, dy = __calc_mag_ang(image)
    mag *= 0.001
    
    # tmp(edmim<magThreshold)=-1;
    temp = np.where(mag < mag_threshold, -1, ang) # set pixels to -1 if magnitude is below the mag_threshold
    
    data_ob = __label_regions(temp, distance_threshold, orientation_threshold)
    lsr_m = data_ob.labeled_regions
    line_idx = np.unique(lsr_m)
    
    # lsfarr = zeros(max(lsrM(:)),5);
    lsf_arr = np.zeros(shape=(np.max(lsr_m), 5))
    
    count = 0
    
    l = 1
    for l in range(1, np.max(line_idx)):
        # idx=find(lsrM==l);
        idx = np.argwhere(lsr_m.ravel() == l) # returns an array of indices

        # eim = zeros(size(im));
        eim = np.zeros(shape=temp.shape)

        # eim(idx) = 1;
        eim = np.where(lsr_m == l, 1, eim)

        # if (sum(eim(:)) <= lsrThreshold)
        if np.sum(eim) <= lsr_threshold: # ignore small line support region
            continue

        # ix_wi = dx(idx)
        # iy_wi = dy(idx)
        Ix_wi = dx.ravel()[idx] # extract elements in dx at index locations where lsr_m == l
        Iy_wi = dy.ravel()[idx]
        grd_wi = mag.ravel()[idx]
        # find major orientation
        ST = [[np.sum(Ix_wi**2), np.sum(Ix_wi*Iy_wi)],
              [np.sum(Ix_wi*Iy_wi), np.sum(Iy_wi**2)]]
        # V, D = eig(ST)
        # matlab returns returns diagonal matrix D of eigenvalues and matrix V whose columns are the corresponding right eigenvectors, so that A*V = V*D.
        D, V = np.linalg.eig(ST) # python's return of D is a 1D array, 
        D = np.diag(D) # make D on the diagonal to conform to matlab's procedure
        # if D(1,1)<D(2,2)
        #     lorn=atan(V(2,1)/V(1,1));
        # else
        #     lorn=atan(V(2,2)/V(1,2));
        # end 
        if D[0][0] < D[1][1]:
            # lorn=atan(V(2,1)/V(1,1));
            lorn = np.arctan(V[1][0]/V[0][0])
        else:
            # lorn=atan(V(2,2)/V(1,2));
            lorn = np.arctan(V[1][1]/V[0][1])

        # vote for r
        # [Ytmp,Xtmp]=ind2sub(size(im),idx);
        Ytmp, Xtmp = np.unravel_index(idx, temp.shape)
        Ytmp+=1 # indices need += 1 for some indexing weirdness...
        Xtmp+=1
        # Raccm=round(Xtmp.*cos(lorn-pi/2)+Ytmp.*sin(lorn-pi/2));
        Raccm=np.round(Xtmp*math.cos(lorn-(math.pi/2))+Ytmp*math.sin(lorn-(math.pi/2)))
        rng=np.arange(Raccm.min(),Raccm.max()+1)
        accm=np.zeros(shape=(len(rng)))
        for k in range(len(idx)):
            rc = np.round(Xtmp[k]*math.cos(lorn-math.pi/2)+Ytmp[k]*math.sin(lorn-math.pi/2))
            accm[np.where(rng==rc)] = accm[np.where(rng==rc)] + grd_wi[k]

        mxid = np.argmax(accm)
        Xmx=max(Xtmp[np.where(Raccm==rng[mxid])])
        Xmn=min(Xtmp[np.where(Raccm==rng[mxid])])
        Ymx=max(Ytmp[np.where(Raccm==rng[mxid])])
        Ymn=min(Ytmp[np.where(Raccm==rng[mxid])])

        lmx = ((Xmx+Xmn)/2) - 1
        lmy = ((Ymx+Ymn)/2) - 1
        llen = math.sqrt((Xmx-Xmn)**2+(Ymx-Ymn)**2)
        lsf_arr[count][0] = llen
        lsf_arr[count][1] = lmx
        lsf_arr[count][2] = lmy
        lsf_arr[count][3] = lorn
        lcon=np.mean(grd_wi[(np.where(Raccm==rng[mxid]))])
        lsf_arr[count][4] = lcon
        lsf_raster[0][int(lmy)][int(lmx)] = llen
        lsf_raster[1][int(lmy)][int(lmx)] = np.rad2deg(lorn)+180
        lsf_raster[2][int(lmy)][int(lmx)] = lcon
        count+=1
    lsf_arr = lsf_arr[0:count,:]
    #lsf_arr[:,1] = ulx + lsf_arr[:,1] * in_cell_width
    #lsf_arr[:,2] = uly + lsf_arr[:,2] * in_cell_height
    if output_lsr_shapefile:
        write_lsr_shapefile(lsf_arr, os.path.join(os.path.dirname(image_name), os.path.basename(image_name)[:-4])+"_lsr_MT"+str(mag_threshold)+"_LT"+str(lsr_threshold)+"_DT"+str(distance_threshold)+"_OT"+str(orientation_threshold)+".shp", geotran)
    return lsf_raster


def __label_regions(orientation_image, distance_threshold, orientation_threshold):
    labeled_regions = np.zeros(shape=orientation_image.shape, dtype=int) # labeled image
    region_support = np.zeros(shape=orientation_image.shape, dtype=int) # counts of pixels supporting the region
    out_l = []
    out_row_idx = []
    out_col_idx = []
    
    ws = distance_threshold*2
    i = 0
    while i < orientation_image.shape[0]:
        j = 0
        while j < orientation_image.shape[1]:
            if orientation_image[i][j] >= 0:
                center_i = i
                center_j = j
                # now get the row and col indices for the kernel
                # (top, bot, left, right). The kernel is centered on orientation_image[i][j]
                if center_i - ws > 0: # avoid indexing out of bounds of the top of the array
                    top = center_i - ws
                else:
                    top = 0
                if center_i + ws < orientation_image.shape[0] - 1: # avoid indexing out of bounds of the bottom of the array
                    bot = center_i + ws
                else:
                    bot = orientation_image.shape[0] - 1
                if center_j - ws > 0: # avoid indexing out of bounds to the left of the array
                    left = center_j - ws
                else:
                    left = 0
                if center_j + ws < orientation_image.shape[1] - 1: # avoid indexing out of bounds to the right of the array
                    right = center_j + ws
                else:
                    right = orientation_image.shape[1] - 1
                
                pixel_count = 0
                ii = top
                while ii <= bot:
                    jj = left
                    while jj <= right:
                        dist = math.sqrt((center_i-ii)*(center_i-ii)+(center_j-jj)*(center_j-jj))
                        if dist <= distance_threshold and orientation_image[ii][jj] >= 0 and (ii != center_i or jj != center_j):
                            abs_orientation_diff = abs(orientation_image[center_i][center_j] - orientation_image[ii][jj])
                            # when abs_orientation_diff is large, it means that the orientations of the pixels are similar
                            c2 = 360 - abs_orientation_diff
                            # when c2 is small, it means that the orientations of the pixels are similar
                            if abs_orientation_diff < c2:
                                c = abs_orientation_diff
                            else:
                                c = c2
                            if c < orientation_threshold:
                                pixel_count += 1
                        jj+=1
                    ii+=1
                region_support[i][j] = pixel_count
                out_l.append(pixel_count)
                out_row_idx.append(i)
                out_col_idx.append(j)
            j+=1
        i+=1
    # sort the support regions based on the number of contributing pixels
    out_l, out_row_idx, out_col_idx = (list(t) for t in zip(*sorted(zip(out_l, out_row_idx, out_col_idx))))
    
    # begin expanding regions with support. start with the line with the most support first and count down from there
    label_id = 0
    data_object = LSR_data_object(orientation_image, region_support, labeled_regions, 0, orientation_threshold)
    for k in range(len(out_l)-1, -1, -1):
        if data_object.labeled_regions[out_row_idx[k]][out_col_idx[k]] == 0: # if values at this location have not been written,
            center_i = out_row_idx[k]
            center_j = out_col_idx[k]
            # if out_m[tgti + tgtj * rows] == 0:
            if data_object.region_support[center_i][center_j] == 0: # there were no pixels with similar orientation connected to eachother
                continue
            # tgt = image[tgti + tgtj * rows]
            orientation = data_object.orientation_image[center_i][center_j] # orientation at this location
            # out_m[tgti + tgtj * rows] = 0
            data_object.region_support[center_i][center_j] = 0
            data_object.label_id+=1
            top = center_i
            bot = center_i
            left = center_j
            right = center_j
            # now expand out from center_i, center_j and label the support region
            __expand(data_object, center_i, center_j, center_i, center_j)
    return data_object
            
def __expand(data_object, origin_i, origin_j, candidate_i, candidate_j):
    """
    label regions with pixels of similar orientation
    """
    abs_orientation_diff = abs(data_object.orientation_image[candidate_i][candidate_j] - data_object.orientation_image[origin_i][origin_j])
    # when abs_orientation_diff is large, it means that the orientations of the pixels are similar
    c2 = 360 - abs_orientation_diff
    # when c2 is small, it means that the orientations of the pixels are similar
    if abs_orientation_diff < c2:
        c = abs_orientation_diff
    else:
        c = c2
    # if c > ot or image[i + j * rows] < 0 or seg_res[i + j * rows] > 0:
    if c > data_object.orientation_threshold or data_object.orientation_image[candidate_i][candidate_j] < 0 or data_object.labeled_regions[candidate_i][candidate_j] > 0:
        return
    # seg_res[i + j * rows] = label_n
    data_object.labeled_regions[candidate_i][candidate_j] = data_object.label_id
    # out_m[i + j * rows] = 0
    data_object.region_support[candidate_i][candidate_j] = 0
    
    # continue expanding until the bounds of the array are reached. Worst case scenario
    # is that a line support region spans the entire image
    if candidate_i + 1 < data_object.orientation_image.shape[0]:
        __expand(data_object, candidate_i, candidate_j, candidate_i + 1, candidate_j)
    if candidate_j + 1 < data_object.orientation_image.shape[1]:
        __expand(data_object, candidate_i, candidate_j, candidate_i, candidate_j + 1)
    if candidate_i - 1 >= 0:
        __expand(data_object, candidate_i, candidate_j, candidate_i - 1, candidate_j)
    if candidate_j - 1 >= 0:
        __expand(data_object, candidate_i, candidate_j, candidate_i, candidate_j - 1)