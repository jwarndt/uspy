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

    parser.add_argument('-imdir',
                        '--image_dir',
                        dest='image_dir',
                        required=True,
                        type=str)
    parser.add_argument('-tf',
    					'--tiles_footprint',
    					dest='tiles_footprint',
    					required=True,
    					type=str)
    parser.add_argument('-o',
                        '--output',
                        dest='output_image',
                        required=True,
                        type=str)
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=True)
    args = parser.parse_args()


    if args.verbose:
        print("---------------- Input ----------------")
        print('image directory:\t' + args.image_dir)
        print('tiles footprint:\t' + args.tiles_footprint)
        print('output image:\t' + args.output_image)
        print("---------------------------------------")
        print()
    
    return args

def __check_args(args):
	return NotImplemented

def mosaic_image_tiles(image_dir, tiles_footprint, output_image):
	image_filenames = []
	filenames = [os.path.join(image_dir, i) for i in os.listdir(image_dir)]
	for i in filenames:
		if i[-4:] == ".tif":
			image_filenames.append(i)

	


def main():
    start_time = time.time()
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    args = __parse_args()
    __check_args(args)
    mosaic_image_tiles(args.image_dir,
                       args.tiles_footprint,
                       args.output_image)

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    print('\nEnd data & time -- (%s)\nTotal processing time -- (%d min %f sec)\n' %
        (time.asctime(time.localtime(time.time())), minutes, sec))

if __name__ == '__main__':
    main()