import psycopg2
import os
from osgeo import gdal
from osgeo import osr

# host='160.91.165.111'
# database='RESFlow_Img_Mod_Gallery'
# user='6jg'
# table='chips'

def main():
    #outfile = open("./RESFLOW_Img_Mod_Gallery.txt", "w")
    #outfile.write("tile_name,row1,col1,row2,col2,ulx,uly,lrx,lry,bucket_id" + "\n")


    conn = psycopg2.connect(dbname='RESFlow_Img_Mod_Gallery',
                            user='6jg',
                            host='160.91.165.111',
                            port='5432')

    cur = conn.cursor()
    cur.execute("SELECT * FROM chips;")
    for record in cur:
        image_name = record[12] +"/" + record[1]
        print(image_name)
        print("image exists: " + str(os.path.exists(image_name)))
        #outfile.write(image_name + "," + record[2] + "," + record[3] + "," + record[4] + "," + record[5] + "," + record[6] + "," + record[7] + "," + record[])
        
        ds = gdal.Open(image_name)
        #print(ds.GetProjection()[-8:-3])
        break
    print("done")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()