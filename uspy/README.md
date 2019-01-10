# nmapy
A python package for creating textural features from satellite images.

# Installation
1. download the repository
2. unzip and place the package in the site-packages folder of the
anaconda environment you would like to use the package with.  
Example: `C:/Users/4ja/AppData/Local/Continuum/miniconda3/envs/py36/Lib/site-packages`  
Example: `C:/Users/4ja/AppData/Local/Continuum/Anaconda3/Lib/site-packages`  
The location of your site-packages folder may be different depending on your
install location of Anaconda/Miniconda and your virtual environments.
3. Once you have moved the contents of nmapy to the site-packages folder, open anaconda, activate your environment, 'cd'
to the appropriate site-packages folder and run  
`pip install nmapy`  
This should make sure that the supporting packages are installed in your environment

# Dependencies
numpy  
GDAL  
scikit-learn  
scikit-image  
scipy  
opencv-python  
opencv-contrib-python  

# Usage and Examples

### Available Textural Features  
Gray-Level Co-occurence Matrix (GLCM) [1]  
Histogram of Oriented Gradients (HOG) [2]  
Built-up Presence Index (PANTEX) [3]  
Lacunarity (Lac) [4,5,6]  
Local Binary Pattern (LBP) [7]  
Scale-Invarient Feature Transform (SIFT) [8]  
Gabor Filters [9]  
Morpohological Building Index (MBI) [10,11]  
Line Support Regions (LSR) - still in progress [12,13,14]  
### Processing Parameters and Fundamentals [15]
This package uses a 'block' as the base unit for computing textural features in the image. When computing textural features, a **block** of size `block x block` (in pixels) will be moved over the image. Blocks do not overlap. At each block location in the image, textural features are computed using all pixels that fall within a given **scale**. A scale defines a window of size `scale x scale` (in pixels) centered on the block. After the textural feature is computed, the value or feature vector is written to a pixel of size `block x block` at the current location of the block. The spatial resolution of the output textural feature image is therefore defined by `block size * input image pixel size`.
  
  
![](docs/images/processing_example.PNG)  
**Figure 1**: An example of how textural features for an image are computed using blocks and scales. This is a WorldView-2 image and has a spatial resolution of 0.46 m. The image is being processed using a block size of 50 and scale size of 100. A textural feature is computed using all the pixels that fall withing the window of scale size 100. The window is centered around the block. Areas shaded in blue are blocks that have already been processed. The output spatial resolution of the textural feature image will be 23 meters.  
### Python
See the docs for more information on the textural features, their implementation, and parameters.
There are more examples and demos in the 'notebooks' folder. :)  
###### Compute HOG features for a single image
```python
from nmapy.features import *

in_image_name = "C:/Users/4ja/data/imagery/dakar_e_wv2_05272018_000015000_000005000_00023.tif"
out_image_name = "C:/Users/4ja/data/imagery/features/hog_image_BK50_SC50.tif"

hog.hog_feature(image_name=in_image_name,
                block=50,
                scale=50,
                output=out_image_name,
                stat=None)
```  
###### Compute GLCM features for a single image
```python
from nmapy.features import *

in_image_name = "C:/Users/4ja/data/imagery/dakar_e_wv2_05272018_000015000_000005000_00023.tif"
out_image_name = "C:/Users/4ja/data/imagery/features/glcm_image_BK50_SC100_corr.tif"

glcm.glcm_feature(image_name=in_image_name,
                  block=50,
                  scale=100,
                  output=out_image_name,
                  stat=None,
                  prop="correlation")
```
###### Compute LBP features for a single image and return the results in memory.
```python
from nmapy.features import *

in_image_name = "C:/Users/4ja/data/imagery/dakar_e_wv2_05272018_000015000_000005000_00023.tif"

lbp_im = lbp.lbp_feature(image_name=in_image_name,
                         block=50,
                         scale=100,
                         output=None,
                         method='default',
                         radius=[4, 4, 8, 8],
                         n_points=[8, 16, 8, 16],
                         stat=["moments"])
```

### Command Line Scripts
(in progress)  

### GUI
###### Using your conda terminal, open the GUI from the nmapy_gui subpackage by typing `python nmapy_gui.py` in the terminal
```console
(py36) C:\Users\4ja\AppData\Local\Continuum\miniconda3_64bit\envs\py36\Lib\site-packages\nmapy> cd nmapy_gui
(py36) C:\Users\4ja\AppData\Local\Continuum\miniconda3_64bit\envs\py36\Lib\site-packages\nmapy\nmapy_gui> python nmapy_gui.py
```
###### The GUI should appear and you can set parameters and begin processing
![](docs/images/nmapy_gui_opening.PNG)  
  
  
![](docs/images/nmapy_gui_availfeatures.PNG) 
  
  
![](docs/images/nmapy_gui_pantexempty.PNG)  
  
  
# References
[1] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification", *IEEE Trans. Syst., Man, Cybern.*, vol. SMC-3, no. 11, pp. 610-621, Nov. 1973.  
[2] N. Dalal, and B. Triggs, "Histogram of oriented gradients for human detection", *Proceedings of the 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2005.  
[3] M. Pesaresi, A. Gerhardinger, and F. Kayitakire, "A robust built-up area presence index by anisotropic rotation-invariance textural measure", *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 1, no. 3, pp. 180-192, Sep. 2008.  
[4] R. E. Plotnick, R. H. Gardner, and R. V. O'Neill, "Lacunarity indices as measures of landscape texture", *Landsace Ecology*, vol. 8, no. 3, pp. 201-211, 1993.  
[5] S. W. Myint, V. Mesev, and N. Lam, "Urban textural analysis from remote sensor data: lacunarity measurements based on the differential box counting method", *Geographical Analysis*, vol. 38, pp. 371-390, Nov. 2006.  
[6] M. N. Barros Filho, and F. J. A. Sobreira, "Accuracy of lacunarity algorithms in texture classification of high spatial resolution images from urban areas", *The International Archives of the Photgrammetry, Remote Sensing and Spatial Information Sciences*, vol. XXXVII, part B3b, pp. 417-422, 2008.    
[7] T. Ojala, M. Pietikainen, and T. Maenpaa, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns", *IEEE Transactions on Patternt Analysis and Machine Intelligence*, vol. 24, no. 7, pp. 971-987, Jul. 2002.   
[8] D. G. Lowe, "Object recognition from local scale-invariant features", *Proc. of the International Conference on Computer Vision*, pp. 1-8, 1999.  
[9]  
[10] X. Huang, and L. Zhang, "A multidirectional and multiscale morphological index for automatic building extraction from multispectral GeoEye-1 imagery", *Photogrammetric Engineering & Remote Sensing*, vol. 77, no. 7, pp. 721-732, Jul. 2011.  
[11] X. Huang, and L. Zhang, "Morphological building/shadow index for building extraction from high-resolution imagery over urban areas", *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 5, no. 1, pp. 161-172, Feb. 2012.  
[12] J. B. Burns, and A. R. Hanson, "Extracting Straight Lines", *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-8, no. 4, pp. 425-455, Jul. 1986.  
[13] C. Unsalan, and K. L. Boyer, "Classifying land development in high-resolution panchromatic satellite images using straight-line statistics", *IEEE Transactions on Geoscience and Remote Sensing*, vol. 42, no. 4, pp. 907-919, Apr. 2004.  
[14] J. Yuan, and A. M. Cheriyadat, "Learning to count buildings in diverse aerial scenes", *Association for Computing Machinery SIGSPATIAL'14 GIS*, 2014.   
[15] J. Graesser, A. Cheriyadat, R. R. Vatsavai, V. Chandola, J. Long, and E. Bright, "Image based characterization of formal and informal neighborhoods in an urban landscape", *IEEE Journal of selected topics in applied earth observations and remote sensing*, vol. 5, no. 4, pp. 1164-1176, 2012.  