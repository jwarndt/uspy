from nmapy.features import *
from nmapy.execution import *

if __name__ == "__main__":
    pantex_params = {"input":"C:/Users/4ja/data/imagery/dakar_west",
                 "output":"C:/Users/4ja/data/imagery/dakar_west_features/pantex",
                 "feature":"Pantex",
                 "block":50,
                 "scale":[50, 100, 200],
                 "jobs":4}            
    
    lbp_params = {"input":"C:/Users/4ja/data/imagery/dakar_west",
                 "output":"C:/Users/4ja/data/imagery/dakar_west_features/lbp",
                 "feature":"LBP",
                 "block":50,
                 "scale":[50, 100, 200],
                 "stat":["mean", "var"],
                 "radius":[1, 2, 3],
                 "n_points":[4, 8, 16],
                 "smooth_factor":1.5,
                 "rescale_factor":40,
                 "lbp_method":'default',
                 "jobs":2}

    mbi_params = {"input":"C:/Users/4ja/data/imagery/dakar_west",
                 "output":"C:/Users/4ja/data/imagery/dakar_west_features/mbi",
                 "feature":"MBI",
                 "postprocess":False,
                 "scale":[None],
                 "jobs":1}

    hog_params = {"input":"C:/Users/4ja/data/imagery/dakar_west",
                 "output":"C:/Users/4ja/data/imagery/dakar_west_features/hog",
                 "feature":"HOG",
                 "block":50,
                 "scale":[50, 100, 200],
                 "stat":["min","max","mean","var","skew"],
                 "jobs":4}

    input_image_dirs = ["C:/Users/4ja/data/imagery/dakar_west"]
    out_gabor_dir = "C:/Users/4ja/data/imagery/dakar_west_features/gabor"
    # gabor.create_gabor_codeword_images(input_image_dirs,
    #                                    out_gabor_dir,
    #                                    n_clusters=32,
    #                                    rand_samp_num=10000)
    gabor.create_gabor_codebook(out_gabor_dir, n_clusters=32, rand_samp_num=2000)

    # execute_texture.execute(hog_params)