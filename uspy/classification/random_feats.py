import random
from itertools import combinations
import math
import copy


def get_all_combs(in_list):
        combs = []
        for n in range(1,len(in_list)+1):
            comb = combinations(in_list,n)
            for i in list(comb):
                combs.append(i)
        return combs

class Random_Feat_Params:
    """
    A Random Feature Parameters class for generating random features to compute on images
    """
    def __init__(self, feature_names=["glcm", "w_hog", "lbp", "lac", "pantex", "sift", "gabor"], scales=["50 meters", "90 meters", "120 meters"]):
        self.avail_feats = feature_names
        self.scales = scales

    def set_avail_feats(self, newfeats):
        assert(type(newfeats) == list), "error: features must be in a list"
        self.avail_feats = newfeats

    def set_scales(self, scales):
        assert(type(scales) == list), "error: scales must be in a list"
        self.scales = scales

    def get_rand_feat(self, feature_name):
        """
        Parameters:
        ------------
        feature_name: str
            the name of the feature (i.e. 'glcm', 'pantex', etc...)
        
        Returns:
        ---------
            a python dictionary of the parameters available for the given feature
        """
        if feature_name == "glcm":
            # stats = None
            # prop = None
            # stat_combs = get_all_combs(['mean', 'var', 'min', 'max', 'sum', 'skew', 'kurtosis'])
            # while stats == None and prop == None:
            #     prop = random.choice([None,"contrast","dissimilarity","correlation","ASM","energy","homogeneity"])
            #     stats = None if random.randint(0,1) == 0 else random.choice(stat_combs)
            prop = random.choice(["contrast","dissimilarity","ASM","energy","homogeneity"]) # took out correlation
            distance = random.choice([[1,2], [1,2,3,4,5,6,7,8,9,10], [2,4,6,8,10], [1,5,10], [1,3,5,7,11,13,17]])
            angle = random.choice([[0.,math.pi/6.,math.pi/4.,math.pi/3.,math.pi/2.,(2.*math.pi)/3.,(3.*math.pi)/4.,(5.*math.pi)/6.],
              [0.,math.pi/4.,math.pi/2.,(3.*math.pi)/4.],
              [0.,math.pi/2.]])
            smooth_factor = random.choice([None, 1.5])
            levels = random.choice([16, 32, 64, 128, 200, None])
            return {"feature":"glcm",
                    "params":{"scales":self.scales,
                    "prop":prop,
                    "distances":distance,
                    "angles":angle,
                    "smooth_factor":smooth_factor,
                    "levels":levels,
                    "stat":None}}
        if feature_name == "w_hog":
            return {"feature":"w_hog",
                    "params":{"scales":self.scales}}
        if feature_name == "pantex":
            distance = random.choice([[1,2], [1,2,3,4,5,6,7,8,9,10], [2,4,6,8,10], [1,5,10], [1,3,5,7,11,13,17]])
            angle = random.choice([[0.,math.pi/6.,math.pi/4.,math.pi/3.,math.pi/2.,(2.*math.pi)/3.,(3.*math.pi)/4.,(5.*math.pi)/6.],
              [0.,math.pi/4.,math.pi/2.,(3.*math.pi)/4.],
              [0.,math.pi/2.]])
            smooth_factor = random.choice([None, 1.5])
            levels = random.choice([16, 32, 64, 128, 200, None])
            return {"feature":"pantex",
                    "params":{"scales":self.scales,
                    "prop":"contrast",
                    "distances":distance,
                    "angles":angle,
                    "smooth_factor":smooth_factor,
                    "levels":levels,
                    "stat":["min"]}}
        if feature_name == "lbp":
            radii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            radius = random.choice(radii)
            points = [4, 8, 12, 16, 20, 24, 32]
            n_points = random.choice(points)
            method = random.choice(["uniform", "default"])
            hist = True
            smooth_factor = random.choice([None, 0.5, 1.5, 2])
            levels = random.choice([16, 32, 64, 128, 200, None])
            return {"feature":"lbp",
                    "params":{"scales":self.scales,
                    "radius":radius,
                    "n_points":n_points,
                    "method":method,
                    "hist":hist,
                    "stat":None,
                    "smooth_factor":smooth_factor,
                    "levels":levels}}
        if feature_name == "lac":
            box_size = None
            while box_size == None:
                # find a random box size, the scales must be divisible by the box_size
                box_size = random.randint(2, 25)
                for n in self.scales:
                    if int(n.split(" ")[0])%box_size != 0:
                        box_size = None
                        break
            slides = [n for n in range(-1, 20)]
            slide_style = slides[random.randint(0, len(slides)-1)]
            smooth_factor = random.choice([None, 0.5, 1.5, 2])
            levels = random.choice([16, 32, 64, 128, 200, None])
            return {"feature":"lac",
                    "params":{"scales":self.scales,
                    "lac_type":"grayscale",
                    "box_size":box_size,
                    "slide_style":slide_style,
                    "smooth_factor":smooth_factor,
                    "rescale_factor":None,
                    "levels":levels}}
        if feature_name == "sift":
            n_clusters = random.choice([8, 16, 32, 64, 128])
            n_clusters = 32
            return {"feature":"sift",
                    "params":{"scales":self.scales,
                    "n_clusters":n_clusters}}
        if feature_name == "gabor":
            n_clusters = random.choice([8, 16, 32, 64, 128])
            n_clusters = 32
            mean_var = random.choice([True, False])
            thetas = random.choice([[0, math.pi/3., math.pi/6., math.pi/2., (2*math.pi)/3., (5*math.pi)/6.], 
                                    [0, math.pi/4., math.pi/2., (3*math.pi)/4]])
            sigmas = random.choice([[1,3],[2, 3.5, 7],[1, 3, 7]])
            frequencies = random.choice([[0.1], [0.9], [0.1, 0.5]])
            return {"feature":"gabor",
                    "params":{"scales":self.scales,
                    "thetas":thetas,
                    "sigmas":sigmas,
                    "frequencies":frequencies,
                    "n_clusters":n_clusters,
                    "mean_var_method":mean_var}}

    def get_n_rand_feats(self, n_feats, max_same_feature):
        """
        Parameters:
        ------------
        n_feats: int
            the number of features and their parameters to return
        max_same_feature: int
            the maximum number of the same primary feature possible
            in the returned features
        
        Returns:
        --------
        out: list
            a list of feature parameters.
            len(out) == n_feats
        """
        feats = [0 for n in range(len(self.avail_feats))] # a list to count number of times each feature is in the set
        feat_names = copy.deepcopy(self.avail_feats) # a list of the feature names
        out = []
        while len(out) < n_feats:
            feat_idx = random.randint(0, len(feat_names)-1)
            if feats[feat_idx] < max_same_feature:
                feats[feat_idx]+=1
                feat_params = self.get_rand_feat(feat_names[feat_idx])
                if feat_params not in out:
                    out.append(feat_params)
                if feat_names[feat_idx] == "pantex" or feat_names[feat_idx] == "sift" or feat_names[feat_idx] == "gabor":
                    feat_names.remove(feat_names[feat_idx])
                    del feats[feat_idx]
        return out