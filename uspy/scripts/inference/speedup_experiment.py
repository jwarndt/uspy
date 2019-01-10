import os

from nmapy.classification import *

if __name__ == "__main__":
    
    in_image = "/mnt/GATES/UserDirs/4ja/data/johannesburg_cw_wv2_000024000_000078000_00114.tif"
    out = None
    block = "30 meters"

    model_dir = "/mnt/GATES/UserDirs/4ja/models"
    classifier_file = os.path.join(model_dir, "svm_model.pkl")
    scaler_file = os.path.join(model_dir, "data_scaler.pkl")
    feature_file = os.path.join(model_dir, "feature_params.pkl")

    sequentials = []
    twos = []
    fours = []
    sixes = []
    eights = []
    tens = []
    twelves = []

    times = [sequentials, twos, fours, sixes, eights, tens, twelves]
    cpus = [-1, 2, 4, 6, 8, 10, 12]

    total_trials = 30

    results_file = open("/mnt/GATES/UserDirs/4ja/experiments/speedup.txt", "w")
    header_str = "t0"
    for n in range(1, total_trials):
        header_str += ", t" + str(n)
    results_file.write(header_str + "\n")

    t = 0
    while t < len(times):
        num_trials = 0
        while num_trials < total_trials:
            print("CPUs: " + str(cpus[t]) + " trial: " + str(num_trials))
            _, tot_time = classify.classify_pixels(in_image,
                             out,
                             block,
                             classifier_file,
                             scaler_file,
                             feature_file,
                             tile_size=None,
                             n_data_chunks=cpus[t],
                             n_jobs=cpus[t])
            results_file.write(str(tot_time) + ",")
            times[t].append(tot_time)
            num_trials+=1
        results_file.write("\n")
        t+=1
    
    results_file.close()
    for r in times:
        print(r)
        print()