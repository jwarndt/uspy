import numpy as np
from scipy.stats import skew, kurtosis

def calc_stat(arr, stat_name, axis=None):
    """
    Parameters:
    -----------
    arr: ndarray
        the input array
    stat_name: str
        the name of the statistics.
        "max", "min", "mean", "var", "std"
    axis: int, optional
        the axis over which the statistics is calculated
        
    Returns:
    --------
    out: ndarray
    """
    if stat_name == "all":
        out = np.array([np.amin(arr, axis), np.amax(arr, axis), np.mean(arr, axis), np.var(arr, axis), np.sum(arr, axis)])
    elif stat_name == "min":
        out = np.amin(arr, axis)
    elif stat_name == "max":
        out = np.amax(arr, axis)
    elif stat_name == "var":
        out = np.var(arr, axis)
    elif stat_name == "mean":
        out = np.mean(arr, axis)
    elif stat_name == "std":
        out = np.std(arr, axis)
    else: # stat_name == "sum":
        out = np.sum(arr, axis)
    return out

def calc_stats(arr, stat_names, axis=None):
    out = []
    for s in stat_names:
        if s == "all":
            out = np.array([np.amin(arr, axis), np.amax(arr, axis), np.mean(arr, axis), np.var(arr, axis), np.sum(arr, axis)])
            return out
        if s == "moments":
            out = [np.mean(arr, axis), np.var(arr, axis)]
            if type(axis) == tuple:
                sk = np.apply_over_axes(skew, arr, axis)
                k = np.apply_over_axes(kurtosis, arr, axis)
                out.append(sk.flatten())
                out.append(k.flatten())
            else: 
                out.extend([skew(arr, axis), kurtosis(arr, axis)])
            return np.array(out)
        if s == "min":
            out.append(np.amin(arr, axis))
        if s == "max":
            out.append(np.amax(arr, axis))
        if s == "mean":
            out.append(np.mean(arr, axis))
        if s == "var":
            out.append(np.var(arr, axis))
        if s == "std":
            out.append(np.std(arr, axis))
        if s == "sum":
            out.append(np.sum(arr, axis))
        if s == "skew":
            if type(axis) == tuple:
                sk = np.apply_over_axes(skew, arr, axis)
                out.append(sk.flatten())
            else:
                out.append(skew(arr, axis))
        if s == "kurtosis":
            if type(axis) == tuple:
                k = np.apply_over_axes(kurtosis, arr, axis)
                out.append(k.flatten())
            else:
                out.append(kurtosis(arr, axis))
    if len(out) == 1:
        if out[0].shape == (): # this means it's a scalar value that needs to be returned
            return out[0]
        else:
            return np.array(out[0]) # this means that there is only one element in the list, but that element is an array
    else:
        return np.array(out)