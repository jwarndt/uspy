import math

import numpy as np

def row_deco_2(in_array, num_chunks, ghost_zone_size, block):
    """
    Returns:
    ---------
    chunks: nested list
        [[dataid, start_idx, end_idx], [dataid, start_idx, end_idx]]
        the start_idx and end_idx that is returned accounts for the 
    """
    tot_rows = in_array.shape[1]
    tot_cols = in_array.shape[2]
    rows_in_interior = tot_rows - ghost_zone_size*2
    start_idx = ghost_zone_size
    extras = 0
    while rows_in_interior%block != 0: # this loop ensures that the data chunks are divisible by block
        rows_in_interior-=1
        extras+=1

    extras_top = math.ceil(extras/2) # so only take half of the extras
    # extras_bot = math.floor(extras/2) # this isn't used
    start_idx+=extras_top # add the half the extra rows to the top of the raster to center the interior

    tot_blocks_in_interior = rows_in_interior / block
    blocks_per_chunk = tot_blocks_in_interior // num_chunks
    remaining = tot_blocks_in_interior % num_chunks

    end = start_idx + rows_in_interior # the last row of pixels in the last datachunk
    chunk_id = 0
    chunks = []
    while start_idx < end:
        end_idx = start_idx+(blocks_per_chunk*block)
        if remaining > 0:
            end_idx += block
            remaining -= 1

        #chunks.append([chunk_id,in_array[:,start_idx:end_idx,:]])
        chunks.append([chunk_id, int(start_idx), int(end_idx)]) # don't actually need the data, just the location for slices
        chunk_id+=1
        start_idx = end_idx
    return chunks

def row_decomposition(in_array, num_chunks, ghost_zone_size):
    """
    Parameters:
    -----------
    in_array: np.ndarray
        the input array. should have shape (bands, rows, columns)
    num_chunks: int
        the number of chunks to split the data into.
    ghost_zone_size: int
        in the case of focal map algebra computations, the ghost_zone_size
        indicates the number of additional rows to append to each chunk to
        ensure the focal operations are computed accurately (i.e. so that
        each computation receives it's neighbors for accurate and complete
        calculations). This should be in number of pixels.

    NOTE: there is the potential of copying a large amount of redundant data
    between data chunks if the num_chunks is not chosen carefully. It should
    be chosen with the size of the input array in mind.

    Returns:
    ---------
    chunks: nested list
        holds information about the bounds of the data chunks that will be processed
        start_row_idx and end_row_idx are indices that indicate the topmost and bottommost
        rows for processing features. These indices also take into account the buffer needed
        for computing multiscale features
        [[dataid, start_row_idx, end_row_idx], [dataid, start_row_idx, end_row_idx]]
    """

    # 1. first get the number of rows in the interior (total rows - buffer pixels for multiscale computations)
    # 2. ensure that the number of interior rows is divisible by the block size
    # 3. if not 
    #
    tot_rows = in_array.shape[1]
    tot_cols = in_array.shape[2]
    rows_in_interior = tot_rows - ghost_zone_size*2

    chunk_size = rows_in_interior // num_chunks
    remaining = rows_in_interior % num_chunks

    start_idx = 0
    end_idx = 0
    chunk_id = 0
    chunks = []
    while end_idx < tot_rows:
        end_idx = start_idx+chunk_size+(ghost_zone_size*2)
        if remaining > 0:
            end_idx += 1
            remaining -= 1

        #chunks.append([chunk_id,in_array[:,start_idx:end_idx,:]])
        chunks.append([chunk_id, start_idx, end_idx]) # don't actually need the data, just the location for slices
        chunk_id+=1
        start_idx = (end_idx)-(ghost_zone_size*2)
    return chunks

def mosaic_chunks(chunks):
    print("mosaicing results...")
    chunks.sort()
    total_rows = 0
    total_cols = chunks[0][1].shape[1]
    for c in chunks:
        total_rows+= c[1].shape[0]
    final_out = np.ndarray(shape=(total_rows,total_cols))
    row_idx = 0
    for n in chunks:
        final_out[row_idx:row_idx+n[1].shape[0],:] = n[1]
        row_idx += n[1].shape[0]
    return final_out