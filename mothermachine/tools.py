import numpy as np
import pandas as pd
import warnings
from functools import reduce
import os

def make_cell_montage(images, n_cols=20):
    """

    :param images: a list of images to append together
    :param n_cols: the number columns of images to display, number of rows will be dictated by the length of images
    :return: a matrix containing the images concatenated together
    """

    images = pd.Series(images)
    
    max_x = np.max([i.shape[0] for i in images.dropna()])
    max_y = np.max([i.shape[1] for i in images.dropna()])

    # put image into full size bounding box
    padded_int_images = []
    for im in images:
        new_im = np.zeros((max_x,max_y))
        if im is not None:
            new_im[:im.shape[0],:im.shape[1]] = im
        padded_int_images.append(new_im)
    
    # unless n_images is divisible by n_cols, add one to n_rows
    n_rows = (len(padded_int_images)//n_cols)
    if (len(padded_int_images) % n_cols) > 0:
        n_rows = n_rows + 1
    
    # add blank images so that it has correct number of rows and columns
    n_blank_imgs = n_rows*n_cols - len(padded_int_images)
    for i in range(n_blank_imgs):
        padded_int_images.append(np.zeros((max_x,max_y)))
        
    np_padded_int_images = np.array(padded_int_images)  
    
    # build list of n_rows total rows, each row has n_columns
    im_rows = []
    for i in range(n_rows):
        im_rows.append(np.concatenate(np_padded_int_images[i*n_cols:(i+1)*n_cols], axis=1))
    
    # combine rows and return
    return np.concatenate(im_rows)

def build_conn_comp(props_clean, lane_num, pos_num, t_frame, return_all=False):
    ''' create a connected components image from the pandas dataframe with a bbox, intensity_image and label
    '''
    props_lpt = props_clean[props_clean.lane_num == lane_num]
    props_lpt = props_lpt[props_lpt.pos_num == pos_num]
    props_lpt = props_lpt[props_lpt.t_frame == t_frame]
    conn_comp = build_conn_comp_fast(props_lpt)
    if return_all:
        return conn_comp, props_lpt
    else:
        return conn_comp 


def build_conn_comp_fast(props_lpt):
    ''' if you already have properties for a specific lane, position and time, use this
    '''
    conn_comp = np.zeros((props_lpt.iloc[0,:].img_height,props_lpt.iloc[0,:].img_width))
    for i in range(props_lpt.shape[0]):
        pt = props_lpt.iloc[i,:]
        bbox = pt.bbox
        conn_comp[bbox[0]:bbox[2],bbox[1]:bbox[3]] += (pt.intensity_image > 0) * pt.label            
    return conn_comp 


def extract_attribute_into_array(props_sort, attribute, cell_pos = 0, idx = -1):
    tsort_moms = props_sort[props_sort['cell_pos'] == cell_pos]
    grouped_moms = tsort_moms.groupby(['pos_num','lineage_idx'])
    final_t_frame = np.max(tsort_moms.t_frame)
    n_lineages = len(grouped_moms)
    prop_array = np.NaN * np.ones((final_t_frame+1,n_lineages))
    iLineage = -1
    for name, group in grouped_moms:
        if name[1] > -1:
            iLineage += 1
            t_frames = np.array(group.t_frame)
            if idx > -1:
                prop_array[t_frames,iLineage] = np.array([g[idx] for g in group.loc[:,attribute]])
            else:
                prop_array[t_frames,iLineage] = group.loc[:,attribute]

    return prop_array

