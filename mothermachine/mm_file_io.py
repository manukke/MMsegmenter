import numpy as np
import pandas as pd
import os
import warnings
from functools import reduce


def save_properties(props, filename):
    """
    saves pandas data-frame as a 'pickled' object
    :param props: any pandas dataframe
    :param filename: a string giving the full file and path where to save dataframe
    :return:
    """
    props.to_pickle(filename)

def load_properties(filename):
    """
    load a pickled pandas dataframe

    :param filename: path to pickled pandas object
    :return: pandas dataframe from file
    """
    return pd.read_pickle(filename)


def detect_image_directories(MotherMachineFile,lane_max=100,pos_max=200,t_max=5000):

    sub_dirs = [x[0] for x in os.walk(MotherMachineFile(0,0,0).basedir)]
    generated_dir_list = [MotherMachineFile(l,p,0).img_dir for l in range(lane_max) for p in range(pos_max)]
    generated_lane_pos_list = [[l,p] for l in range(lane_max) for p in range(pos_max)]

    matched_dirs = list(set(sub_dirs) & set(generated_dir_list))

    lane_pos_list = []
    for md in matched_dirs:
        idx = generated_dir_list.index(md)
        lane_pos_list.append(generated_lane_pos_list[idx])


    lane_pos_time_list = []
    for i,lp in enumerate(lane_pos_list):
        generated_file_list = [MotherMachineFile(lp[0],lp[1],t).filename for t in range(0,t_max)]
        generated_lpt_list = [[lp[0],lp[1],t] for t in range(0,t_max)]
        listed_files = os.listdir(matched_dirs[i])
        matched_files = set(listed_files) & set(generated_file_list)
        for mf in matched_files:
            idx = generated_file_list.index(mf)
            lane_pos_time_list.append(generated_lpt_list[idx])

    lane_pos_time_list.sort()

    return lane_pos_time_list


def find_shared_lane_pos_time_indcs(lane_pos_time_list, warn_drop=True,warn_missing=True):
    """
    finds the set of positions shared by all the lanes and the set of timepoints shared by all those positions
    :param lane_pos_time_list: an array of triples of the form [[lane_num,pos_num,t_frame],...]
    :param warn_drop: should a message be printed when an index must be dropped
    :param warn_missing: should a message be printed when the returned values are not contiguous
    :return: array for the lanes, array for the positions and array for the times
    """
    lpt = np.array(lane_pos_time_list)

    all_lanes = np.unique(lpt[:,0])
    all_pos = np.unique(lpt[:,1])
    all_t_frames = np.unique(lpt[:,2])

    pos_by_lane = [np.unique(lpt[lpt[:,0]==l,1]) for l in all_lanes]
    overlapping_pos = reduce(np.intersect1d,pos_by_lane)
    t_by_pos = []
    for l in all_lanes:
        pt = lpt[lpt[:,0]==l]
        t_by_pos.extend([np.unique(pt[pt[:,1]==p,2]) for p in overlapping_pos])
    overlapping_t_frames = reduce(np.intersect1d,t_by_pos)

    #hypothetical range of values for the lanes,positions and t_frames if they are contiguous
    lanes_range = range(np.min(all_lanes),np.max(all_lanes)+1)
    pos_range = range(np.min(overlapping_pos),np.max(overlapping_pos)+1)
    t_frames_range = range(np.min(overlapping_t_frames),np.max(overlapping_t_frames)+1)

    if warn_drop:

        pos_dropped = list(set(all_pos)-set(overlapping_pos))
        t_frames_dropped = list(set(all_t_frames)-set(overlapping_t_frames))

        if len(pos_dropped) > 0:
            pos_dropped_warning = "positions {pos} were dropped from the file list".format(pos=pos_dropped)
            warnings.warn(pos_dropped_warning)
        if len(t_frames_dropped)>0:
            t_frames_dropped_warning = "time frames {t_frames} were dropped from the file list".format(t_frames=t_frames_dropped)
            warnings.warn(t_frames_dropped_warning)
    if warn_missing:

        lanes_missing = list(set(lanes_range)-set(all_lanes))
        pos_missing = list(set(pos_range)-set(overlapping_pos))
        t_frames_missing = list(set(t_frames_range)-set(overlapping_t_frames))

        if len(lanes_missing) > 0:
            lanes_missing_warning = "lanes {lanes} are not in the file list".format(lanes=lanes_missing)
            warnings.warn(lanes_missing_warning)
        if len(pos_missing) > 0:
            pos_missing_warning = "positions {pos} are not in the file list".format(pos=pos_missing)
            warnings.warn(pos_missing_warning)
        if len(t_frames_missing) > 0:
            t_frames_missing_warning = "positions {t_frames} are not in the file list".format(t_frames=t_frames_missing)
            warnings.warn(t_frames_missing_warning)


    return all_lanes, overlapping_pos, overlapping_t_frames

