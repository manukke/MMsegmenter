import numpy as np
from mothermachine.DetectPeaks import detect_peaks
import pandas as pd


def stack_mother_properties(props_sort):
    """
    manipulate normal linear mother machine props data-frame so that it is indexed by the lineage and when a
    property is needed it returns a matrix of values indexed by lineage and time_frame
    :param props_sort: a pandas data_frame with:
    linear_lineage_idx (a unique integer specifying that the props are for cells in a given lineage for each lineage)
    t_frame: the integer timepoint for the measurement
    cell_pos: the position of the cell in the trench
    :return:
    """
    multi_idx_props = props_sort.set_index([props_sort.linear_lineage_idx,props_sort.t_frame,props_sort.cell_pos])
    return multi_idx_props.loc[pd.IndexSlice[:,:,0],:].unstack(0).reset_index(drop=True)

def unstack_mother_properties(mother_props, drop_na = False):
    unstack_mommy = mother_props.swaplevel(axis=1).T.unstack(level=0).T
    if drop_na:
        unstack_mommy = unstack_mommy.swaplevel()
        unstack_mommy = unstack_mommy.reset_index(drop=True)
    else:
        unstack_mommy = unstack_mommy.swaplevel().drop('linear_lineage_idx',axis=1).reset_index()
        unstack_mommy = unstack_mommy.drop('level_1',axis=1)
    return unstack_mommy


def add_mother_property(mother_props, prop_value, prop_name):
    """
    a helper function to add properties to a the pandas dataframe created by stack_mother_properties
    :param mother_props: a dataframe in the format output by stack_mother_properties
    :param prop_value: A matrix that has shape (num t_frames, num lineages)
    :param prop_name: the name to assign to this new property
    :return: the input dataframe appended with the new property
    """
    new_idx = pd.MultiIndex.from_product([[prop_name], mother_props.columns.levels[1]])
    df_clf = pd.DataFrame(prop_value ,columns = new_idx)
    return mother_props.join(df_clf)


def find_peaks(cell_len, peak_offset_threshold=-2, peak_threshold = 0, min_peak_height = 0):
    """

    :param cell_len: 1-d list of cell lengths
    :param peak_offset_threshold: the minimum difference between the value after the found peak and the value before
    Cell division is characterized by a slow increase followed by a sharp drop-off, so a negative value tends to
    select for cell division events, if too extreme it will be overly selective
    :param peak_threshold: The minimum peak prominence
    :param min_peak_height: The minimum absolute peak height
    :return: three 1-d lists:
    is_dividing: binary mask which indicates if the cell is dividing at that timepoint
    division_times: division time for proceeding division interval, i.e. time between current and previous peak
    doubling_times: exponential-fit doubling time for proceeding division interval
    """
    peak_inds = detect_peaks(cell_len, offset_threshold = peak_offset_threshold,
                             threshold = peak_threshold, mph = min_peak_height)
    
    is_dividing = np.zeros(cell_len.shape,dtype=np.bool)
    division_times = np.nan*np.ones(cell_len.shape)
    doubling_times = np.nan*np.ones(cell_len.shape)

    delta_div = np.diff(peak_inds)
    peak_ranges = [range(peak_inds[i-1]+1,peak_inds[i]+1) for i in range(1,len(peak_inds))]
    g_double = []
    for peak in peak_ranges:
        if not np.any(np.isnan(np.log(cell_len[peak]))):
            g_double.append(np.log(2)/np.polyfit(peak,np.log(cell_len[peak]),1)[0])
        else:
            g_double.append(np.nan)

    is_dividing[peak_inds] = True
    division_times[peak_inds[1:]] = delta_div
    doubling_times[peak_inds[1:]] = g_double

    return is_dividing, division_times, doubling_times

