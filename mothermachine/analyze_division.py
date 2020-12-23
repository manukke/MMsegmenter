import numpy as np
from mothermachine.DetectPeaks import detect_peaks
import pandas as pd
from mothermachine.post_cluster_analysis import add_mother_property, find_peaks


def calculate_and_append_divisions(props, cell_pos = 0, peak_column = 'major_axis_length', suffix='', peak_offset_threshold=-2, peak_threshold = 0, min_peak_height = 0):
    """
    determines division points, division times and doubling times and append to 
    props pandas dataframe
    :param props: pandas dataframe with cell properties
    :param cell_pos: postion of cell in trench (use 0 for mothers)
    :param peak_column: column of dataframe to run division analysis on
    :param suffix: a string suffix to add to the generated property names
    :param peak_offset_threshold: min difference between point(s) after peak and before peak
    :param peak_threshold: min peak prominence
    :param min_peak_height: min absolute peak height
    :return:
    """
    
    name_is_dividing = 'is_dividing' + suffix
    name_division_time = 'division_time' + suffix
    name_doubling_time = 'doubling_time' + suffix
    
    def calculate_divisions(prop):
        cell_length = prop[peak_column].values
        is_dividing, division_time, doubling_time = find_peaks(cell_length, 
                                                               peak_offset_threshold=peak_offset_threshold,
                                                               peak_threshold = peak_threshold, 
                                                               min_peak_height = min_peak_height)
        
        prop[name_is_dividing] = is_dividing
        prop[name_division_time] = division_time
        prop[name_doubling_time] = doubling_time
        return prop
        
    props[name_is_dividing] = np.nan
    props[name_division_time] = np.nan
    props[name_doubling_time] = np.nan

    mother_idx = props[props['cell_pos'] == cell_pos].index
    moms = props.loc[mother_idx,:]
    moms_div = moms.groupby('linear_lineage_idx').apply(lambda x: calculate_divisions(x))
    props.loc[mother_idx, name_is_dividing] = np.array(moms_div[name_is_dividing])
    props.loc[mother_idx, name_division_time] = np.array(moms_div[name_division_time])
    props.loc[mother_idx, name_doubling_time] = np.array(moms_div[name_doubling_time])

   
    return props

def _is_jumper(before_pk,after_pk,cutoff = 0):
    if (before_pk >= cutoff and after_pk < 0):
        return True
    else:
        return False


def _is_faller(before_pk,after_pk,cutoff=0):
    if (before_pk < 0 and after_pk >= cutoff):
        return True
    else:
        return False

def _detect_bad_timepoint(pk_region, cutoff=0):
    """
    designed to determine which timepoint found with cleanup lengths created the
    ugly peak, if any, still could use improvement, it's a bit of hack at the moment
    :param pk_region: frames before, during and after possible bad peak
    :return:
    """
    before_pk = pk_region[0]
    after_pk = pk_region[2]

    if _is_jumper(before_pk,after_pk,cutoff=cutoff):
        return 2, "jumper"
    elif _is_faller(before_pk,after_pk,cutoff=cutoff):
        return 1, "faller"
    else:
        return np.nan, "unknown"


def cleanup_lengths(props, minimum_peak_height = 7, cutoff_scale = 2, cell_pos=0):
    """
    tries to find spots where watershed has cut mask incorrectly or other anamolies
    have occured, it does this by simply looking for locations where there is a
    very sharp increase in the cell length in a single time-frame, such as when 
    recovering from a timepoint with a bad watershed slice
    :param props: pandas dataframe of cell properties
    :param minimum_peak_height: minimum height of peak for detected 'diff' peak
    :param cutoff_scale: scale the choice of cutoff for what is a bad frame (default: 3)
    :param cell_pos: the position of cell to run this analysis on (doesn't do lineage tracking)
    :return: properties with corrected axis lengths appended
    """
    def fix_cell_length(prop):
        cell_length = np.array(prop.major_axis_length)
        cell_length_fixed = cell_length.copy()
        diff_length = np.diff(cell_length)
        bad_peaks = detect_peaks(diff_length, mph = minimum_peak_height, show=False)

        # needed to try to handle the case where the 
        normal_peaks = detect_peaks(-diff_length, mph = 5, show=False)
        cutoff = np.median(diff_length[normal_peaks])*cutoff_scale

        for bpeak in bad_peaks:
            pk_region = diff_length[(bpeak-1):(bpeak+2)]
            bad_t_frame, _ = _detect_bad_timepoint(pk_region, cutoff=cutoff)
            if bad_t_frame is not np.nan:
                cell_length_fixed[bpeak + bad_t_frame - 1] = np.nan
        
        #interpolate the nans to fix the bad peaks that are found
        cell_length_fixed = np.array(pd.Series(cell_length_fixed.tolist()).interpolate())
        prop['major_axis_length_corrected'] = cell_length_fixed
        return prop
        
    
    if 'major_axis_length_corrected' not in props:
        props['major_axis_length_corrected'] = np.nan

    mother_idx = props[props['cell_pos'] == cell_pos].index
    moms = props.loc[mother_idx,:]
    moms_fixed = moms.groupby('linear_lineage_idx').apply(lambda x: fix_cell_length(x))
    props.loc[mother_idx,'major_axis_length_corrected'] = np.array(moms_fixed['major_axis_length_corrected'])
    return props

