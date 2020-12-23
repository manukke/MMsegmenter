import numpy as np
from .trenchlocs import TrenchLocs


def select_cells_in_trenches(props_all, trenchLocs = TrenchLocs.MIDDLE, below_trench_quantile = 90, above_trench_quantile = 100, mother_cell_y_offset=10,inversion_mult = 1):
    """
    selects rows from props_all that have centroids that are in the region of the trenches
    for example if the trenchlocs is MIDDLE, it selects those centroids that fall
    below 'below_trench_quantile percent' of all centroids and also fall above
    '100 - above_trench_quantile percent' of all centroids
    if any other trenchlocs are specified, the centroids are divided by whether
    they fall above or below the middle point of the image, inverted if needed,
    then the same procedure as used in MIDDLE is applied
    :param props_all: properties pandas data frame
    :param trenchLocs: which trench positions to analyze
    :param below_trench_quantile: effectively how close toward the feeding channel
    to allow cells to be
    a smaller number will keep fewer daughters and allow the code to run faster but risks losing low mothers
    a high number will cause code to run slower and risks causing trenches to merge
    due to junk in the feeding channel
    :param above_trench_quantile: how much of the trench away from feeding channel to keep
    use the highest number possible to avoid losing mothers, set lower if junk above
    mothers is causing lineages to merge
    :param mother_cell_y_offset: a bs factor to use in conjunction with above_trench_quantile
    :param inversion_mult: if for some reason your trenches are inverted compared to standard
    i.e. if a 'MIDDLE' trench faces up or a 'BOTTOM' trench faces down, then set to -1, otherwise
    don't touch it
    :return:
    """
    def indcs_in_trenches(centy,cell_indcs, invert):
        
        cy = centy[cell_indcs]
        above_trench_cut = np.percentile(invert*cy,100 - above_trench_quantile)
        below_trench_cut = np.percentile(invert*cy,below_trench_quantile)
        idx_above = (invert*centy) > above_trench_cut - mother_cell_y_offset
        idx_below = (invert*centy) < below_trench_cut
        idx_select = np.all(np.vstack((cell_indcs,idx_above,idx_below)),axis=0)
        return idx_select
        
    
    img_height = props_all.img_height
    centy = np.array(props_all.centy)
    props_all['trench_inversion_mult'] = 1*inversion_mult
    ### Note that the y-indx is flipped in image compared to matrix coords
    if trenchLocs == TrenchLocs.MIDDLE:
        idx_select = np.zeros(centy.shape,dtype=bool)
        for pos in np.unique(props_all.pos_num):
            idx_pos = np.array(props_all.pos_num == pos)
            idx_cell_pos = indcs_in_trenches(centy,idx_pos, 1*inversion_mult)
            idx_select = np.any(np.vstack((idx_select,idx_cell_pos)),axis=0)
        
        props_clean = props_all[idx_select]
    else:
        # top position in actual picture (smallest y value in matrix)
        idx_top = centy < (img_height/2) 
        idx_bottom = centy > (img_height/2)
        props_all.loc[idx_bottom,'trench_inversion_mult'] = -1*inversion_mult
        
        idx_select_top = np.zeros(idx_top.shape,dtype=bool)
        idx_select_bottom = np.zeros(idx_bottom.shape,dtype=bool)

        for pos in np.unique(props_all.pos_num):        
            idx_pos = (props_all.pos_num == pos)
            idx_top_pos = np.all(np.vstack((idx_top,idx_pos)),axis=0)
            idx_bottom_pos = np.all(np.vstack((idx_bottom,idx_pos)),axis=0)

            idx_select_top_pos = indcs_in_trenches(centy,idx_top_pos, 1*inversion_mult)
            idx_select_bottom_pos = indcs_in_trenches(centy,idx_bottom_pos, -1*inversion_mult)
            idx_select_top = np.any(np.vstack((idx_select_top,idx_select_top_pos)),axis=0)
            idx_select_bottom = np.any(np.vstack((idx_select_bottom,idx_select_bottom_pos)),axis=0)

        
        if trenchLocs == TrenchLocs.TOP:
            idx_reasonable_cells = idx_select_top
        elif trenchLocs == TrenchLocs.BOTTOM:
            idx_reasonable_cells = idx_select_bottom
        elif trenchLocs == TrenchLocs.TOP_AND_BOTTOM:
            idx_reasonable_cells = np.any(np.vstack((idx_select_top,idx_select_bottom)),axis=0)        
        
        props_clean = props_all.loc[idx_reasonable_cells,:]

    return props_clean



def select_reasonable_cells(props_all, property_bounds_dict, img_pad_pixels = 20):
    """
    choose cells that fall between certain parameters
    :param props_all: pandas dataframe in standard flattened format
    :param property_bounds_dict: a dictionary of values with the key refering to the
    scalar property to bound and the value being the range of values to accept as
    reasonable, i.e. property_bounds_dict['area'] = [10,1000] would select only
    cells that have areas > 10 and <1000
    :param img_pad_pixels: This function also gets rid of cells that are too close to the
    edges of the image, as defined by img_pad in pixels
    :return: props matrix with unreasonable cells removed
    """
    
    img_width = props_all.iloc[0,:].img_width
    img_height = props_all.iloc[0,:].img_height
    
    good_idx = []
    # first get rid of cells on edges of image
    good_idx.append([(img_pad_pixels <= x <= (img_width - img_pad_pixels) ) for x in props_all.centx])
    good_idx.append([(img_pad_pixels <= y <= (img_height - img_pad_pixels) ) for y in props_all.centy])
    
    for key, prop in property_bounds_dict.items():
        good_idx.append([prop[0] <= p <= prop[1] for p in props_all[key]])

    idx_reasonable = np.all(np.array(good_idx),axis=0)
  
    return props_all[idx_reasonable]
 
