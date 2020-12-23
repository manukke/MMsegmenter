import numpy as np
import mahotas as mh
import pandas as pd
import skimage as sk
import skimage.morphology
from skimage import measure
from skimage.filters import threshold_otsu, threshold_niblack
import scipy.ndimage.morphology as morph


def perform_watershed(threshed, maxima):
    
    distances = mh.stretch(mh.distance(threshed))
    spots, n_spots = mh.label(maxima,Bc=np.ones((3,3)))
    surface = (distances.max() - distances)
    return sk.morphology.watershed(surface, spots, mask=threshed)


def detect_rough_trenches(img,max_perc_contrast = 0.97):

    def fill_holes(img):
        seed = np.copy(img)
        seed[1:-1, 1:-1] = img.max()
        strel = sk.morphology.square(3, dtype=bool)
        img_filled = sk.morphology.reconstruction(seed, img, selem=strel, method='erosion')
        img_filled = img_filled.astype(bool)
        return img_filled

    max_val = np.percentile(img,max_perc_contrast)
    img_contrast = sk.exposure.rescale_intensity(img, in_range=(0,max_val))

    img_median = mh.median_filter(img_contrast,Bc=np.ones((2,2)))
    img_edges = sk.filters.sobel(img_median)

    T_otsu = sk.filters.threshold_otsu(img_edges)
    img_otsu = img_edges > T_otsu
    img_close = morph.binary_closing(img_otsu, structure = np.ones((3,3)),iterations=6)
    img_filled = fill_holes(img_close)
    img_open = morph.binary_opening(img_filled, structure = np.ones((3,3)),iterations=2)
    trench_masks = morph.binary_dilation(img_open,structure = np.ones((9,1)))
    return trench_masks

    
def get_trench_cutbox(trench_masks, flip_trenches = False, cut_from_bottom = 90, above_trench_pad = 30):
    reg_props = sk.measure.regionprops(trench_masks*1)
    # make sure that in case there is more than one region, we grab the largest region
    rp_area = [r.area for r in reg_props]
    prop = reg_props[np.argmax(rp_area)]
    cutbox = np.zeros(trench_masks.shape,trench_masks.dtype)
    if flip_trenches:
        cutbox[prop.bbox[0]+cut_from_bottom:prop.bbox[2]+above_trench_pad,:] = 1
    else:
	start = np.max([prop.bbox[0]-above_trench_pad,0])
	end = np.max([prop.bbox[2]-cut_from_bottom,0])
        cutbox[start:end,:] = 1
    return cutbox

def detect_trenches(img, flip_trenches = False, max_perc_contrast = 97, cut_from_bottom = 90, above_trench_pad = 30):
    trench_masks = detect_rough_trenches(img,max_perc_contrast = max_perc_contrast)
    cutbox = get_trench_cutbox(trench_masks, flip_trenches = flip_trenches, cut_from_bottom = cut_from_bottom, above_trench_pad = above_trench_pad)

    trench_masks = trench_masks * cutbox
    trench_masks = morph.binary_closing(trench_masks,structure=np.ones((15,1)),iterations=6)
    trench_masks = morph.binary_erosion(trench_masks,structure=np.ones((3,3)),iterations=6)
    return trench_masks
    
    
def extract_connected_components_phase(img, trench_masks = [], flip_trenches = False, 
                                       cut_from_bottom = 105, above_trench_pad = 70,
                                       init_smooth_sigma = 3, init_niblack_window_size = 13,
                                       init_niblack_k = -0.35, maxima_smooth_sigma = 2,
                                       maxima_niblack_window_size = 11, maxima_niblack_k = -0.2,
                                       min_cell_size = 10, max_perc_contrast = 97,
                                       return_all = False):

    """
    phase segmentation and connected components detection algorithm

    :param img: numpy array containing image
    :param trench_masks: you can supply your own trench_mask rather than computing it each time
    :param flip_trenches: if mothers are on bottom of image set to True
    :param cut_from_bottom: how far to crop the bottom of the detected trenches to avoid impacting segmentation
    :param above_trench_pad: how muching padding above the mother cell
    :param init_smooth_sigma: how much smoothing to apply to the image for the initial niblack segmentation
    :param init_niblack_window_size: size of niblack window for segmentation
    :param init_niblack_k: k-offset for initial niblack segmentation
    :param maxima_smooth_sigma: how much smoothing to use for image that determines maxima used to seed watershed
    :param maxima_niblack_window_size: size of niblack window for maxima determination
    :param maxima_niblack_k: k-offset for maxima determination using niblack
    :param min_cell_size: minimum size of cell in pixels
    :param max_perc_contrast: scale contrast before median filter application
    :param return_all: whether just the connected component or the connected component, 
    thresholded image pre-watershed and maxima used in watershed
    :return: 
        if return_all = False: a connected component matrix
        if return_all = True: connected component matrix, 
        thresholded image pre-watershed and maxima used in watershed

    """

    # makes it directly compatible with fluorescense visualizations (just inverts black and white)
    init_niblack_k = -1*init_niblack_k
    maxima_niblack_k = -1*maxima_niblack_k


    def findCellsInTrenches(img,mask,sigma,window_size,niblack_k):
        img_smooth = img

        if sigma > 0:    
            img_smooth = sk.filters.gaussian(img,sigma=sigma,preserve_range=True,mode='reflect')

        thresh_niblack = sk.filters.threshold_niblack(img_smooth, window_size = window_size, k= niblack_k)
        threshed = img > thresh_niblack
        threshed = sk.util.invert(threshed)
        threshed = threshed*mask
        return threshed

    def findWatershedMaxima(img,mask):
        maxima = findCellsInTrenches(img,mask,maxima_smooth_sigma,maxima_niblack_window_size,maxima_niblack_k)
        maxima = mh.erode(maxima,Bc=np.ones((7,5)))
        reg_props = sk.measure.regionprops(sk.measure.label(maxima,neighbors=4))
        # make sure that in case there is more than one region, we grab the largest region
        rp_area = [r.area for r in reg_props]
        med_size = np.median(rp_area)
        std_size = np.std(rp_area)
        cutoff_size = int(max(0,med_size/6))
        
        maxima = sk.morphology.remove_small_objects(maxima,min_size=cutoff_size)
        return maxima

    def findWatershedMask(img,mask):
        img_mask = findCellsInTrenches(img,mask,init_smooth_sigma,init_niblack_window_size,init_niblack_k)
        img_mask = mh.dilate(mh.dilate(img_mask),Bc=np.ones((1,3),dtype=np.bool))
        img_mask = sk.morphology.remove_small_objects(img_mask,min_size=4)
        return img_mask
    
    
    if len(trench_masks) == 0:
        trench_masks = detect_trenches(img, flip_trenches = flip_trenches, max_perc_contrast = max_perc_contrast, cut_from_bottom = cut_from_bottom, above_trench_pad = above_trench_pad)
        
    img_median = mh.median_filter(img,Bc=np.ones((3,3)))
    img_mask = findWatershedMask(img_median,trench_masks)
    maxima = findWatershedMaxima(img_median,img_mask)

    conn_comp = perform_watershed(img_mask, maxima)

    # re-label in case regions are split during multiplication
    conn_comp = sk.measure.label(conn_comp,neighbors=4)
    conn_comp = sk.morphology.remove_small_objects(conn_comp,min_size=min_cell_size)

    if return_all:
        return conn_comp, trench_masks, img_mask, maxima
    else:
        return conn_comp


def extract_connected_components_standard(img, rough_thresh_otsu_mult = 0.95, rough_min_size = 30,
                                          init_smooth_sigma = 1, init_niblack_window_size = 9,
                                          init_niblack_k = -0.2, maxima_smooth_sigma = 2,
                                          maxima_niblack_window_size = 5, maxima_niblack_k = -0.75,
                                          min_cell_size = 18, return_all = False):
    """
    standard segmentation and connected components detection algorithm
    
    :param img: numpy array containing image
    :param rough_thresh_otsu_mult: parameter to adjust rough thresholding extent
    :param rough_min_size: minimum size of object after performing the rough threshold
    :param init_smooth_sigma: how much smoothing to apply to the image for the initial niblack segmentation
    :param init_niblack_window_size: size of niblack window for segmentation
    :param init_niblack_k: k-offset for initial niblack segmentation
    :param maxima_smooth_sigma: how much smoothing to use for image that determines maxima used to seed watershed
    :param maxima_niblack_window_size: size of niblack window for maxima determination
    :param maxima_niblack_k: k-offset for maxima determination using niblack
    :param min_cell_size: minimum size of cell in pixels
    :param return_all: whether just the connected component or the connected component, 
    thresholded image pre-watershed and maxima used in watershed
    :return: 
        if return_all = False: a connected component matrix
        if return_all = True: connected component matrix, 
        thresholded image pre-watershed and maxima used in watershed
            
    """
    
    def roughGlobalSegmentation(img):
        # rough large threshold to multiply out background junk
        #T_otsu = mh.otsu(img)
        T_otsu = sk.filters.threshold_otsu(img)
        rough_thresh  = (img > rough_thresh_otsu_mult*T_otsu)
        rough_thresh = sk.morphology.remove_small_objects(rough_thresh,min_size = rough_min_size)

        for i in range(3):
            rough_thresh = mh.dilate(rough_thresh)

        return rough_thresh
    
    def cleanMorphology(bw_img):
        bw_img = mh.open(bw_img)
        bw_img = mh.open(bw_img)
        return bw_img
    
    def initialSegmentation(img):
        img_smooth = img
        if init_smooth_sigma > 0:    
            img_smooth = sk.filters.gaussian(img,sigma=init_smooth_sigma,preserve_range=True,mode='reflect')

        thresh_niblack = threshold_niblack(img_smooth, window_size = init_niblack_window_size, k=init_niblack_k)
        threshed = img_smooth > thresh_niblack
        return threshed
    
    def findWatershedMaxima(img):
        img_smooth = img
        if maxima_smooth_sigma > 0:    
            img_smooth = sk.filters.gaussian(img,sigma=maxima_smooth_sigma,preserve_range=True,mode='reflect')

        thresh_niblack = threshold_niblack(img_smooth, window_size = maxima_niblack_window_size, k=maxima_niblack_k)
        maxima = img_smooth > thresh_niblack
        return maxima
        
    # mask to try to reduce background crap in final segmentation
    rough_thresh = roughGlobalSegmentation(img)
    
    threshed = initialSegmentation(img)
    threshed *= rough_thresh
    threshed = cleanMorphology(threshed)
    threshed = sk.morphology.remove_small_objects(threshed,min_size=min_cell_size)

    maxima = findWatershedMaxima(img)
    maxima *= rough_thresh
    
    # apply watershed to find boundaries
    conn_comp = perform_watershed(threshed, maxima)
    
    # re-label in case regions are split during multiplication
    conn_comp = sk.measure.label(conn_comp,neighbors=4)
    conn_comp = sk.morphology.remove_small_objects(conn_comp,min_size=min_cell_size)

    if return_all:
        return conn_comp, threshed, maxima
    else:
        return conn_comp


def calc_mean_top_n_percent(prop, n_percent):
    """ 
    calculate the mane of the top n most intense pixels in the intentsity image
    :param prop:
    :param n_percent:
    :return:
    """
    topN = np.percentile(prop.intensity_image[:], 100 - n_percent)
    meanTopN = np.mean(prop.intensity_image[prop.intensity_image > topN])
    return meanTopN

def set_region_properties(prop_dict,prop_values,prop_names,prefix=''):
    """
    add the given properties of the region to the prop dict
    :param prop_dict: dictionary of properties to append to
    :param prop_values: list of property values to add
    :param prop_names: list of names of properties being added
    :param prefix: any prefix to append to the property names
    :return: prop_dict with appended properties
    """
    
    if type(prop_values) is np.ndarray:
        assert len(prop_values) == len(prop_names), "prop values and prop names are not the same length"
        for i,pname in enumerate(prop_names):
            prop_dict[prefix + pname] = prop_values[i]
    else:
        for pname in prop_names:
            prop_dict[prefix + pname] = getattr(prop_values,pname)
            
    return prop_dict
        
def set_file_properties(prop_dict, prop, prefix=''):
    """
    append the segmentation file properties from prop to prop_dict

    :param prop_dict:
    :param prop:
    :param prefix:
    :return:
    """

    file_prop_names = ['filename','img_dir','lane_num','pos_num','t_frame']

    return set_region_properties(prop_dict,prop,file_prop_names,prefix=prefix)

def set_image_dim_properties(prop_dict, img_height=np.nan, img_width=np.nan, prefix=''):

    image_dim_prop_names = ['img_height','img_width']

    return set_region_properties(prop_dict,np.array([img_height,img_width]),image_dim_prop_names,prefix=prefix)


def set_fl_file_properties(prop_dict, props, prefixes=['fl0_','fl1_','fl2_']):
    """
    append the fluorescent file properties from props to prop_dict

    :param prop_dict:
    :param props:
    :param prefixes:
    :return:
    """

    file_prop_names = ['filename']
    n_fluor_images = len(props)
    
    for i in range(len(prefixes)):    
        if i <= (n_fluor_images - 1):
            fl_file = props[i]
        else:
            fl_file = np.array([''])
        prop_dict = set_region_properties(prop_dict, fl_file, file_prop_names, prefixes[i])
    
    prop_dict = set_region_properties(prop_dict,np.array([n_fluor_images]),['n_fluorescent_images'])
    
    return prop_dict

def set_morphological_properties(prop_dict, prop, prefix='',props_to_grab='all'):
    """
    append the morphological properties from prop to prop_dict
    :param prop_dict:
    :param prop:
    :param prefix:
    :return:
    """
    minimal_prop_names = ['area','bbox','centx','centy','label',
                          'major_axis_length','minor_axis_length']

    supp_prop_names = ['convex_area','eccentricity','equivalent_diameter',
                       'euler_number','extent','filled_area',
                       'orientation','perimeter','solidity']

    if props_to_grab == 'min':
       morphological_prop_names = minimal_prop_names
    elif props_to_grab == 'supp':
       morphological_prop_names = supp_prop_names
    else:
       minimal_prop_names.extend(supp_prop_names)
       morphological_prop_names = minimal_prop_names

    prop.centx = prop.centroid[1]
    prop.centy = prop.centroid[0]

    return set_region_properties(prop_dict,prop,morphological_prop_names,prefix=prefix)

def set_intensity_properties(prop_dict,prop,prefix=''):
    """
    append the intensity properties from prop to prop_dict
    :param prop_dict:
    :param prop:
    :param prefix:
    :return:
    """

    intensity_prop_names = ['intensity_image','max_intensity','mean_intensity',
                        'mean_top_30_percent','min_intensity']
    
    if prop is np.nan:
        prop = np.array([np.nan]*len(intensity_prop_names))
    elif type(prop) is skimage.measure._regionprops._RegionProperties:
        prop.mean_top_30_percent = calc_mean_top_n_percent(prop, 30)
    else:
        print(type(prop))
    
    return set_region_properties(prop_dict, prop, intensity_prop_names, prefix=prefix)

def extract_cells(segmentation_file, connected_components, fluorescent_files = [],props_to_grab='all'):
    """
    takes connected components, finds relevant region properties for each cell
    given all images, and assembles them into a dictionary, which can then
    be combined into a pandas dataframe -- much, much faster than creating a pandas dataframe
    each time this is run
    :param segmentation_file: AbstractMotherMachineFile object type specifying which
    file is used for segmentation
    :param connected_components: connected components matrix specifying the potential
    cell regions
    :param fluorescent_files: any additional fluorescent images of type AbstractMotherMachinFile
    :return: dictionary of properties for each region in conn_comp
    """

#    max_n_fluorescent_files = 3
#    
#    assert len(fluorescent_files) < max_n_fluorescent_files, \
#        "you can have at most %d fluorescent images" % max_n_fluorescent_files
    
    fl_prefixes = []
    for i in range(len(fluorescent_files)):
        fl_prefixes.append('fl'+str(i)+'_')
        
    seg_img = segmentation_file.getImage()
    seg_props = measure.regionprops(connected_components,intensity_image=seg_img)

    fl_props = []
    for fl_file in fluorescent_files:
        fl_props.append(measure.regionprops(connected_components,intensity_image=fl_file.getImage()))

    # intialize property dictionary so that all regions have file info
    prop_dict0 = {}
    prop_dict0 = set_file_properties(prop_dict0, segmentation_file)
    prop_dict0 = set_image_dim_properties(prop_dict0, img_height=seg_img.shape[0], img_width = seg_img.shape[1])
    prop_dict0 = set_fl_file_properties(prop_dict0, fluorescent_files, prefixes = fl_prefixes)
    
    for i in range(len(fluorescent_files)):
        prop_dict0 = set_intensity_properties(prop_dict0,np.nan,prefix=fl_prefixes[i])
        
    props = []
    for cell_idx, prop in enumerate(seg_props):
        prop_dict = dict(prop_dict0) # do not change this or it will overwrite everytime
        prop_dict = set_morphological_properties(prop_dict, prop,props_to_grab=props_to_grab)
        prop_dict = set_intensity_properties(prop_dict,prop)
        
        for i in range(len(fluorescent_files)):
            prop_dict = set_intensity_properties(prop_dict,fl_props[i][cell_idx],fl_prefixes[i])
            
        props.append(prop_dict)

    return props


