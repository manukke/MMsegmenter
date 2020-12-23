import numpy as np
import holoviews as hv
from holoviews.operation.datashader import datashade, shade
from colorcet import fire, gray

# randomly shuffle values for colormap to make different cells more obvious
fire_cmap = np.array(fire)
np.random.shuffle(fire_cmap[1:])
fire_cmap = fire_cmap.tolist()

def rescale(arr):
    im_range = np.max(arr)-np.min(arr) 
    if im_range == 0:
        im_range = 1
    return (arr - np.min(arr))/im_range 

def plot_raw_image(img):
    return shade(hv.Image(rescale(img)),normalization='linear',cmap=gray)   

def plot_conn_comp(conn_comps, cmap=[]):
    if len(cmap)==0:
        return shade(hv.Image(rescale(conn_comps)), normalization='linear', cmap=fire_cmap)
    else:
        return shade(hv.Image(rescale(conn_comps)), normalization='linear', cmap=cmap)

