import numpy as np
import holoviews as hv
from mothermachine.record_changes import RecordChanges
import skimage as sk
from mothermachine.plot_images import plot_raw_image,plot_conn_comp
from mothermachine.props_plotter import PropsPlotter
from mothermachine.tools import build_conn_comp_fast
from bokeh.models import HoverTool
from colorcet import gray, fire
import os

fire_cmap = np.array(fire[1:])
np.random.shuffle(fire_cmap[1:])
fire_shuffle = fire_cmap.tolist()


class PlotConnCompResults(PropsPlotter):

    def __init__(self, lane_indcs, pos_indcs, tindcs, props):
        
        PropsPlotter.__init__(self, lane_indcs, pos_indcs, tindcs, props.copy())
        
        cent_x_scaled = (self.props.centx+0.5)/self.img_width-0.5
        cent_y_scaled_flipped = abs(1-(self.props.centy+0.5)/self.img_height)-0.5
        
        self.props['cent_x_scaled'] = cent_x_scaled
        self.props['cent_y_scaled_flipped'] = cent_y_scaled_flipped
        self.props['label_orig'] = self.props.label.values.copy()
        
        
        t_tips = [("centx", "@centx"),("centy", "@centy"),("area", "@area"),
                  ("solidity", "@solidity"),("label","@label_orig")]
        
        vdims = ["centx","centy","area","solidity","pos_num","colorby","label","label_orig"]

        if "linear_lineage_idx" in props.columns:
            t_tips.append(("lineage","@linear_lineage_idx"))
            vdims.extend(["linear_lineage_idx","lineage_idx"])
            self.props['colorby'] = self.props['lineage_idx'].values + 1
            self.c_cmap = fire_shuffle
        else:
            abs_y = abs(cent_y_scaled_flipped)
            colorby = (abs_y - min(abs_y))/(max(abs_y)-min(abs_y))
            self.props['colorby'] = np.array(colorby.values*10000,dtype=np.uint16).tolist()
            self.c_cmap = fire
        
        self.max_dim = np.max(self.props['colorby'])
        self.props['label'] = self.props['colorby']
        self.hover = HoverTool(tooltips=t_tips)
        self.vdims = vdims
        basic_prop_dict = dict(width=self.img_cropped_width, height=self.img_cropped_height, 
                                  fontsize={'title':0, 'xlabel':0, 'ylabel':0, 'ticks':0})
        
        img_prop_dict = basic_prop_dict.copy()
        img_prop_dict['cmap'] = self.c_cmap
        
        scatter_prop_dict = basic_prop_dict.copy()
        scatter_prop_dict['color'] = 'g'
        scatter_prop_dict['size'] = 3
        
        self.layout_prop_dict = basic_prop_dict
        self.img_prop_dict = img_prop_dict
        self.scatter_prop_dict = scatter_prop_dict

            
    def show_images_and_centroids(self,lane_num,pos_num,t_frame):
        
        pt_all = self.get_props_lpt(lane_num,pos_num,t_frame)        
        filename = os.path.join(pt_all.iloc[0].img_dir,pt_all.iloc[0].filename)
        img = sk.io.imread(filename,as_grey=True)

        pt_no_obj = pt_all.select_dtypes(exclude=np.object).copy()
        
        hv_img = plot_raw_image(img)        
        conn_comp = build_conn_comp_fast(pt_all)
        hv_conn_comp = hv.Image(conn_comp).opts(plot=dict(cmap=self.c_cmap))
        pt_ds = hv.Dataset(pt_no_obj,kdims=['cent_x_scaled','cent_y_scaled_flipped'], vdims=self.vdims)
        hv_cents = hv.Scatter(pt_ds).opts(plot=dict(tools=[self.hover]))

        ims = (hv_conn_comp*hv_cents + hv_img*hv_cents).cols(1)
        ims = ims.options({'Image':self.img_prop_dict,'Scatter':self.scatter_prop_dict,'Layout':self.layout_prop_dict})
        return ims

    def plot(self):
        dmap = hv.DynamicMap(self.show_images_and_centroids, kdims = self.indcs_kdims)
        dmap = dmap.redim.range(y=(0.5-self.scale_factor_h,0.5), x=(-0.5,-0.5+self.scale_factor_w))
        dmap = dmap.redim.range(cent_y_scaled_flipped=(0.5-self.scale_factor_h,0.5), cent_x_scaled=(-0.5,-0.5+self.scale_factor_w))
        dmap = dmap.redim.range(z=(0,self.max_dim))
        return dmap

