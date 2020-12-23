import numpy as np
import holoviews as hv
#from mothermachine.record_changes import RecordChanges
from mothermachine.plot_images import plot_raw_image, plot_conn_comp
from mothermachine.mm_plotter import MotherMachinePlotter
from colorcet import gray, fire

class PlotConnCompSettings(MotherMachinePlotter):

    init_niblack_k_def = -0.3
    maxima_niblack_k_def = -0.75
    init_smooth_sigma_def = 1
    maxima_smooth_sigma_def = 2
    maxima_niblack_window_size_def = 5
    init_niblack_window_size_def = 9

    param_kdims = [
         hv.Dimension("init_niblack_k", range=(-0.7,0), step=0.1, default=init_niblack_k_def),
         hv.Dimension("maxima_niblack_k", range=(-1.55,0), step=0.1, default=maxima_niblack_k_def),
         hv.Dimension("init_smooth_sigma", range=(0,3), step=0.5, default=init_smooth_sigma_def),
         hv.Dimension("maxima_smooth_sigma", range=(0,3), step=0.5, default=maxima_smooth_sigma_def),
         hv.Dimension("init_niblack_window_size", range=(3,15), step=2, default=init_niblack_window_size_def),
         hv.Dimension("maxima_niblack_window_size", range=(3,9), step=2, default=maxima_niblack_window_size_def)
         ]

    @staticmethod
    def extract_image(t_frame,lane_num,pos_num, extract_conn_comp_func,MotherMachineFile,**kwargs):
        #static is required for dask to pickle it !!!!
        s_file = MotherMachineFile(lane_num, pos_num, t_frame)
        img = s_file.getImage()
        conn_comp = extract_conn_comp_func(img,**kwargs)
        return img, conn_comp


    def __init__(self, lane_indcs, pos_indcs, tindcs, MotherMachineFile, extract_conn_comp_func, client, has_two_lanes=False):
        MotherMachinePlotter.__init__(self, lane_indcs, pos_indcs, tindcs, MotherMachineFile)
        self.extract_conn_comp_func = extract_conn_comp_func
        self.client = client
        if has_two_lanes:
            self.cmap = fire[1:]*2
        else:
            self.cmap = fire[1:]
        
        basic_prop_dict = dict(width=self.img_cropped_width, height=self.img_cropped_height,
                                  fontsize={'title':0, 'xlabel':0, 'ylabel':0, 'ticks':0})

        img_prop_dict = basic_prop_dict.copy()
        img_prop_dict['cmap'] = self.cmap
        self.img_prop_dict = img_prop_dict
        self.rgb_prop_dict = basic_prop_dict.copy()
        self.layout_prop_dict = basic_prop_dict.copy()


    def construct_image_arrays(self):

        # need to make copies to make dask happy
        extract_conn_comp_func = self.extract_conn_comp_func
        lane_num = self.recC.lane_num.copy()
        pos_num = self.recC.pos_num.copy()
        MotherMachineFile = self.MotherMachineFile
        kwargs = self.recC.kwargs.copy()
        tindcs = self.tindcs.copy()

        g_image = lambda t_frame: PlotConnCompSettings.extract_image(t_frame,lane_num, pos_num,
                  extract_conn_comp_func,
                  MotherMachineFile,**kwargs)
        img0, conn_comp0 = g_image(self.recC.t_frame)
        fut = self.client.map(g_image, tindcs)

        return fut, img0, conn_comp0

    def show_images(self,lane_num, pos_num, t_frame, *args):
        keys = [pk.label for pk in PlotConnCompSettings.param_kdims]
        kwargs = dict(zip(keys,args))
        # only re-calculate if something other than t_frame is called
        if self.recC.has_changed(lane_num, pos_num, **kwargs):
            self.recC.just_changed = True
            self.recC.update(lane_num, pos_num, t_frame, **kwargs)
            fut, img0, conn_comp0 = self.construct_image_arrays()
            self.recC.fut = fut
        else:
            self.recC.just_changed = False

        # simply accelerates first value change
        if self.recC.just_changed:
            ri = img0
            cc = conn_comp0
        else:
            ri, cc = self.recC.fut[np.where(self.tindcs == t_frame)[0][0]].result()

        ims = (plot_raw_image(ri) + plot_conn_comp(cc,cmap=self.cmap)).cols(1)
        ims = ims.options({'RGB':self.rgb_prop_dict,'Image':self.img_prop_dict,'Layout':self.layout_prop_dict})

        return ims

    def plot(self):
        kdims = self.indcs_kdims + PlotConnCompSettings.param_kdims
        dmap = hv.DynamicMap(self.show_images, kdims = kdims)
        dmap = dmap.redim.range(y=(0.5-self.scale_factor_h,0.5),x=(-0.5,-0.5+self.scale_factor_w))
        return dmap
