import holoviews as hv
from abc import ABC, abstractmethod
from mothermachine.record_changes import RecordChanges


class AbstractPlotter(ABC):
    ''' base class for making standard mother machine image plots
    '''

    default_img_cropped_height = 450
    default_img_cropped_width = 700


    def __init__(self, lane_indcs, pos_indcs, tindcs):
        self.set_lpt_indcs(lane_indcs, pos_indcs, tindcs)
        self.recC = RecordChanges(lane_indcs[0],pos_indcs[0],tindcs[0])
        self.update_image_dimensions()
        self.update_desired_crop(AbstractPlotter.default_img_cropped_height,
                AbstractPlotter.default_img_cropped_width)

    @abstractmethod
    def update_image_dimensions(self):
        ''' updates values for self.img_height and self.img_width, using self.recC 
        '''

    def set_lpt_indcs(self, lane_indcs, pos_indcs, tindcs):
        self.lane_indcs = lane_indcs
        self.pos_indcs = pos_indcs
        self.tindcs = tindcs
        self.indcs_kdims = [hv.Dimension("lane_num",values=lane_indcs),
                           hv.Dimension("pos_num",values=pos_indcs),
                           hv.Dimension("t_frame", values=tindcs)]

    def update_desired_crop(self,img_height,img_width):
        self.img_cropped_height = img_height
        self.img_cropped_width = img_width
        self.scale_factor_h = self.img_cropped_height/self.img_height
        self.scale_factor_w = self.img_cropped_width/self.img_width
