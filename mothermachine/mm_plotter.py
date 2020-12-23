import holoviews as hv
from mothermachine.record_changes import RecordChanges
from mothermachine.abstract_plotter import AbstractPlotter
import PIL

class MotherMachinePlotter(AbstractPlotter):
    ''' base class to plot images using the MotherMachineFile class
    '''

    def __init__(self, lane_indcs, pos_indcs, tindcs, MotherMachineFile):
        self.MotherMachineFile = MotherMachineFile
        AbstractPlotter.__init__(self,lane_indcs,pos_indcs,tindcs)

    def update_image_dimensions(self):
        mm_file = self.MotherMachineFile(self.recC.lane_num, self.recC.pos_num, self.recC.t_frame)
        popen = PIL.Image.open(mm_file.fullfile,mode='r')
        self.img_height = popen.height
        self.img_width = popen.width

    def get_mmfile(self,lane_num,pos_num,tindx):
        return self.MotherMachineFile(lane_num, pos_num, tindx)
        
