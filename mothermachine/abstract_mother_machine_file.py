from abc import ABC, abstractmethod
import skimage.io
import os

class AbstractMotherMachineFile(ABC):
    
    def __init__(self, lane_num, pos_num, t_frame, img_channel = 0):
        self.lane_num = lane_num
        self.pos_num = pos_num
        self.t_frame = t_frame
        self.img_channel = img_channel
        self.filename = self.construct_filename()
        self.basedir = self.set_base_directory()
        self.img_dir = self.construct_image_directory()
        self.fullfile = os.path.join(self.img_dir, self.filename)
        self._image = []
        self.is_image_extracted = False
         
    @abstractmethod
    def set_base_directory(self):
        return ""   

    @abstractmethod
    def construct_image_directory(self):
        return ""
    
    @abstractmethod
    def construct_filename(self):
        return ""
           
    def getImage(self):
        if not self.is_image_extracted:
            self._image = skimage.io.imread(self.fullfile, as_grey=True)
            self.is_image_extracted = True
        return self._image
