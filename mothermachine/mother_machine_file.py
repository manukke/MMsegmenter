import mahotas as mh
import skimage.io
import os
class MotherMachineFile():
    
    def __init__(self, lane_num, pos_num, t_frame):
        self.lane_num = lane_num
        self.pos_num = pos_num
        self.t_frame = t_frame
        self.filename = ""
        self.basedir = ""
        self.im_dir = ""
        self.fullfile = os.path.join(self.im_dir,self.filename)
        self._image = []
        self.is_image_extracted = False
        self.is_configured = False
        self.use_skimage = False

    def configure_file(self, basedir, imagedir, filename):
        self.is_configured = True
        self.basedir = basedir
        self.im_dir = imagedir
        self.filename = filename
        self.fullfile = os.path.join(imagedir, filename)

           
    def getImage(self):
        assert self.is_configured, "file has not been properly configured, please use the configure_file method to configure the file"
        if not self.is_image_extracted:
            if self.use_skimage:
                self._image = skimage.io.imread(self.fullfile, as_grey=True)
                self.is_image_extracted = True
            else:
                self._image = mh.imread(self.fullfile, as_grey=True)
                self.is_image_extracted = True
        return self._image
