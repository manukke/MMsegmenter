from mothermachine.abstract_plotter import AbstractPlotter

class PropsPlotter(AbstractPlotter):
    ''' base class designed to plot properties from a pandas dataframe extracted for mother machine data
    '''

    def __init__(self, lane_indcs, pos_indcs, tindcs, props):
        self.props = props
        AbstractPlotter.__init__(self,lane_indcs,pos_indcs,tindcs)
        
    def get_props_lpt(self,lane_num,pos_num,tindx,q_return=True):
        if self.recC.has_changed(lane_num,pos_num,tindx):
            props_lpt = self.props[self.props.lane_num == lane_num]
            props_lpt = props_lpt[props_lpt.pos_num == pos_num]
            self._props_lpt = props_lpt[props_lpt.t_frame == tindx]
            
        if q_return:
            return self._props_lpt.copy()
        
    def update_image_dimensions(self):
        self.get_props_lpt(self.recC.lane_num, self.recC.pos_num, self.recC.t_frame,q_return=False)
        self.img_height = self._props_lpt.iloc[0].img_height
        self.img_width = self._props_lpt.iloc[0].img_width
