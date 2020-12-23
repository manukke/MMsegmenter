class RecordChanges():
    ''' crappy way of passing by value and storing values during func call
    '''
    def __init__(self,lane_num,pos_num,t_frame):
        self.just_changed = True
        self.just_created = True
        self.lane_num = lane_num
        self.pos_num = pos_num
        self.t_frame = t_frame
        self.kwargs = {}
        self.fut = []
        
        
    def update(self,lane_num,pos_num,t_frame,**kwargs):
        self.just_changed = True
        self.just_created = False
        self.lane_num = lane_num
        self.pos_num = pos_num
        self.t_frame = t_frame
        self.kwargs = kwargs

    def has_changed(self, lane_num, pos_num, check_t_frame = -1, **kwargs):
        # check if values input are different than that stored
        vals_stored = set(self.kwargs.items())
        vals = set(kwargs.items())
        vals_changed = vals_stored - vals

        #return (self.kwargs != kwargs or
 
        has_changed = (len(vals_changed) > 0 or 
                      self.pos_num != pos_num or 
                      self.lane_num != lane_num or 
                      self.just_created)
        if check_t_frame > -1:
            has_changed = has_changed or (self.t_frame != check_t_frame)
        
        return has_changed
        
