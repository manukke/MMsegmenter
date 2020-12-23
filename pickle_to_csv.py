import os
import pandas as pd
main_dir = "/Volumes/SysBio-1/PAULSSON LAB/Leoncini/DATA_Ti3/20180406--HYSTERESIS_1min/HYSTERESYS_SPEEDUP_1min--Round2/Lane_01/Kymographs_Selected/Lane_001"
os.chdir(main_dir)
props = pd.read_pickle( "props_filtered.pkl" )

import glob  # pathname pattern

file_dir = "/Volumes/SysBio-1/PAULSSON LAB/Leoncini/DATA_Ti3/20180406--HYSTERESIS_1min/HYSTERESYS_SPEEDUP_1min--Round2/Lane_01/Kymographs_Selected/GFP"
# file name: Stack_Lane_01_pos_002_trench_07_top_c_RFP.tiff
os.chdir(file_dir)
file_list = glob.glob("Stack*")
order_maps = {}
for i in range(len(file_list)):
    order_maps[i] = file_list[i][:-11] # remove the extension


new_props = props.drop(columns=['bbox', 'equivalent_diameter', 'euler_number', 'extent', 'filename',
       'filled_area', 'fl0_filename', 'fl0_intensity_image','img_dir','intensity_image', 'label', 'lane_num','n_fluorescent_images',
       'orientation', 'trench_inversion_mult','centy_flipped'])

new_props = new_props.loc[props.cell_pos==0]
new_props.columns
save_dir = os.path.join(main_dir, "CSV/")
try:
    os.mkdir(save_dir)
except:
    pass
num_trench = len(file_list)

for i in range(num_trench):
    trench_info = order_maps[i]
    sub_data = new_props.loc[new_props.pos_num == i]
    sub_data.to_csv(os.path.join(save_dir, trench_info + ".csv"))