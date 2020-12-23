import glob  # pathname pattern

from PIL import Image

# from ND2 extractor

import os
import skimage.io
import PIL
import numpy as np

from tifffile import imsave
import multiprocessing
import shutil

main_dir = r"/Volumes/Samsung_T3/20170808--Sampler--1_5--1_7/Lane_01_MANUAL"

# go through all sub folders





exp_name = "plasmid_loss"

# out_dir_base =os.path.join(main_dir, "single_frames")

all_kymo_dir =os.path.join(main_dir,  "single_frames")

try:
    os.mkdir(all_kymo_dir)
except:
    pass

lanes = range(1,7) # has to be a list
poses = range(1,30)  # second value exclusive
channels = ["-c_", "-y_", "-r_"]


# # first thing to do: put all kymo stacks from one lane to a folder
#
# for l in lanes:
#     lane_dir   = os.path.join(main_dir, "Lane" + str(l))
#     target_dir = os.path.join(all_kymo_dir, "Lane_" + str(l).zfill(2))
#     try:
#         os.mkdir(target_dir)
#     except:
#
#         pass
#
#     for p in poses:
#         # pos_dir = os.path.join(lane_dir, "pos" + str(p))
#         file_dir = os.path.join(main_dir, lane_dir, "pos" + str(p), "Kymographs")
#         os.chdir(file_dir)
#         file_list = glob.glob("Stack*")
#         for f in file_list:
#             cur_file = os.path.join(file_dir, f)
#             target_file = os.path.join(target_dir, f)
#             shutil.move(cur_file, target_file)

# second, generate sing frames from each

for l in lanes:
    target_dir = os.path.join(all_kymo_dir, "Lane_" + str(l).zfill(2))
    os.chdir(target_dir)
    c = "-c_"
    c_file_list = glob.glob("*" + c + "*")
    c_file_list.sort()
    num_trenches = len(c_file_list)



    def to_frames(i):
        t_pos_dir = os.path.join(target_dir, "pos"+str(i))
        try:
            os.mkdir(t_pos_dir)
        except:
            pass
        f = c_file_list[i]
        file_prefix = f[:-8]
        # cur_pos_files = glob.glob(file_prefix + "*")
        for c in channels:
            f = file_prefix +c+".tiff"
            im = skimage.io.imread(f)
            tframes, im_height, im_width = im.shape
            half_h = int(im_height / 2)
            for t in range(tframes):
                new_im = np.zeros((half_h, im_width * 3))
                new_im[:, im_width:im_width * 2] = im[t][:half_h, :]

                base_name = exp_name + "_" + c + "_"

                new_name = base_name + "_pos_" + str(i).zfill(4) + "_t" + str(t).zfill(3) + ".tiff"
                new_name = os.path.join(t_pos_dir, new_name)
                # print(new_name)
                imsave(new_name, new_im.astype(np.uint16))

    cores = multiprocessing.cpu_count() - 1
    jobs = []
    batch_num = int(num_trenches / cores) + 1
    p_list = range(num_trenches)

    for i in range(batch_num):
        start_ind = i * cores
        end_ind = start_ind + cores
        partial_poses =p_list[start_ind:end_ind]

        for p in partial_poses:
            j = multiprocessing.Process(target=to_frames, args=(p,))
            jobs.append(j)
            j.start()
            print(p, j.pid)

        for job in jobs:
            print(job.pid)
            job.join()


