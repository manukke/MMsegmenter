# MMsegmenter
Segmentation code for Mother Machine experiments (fluorescence)

the main file you will run is a jupyter notebook called segment_images-official-training.ipynb
- to help you learn how to use the code
- when you know what you are doing, you can use the template notebook for your experiments

However, before you can jump right in, you need to install some stuff. 

Begin by downloading anaconda for python 3, on your computer. 

Once installed, look at the script: imports.sh and ensure that the path variables are correct. then run bash imports.sh

If it is successful, it should install and update several items.

You should now be ready to run things locally using the command:
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10 --port=4940
