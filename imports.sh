#!/bin/bash
~/anaconda3/bin/pip install dask_jobqueue more_itertools
conda config --add channels conda-forge
conda update conda
conda install pandas numpy colorcet paramiko holoviews zarr xarray datashader qgrid tqdm peakutils nd2reader natsort line_profiler jupyterthemes bokeh matplotlib mahotas
conda update bokeh pandas dask cloudpickle
