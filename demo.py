# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:51:07 2022

@author: janousu
"""

from tools import dem_from_mml, delineate_catchment_from_dem
import pandas as pd

# inputs
subset = [370000,7537500,390000,7557500]
outlet_file = 'C:\SpaFHy_GISPUTS\input\stream_outlets.csv'
streams = pd.read_csv(outlet_file, sep=';', usecols=['stream'], encoding = "ISO-8859-1")['stream'].to_list()

outpath = r'F:\Pallaslake_Catchment\GIS_inputs'
dem, out_fp = dem_from_mml(outpath, subset)

# single catchment test
delineate_catchment_from_dem(dem_path=out_fp,
                             catchment_name=streams[0],
                             outfolder=outpath,
                             outlet_file=outlet_file,
                             clip_catchment=True)

# loop catchment test
for catchment_name in streams:
    delineate_catchment_from_dem(dem_path=out_fp,
                             catchment_name=catchment_name,
                             outfolder=outpath,
                             outlet_file=outlet_file,
                             clip_catchment=True)
