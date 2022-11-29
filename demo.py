# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:51:07 2022

@author: janousu
"""

from tools import delineate_catchment
import pandas as pd
# inputs
subset = [370000,7537500,390000,7557500]
outlet_file = 'F:\Pallaslake_Catchment\stream_outlets.csv'
streams = pd.read_csv(outlet_file, sep=';', usecols=['stream'], encoding = "ISO-8859-1")['stream'].to_list()

outfolder = 'F:\Pallaslake_Catchment\GIS_inputs'
for catchment_name in streams:
    print(catchment_name)
    delineate_catchment(outfolder, catchment_name, subset, outlet_file)

