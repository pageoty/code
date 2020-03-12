#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:29:18 2020

@author: pageot
"""


import os
import sqlite3
import geopandas as geo
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import TEST_ANALYSE_SIGNATURE
import shapely.geometry as geom
import descartes
import pickle




if __name__ == '__main__':
    
    df=pickle.load(open('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/test_SAMIR_labo/Inputdata/outputtest_SAMIR_labo_2018.df','rb'))
   
    ETsum = (df.groupby(['LC', 'id'])['ET'].sum()).reset_index()
    LCclasses = df.LC.cat.categories
    
    ETmin = ETsum.ET.min()
    ETmax = ETsum.ET.max()
    ET= {}
    for lc in LCclasses:
        ET[lc] = ETsum.loc[ETsum.LC == lc].reset_index()
        ET[lc].drop(columns = ['LC', 'index'], inplace = True)
    ETmin, ETmax
    LCclasses = df.LC.cat.categories

    gdf = {}
    for lc in LCclasses:


        gdf[lc] = geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/test_SAMIR_labo/Inputdata/shapefiles/PARCELLE_LABO_ref.shp")
        gdf[lc] = gdf[lc].merge(ET[lc], on='id') # attention ID en minicuel dans le shape
        
        gdf[lc].plot(column='ET',figsize=(10,10), vmin=ETmin, vmax=ETmax, cmap='RdYlGn', legend=True)
        plt.title(lc + '   : Evapotranspiration')
        
# =============================================================================
        # crée plot for 
# =============================================================================
