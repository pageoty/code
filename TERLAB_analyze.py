#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:19:47 2019

@author: pageot
"""

import os
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import geopandas as geo
from scipy import stats
from  TEST_ANALYSE_SIGNATURE import *


if __name__ == '__main__':
    data_ITK=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_ITK_ADOUR_2017/ITK_STAT_2017_ADOUR.shp")
    data_ITK=data_ITK.groupby("originfid").mean()
    data_TERLAB=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_GROUND_2017/DATA_TERLABOUR/HAUTE_PYR/SURFACES-2017-PARCELLES-GRAPHIQUES-CONSTATEES_066_20180210.shp")
#    df_ITK=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_ITK_ADOUR_2017/SampleExtractionSentinel2_T30TYP_Features.tif.sqlite","df_ITK")
    Features=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_OPTIC.txt",header=None)
    Features=Features.T
    Features.columns=["band_name"]
    colnames=list(Features.band_name.apply(lambda s: s[2:-1]))
    labcroirr=data_ITK.label
    data_ITK.drop(columns=["label"],inplace=True)
    data_ITK=data_ITK.T
    data_ITK["band_names"]=colnames
    data_ITK["date"] = data_ITK.band_names.apply(lambda s: s[-8:])
    data_ITK.set_index("band_names",inplace=True)
    data_ITK=data_ITK.T
    data_ITK["labcroirr"]= labcroirr
