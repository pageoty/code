#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:10:06 2020

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
from mpl_toolkits import mplot3d
from IPython.display import HTML
import matplotlib.animation as animation
import statistics as stat

if __name__ == '__main__':
    TS_2018=pd.read_csv('/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/SampleExtractionLC08_L1TP_199030_20180701_20180716_01_T1_TST_Mask.tif.csv')
    a=TS_2018.groupby('originfid').mean()
    a[a.value_0 <=30000]=np.nan
    sns.boxplot(a.dn,a.value_0)
    TS_2017=pd.read_csv('/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/SampleExtractionLC08_L1TP_199030_20170527_20170615_01_T1_TST_Mask.tif.csv')
    a=TS_2017.groupby('originfid').mean()
    a[a.value_0 <=30000]=np.nan
    sns.boxplot(a.dn,a.value_0)