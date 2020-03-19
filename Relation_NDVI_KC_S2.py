# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:59:32 2020

@author: Yann Pageot
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
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression



if __name__ == '__main__':
    """ Relation NDVI-KC issu de Sentinel 2 et SWC Lamothe """
    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/R1/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    
    df=pd.read_csv("")