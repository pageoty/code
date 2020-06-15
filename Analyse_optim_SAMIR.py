# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:34:29 2020

@author: Yann Pageot
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:25:10 2020

@author: Yann Pageot
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
from scipy.optimize import minimize
from sklearn.metrics import *
from scipy.optimize import linprog
from scipy import optimize
import random
import seaborn as sns
import pickle
from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    
    name_run="RUNS_optim_LUT_LAM_ETR/"
    d={}
    d['SAMIR_run']="/mnt/d/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    d['SAMIR_run_Wind']="D:/THESE_TMP/RUNS_SAMIR/"+name_run+"/"
    
    resultat=pd.read_csv(d["SAMIR_run_Wind"]+"param_RMSE.csv")    
    gp_resu=resultat.groupby(["years"])
    for y in list(set(resultat.years)):
        a=gp_resu.get_group(y)
        min_value=a.loc[a['RMSE'].idxmin()]
        print(min_value)
