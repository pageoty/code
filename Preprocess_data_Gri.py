#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:58:22 2020

@author: pageot
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
import seaborn as sns

SWC_gri=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/DATA_PARCELLE_GRIGNON/RAW_DATA/DATA_ICOS_ECO/FR-Gri_Meteo_2019.csv")
SWC_gri["time"]=SWC_gri["TIMESTAMP"].astype(str)
SWC_gri["date"]=SWC_gri["time"].apply(lambda x:x[0:8])
SWC_gri["date"]=pd.to_datetime(SWC_gri["date"],format="%Y%m%d")
SWC_gri.drop(columns=["TIMESTAMP","time"],inplace=True)
SWC_GRI=SWC_gri[["date","SWC_1_1_1",'SWC_1_2_1', 'SWC_1_3_1', 'SWC_1_4_1', 'SWC_1_5_1',"SWC_1_6_1",'SWC_2_1_1',"SWC_2_2_1", 'SWC_2_3_1', 'SWC_2_4_1', 'SWC_2_5_1',"SWC_2_6_1"]]
SWC_GRI=SWC_GRI.iloc[:-1]
for col in SWC_GRI.columns[1:]:
    SWC_GRI[SWC_GRI[str(col)]<=-9998]=np.nan
SWC_GRI=SWC_GRI.groupby("date").mean()
SWC_GRI["SWC_5_moy"]=SWC_GRI[["SWC_1_1_1","SWC_2_1_1"]].mean(axis=1)
SWC_GRI["SWC_10_moy"]=SWC_GRI[["SWC_1_2_1","SWC_2_2_1"]].mean(axis=1)
SWC_GRI["SWC_20_moy"]=SWC_GRI[["SWC_1_3_1","SWC_2_3_1"]].mean(axis=1)
SWC_GRI["SWC_30_moy"]=SWC_GRI[["SWC_1_4_1","SWC_2_4_1"]].mean(axis=1)
SWC_GRI["SWC_60_moy"]=SWC_GRI[["SWC_1_5_1","SWC_2_5_1"]].mean(axis=1)
SWC_GRI["SWC_90_moy"]=SWC_GRI[["SWC_1_6_1","SWC_2_6_1"]].mean(axis=1)
SWC_GRI.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/DATA_VALIDATION/DATA_SWC/SWC_GRI/SWC_GRI_2019.csv")


