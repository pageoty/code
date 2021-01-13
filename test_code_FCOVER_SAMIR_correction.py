#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:36:39 2021

@author: pageot
"""

import os 
import pandas as pd
import numpy as np
d={}
d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"


offset=-0.25
slpoe=1.39
stop_pal=0.15
ndviPlateau=70
minVar = 0
maxVar = 1
NDVI=pd.read_csv(d["PC_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_2019.csv",decimal=".")
NDVI.date=pd.to_datetime(NDVI.date,format="%Y-%m-%d")
NDVI["id"]=1
maxNDVI=NDVI.loc[NDVI['NDVI'].idxmax()][['id','date','NDVI']]
maxNDVI['endDate'] = maxNDVI.date + pd.Timedelta(ndviPlateau, unit='D')
NDVI["max_NDVI"]=maxNDVI.NDVI
A_NDVI=NDVI.max_NDVI.where((NDVI.date > maxNDVI.date) & (NDVI.date < maxNDVI.endDate), NDVI.NDVI)

Fcover=NDVI.NDVI*slpoe+offset
Fcover_corr=[]

for i in np.arange(Fcover.shape[0]):
    if (Fcover.iloc[i] > stop_pal) and (A_NDVI.iloc[i] >= A_NDVI.max()) :
        print(Fcover.iloc[i])
        a=A_NDVI.max()*slpoe+offset
    else:
        a=NDVI.NDVI.iloc[i]*slpoe+offset
    Fcover_corr.append(a)
test=pd.DataFrame(Fcover_corr)
test.plot()
test.iloc[220:280]

Fcover_con=Fcover > stop_pal
Fcover_con.iloc[211:282]
