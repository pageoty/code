#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:33:21 2019

@author: pageot
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import csv
import zipfile
from scipy import stats
from datetime import datetime




        
list_bd_drop=['ogc_fid', 'surf_parc','label']
#    date=["01","02","03"]

#if "Plot_result" in os.listdir(d['output_file']):
#    print("existing file")
#else:
#    os.mkdir("%s/Plot_result"%d['output_file'])




dfnames=pd.read_csv('/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_OPTIC.txt',sep=',', header=None) 
df1=dfnames.T
df1.columns=["band_name"]
colnames=list(df1.band_name.apply(lambda s: s[2:-1]))
sql=sqlite3.connect("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/SampleExtractionSentinel2_T30TYP_Features.tif.sqlite")
df=pd.read_sql_query("SELECT * FROM output", sql)
dfpar=df.groupby("originfid").mean()
labcroirr=dfpar["label"]
dfpar.drop(columns=list_bd_drop,inplace=True)
dfpar1=dfpar.T
dfpar1["band_names"]=colnames
dfpar1["date"] = dfpar1.band_names.apply(lambda s: s[-8:])
dfpar1.set_index("band_names",inplace=True)
dfpar2=dfpar1.T
dfpar2["label"]= labcroirr
dfNDWI=dfpar2[dfpar2.columns[396:432]]
dfNDVI=dfpar2[dfpar2.columns[360:396]]
dfNDVI["label"]= labcroirr
dfNDWI=dfNDWI[dfNDWI.columns[:-1]]*-1
dfNDWI["label"]=labcroirr

for i in set(labcroirr):
    globals()['cropslab%s' % int(i)] = pd.DataFrame(dfNDWI[dfNDWI.label==i]).T
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.plot(globals()['cropslab%s' %(int(i))].iloc[:-1].T.mean())
    plt.fill_between(globals()['cropslab%s' %(int(i))].iloc[:-1].index, globals()['cropslab%s' %(int(i))].iloc[:-1].T.mean()-  globals()['cropslab%s' %(int(i))].iloc[:-1].T.std() , globals()['cropslab%s' %(int(i))].iloc[:-1].T.mean() + globals()['cropslab%s' %(int(i))].iloc[:-1].T.std(), alpha=0.2)
    plt.xticks(rotation=90,size=9)
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/PLOT_ndvi_non_interpoler/Plot_NDWI_%s.png"%(int(i)))


#for i in set(lab):
#    globals()['dfpar%s' % (int(i))] = pd.DataFrame(dfpar[dfpar["label"]==int(i)]).T
#    plt.figure(figsize=(10,10))
#    sns.set(style="darkgrid")
#    sns.set_context('paper')
#    plt.plot( globals()['dfpar%s' % (int(i))].iloc[:-1].T.mean())
#    plt.fill_between(globals()['dfpar%s' % (int(i))].iloc[:-1].index, globals()['dfpar%s' % (int(i))].iloc[:-1].T.mean()-  globals()['dfpar%s' % (int(i))].iloc[:-1].T.std() , globals()['dfpar%s' % (int(i))].iloc[:-1].T.mean() + globals()['dfpar%s' % (int(i))].iloc[:-1].T.std(), alpha=0.2)
#    plt.xticks(rotation=90,size=9)
#    plt.savefig(d["output_file"]+"Plot_result/Plot_%s_%s.png"%(k[-11:-7],int(i)))
ending=datetime.now() - start
print ("===============================")
print ("time process : %s" % ending)
print ("===============================")
#i=4
#k='NDRE'
#plt.figure(figsize=(10,10))
#sns.set(style="darkgrid")
#sns.set_context('paper')
#plt.title("%s%s"%(k[-11:-7],int(i)))
#plt.plot( globals()['df%s%s' % (k,int(i))].iloc[:-1])