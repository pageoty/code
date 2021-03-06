#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:11:07 2020

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

if __name__ == '__main__':
    Parcellaire=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/Parcelle_labo/PARCELLE_CESBIO_L93.shp")
    years='2017'
    list_bd_drop=['ogc_fid', 'centroidx', 'centroidy', 'shape_leng', 'shape_area', 'id']
    
    if years =="2017":
        dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt",sep=',', header=None) 
    else:
        dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_TYN2018.txt",sep=',', header=None) 
    df1=dfnames.T
    df1.columns=["band_name"]
    colnames=list(df1.band_name.apply(lambda s: s[2:-1]))
    
    sql=sqlite3.connect('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SAR_parcelle/SampleExtractionVH_VV0112_ASC_'+str(years)+'.tif.sqlite')
    df=pd.read_sql_query("SELECT * FROM output", sql)
    df2017=df.groupby("originfid").mean()
    lab=df2017["id"]
    df2017.drop(columns=list_bd_drop,inplace=True)
    df2017=df2017.T
    if years == '2017':
        df2017["band_names"]=colnames[146:183]
    else:
        df2017["band_names"]=colnames[148:185]
    df2017["date"] = df2017.band_names.apply(lambda s: s[-8:])
    df2017.set_index("date",inplace=True)
    df2017.index=pd.to_datetime(df2017.index,format="%Y%m%d")
    df2017.columns=['LAM','AUR','band']

    # recupere la valeurs min et la ligne 
    datemin_VH_VV=df2017.iloc[np.where(df2017.LAM[0:20]==df2017.LAM[0:20].min())]
    if years == '2017':
        meteo=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/DATA_SAFRAN_2017_EMPIRSE_ALL.shp")
    else:
        meteo=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SAFRAN_2018_EMPRISE_L93.shp")
    meteo.drop(columns=['field_1', 'Unnamed_ 0', 'LAMBX', 'LAMBY', 'PRENEI_Q',
        'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q', 'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
       'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
       'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
       'lambX_1', 'lambY_1'],inplace=True)
    dfmeteo=meteo.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
    meteo.geometry=dfmeteo
    # meteo.set_index("DATE",inplace=True)
    meteo.DATE=pd.to_datetime(meteo.DATE,format='%Y%m%d')   
    
    Parcel_LABO=geo.overlay(Parcellaire,meteo,how='intersection')
    Parcel_LABO.set_index('DATE',inplace=True)
    min_signal_labo=Parcel_LABO.iloc[np.where(Parcel_LABO.index.values==datemin_VH_VV.index.values)]
    
    #  Multipilier la température par 500 pour otbeir
    # min_signal_labo.iloc[1].T_Q*500 #=3150
    # Cumsum_T=Parcel_LABO.T_Q.cumsum()
    # res=Parcel_LABO.loc[np.where((Cumsum_T>=min_signal_labo.iloc[0].T_Q*500) & (Cumsum_T<=min_signal_labo.iloc[0].T_Q*500))]
    



#  test
    LAM=Parcel_LABO[Parcel_LABO.NOM_PARCEL=="Lamothe"]
    # LAM.set_index("DATE",inplace=True)
    LAM.sort_index(ascending=True,inplace=True)
    Tcum_stat=LAM.loc[LAM.index>=min_signal_labo.index[0]].T_Q.cumsum()
    # Tcum=LAM.T_Q.cumsum()
    res=Tcum_stat.where(Tcum_stat >min_signal_labo.iloc[0].T_Q+500)
    res=res.dropna()
    
    min_signal_labo.index-res.index[0]