#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:35:03 2020

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
import random
import shapely.geometry as geom
import descartes

if __name__ == "__main__":
    """ mise en place df pour SAMIR """
    
    # A adpater en focntion du test
    # RPG=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/RPG/RPG_BV/RPG_SUMMER_2018_ADOUR_AMONT.shp")
    

    list_bd_drop=['originfid', 'ogc_fid','centroidx', 'centroidy', 'shape_leng','shape_area']
    list_col_drop=['originfid',   'ogc_fid', ' n° sonde',' x',' y',]

    dfnames=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_2017.txt",sep=',', header=None) 
    
    # RPG=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/RPG/RPG_BV/RPG_SUMMER_2018_ADOUR_AMONT_only_maize.shp")
    LABO=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/PARCELLE_LABO_ref.shp")

    NDVI=pd.DataFrame()
    dfNDVI_interTYN=pd.DataFrame()
    dfNDVI_interTYP=pd.DataFrame()
    dfNDVI_interTCJ=pd.DataFrame()
    for n in os.listdir("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO"):
        if "SampleExtractionNDVI" in n and "2017" in n: 
            df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/"+str(n))
            tuile=n[-16:-13]
            #  gestion des colonnes du tableau
            df.columns=list(df.columns[0:7])+list(dfnames[0])
            df.set_index("id",inplace=True)
            dataN=df.drop(columns=list_bd_drop)
            dataN.columns=pd.to_datetime(dataN.T.index,format="%Y%m%d")
            globals()["dfNDVI_inter%s"%tuile]=dataN.where(dataN!=0.0)
            globals()["dfNDVI_inter%s"%tuile].dropna(inplace=True)

    test=set(dfNDVI_interTYP.index)-set(dfNDVI_interTYN.index)
    non=dfNDVI_interTYP.loc[test]
    all_NDVI=pd.concat([dfNDVI_interTCJ])
    tatar=all_NDVI.T.resample("D").asfreq().interpolate()
    tar=tatar.T.sort_index(ascending=True)

    for i in list(tar.index):
        print(i)
        t=tar[tar.T.columns==[i]]/1000
        NDVI=NDVI.append(t.T.values.tolist(),ignore_index=True)
    identity=list(np.repeat(tar.index,tar.shape[1]))
    date=list(tar.columns)*len(tar.index)
                
    NDVI["id"]=identity
    NDVI["date"]=date
    NDVI["date"]=pd.to_datetime(NDVI.date,format="%Y%m%d")
    NDVI.columns=["NDVI","id","date"]
   

    #  Select only maize
    
    # maize=RPG.loc[RPG.CODE_CULTU.str.startswith("MI")]
    # NDVIMAIZE=NDVI.loc[NDVI['id'].isin(maize.ID)]
    
    NDVI.to_pickle("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/test_SAMIR_labo/Inputdata/maize/NDVI.df")
    
# =============================================================================
#     Build df FC and WP
# =============================================================================
    Parcellaire=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/Parcelle_labo/PARCELLE_CESBIO_L93.shp")
    for i in ["WP",'FC']:
        soil=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/SOIL_'+str(i)+'_1m.shp')
        soil.drop(columns=['NOM_PARCEL', 'LABO', 'WP_0_30cm', 'WP_40_50m', 'FC_0_30cm',
       'FC_40_50cm', 'pt_sat0_30', 'pt_sat40_5', 'RU_0_30', 'RU_40_50',
       'RU_SG_60cm', 'sdRU_SG_60', 'RU_SG_0_30', 'sdRU_SG0_3', 'count', 'min_0', 'max_0','geometry'],inplace=True)
        soil.columns=["id",str(i),str(i+'std')]
        soil.to_pickle('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/test_SAMIR_labo/Inputdata/maize/'+str(i)+'.df')
    # FC.to_pickle("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/test_SAMIR_2018/Inputdata/maize/FC.df")
    
# =============================================================================
#     Build METEO spatialieser 
# =============================================================================
    # meteo=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/test_SAMIR_2018/meteo_2018_ADOUR.csv")
    # meteo=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SAFRAN2018_ALL.csv")
    # meteo.drop(columns=['X', 'Y', 'field_1', 'LAMBX', 'LAMBY', 'DATE', 'PRENEI_Q', 'PRELIQ_Q',
    #    'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'ETP_Q',
    #    'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE_Q', 'RESR_NEIGE6_Q',
    #    'HTEURNEIGE_Q', 'HTEURNEIGE6_Q', 'HTEURNEIGEX_Q', 'SNOW_FRAC_Q',
    #    'ECOULEMENT_Q', 'WG_RACINE_Q', 'WGI_RACINE_Q', 'TINF_H_Q', 'TSUP_H_Q',
    #    'lambX_1', 'lambY_1'],inplace=True)
    # meteo["DATE"]=pd.to_datetime(meteo.DATE,format="%Y%m%d")
    # meteo["Irrig"]=0.0
    # meteo.columns=["date","Prec",'ET0','Irrig']
    # meteo.to_pickle("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/test_SAMIR_2018/Inputdata/meteo.df")
    Parcellaire=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/Parcelle_labo/PARCELLE_CESBIO_L93.shp")
    # meteo=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SAFRAN_2018_EMPRISE_L93.shp")
    meteo=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/DATA_SAFRAN_2017_EMPIRSE_ALL.shp")
    meteo.drop(columns=['field_1', 'Unnamed_ 0', 'LAMBX', 'LAMBY', 'PRENEI_Q',
        'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q',
       'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
       'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
       'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
       'lambX_1', 'lambY_1'],inplace=True)
    
    # meteo.drop(columns=['X', 'Y', 'field_1_1', 'LAMBX', 'LAMBY', 'PRENEI_Q',
    #     'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q','PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE', 'RESR_NEI_1',
    #     'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_', 'ECOULEMENT',
    #     'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q', 'lambX_1','EVAP_Q',
    #     'lambY_1'],inplace=True)
    dfmeteo=meteo.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
    meteo.geometry=dfmeteo
    
    meteo.DATE=pd.to_datetime(meteo.DATE,format='%Y%m%d')   
    # meteo.set_index("DATE",inplace=True)
    
    Meteo_par=geo.overlay(meteo,Parcellaire,how='intersection')
    
    Meteo_par.drop(columns=["geometry",'NOM_PARCEL', 'LABO','WP_0_30cm',
       'WP_40_50m', 'FC_0_30cm', 'FC_40_50cm', 'pt_sat0_30', 'pt_sat40_5',
       'RU_0_30', 'RU_40_50', 'RU_SG_60cm', 'sdRU_SG_60', 'RU_SG_0_30',
       'sdRU_SG0_3', 'geometry'],inplace=True)
    # Meteo_par["DATE"]=pd.to_datetime(Meteo_par.DATE,format="%Y%m%d")
    Meteo_par["Irrig"]=0.0
    Meteo_par.columns=["date","Prec",'ET0',"id",'Irrig']
    Meteo_par.info()
    lam=Meteo_par.loc[np.where(Meteo_par.id==1)] # if 2018 .iloc[:-31]
    lam.drop(columns="id",inplace=True)
    lam.to_pickle("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/test_SAMIR_labo/Inputdata/meteo.df")
    

