#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:13:23 2021

# Extraction données SAFRAN 
@author: pageot
"""

import os
import pandas as pd
import geopandas as geo
import numpy as np
from osgeo import osr

if __name__ == '__main__':
    d={}
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    years="2018"
    # name_run="RUNS_SAMIR/RUNS_SENSI_DATA_RAINFALL/DATA_STATION/"+str(years)+"/Inputdata/"
    name_run="RUNS_SAMIR/DATA_SCP_ICOS/CACG_SAFRAN/"+str(years)+"/Inputdata/"
    # mode="CSV"
    d["path_run"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    d["path_run_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
# =============================================================================
# Conversion csv en shp (geopandas)
# =============================================================================
    # df=pd.read_csv(/file/meteo.csv)
    # LAMBX=df.LAMBX*100
    # LAMBY=df.LAMBY*100
    # df["lambX"]=LAMBX
    # df['lambY']=LAMBY
    # df=df.loc[(df.DATE >= 20090101) &(df.DATE <= 20091231)]
    # geometry = df['geometry'].map(shapely.wkt.loads)
    # meteo_spa=geo.df(df, crs="EPSG:", geometry=geometry)
    
    #  Pour le reprojection pas encore regarder
# =============================================================================
#     Extaction de la donnée METEO par rapport centoide parcelle (methode NN)
# =============================================================================
#     meteo=geo.read_file("H:/Yann_THESE/BESOIN_EAU//BESOIN_EAU/DONNEES_RAW/DONNES_METEO/SAFRAN_ZONE_2019_L93.shp")
#     parcelle=geo.read_file("H:/Yann_THESE/BESOIN_EAU//BESOIN_EAU/DONNEES_RAW/DONNEES_CACG_PARCELLE_REF/Parcelle_CAGC.shp")
#     # meteo.DATE=meteo.DATE.astype(int)
#     meteo.DATE=pd.to_datetime(meteo.DATE,format="%Y%m%d")
#     meteo.set_index("field_1",inplace=True)
#     parcelle.set_index("ID",inplace=True)
#     resu=pd.DataFrame()
#     idgeom=[]

#     for par in parcelle.index:
#           extart_meteo=meteo.loc[meteo["geometry"].distance(parcelle["geometry"].iloc[0])==meteo["geometry"].distance(parcelle["geometry"].iloc[0]).min()][['DATE',"PRELIQ_Q","T_Q","ETP_Q"]]
#           idgeom.append(np.repeat(par,extart_meteo.shape[0]))
#           resu=resu.append(extart_meteo)
#     idpar=pd.DataFrame(idgeom).stack().to_list()
#     resu["ID"]=idpar
#     test=pd.merge(parcelle,resu[["DATE","ETP_Q","PRELIQ_Q","T_Q",'ID']],on="ID")

# # mettre cela en df SAMIR    
#     meteo=test.filter(['ID',"DATE","ETP_Q", 'PRELIQ_Q'])
#     meteo.columns=["id",'date',"ET0",'Prec']
#     meteo["Irrig"]=0.0
#     meteo.to_csv('H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_CACG/meteo_'+years+'.csv')
    meteo2=pd.read_csv('H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_CACG/meteo_'+years+'.csv')
    meteo2.drop(columns=["Unnamed: 0"],inplace=True)
    meteo2=meteo2.loc[(meteo2.id<14.0)&(meteo2.id!=6.0) & (meteo2.id!=8.0) & (meteo2.id!=2) & (meteo2.id!=3)]
    meteo2.date=pd.to_datetime(meteo2.date,format="%Y-%m-%d")
    meteo2.to_pickle(d["path_run_disk"]+"/maize_irri/meteo.df")
# =============================================================================
#     Ancienne version avec intersection 
# =============================================================================
    # Lecture data SAFRAN
    # SAF=geo.read_file("D:/THESE_TMP/DONNEES_RAW/DONNES_METEO/SAFRAN_ZONE_"+str(years)+"_L93.shp")
    # # Lecture parcellaire
    # parce=geo.read_file("D:/THESE_TMP/DONNEES_RAW/PARCELLE_LABO/PARCELLE_LABO_LAM_L93.shp")
    # SAF.drop(columns=['field_1', 'LAMBX', 'LAMBY', 'PRENEI_Q', 'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q',
    #         'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
    #         'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
    #         'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
    #         'Y', 'X'],inplace=True)
    # dfmeteo=SAF.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
    # SAF.geometry=dfmeteo
    # SAF.DATE=pd.to_datetime(SAF.DATE,format='%Y%m%d')
    # SAF_par=geo.overlay(SAF,parce,how='intersection')
    # SAF_par.id=1
    # SAF_par.drop(columns=['NOM_PARCEL', 'EVAP_Q', 'LABO','geometry'],inplace=True)
    # SAF_par["Irrig"]=0.0
    # SAF_par.columns=["date","Prec",'ET0',"id",'Irrig']
    # SAF_par.to_pickle(d["path_run_home"]+"/maize_irri/meteo.df")