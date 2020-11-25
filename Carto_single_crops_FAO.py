# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:12:04 2020

@author: yann
"""
import os
import sqlite3
import geopandas as geo 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import seaborn as sns 
import TEST_ANALYSE_SIGNATURE
import shapely.geometry as geom
import shapely.wkt
import descartes
import pickle
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
import sys
import shapely.wkt



if __name__ == '__main__':
    d={}
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["clu"]="/work/CESBIO/projects/Irrigation/Irrigation_Sud_Ouest/MODELISATION/"
    years=["2017"]
           
    # #  lecture fichier NDVI 
    dfnames=pd.read_csv( d["PC_home_Wind"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_2017.txt",header=None)
    NDVI_TYN=pd.read_csv(d["PC_home_Wind"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/ADOUR/RPG/SampleExtractionNDVI_TYN_2017.tif.csv")
    NDVI_TYN.drop(columns=['id_parcel', 'surf_parc', 'code_cultu', 'code_group', 'culture_d1','culture_d2', 'summer', 'minx', 'miny', 'maxx', 'maxy', 'cntx', 'cnty','perim', 'height', 'width', 'name', 'surfha', 'area', 'originfid'],inplace=True)
    NDVI_TYN=NDVI_TYN.groupby("id").mean()
    NDVI_TYN.columns=dfnames[0]
    NDVI_TYP=pd.read_csv(d["PC_home_Wind"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/ADOUR/RPG/SampleExtractionNDVI_TYP_2017.tif.csv")
    NDVI_TYP.drop(columns=['id_parcel', 'surf_parc', 'code_cultu', 'code_group', 'culture_d1','culture_d2', 'summer', 'minx', 'miny', 'maxx', 'maxy', 'cntx', 'cnty','perim', 'height', 'width', 'name', 'surfha', 'area', 'originfid'],inplace=True)
    NDVI_TYP=NDVI_TYP.groupby("id").mean()
    NDVI_TYP.columns=dfnames[0]
    NDVI=pd.concat([NDVI_TYP,NDVI_TYN])
    NDVI.columns=pd.to_datetime(NDVI.columns,format="%Y%m%d")
    NDVI=NDVI.T.resample("D").interpolate()
    NDVI=NDVI.loc[(NDVI.index >= "2017-03-02") &(NDVI.index <= "2017-10-31")]
    NDVI=NDVI.T
    NDVI["ID"]=NDVI.index
    #  Merge ficher et drop duplicates (first)
    NDVI.drop_duplicates(subset=['ID'],keep="first",inplace=True)
    NDVI.sort_values("ID",ascending=True,inplace=True)
    # Clacul du Kc 
    # #  Lecture classif 
    Classif=geo.read_file(d["PC_home_Wind"]+"/TRAITEMENT/vector_Classif_ADOUR_Fusion_demps_Accu_2017.shp")
    # #  selction class 1 et 11
    crops=Classif[Classif.majority==1.0]
    crops.drop(columns=['id_parcel', 'surf_parc', 'code_cultu', 'summer', 'majority', 'geometry'])
    kc=pd.DataFrame()
    for i in crops.ID:
        a=NDVI[NDVI.ID==i]*1.25-0.17
        kc=kc.append(a/1000)
    kc["ID"]=kc.index.astype(float)
    # Interpoler les Kc 

    #  lier NDVI et meteo
    # Lecture fichier meteo
    meteo=geo.read_file(d["PC_home_Wind"]+"/DONNEES_RAW/DONNES_METEO/SAFRAN_ZONE_2017_L93_Adour.shp")
    meteo.drop(columns=['field_1', 'LAMBX', 'LAMBY', 'PRENEI_Q', 'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q',
    'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
    'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
    'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
    'X', 'Y'],inplace=True)
    dfmeteo=meteo.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
    meteo.geometry=dfmeteo
    meteo.DATE=pd.to_datetime(meteo.DATE,format='%Y%m%d')
    Meteo_par=geo.overlay(meteo,crops,how='intersection')
    Meteo_par.drop(columns=['id_parcel', 'surf_parc','code_cultu', 'summer','majority', '1', '11', '44', '33', '2', '22'],inplace=True)
    Meteo_par.sort_values(["ID","DATE"],ascending=True,inplace=True)
    Meteo_par=Meteo_par.loc[(Meteo_par.DATE >= "2017-03-02") &(Meteo_par.DATE <= "2017-10-31")]
    #  Calcul besion pour chaque parcelle 
    Bes_irri=[]
    id_p=[]
    time_p=[]
    geome=[]
    for i in sorted(list(set(Meteo_par.ID))):
        print("ici i :%s"%i)
        for j in np.arange(0,244):
            ge=Meteo_par.geometry[Meteo_par.ID==i].iloc[j]
            t=Meteo_par.DATE[Meteo_par.ID==i].iloc[j]
            # b=Meteo_par.ETP_Q[Meteo_par.ID==i].iloc[j]*kc.T[:-1][i][j]-Meteo_par.PRELIQ_Q[Meteo_par.ID==i].iloc[j]
            b=Meteo_par.ETP_Q[Meteo_par.ID==i].iloc[j]
            Bes_irri.append(b)
            id_p.append(i)
            time_p.append(t)
            geome.append(ge)
    id_p=pd.DataFrame(id_p)
    Bes_irri=pd.DataFrame(Bes_irri)
    time_p=pd.DataFrame(time_p)
    geome=pd.DataFrame(geome)
    data_bes=pd.concat([id_p,time_p,Bes_irri],axis=1)
    data_bes.columns=["ID",'DATE',"bes"]
    data_bes["month"]=data_bes.DATE.dt.strftime("%m")
    # month_data=data_bes
    data_bes["geometry"]=geome
    data_bes.columns=["ID",'DATE',"bes","month","geometry"]
    data_bes_spa=geo.GeoDataFrame(data_bes, crs="EPSG:2154")
    data_bes_spa[data_bes_spa.bes<=0]=np.nan
    data_bes_spa.dropna(inplace=True)
    data_bes_spa.reset_index(inplace=True)
    data_bes_spa.to_csv(d["PC_home_Wind"]+"/RESULT/PLOT/CARTE_Single_crops_ADOUR/NEW_code/CSV_data_bes_Irr_spa_mais_irri_2017.csv")
    # for dates in sorted(list(set(data_bes_spa.DATE))):
        # a=data_bes_spa[data_bes_spa.DATE==dates]
        # a.plot(figsize=(10,10),column='bes',legend=True)
        # plt.savefig(d["PC_home_Wind"]+"carte_ADOUR_mais_irr_2017_"+dates+".png")

    data_bes_spa=pd.read_csv("D:/THESE_TMP/RESULT/PLOT/CARTE_Single_crops_ADOUR/DATA_BES_EAU/csv_bes_eau_mais_irri_2017.csv")
    geometry = data_bes_spa['geometry'].map(shapely.wkt.loads)
    data_bes_spa_tes=geo.GeoDataFrame(data_bes_spa, crs="EPSG:2154", geometry=geometry)

#   Clum mensuel
    # id_month=data_bes_spa["ID"][data_bes_spa.month=='10']
    for m in ["3","4","5","6","7","8","9"]:
        id_m=pd.DataFrame()
        bes_m=pd.DataFrame()
        geom_d=pd.DataFrame()
        id_month=data_bes_spa_tes["ID"][data_bes_spa_tes.month==int(m)]
        for j in list(set(id_month)):
            print(j)
            id_geom=data_bes_spa_tes["geometry"][(data_bes_spa_tes.month==int(m))&(data_bes_spa_tes.ID==j)].iloc[0]
            # id_month=data_bes_spa["ID"][(data_bes_spa.month==str(m))&(data_bes_spa.ID==j)]
            tes=data_bes_spa_tes["bes"][(data_bes_spa_tes.month==int(m))&(data_bes_spa_tes.ID==j)].cumsum().astype(float).iloc[-1]
            id_m=id_m.append([j])
            bes_m=bes_m.append([tes])
            geom_d=geom_d.append([id_geom])
        test_m=pd.concat([bes_m,id_m],axis=1)
        test_m["geometry"]=geom_d[0]
        test_m.columns=['bes','ID','geometry']
        # test_m.reset_index(inplace=True)
        test_m_spa=geo.GeoDataFrame(test_m, crs="EPSG:2154")
        # test_m.plot(figsize=(10,10),column='bes',legend=True)
        # plt.title("month : %s"%str(m))
        # plt.savefig(d["PC_home_Wind"]+"RESULT/PLOT/CARTE_Single_crops_ADOUR/NEW_code/carte_ADOUR_mais_irr_2017_"+str(m)+".png")
        test_m_spa.to_file(d["PC_home_Wind"]+'RESULT/PLOT/CARTE_Single_crops_ADOUR/NEW_code/ET0_satfran'+str(m)+'_v2.shp')
# Cmul sur periode vege
    data_bes_spa['mois']=data_bes_spa.month.astype(int)
    est=data_bes_spa[(data_bes_spa.mois>=4)&(data_bes_spa.mois<=9)]
    geoms=pd.DataFrame()
    cumsais=pd.DataFrame()
    idr=pd.DataFrame()
    for ids in list(set(est.ID)):
        print(ids)
        g=est.bes[est.ID==ids].cumsum().iloc[-1]
        s=est["geometry"][est.ID==ids].iloc[0]
        geoms=geoms.append([s])
        cumsais=cumsais.append([g])
        idr=idr.append([ids])
    testd=pd.concat([idr,cumsais],axis=1)
    testd["geometry"]=geoms
    testd.columns=['ID','bes','geometry']
    testd.reset_index(inplace=True)
    testd_spa=geo.GeoDataFrame(testd, crs="EPSG:2154")
    testd_spa.columns=["index","ID","BES",'geometry']
    testd_spa.plot(figsize=(10,10),column='BES',legend=True)
    testd_spa.to_file(d["PC_home_Wind"]+'RESULT/PLOT/CARTE_Single_crops_ADOUR/NEW_code/test_cumsais_Avril_sept_v2.shp')
