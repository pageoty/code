#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:59:59 2020

@author: pageot

Code pour estimer les besoins n eau selon la FAO-56 single crops 
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
import seaborn as sns
import TEST_ANALYSE_SIGNATURE
import random
import shapely.geometry as geom
import descartes
from shapely.geometry import Point, Polygon
import argparse
if __name__ == "__main__":
    """ Cartographie Besion Irrigation BV single crop """

    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_disk_labo"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/DONNEES_RAW/DONNES_METEO/"
  
    for years in ["2018"]:
        Parcellaire=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/vector_Classif_ADOUR_Fusion_demps_Accu_"+years+".shp")
        if years=="2018":
            Parcellaire.columns=['code_cultu', 'Area', 'id', 'majority', '11', '1', '44', '22', '2','33', 'geometry']
        else:
            Parcellaire.columns=['id_parcel', 'surf_parc', 'code_cultu', 'summer', 'id', 'majority', '1','11', '44', '33', '2', '22', 'geometry']
        id_mais=Parcellaire.id[Parcellaire.majority==1.0]
        id_sunflower=Parcellaire.id[Parcellaire.majority==44.0]
        meteo=geo.read_file(d["PC_disk_labo"]+"/SAFRAN_ZONE_"+str(years)+"_L93.shp")
        meteo.drop(columns=['field_1', 'LAMBX', 'LAMBY', 'PRENEI_Q', 'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q',
       'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
       'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
       'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
       'X', 'Y'],inplace=True)
        dfmeteo=meteo.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
        meteo.geometry=dfmeteo
        meteo.DATE=pd.to_datetime(meteo.DATE,format='%Y%m%d')
        Meteo_par=geo.overlay(meteo,Parcellaire,how='intersection',make_valid=True, use_sindex=None)
        # Calcul relation NDVI/kc par parcelle.  
        dfnames=pd.read_csv(d["path_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_"+str(years)+".txt",sep=',', header=None) 
        NDVI_BV=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/ADOUR/RPG/SampleExtractionNDVI_TYN_"+str(years)+".tif.csv")
        NDVI_BV=NDVI_BV.groupby("originfid").mean()
        NDVI_BV2=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/ADOUR/RPG/SampleExtractionNDVI_TYP_"+str(years)+".tif.csv")
        NDVI_BV2=NDVI_BV2.groupby("originfid").mean()
        NDVI=pd.concat([NDVI_BV,NDVI_BV2])
        if years =="2017":
            NDVI.drop(columns=['id_parcel', 'surf_parc', 'code_group', 'summer', 'minx', 'miny',
                               'maxx', 'maxy', 'cntx', 'cnty', 'perim', 'height', 'width', 'surfha','area'],inplace=True)
        else:
             NDVI.drop(columns=['area'],inplace=True)
        NDVI.drop_duplicates(subset=["id"],keep='last',inplace=True)
        for crops in ["mais","maize_rain"]:
            if crops =="mais":
                id_crops=Parcellaire.id[Parcellaire.majority==1.0]
            else:
                id_crops=Parcellaire.id[Parcellaire.majority==11.0]
            NDVI_crops=pd.DataFrame()
            for i,f in enumerate(id_crops):
                a=NDVI[NDVI.id==f]
                NDVI_crops=NDVI_crops.append(a)
            NDVI_crops.columns=list(NDVI_crops.columns[0:1])+list(dfnames[0]) 
            NDVI_crops.set_index("id",inplace=True)
            NDVI_crops.columns=pd.to_datetime(NDVI_crops.columns,format="%Y%m%d")
            NDVI_inter=NDVI_crops.T.resample("D").asfreq().interpolate()
            NDVI_inter=NDVI_inter/1000
            kc=pd.DataFrame()
            for i in NDVI_inter.columns:
                if crops =="mais":
                    a=NDVI_inter[int(i)]*1.25+0.2
                elif crops=="maize_rain":
                    a=NDVI_inter[int(i)]*1.46-0.25
                kc=kc.append([a])
            kc["id"]=kc.index
        ## Récuper uniquement le Mais Irriguée
            Meteo_par["Area"]=Meteo_par.area/10000
            if crops =="mais":
                MIRR=Meteo_par[Meteo_par.majority==1.0]
            else:
                MIRR=Meteo_par[Meteo_par.majority==11.0]
            MIRR[MIRR.Area < 0.05]=np.nan
            MIRR.dropna(subset=["Area"],inplace=True)
            bes_irri_kc=pd.DataFrame()
            id_p=[]
            for j in kc.T.columns:
                test=MIRR[MIRR.id==j]
                if test.shape[0] != 0 :
                    test.reset_index(inplace=True)
                    b=test.ETP_Q[:kc.shape[1]]*kc.T[j].values-test.PRELIQ_Q[:kc.shape[1]] *10# a corrigée modulable dans le tmpes  size de kc.shape[1]]
                    b[b<=0]=np.nan
                    bes_irri_kc=bes_irri_kc.append(b,ignore_index=True)
                    id_p.append(j)
                else:
                    print("pas de data")
            
            bes_irri_kc["id"]=id_p
            bes_irri_kc=pd.merge(bes_irri_kc,Parcellaire[["id","geometry"]],on=["id"])
            # bes_irri_kc.columns=pd.to_datetime(np.arange(0,bes_irri_kc.shape[1]-3), unit='D',origin=pd.Timestamp(str(years)+'-01-01')),"id","geometry"
            bes_irri_kc_spa=geo.GeoDataFrame(bes_irri_kc,geometry=bes_irri_kc.geometry)
            # for d in bes_irri_kc_spa.columns[94:300]:
            #     if bes_irri_kc_spa[d].iloc[0] > 0 :
            #         # fig,ax=plt.subplots(figsize=(15,15))
            #         # ax.set_xlim([440000, 470000])
            #         # ax.set_ylim([6.23*1e6, 6.30*1e6])
            #         # meteo[meteo.DATE=="2018-07-18"].plot(ax=ax,column='PRELIQ_Q')
            #         # Parcellaire.plot(ax=ax,cmap='OrRd')
            #         # bes_irri_kc_spa.plot(column=d,ax=ax,legend=True,legend_kwds={'label': "Besoin en irrigation"})
            #         bes_irri_kc_spa.plot(column=d,legend=True,figsize=(10,10),legend_kwds={'label': "Besoin en irrigation"})
            #         plt.title("date en jj :%s "%d)
            #         plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/RESULT/PLOT/CARTE_Single_crops_ADOUR/carte_adour_besoin_Irr_single_crop_"+years+"_"+crops+"_"+str(d)+".png",dpi=600)
                # else :
                #     print('pas de data')
        # Somme tous les dix jours 
            bes_irri_kc_spa.replace(np.nan,0,inplace=True)
            bes_irri_kc_spa.T[:-3].mean(axis=1).cumsum().plot(legend=True)
            plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/RESULT/PLOT/CARTE_Single_crops_ADOUR/mensuel_adour/cumul_besoin_Irr_single_crop_"+years+"_"+crops+"_.png",dpi=600)
            # for d,c in zip(np.arange(0,bes_irri_kc_spa.shape[1]-10,10),np.arange(11,bes_irri_kc_spa.shape[1],10)):
            #     print(d,c)
            #     a=bes_irri_kc_spa.T[d:c].cumsum()
            #     a=a.T.astype(float)
            #     a["id"]=bes_irri_kc_spa.id
            #     a["geometry"]=bes_irri_kc_spa.geometry
            #     a_spa=geo.GeoDataFrame(a,geometry=a.geometry)
            #     a_spa.plot(column=c-1,figsize=(10,10),legend=True)
            #     plt.title("date en jj :%s "%c)
            #     plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/RESULT/PLOT/CARTE_Single_crops_ADOUR/decade_adour/carte_adour_besoin_Irr_single_crop_"+years+"_"+crops+"_"+str(c)+".png",dpi=600)
            # #  Somme mensuel
            # for d,c in zip(np.arange(0,bes_irri_kc_spa.shape[1]-30,30),np.arange(31,bes_irri_kc_spa.shape[1],30)):
            #     print(d,c)
            #     a=bes_irri_kc_spa.T[d:c].cumsum()
            #     a=a.T.astype(float)
            #     a["id"]=bes_irri_kc_spa.id
            #     a["geometry"]=bes_irri_kc_spa.geometry
            #     a_spa=geo.GeoDataFrame(a,geometry=a.geometry)
            #     a_spa.plot(column=c-1,figsize=(10,10),legend=True)
            #     plt.title("date en jj :%s "%c)
            #     plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/RESULT/PLOT/CARTE_Single_crops_ADOUR/mensuel_adour/carte_adour_besoin_Irr_single_crop_"+years+"_"+crops+"_"+str(c)+".png",dpi=600)
                # plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/RESULT/PLOT/CARTE_Single_crops_ADOUR/mensuel_adour/cumul_besoin_Irr_single_crop_"+years+"_"+crops+"_.png",dpi=600)
    #     MIRR["BES_IRR"]=MIRR.eval("ETP_Q*1.2-PRELIQ_Q")
    #     MIRR["BES_EAU"]=MIRR.eval("ETP_Q*1.2")
    #     if years =="2018":
    #         BESOIN_JOUR=MIRR.groupby("DATE")["BES_IRR"].mean()*sum(MIRR.Area[MIRR.DATE==str(years)+'-10-27'])
    #         BESOIN_JOUR[BESOIN_JOUR<0]=np.nan
    #         BESOIN_2018=pd.concat([MIRR.groupby("DATE")["PRELIQ_Q"].sum(),BESOIN_JOUR],axis=1)
    #         BESOIN_2018.columns=["Prec",'bes_irr_en_m3']
    #         BESOIN_2018["month"]=BESOIN_2018.index.strftime('%m')
    #         Besoin_sum_month_2018=BESOIN_2018.groupby("month").sum()
    #     else: 
    #         BESOIN_JOUR=MIRR.groupby("DATE")["BES_IRR"].mean()*sum(MIRR.Area[MIRR.DATE==str(years)+'-10-27'])*10
    #         BESOIN_JOUR[BESOIN_JOUR<0]=np.nan
    #         BESOIN_2017=pd.concat([MIRR.groupby("DATE")["PRELIQ_Q"].sum(),BESOIN_JOUR],axis=1)
    #         BESOIN_2017.columns=["Prec",'bes_irr_en_m3']
    #         BESOIN_2017["month"]=BESOIN_2017.index.strftime('%m')
    #         Besoin_sum_month_2017=BESOIN_2017.groupby("month").sum()
    
    # plt.figure(figsize=(7,7))
    # plt.plot(Besoin_sum_month_2018.index,Besoin_sum_month_2018.bes_irr_en_m3/1e6,marker='o',label="2018")
    # plt.plot(Besoin_sum_month_2017.index,Besoin_sum_month_2017.bes_irr_en_m3/1e6,marker='o',label="2017")
    # plt.ylabel("Besoin en eau en Mm3")
    # plt.xlabel("mois")
    # plt.legend()
    # plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/RESULT/PLOT/plt_besoin_Irr_adour.png")
    

