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
import seaborn as sns
import TEST_ANALYSE_SIGNATURE
import random
import shapely.geometry as geom
import descartes
from shapely.geometry import Point, Polygon
if __name__ == "__main__":
    """ mise en place df pour SAMIR """
    
    
    bv = "Fusion" # Fusion PARCELLE_CESBIO
    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/RUN_STOCK_DATA_2018_partenaire/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    d["PC_disk_labo"]="/run/media/pageot/ADATAHD650/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/"
    years="2018"


    list_bd_drop=['originfid', 'ogc_fid','centroidx', 'centroidy', 'shape_leng','shape_area']
    list_col_drop=['originfid',   'ogc_fid', ' n° sonde',' x',' y'] ;
    list_col_drop_cacg=[ '54787', 'originfid','ogc_fid']
    list_col_drop_tarn=['originfid','ogc_fid', 'num']
    list_col_drop_fus=['originfid', 'ogc_fid']

    dfnames=pd.read_csv(d["PC_disk"]+"TRAITEMENT/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_"+str(years)+".txt",sep=',', header=None) 
    
    for bv in [bv]:
        NDVI=pd.DataFrame()
        if bv =="CACG":
            dfNDVI_interTYP=pd.DataFrame()
            dfNDVI_interTCJ=pd.DataFrame()
        elif bv == "TARN":
            dfNDVI_interTCJ=pd.DataFrame()
            dfNDVI_interTDJ=pd.DataFrame()
        elif bv == "labo":
            dfNDVI_interTCJ=pd.DataFrame()
        else:
            dfNDVI_interTYN=pd.DataFrame()
            dfNDVI_interTYP=pd.DataFrame()
            dfNDVI_interTCJ=pd.DataFrame()
            dfNDVI_interTDJ=pd.DataFrame()
        for n in os.listdir(d["PC_disk"]+"/TRAITEMENT/NDVI_parcelle/Parcelle_ref/"+str(bv)):
            if "SampleExtractionNDVI" in n and years in n: 
                df=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/NDVI_parcelle/Parcelle_ref/"+str(bv)+"/"+str(n))
                tuile=n[-16:-13]
                #  gestion des colonnes du tableau
                if bv == "CACG":
                    df.columns=list(df.columns[0:4])+list(dfnames[0])
                    df.set_index("id",inplace=True)
                    dataN=df.drop(columns=list_col_drop_cacg)
                elif bv == "TARN":
                    df.columns=list(df.columns[0:4])+list(dfnames[0])
                    df.set_index("id",inplace=True)
                    dataN=df.drop(columns=list_col_drop_tarn)
                elif bv == "PARCELLE_CESBIO":
                    df.columns=list(df.columns[0:7])+list(dfnames[0])
                    df.set_index("id",inplace=True)
                    dataN=df.drop(columns=list_bd_drop)
                elif bv=='Fusion':
                    df.columns=list(df.columns[0:3])+list(dfnames[0])
                    df.set_index("id",inplace=True)
                    dataN=df.drop(columns=list_col_drop_fus)
                dataN.columns=pd.to_datetime(dataN.T.index,format="%Y%m%d")
                globals()["dfNDVI_inter%s"%tuile]=dataN.where(dataN!=0.0)
                globals()["dfNDVI_inter%s"%tuile].dropna(inplace=True)
                
                
        if bv == "CACG":
            id_nn_trai=set(dfNDVI_interTCJ.index)-set(dfNDVI_interTYP.index)
            non_traiter=dfNDVI_interTCJ.loc[id_nn_trai]
            all_NDVI=pd.concat([dfNDVI_interTYP,non_traiter])
            tatar=all_NDVI.T.resample("D").asfreq().interpolate()
            tar=tatar.T.sort_index(ascending=True)
        elif bv == "TARN" :
            id_nn_trai=set(dfNDVI_interTCJ.index)-set(dfNDVI_interTDJ.index) # non traiter par la tuile TDJ
            non_traiter=dfNDVI_interTCJ.loc[id_nn_trai]
            all_NDVI=pd.concat([dfNDVI_interTDJ,non_traiter])
            # all_NDVI.drop([5,7,11],inplace=True)
            tatar=all_NDVI.T.resample("D").asfreq().interpolate()
            tar=tatar.T.sort_index(ascending=True)
            
        elif bv == "Fusion" :
            id_nn_trai=set(dfNDVI_interTCJ.index)-set(dfNDVI_interTDJ.index)-set(dfNDVI_interTYP.index)
#            in_nn_trai_2=set(dfNDVI_interTCJ.index)-set(dfNDVI_interTDJ.index)
            non_traiter=dfNDVI_interTCJ.loc[id_nn_trai]
            all_NDVI=pd.concat([dfNDVI_interTDJ,non_traiter,dfNDVI_interTYP])
            # all_NDVI.drop([5,7,11,26,27],inplace=True) # verifier suppression parcelle
            tatar=all_NDVI.T.resample("D").asfreq().interpolate()
            tar=tatar.T.sort_index(ascending=True)
        elif bv == "PARCELLE_LABO":
            tatar=dfNDVI_interTCJ.T.resample("D").asfreq().interpolate()
            tar=tatar.T.sort_index(ascending=True)
        else:
            test=set(dfNDVI_interTCJ.index)-set(dfNDVI_interTYP.index)
            non=dfNDVI_interTCJ.loc[test]
            all_NDVI=pd.concat([dfNDVI_interTYP,non])
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
    # NDVI.to_csv("G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/Fusion/NDVI_ref_parcelle_"+str(years)+".csv")


    #  Select only maize
    # maize=RPG.loc[RPG.CODE_CULTU.str.startswith("MI")]
    # NDVIMAIZE=NDVI.loc[NDVI['id'].isin(maize.ID)]
    
    # NDVI.to_pickle(d["path_PC"]+"maize/NDVI.df")
    
# =============================================================================
#     Build df FC and WP
# =============================================================================

    if bv == "PARCELLE_CESBIO":
        Parcellaire=geo.read_file(d["PC_disk"]+"PARCELLE_LABO/PARCELLE_LABO_ref.shp")
        for i in ["WP",'FC']:
            soil=geo.read_file(d["PC_disk"]+'TRAITEMENT/'+str(i)+'_0_2m.shp')
            soil.drop(columns=['NOM_PARCEL', 'LABO', 'WP_0_30cm', 'WP_40_50m', 'FC_0_30cm',
          'FC_40_50cm', 'pt_sat0_30', 'pt_sat40_5', 'RU_0_30', 'RU_40_50',
          'RU_SG_60cm', 'sdRU_SG_60', 'RU_SG_0_30', 'sdRU_SG0_3', 'WP_SG_60',
          'sdWP_SG_60', 'FC_SG_60', 'sdFC_SG_60', 'count',
          'min_0', 'max_0', 'geometry'],inplace=True)
            soil.columns=["id",str(i),str(i+'std')]
            # soil.to_pickle(d["path_PC"]+'/maize/'+str(i)+'.df')
            
    elif bv =="Fusion":
        Parcellaire=geo.read_file(d["PC_disk"]+"TRAITEMENT/DONNEES_VALIDATION_SAMIR/Parcelle_"+str(years)+".shp")
        for i in ["WP",'FC']:
            soil=geo.read_file(d["PC_disk"]+'TRAITEMENT/'+str(i)+'_0_2m_all_data.shp')
            soil.drop(columns=[ 'NOM', 'CULTURE', 'CULTURES','NUM', 'count',
       'min_0', 'max_0', 'geometry'],inplace=True)
            soil.columns=["id",str(i),str(i+'std')]
            soil.to_pickle(d["path_PC"]+str(i)+'.df')

# =============================================================================
#  SOIL data Lamothe
# =============================================================================
#        if i =="FC":
#            soil[str(i)].loc[0]=0.3635
#            soil[str(i)].loc[1]=np.mean([0.310,0.392])
#        else:
#            soil[str(i)].loc[0]=np.mean([0.175,0.169])
#            soil[str(i)].loc[1]=np.mean([0.131,0.185])
#        soil.to_pickle(d["path_PC"]+'/maize/'+str(i)+'.df')
        
    
    
# =============================================================================
#     Build METEO spatialieser 
# =============================================================================

    meteo=geo.read_file(d["PC_disk"]+"DONNES_METEO/SAFRAN_ZONE_"+str(years)+"_L93.shp")
    meteo.drop(columns=['field_1', 'LAMBX', 'LAMBY', 'PRENEI_Q',
    'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q',
    'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
    'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
    'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
    'X', 'Y'],inplace=True)
   
    dfmeteo=meteo.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
    meteo.geometry=dfmeteo
    # meteo.iloc[100:1500].to_file(d['PC_disk']+"/meteo_test.shp")
    meteo.DATE=pd.to_datetime(meteo.DATE,format='%Y%m%d')   
    # Parcellaire.geometry=Parcellaire.centroid
    # Parcellaire.to_file(d["PC_disk"]+'/parcellaire_test.shp')
    Meteo_par=geo.overlay(meteo,Parcellaire,how='intersection') # probleme ne prend pas la majoritaire, test avec centroide error (pb version geopandas)
    # Meteo_par.to_file(d['PC_disk']+"/inter_meteo_par.shp")
    if bv == "PARCELLE_CESBIO":
#     Drop lam
        Meteo_par.drop(columns=["geometry",'NOM_PARCEL', 'LABO','WP_0_30cm','WP_40_50m', 'FC_0_30cm', 
                                'FC_40_50cm', 'pt_sat0_30', 'pt_sat40_5','RU_0_30', 'RU_40_50', 'RU_SG_60cm',
                                'sdRU_SG_60', 'RU_SG_0_30', 'sdRU_SG0_3', 'WP_SG_60', 'sdWP_SG_60', 'FC_SG_60',
           'sdFC_SG_60', 'geometry'],inplace=True)
        Meteo_par["Irrig"]=0.0
        Meteo_par.columns=["date","Prec",'ET0',"id","Irrig"]
        Meteo_par.info()
        lam=Meteo_par.loc[np.where(Meteo_par.id==0)] # if 2018 .iloc[:-31] # id 0 == lamothe
        lam.drop(columns="id",inplace=True)
        lam.to_pickle(d["path_PC"]+"/meteo.df")
    else:
#     Drop fusion
        Meteo_par.drop(columns=['Unnamed_ 0','NOM', 'CULTURE', 'CULTURE'],inplace=True)
        Meteo_par["Irrig"]=0.0
        # Meteo_par.columns=["date","Prec",'ET0',"id",'Irrig']
        Meteo_par.info()
        Meteo_par.to_pickle(d["path_PC"]+"/meteo.df")


    # Select data SAFRAN data 
    Safran=pd.read_csv(d["PC_disk_labo"]+"/SIM2_2018_202002.csv",sep=";")
    SAF2017=Safran.loc[(Safran.DATE >= 20180101)& (Safran["DATE"] < 20190101)]
    SAF2017["X"]=SAF2017.LAMBX*100
    SAF2017["Y"]=SAF2017.LAMBY*100
    # geometry = SAF2017.apply(lambda row: Point(row.X, row.Y), axis=1)
    # crs = {'init': 'SR-ORG:7528'} 
    # geo_df = geo(SAF2017, crs=crs, geometry=geometry)
    # geo_df.to_file(filename = '/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/DONNES_METEO/SAFRAN_test_2018_l2.shp', driver ='ESRI Shapefile')
    SAF2017.to_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/DONNES_METEO/SAFRAN_2018.csv")

    
    saf2018=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/DONNES_METEO/SAFRAN_2018.csv")
    saf2018["geometry"] = SAF2017.apply(lambda row: Point(row.X, row.Y), axis=1)
    saf_2019=geo.read_file("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/DONNES_METEO/SAFRAN_ZONE_2019_L2.shp")
    saf_zone=saf_2019.loc[saf_2019.DATE==20190101.0]
    
    a=list(saf_zone.X.astype(int))
    saf_zone_test_2018=saf2018.loc[a]
    
    
    df1.sort_index().sort_index(axis=1) == df2.sort_index().sort_index(axis=1)
    saf2018[saf2018["X"]==saf_zone["X"].astype(int)]

# =============================================================================
# NDVI2014
# =============================================================================
    # NDVI2014=pd.read_csv('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/NDV_2014_lAND_FORM.csv',decimal=",")
#     NDVI2014_form=NDVI2014.iloc[2,1:13].astype(float)
#     NDVI2014_form.index=pd.to_datetime(NDVI2014_form.index,format="%Y-%m-%d")
#     NDVI2014_f=NDVI2014_form.resample("D").asfreq().interpolate()
#     NDVI2014_land=pd.read_csv('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/NDV_2014_lAND.csv',decimal=",")
#     NDVI2014_land=NDVI2014_land.iloc[2,1:].astype(float)
#     NDVI2014_land.index=pd.to_datetime(NDVI2014_land.index,format="%Y-%m-%d")
#     NDVI2014_L=NDVI2014_land.resample("D").asfreq().interpolate()
    
#     NDVI2014_f.iloc[:-3].plot(label="formosat")
#     NDVI2014_L.iloc[-71:].plot(label='Landsat')
#     NDVI2014=NDVI2014_f.iloc[:-3].append(NDVI2014_L.iloc[-71:])
#     NDVI2014.to_csv("G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/NDVI2014_FORMO_LAND.csv")
#     # plt.legend()
#     # plt.savefig('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/PLOT_NDVI_2014_interpo.png')

    # NDVI2015=pd.read_csv('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/NDVI_parcelles_2012.csv',decimal=",")
    # NDVI2015_Spot=NDVI2015.T.astype(float)
    # NDVI2015_Spot.index=pd.to_datetime(NDVI2015_Spot.index,format="%Y-%m-%d")
    # NDVI2015_S=NDVI2015_Spot.resample("D").asfreq().interpolate()
    # NDVI2015_S.plot()
    # NDVI2015_land=pd.read_csv('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/NDV_2015_lAND.csv',decimal=",")
    # NDVI2015_land=NDVI2015_land.T.astype(float)
    # NDVI2015_land.index=pd.to_datetime(NDVI2015_land.index,format="%Y-%m-%d")
    # NDVI2015_L=NDVI2015_land.resample("D").asfreq().interpolate()
    
    # NDVI2015_S.to_csv("G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/NDVI2012.csv")

# # =============================================================================
# #  Préparation des Fcovers
# # =============================================================================
#     for f in os.listdir('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/FCOVER_parcelle/'):
#         print(f)
#         df=pd.read_csv('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/FCOVER_parcelle/'+f,decimal=",")
#         df=df.T.astype(float)
#         # df.set_index(['Unnamed: 0'],inplace=True)
#         df.index=pd.to_datetime(df.index,format="%Y-%m-%d")
#         df_inter=df.resample("D").asfreq().interpolate()
#         df_inter.plot()
#         plt.title(f[:10])
#         df_inter.to_csv('G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/FCOVER_parcelle/interpol'+str(f[:-32]+'.csv'))
        
# =============================================================================
#   Prépartion data filecsv 
# =============================================================================
    for y in os.listdir("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_NDVI/"):
        years=y[-8:-4]
        df=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_NDVI/LAMOTHE_NDVI_"+str(years)+".csv",decimal=".")
        df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
        NDVI=df
        # NDVI=df[["Date","NDVI"]]
        # NDVI.columns=["date",'NDVI']
        NDVI["id"]=1
        NDVI.NDVI=0 # test sans végétation 
        NDVI.to_pickle("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/Bilan_hydrique/DATA_NDVI_SOIL_NU/NDVI%s.df"%(years))
        # NDVI.to_pickle("D:/THESE_TMP/RUNS_SAMIR/RUN_SENSI_SOL/"+str(years)+"/Inputdata/maize/NDVI.df")
#     for y in os.listdir("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_METEO/"):
#         years=y[-8:-4]
#         df=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_METEO/Meteo_lam_"+str(years)+".csv",decimal=".")
#         Meteo=df[["date","ET0","Prec","Irrig"]]
#         Meteo.date=pd.to_datetime(Meteo.date,format="%Y-%m-%d")
#         # Meteo.columns=["date",'ET0',"Prec","Irrig"]
#         Meteo["id"]=1
#         # Meteo.set_index('date',inplace=True)
#         Meteo.to_pickle("D:/THESE_TMP/RUNS_SAMIR/RUN_SENSI_SOL/"+str(years)+"/Inputdata/meteo.df")
        

    for y in os.listdir("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_FCOVER/"):
        years=y[-8:-4]
        df=pd.read_csv("G:/Yann_THESE/BESOIN_EAU/Calibration_SAMIR/DONNEES_CALIBRATION/DATA_FCOVER/FCOVER_parcelles_"+str(years)+".csv",decimal=".")
        df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
        FCOVER=df
        # FCOVER=df[["Date","FCOVER"]]
        FCOVER.columns=["date","FCov"] # nom de la colonne importante
        FCOVER["id"]=1
        # FCOVER['OS']="maize"
        FCOVER.to_pickle("D:/THESE_TMP/RUNS_SAMIR/RUNS_optim_LUT_LAM_ETR/Run_opti_param_v2_ss_spin_Fcover/"+str(years)+"/Inputdata/maize/FCOVER.df")