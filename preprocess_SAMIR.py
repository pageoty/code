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
import argparse
if __name__ == "__main__":
    """ mise en place df pour SAMIR """
    # parser = argparse.ArgumentParser(description='Preprocess data SAMIR ')
    # parser.add_argument('-path', dest='path',nargs='+',help="path file ",required = True)
    # parser.add_argument('-zone',dest='zone',nargs='+',help='optimisation value',required = True)
    # parser.add_argument('-name',dest='name_run',nargs='+',help='name run',required = True)
    # parser.add_argument('-REW',dest='REW',nargs='+',help='name run',required = True)
    # parser.add_argument('-RU_start',dest='IniRU',nargs='+',help='name run',required = True)
    # args = parser.parse_args()
    # print (args.optim)
    # print(args.name_run)
   
    years="2018"
    ZONE =["CACG"] # Fusion PARCELLE_CESBIO
    # name_run="RUNS_SAMIR/RUNS_SENSI_DATA_RAINFALL/DATA_STATION/"+str(years)+"/Inputdata/"
    name_run="RUNS_SAMIR/DATA_SCP_ICOS/CACG_SAFRAN/"+str(years)+"/Inputdata/"
    # mode="CSV"
    Meteo="SAFRAN"
    d={}
    d["path_run"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    # d["path_run_home"]="D:/THESE_TMP/TRAITEMENT/"+name_run+"/"
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/RUN_STOCK_DATA_2018_partenaire/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    d["PC_disk_labo"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/DONNES_METEO/"
    d["path_usb_PC"]="H:/YANN_THESE/BESOIN_EAU//BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/SAFRAN/"+str(years)+"/Inputdata/"
    d["PC_disk_home"]="D:/THESE_TMP/"
    

    list_bd_drop=['originfid', 'ogc_fid','centroidx', 'centroidy', 'shape_leng','shape_area']
    list_col_drop=['originfid',   'ogc_fid', ' n° sonde',' x',' y'] ;
    list_col_drop_cacg=[ '54787', 'originfid','ogc_fid']
    list_col_drop_tarn=['originfid','ogc_fid', 'num']
    list_col_drop_fus=['originfid', 'ogc_fid']

    dfnames=pd.read_csv(d["path_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_"+str(years)+".txt",sep=',', header=None) 
    
    for bv in ZONE:
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
        for n in os.listdir(d["path_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/"+str(bv)):
            if "SampleExtractionNDVI" in n and years in n and ".csv" in n: 
                df=pd.read_csv(d["path_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/"+str(bv)+"/"+str(n),sep=',')
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
            all_NDVI=all_NDVI.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
            tatar=all_NDVI.interpolate(method='time',limit_direction='both')
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
        NDVI.to_pickle(d["path_run"]+'/maize_irri/NDVI'+str(years)+'.df')
# =============================================================================
#   NDVI 
# =============================================================================
    # for bv in ZONE:
    #       if bv =="PARCELLE_CESBIO":
    #             df=pd.read_csv("D:/THESE_TMP/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_"+str(years)+".csv",sep=";")
    #             # df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_"+str(years)+".csv",sep=";")
    #             df.date=pd.to_datetime(df.date,format="%d/%m/%Y")
    #             df.set_index('date',inplace=True)
    #             df=df.resample("D").interpolate()
    #             df.reset_index(inplace=True)
    #             # df.date=pd.to_datetime(df.date,format="%d/%m/%Y")
    #             # df.set_index('date',inplace=True)
    #             meteo=df
    #             meteo["id"]=1
    #             meteo.to_pickle(d["path_run_home"]+"/maize_irri/NDVI"+str(years)+".df")
          # elif bv =="PARCELLE_GRIGNON":
          #         df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_Grignon/NDVI_Grignon_"+str(years)+".csv")
          #         df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
          #         df.set_index('date',inplace=True)
          #         df=df.resample("D").interpolate()
          #         df.reset_index(inplace=True)
          #         # df.set_index('date',inplace=True)
          #         NDVI=df
          #         NDVI["id"]=2
          #         NDVI.to_pickle(d["path_run"]+"/maize_rain/NDVI"+str(years)+".df")
# =============================================================================
# LAI
# =============================================================================
    # for bv in ZONE:
    #      if bv =="PARCELLE_CESBIO":
    #             df=LAI_sigmo=pd.read_csv(d["PC_disk_home"]+'/TRAITEMENT/INPUT_DATA/LAI_parcelle/PARCELLE_LABO/LAI_pre_inter_OTB/LAI_inter_dat_date_'+str(years)+'.csv')
    #             df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
    #             # df.set_index('date',inplace=True)
    #             meteo=df
    #             meteo["id"]=1
    #             meteo.to_pickle(d["path_run_home"]+"/maize_irri/LAI"+str(years)+".df")
         # elif bv =="PARCELLE_GRIGNON":
         #         df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_Grignon/NDVI_Grignon_"+str(years)+".csv")
         #         df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
         #         # df.set_index('date',inplace=True)
         #         NDVI=df
         #         NDVI["id"]=2
         #         NDVI.to_pickle(d["path_run"]+"/maize_rain/NDVI"+str(years)+".df")
# =============================================================================
#     Build df FC and WP
# =============================================================================
    # for bv in ZONE:
    #     if bv =="Fusion":
    #         Parcellaire=geo.read_file(d["PC_disk"]+"TRAITEMENT/DONNEES_VALIDATION_SAMIR/Parcelle_"+str(years)+".shp")
    #         for i in ["WP",'FC']:
    #             soil=geo.read_file(d["PC_disk"]+'TRAITEMENT/'+str(i)+'_0_2m_all_data.shp')
    #             soil.drop(columns=[ 'NOM', 'CULTURE', 'CULTURES','NUM', 'count',
    #         'min_0', 'max_0', 'geometry'],inplace=True)
    #             soil.columns=["id",str(i),str(i+'std')]
    #             soil.to_pickle(d["path_run"]+str(i)+'.df')
    
    # # =============================================================================
    # #  SOIL data Lamothe
    # # =============================================================================
    #     elif bv =="PARCELLE_CESBIO":
    #         for i in ["WP",'FC']:
    #             soil=pd.DataFrame({"id": [1], i: [np.nan],i+"std":[np.nan]})
    #             if i=="FC":
    #                 soil[str(i)].loc[0]=0.3635
    #                 soil[str(i)].loc[1]=np.mean([0.310,0.392])
    #             else:
    #                 soil[str(i)].loc[0]=np.mean([0.175,0.169])
    #                 soil[str(i)].loc[1]=np.mean([0.131,0.185])
    #             soil.to_pickle(d["path_run"]+'/maize_irri/'+str(i)+'.df') 
    # # =============================================================================
    # #     Soil Grignon 
    # # =============================================================================
    #     elif bv == "PARCELLE_GRIGNON":
    #         for i in ["WP",'FC']:
    #             soil=pd.DataFrame({"id": [2], i: [np.nan],i+"std":[np.nan]})
    #             if i=="FC":
    #                 soil[str(i)].loc[0]=0.48
    #                 soil[str(i)].loc[1]=np.nan
    #             else:
    #                 soil[str(i)].loc[0]=0.25
    #                 soil[str(i)].loc[1]=np.nan
    #             soil.to_pickle(d["path_run"]+'/maize_rain/'+str(i)+'.df')
# =============================================================================
#     Texture soil
# # =============================================================================
#     for bv in ZONE:
#         if bv =="Fusion":
#             Parcellaire=geo.read_file(d["PC_disk"]+"TRAITEMENT/DONNEES_VALIDATION_SAMIR/Parcelle_"+str(years)+".shp")
#             for i in ["WP",'FC']:
#                 soil=geo.read_file(d["PC_disk"]+'TRAITEMENT/'+str(i)+'_0_2m_all_data.shp')
#                 soil.drop(columns=[ 'NOM', 'CULTURE', 'CULTURES','NUM', 'count',
#             'min_0', 'max_0', 'geometry'],inplace=True)
#                 soil.columns=["id",str(i),str(i+'std')]
#                 soil.to_pickle(d["path_run"]+str(i)+'.df')
#     # =============================================================================
#     #  SOIL data Lamothe
#     # =============================================================================
#         elif bv =="PARCELLE_CESBIO":
#             soil=pd.DataFrame({"id": [1], "Clay": [np.nan],"Clay_std":[np.nan],"Sand":[np.nan], "Sand_std" : [np.nan]})
#             for i in ["Clay",'Sand']:
#                 if i=="Clay":
#                     soil[str(i)].loc[0]=0.5026
#                     soil[str(i)].loc[1]=np.mean([0,0])
#                 else:
#                     soil[str(i)].loc[0]=0.5585
#                     soil[str(i)].loc[1]=np.mean([0,0])
#             soil.to_pickle(d["path_run"]+'/maize_irri/Soil_texture.df') 
#     # =============================================================================
#     #     Soil Grignon 
#     # =============================================================================
#         elif bv == "PARCELLE_GRIGNON":
#             for i in ["WP",'FC']:
#                 soil=pd.DataFrame({"id": [2], i: [np.nan],i+"std":[np.nan]})
#                 if i=="FC":
#                     soil[str(i)].loc[0]=0.48
#                     soil[str(i)].loc[1]=np.nan
#                 else:
#                     soil[str(i)].loc[0]=0.25
#                     soil[str(i)].loc[1]=np.nan
#                 soil.to_pickle(d["path_run"]+'/maize_rain/'+str(i)+'.df')
# # =============================================================================
# #     Build METEO spatialieser 
# # =============================================================================
    # for bv in ZONE:
    #       if bv =="Fusion":
    #           # Parcellaire=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/vector_Classif_ADOUR_Fusion_demps_Accu.shp")
    #           meteo=geo.read_file(d["PC_disk_labo"]+"/SAFRAN_ZONE_"+str(years)+"_L93.shp")
    #           meteo.drop(columns=['field_1', 'LAMBX', 'LAMBY', 'PRENEI_Q', 'T_Q', 'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q',
    #         'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
    #         'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
    #         'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
    #         'X', 'Y'],inplace=True)
    #           dfmeteo=meteo.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
    #           meteo.geometry=dfmeteo
    #           meteo.DATE=pd.to_datetime(meteo.DATE,format='%Y%m%d')
    #           Meteo_par=geo.overlay(meteo,Parcellaire,how='intersection')
    #           Meteo_par.drop(columns=['Unnamed_ 0','NOM', 'CULTURE', 'CULTURE'],inplace=True)
    #           Meteo_par["Irrig"]=0.0
    #           Meteo_par.columns=["date","Prec",'ET0',"id",'Irrig']
    #           Meteo_par.info()
    #           Meteo_par.to_pickle(d["path_run"]+"/maize_irri/meteo.df")
    #       elif bv =="PARCELLE_CESBIO":
    #           if "SAFRAN" in Meteo:
    #               df=pd.read_csv("D:/THESE_TMP/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(years)+".csv",sep=";")
    #               # Irri=pd.read_csv("H:/YANN_THESE/BESOIN_EAU//BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(years)+".csv")
    #               df.date=pd.to_datetime(df.date,format="%d/%m/%Y")
    #               df.set_index('date',inplace=True)
    #               df.reset_index(inplace=True)
    #               df.drop(columns=["Unnamed: 0"],inplace=True)
    #               meteo=df
    #               meteo["id"]=1
    #               meteo.Irrig=df.Irrig
    #               meteo.to_pickle(d["path_run_home"]+"/maize_irri/meteo.df")
#               else: 
#                   ET0=pd.read_csv("H:/YANN_THESE/BESOIN_EAU//BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_"+str(years)+".csv")
#                   ET0.date=pd.to_datetime(ET0.date,format="%Y-%m-%d")
#                   ET0.set_index('date',inplace=True)
#                   ET0.reset_index(inplace=True)
#                   ET0.drop(columns=["Unnamed: 0"],inplace=True)
#                   meteo_stat=pd.read_csv("H:/YANN_THESE/BESOIN_EAU/BESOIN_EAU//TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/Meteo_station_"+str(years)+".csv")
#                   meteo_stat.date=pd.to_datetime(meteo_stat.date,format="%Y-%m-%d")
#                   # meteo_stat.drop(columns=["Unnamed: 0"],inplace=True)
#                   meteo_stat["ET0"]=ET0.ET0
#                   meteo_stat["id"]=1
#                   meteo_stat["Irrig"]=ET0.Irrig.astype(int)
#                   meteo_stat["Irrig"]=0
#                   meteo_stat.to_pickle(d["path_usb_PC"]+"/maize_irri/meteo.df")
   
# #          elif bv =="PARCELLE_GRIGNON":
# #              if "SAFRAN" in Meteo:
# #                  df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_GRI/meteo_gri_"+str(years)+".csv")
# #                  df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
# #                  df.set_index('date',inplace=True)
# #                  df.reset_index(inplace=True)
# #                  df.drop(columns=["Unnamed: 0"],inplace=True)
# #                  meteo=df
# #                  meteo["id"]=2
# #                  meteo.to_pickle(d["path_run"]+"/maize_rain/meteo.df")
# #              else: 
# #                  ET0=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_GRI/meteo_gri_"+str(years)+".csv")
# #                  ET0.date=pd.to_datetime(ET0.date,format="%Y-%m-%d")
# #                  ET0.set_index('date',inplace=True)
# #                  ET0.reset_index(inplace=True)
# #                  ET0.drop(columns=["Unnamed: 0"],inplace=True)
# #                  meteo_stat=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_GRI/Meteo_station_"+str(years)+".csv")
# #                  meteo_stat.date=pd.to_datetime(meteo_stat.date,format="%Y-%m-%d")
# #                  meteo_stat["ET0"]=ET0.ET0
# #                  meteo_stat["id"]=2
# #                  meteo_stat["Irrig"]=0
# #                  meteo_stat.to_pickle(d["path_run"]+"/maize_rain/meteo.df")
   
# # =============================================================================
# #  Préparation des Fcovers
# # =============================================================================
    # for bv in ZONE:
    #     if bv =="Fusion":
    #         print("pas pret")
    #     elif bv == "PARCELLE_CESBIO":
    #         df=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CESBIO/FCOVER_parcelles_"+str(years)+".csv",sep=";")
    #         # df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CESBIO/FCOVER_parcelles_"+str(years)+".csv",sep=",")
    #         # df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
    #         df.date=pd.to_datetime(df.date,format="%d/%m/%Y")
    #         df.set_index('date',inplace=True)
    #         df=df.resample("D").interpolate()
    #         df.reset_index(inplace=True)
    #         FCOVER=df
    #         FCOVER.columns=["date","FCov"]
    #         FCOVER["id"]=1
            # FCOVER.to_pickle(d["path_run"]+"/maize_irri/FCOVER.df")
        # elif bv =="PARCELLE_GRIGNON":
        #     df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_Grignon/FCOVER_Grignon_"+str(years)+".csv")
        #     df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
        #     df.set_index('date',inplace=True)
        #     df=df[df.index !='2019-08-24'] # data nuageuse sur la parcelle de Grignon
        #     df=df.resample("D").interpolate()
        #     df.reset_index(inplace=True)
        #     FCOVER=df
        #     FCOVER.columns=["date","FCov"]
        #     FCOVER["id"]=2
            # FCOVER.to_pickle(d["path_run"]+"/maize_rain/FCOVER.df")



# =============================================================================
#   Météo_SAFRAN
# =============================================================================
    # # Lecture data SAFRAN
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
    # SAF_par.to_csv("D:/THESE_TMP/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/Meteo_station_"+str(years)+".csv")
    
    # FCover=pd.read_csv("D:/THESE_TMP/DONNEES_RAW/PARCELLE_LABO/FCOVER/FCOVER2018/FCOVER_LAM_2018_OTB_INTERPO.csv",sep=";")
    # FCover.drop(columns=["Parcelle"],inplace=True)
    # FCover=FCover.mean()
    # Fcover=pd.DataFrame(FCover)
    # Fcover["date"]=pd.date_range("2011-01-01", periods=365)
    # Fcover.columns=['FCov',"date"]
    # Fcover.to_pickle(d["path_run_home"]+"/maize_irri/FCOVER.df")
