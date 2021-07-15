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
   
    years="2017"
    ZONE =["GERS"] # Fusion PARCELLE_CESBIO
    # name_run="RUNS_SAMIR/RUNS_SENSI_DATA_RAINFALL/DATA_STATION/"+str(years)+"/Inputdata/"
    name_run="RUNS_SAMIR/DATA_SCP_ICOS/CLASSIF_ALL_MAIS/"+str(years)+"/Inputdata/"
    # mode="CSV"
    Meteo="SAFRAN"
    d={}
    # d["path_run"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    # d["path_run"]="H:/YANN_THESE/BESOIN_EAU//BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/RUN_STOCK_DATA_2018_partenaire/Inputdata/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_disk_labo"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["path_usb_PC"]="H:/YANN_THESE/BESOIN_EAU//BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/SAFRAN/"+str(years)+"/Inputdata/"
    d["PC_disk_home"]="D:/THESE_TMP/"
    d["path_run_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    

    list_bd_drop=['originfid', 'ogc_fid','centroidx', 'centroidy', 'shape_leng','shape_area']
    list_col_drop=['originfid',   'ogc_fid', ' n° sonde',' x',' y'] ;
    list_col_drop_cacg=[ '54787', 'originfid','ogc_fid']
    list_col_drop_tarn=['originfid','ogc_fid', 'num']
    list_col_drop_fus=['originfid', 'ogc_fid']

    dfnames=pd.read_csv(d["PC_disk"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_"+str(years)+".txt",sep=',', header=None) 
    
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
        for n in os.listdir(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/"+str(bv)):
            if "SampleExtractionNDVI" in n and years in n and ".csv" in n: 
                df=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/"+str(bv)+"/"+str(n),sep=',')
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
        NDVI=NDVI.loc[(NDVI.id<14.0)&(NDVI.id!=6.0) & (NDVI.id!=8.0) & (NDVI.id!=2) & (NDVI.id!=3)]
        NDVI.columns=["NDVI","id","date"]
        # NDVI.to_pickle(d["path_run"]+'/maize_irri/NDVI'+str(years)+'.df')
        
        
        # for i in list(set(NDVI.id)):
        #     plt.figure(figsize=(10,10))
        #     plt.plot(NDVI.loc[NDVI.id==i]["date"],NDVI.loc[NDVI.id==i]["NDVI"],label=i)
        #     plt.legend()
            
# =============================================================================
#   NDVI 
# =============================================================================
    for bv in ZONE:
          if bv =="PARCELLE_CESBIO":
                df=pd.read_csv("D:/THESE_TMP/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_"+str(years)+".csv",sep=";")
                # df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_"+str(years)+".csv",sep=";")
                df.date=pd.to_datetime(df.date,format="%d/%m/%Y")
                df.set_index('date',inplace=True)
                df=df.resample("D").interpolate()
                df.reset_index(inplace=True)
                # df.date=pd.to_datetime(df.date,format="%d/%m/%Y")
                # df.set_index('date',inplace=True)
                meteo=df
                meteo["id"]=1
                meteo.to_pickle(d["path_run_home"]+"/maize_irri/NDVI"+str(years)+".df")
          elif bv =="PARCELLE_GRIGNON":
                df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_Grignon/NDVI_Grignon_"+str(years)+".csv")
                df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
                df.set_index('date',inplace=True)
                df=df.resample("D").interpolate()
                df.reset_index(inplace=True)
                # df.set_index('date',inplace=True)
                NDVI=df
                NDVI["id"]=2
                NDVI.to_pickle(d["path_run"]+"/maize_rain/NDVI"+str(years)+".df")
          elif bv =="PKGC":
                dfnames=pd.read_csv(d["PC_disk_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_"+str(years)+".txt",sep=',', header=None)
                dfs=pd.DataFrame(dfnames)
                dates=pd.to_datetime(dfnames[0],format="%Y%m%d")
                df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_PKGC/PARCELLE_PKGC_TYP_NDVI.csv",sep=",")
                tmp=df[["ID"]]
                tmp1=pd.DataFrame()
                for i in np.arange(0,36,1): #♣ 2018 : 49 :  2017 : 41
                      a=df["mean_"+str(i)]/1000
                      tmp1=tmp1.append(a)
                Fcover=tmp1.T
                Fcover.columns=list(dates)
                # Fcover=Fcover.T
                Fcover.T.sort_index(inplace=True)
                Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
                Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
                Fcover=Fcover.append(df.ID)
                Fcover=Fcover.T
                Fcover.set_index("ID",inplace=True)
                FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
                FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'NDVI'}, inplace=True)
                FCOVER.to_pickle(d["path_run_disk"]+"/maize_irri/NDVI2017.df")
          elif bv == 'ASA' :
                dfnames=pd.read_csv(d["PC_disk_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_"+str(years)+".txt",sep=',', header=None)
                dfs=pd.DataFrame(dfnames)
                dates=pd.to_datetime(dfnames[0],format="%Y%m%d")
                df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/PARCELLE_ASA/NDVI_ASA_2018.csv",sep=",")
                tmp=df[["ID"]]
                tmp1=pd.DataFrame()
                for i in np.arange(0,37,1): #♣ 2018 : 49 :  2017 : 41
                      a=df["mean_"+str(i)]/1000
                      tmp1=tmp1.append(a)
                Fcover=tmp1.T
                Fcover.columns=list(dates)
                # Fcover=Fcover.T
                Fcover.T.sort_index(inplace=True)
                Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
                Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
                Fcover=Fcover.append(df.ID)
                Fcover=Fcover.T
                Fcover.set_index("ID",inplace=True)
                FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
                FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'NDVI'}, inplace=True)
                FCOVER.to_pickle(d["path_run"]+"/maize_irri/NDVI2018.df")
          elif bv == "FUSION":
                dfnames=pd.read_csv(d["PC_disk_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_"+str(years)+".txt",sep=',', header=None)
                dfs=pd.DataFrame(dfnames)
                dates=pd.to_datetime(dfnames[0],format="%Y%m%d")
                df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref/FUSION/NDVI_2017_FUSION_v2.csv",sep=";")
                tmp=df[["ID"]]
                tmp1=pd.DataFrame()
                for i in np.arange(0,36,1): #♣ 2018 : 49 :  2017 : 41
                      a=df["mean_"+str(i)]/1000
                      tmp1=tmp1.append(a)
                Fcover=tmp1.T
                Fcover.columns=list(dates)
                # Fcover=Fcover.T
                Fcover.T.sort_index(inplace=True)
                Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
                Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
                Fcover=Fcover.append(df.ID)
                Fcover=Fcover.T
                Fcover.set_index("ID",inplace=True)
                FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
                FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'NDVI'}, inplace=True)
                # FCOVER.to_pickle(d["path_run"]+"/maize_irri/NDVI2017.df")
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
    for bv in ZONE:
    #     if bv =="Fusion":
    #         Parcellaire=geo.read_file(d["PC_disk"]+"TRAITEMENT/DONNEES_VALIDATION_SAMIR/Parcelle_"+str(years)+".shp")
            # for i in ["WP",'FC']:
            #     soil=geo.read_file(d["PC_disk"]+'TRAITEMENT/'+str(i)+'_0_2m_all_data.shp')
            #     soil.drop(columns=[ 'NOM', 'CULTURE', 'CULTURES','NUM', 'count',
            # 'min_0', 'max_0', 'geometry'],inplace=True)
            #     soil.columns=["id",str(i),str(i+'std')]
            #     soil.to_pickle(d["path_run"]+str(i)+'.df')
    
    # # =============================================================================
    # #  SOIL data Lamothe
    # # =============================================================================
        if bv =="CACG":
            for i in ["WP",'FC']:
                soil=pd.DataFrame()
                for j in np.arange(1,14):
                    a=pd.DataFrame({"id": j, i: [np.nan],i+"std":[np.nan]})
                    soil=soil.append(a)
                # if i=="FC":
                #     soil[str(i)].loc[0]=0.3635
                #     soil[str(i)].loc[1]=np.mean([0.310,0.392])
                # else:
                #     soil[str(i)].loc[0]=np.mean([0.175,0.169])
                #     soil[str(i)].loc[1]=np.mean([0.131,0.185])
                soil=soil.loc[(soil.id<14.0)&(soil.id!=6.0) & (soil.id!=8.0) & (soil.id!=2) & (soil.id!=3)]
                # soil.to_pickle(d["path_run"]+'/maize_irri/'+str(i)+'.df') 
        elif bv == "PKGC":
            for i in ["WP",'FC']:
                soil=pd.DataFrame()
                for j in np.arange(1,45):
                    a=pd.DataFrame({"id": j, i: [np.nan],i+"std":[np.nan]})
                    soil=soil.append(a)
                # soil.to_pickle(d["path_run_disk"]+'/maize_irri/'+str(i)+'.df')
        elif bv == "ASA":
            for i in ["WP",'FC']:
                soil=pd.DataFrame()
                for j in np.arange(1,387):
                    a=pd.DataFrame({"id": j, i: [np.nan],i+"std":[np.nan]})
                    soil=soil.append(a)
                # soil.to_pickle(d["path_run"]+'/maize_irri/'+str(i)+'.df')
        elif bv == "Adour_Tarn":
            for i in ["WP",'FC']:
                soil=pd.DataFrame()
                for j in list(set(FCOVER_Gers.id)):
                    a=pd.DataFrame({"id": j, i: [np.nan],i+"std":[np.nan]})
                    soil=soil.append(a)
                    # soil=soil[soil.id!=96]
                soil.to_pickle(d["path_run_disk"]+'/maize_irri/'+str(i)+'.df')
                # if i=="FC":
                # if i=="FC":
                #     soil[str(i)].loc[0]=0.3635
                #     soil[str(i)].loc[1]=np.mean([0.310,0.392])
                # else:
                #     soil[str(i)].loc[0]=np.mean([0.175,0.169])
                #     soil[str(i)].loc[1]=np.mean([0.131,0.185])
                # soil=soil.loc[(soil.id<14.0)&(soil.id!=6.0) & (soil.id!=8.0) & (soil.id!=2) & (soil.id!=3)]
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
                # soil.to_pickle(d["path_run"]+'/maize_rain/'+str(i)+'.df')
# =============================================================================
#     Texture soil
# # =============================================================================
    for bv in ZONE:
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
        if bv =="CACG":
                soil=pd.DataFrame()
                for j in np.arange(1,14):
                    a=pd.DataFrame({"id": j, "Clay": [np.nan],"Clay_std":[np.nan],"Sand":[np.nan], "Sand_std" : [np.nan]})
                    soil=soil.append(a)
                    soil=soil.loc[(soil.id<14.0)&(soil.id!=6.0) & (soil.id!=8.0) & (soil.id!=2) & (soil.id!=3)]
        elif bv =="PKGC" :
                soil=pd.DataFrame()
                for j in np.arange(1,45):
                    a=pd.DataFrame({"id": j, "Clay": [np.nan],"Clay_std":[np.nan],"Sand":[np.nan], "Sand_std" : [np.nan]})
                    soil=soil.append(a)
                soil.to_pickle(d["path_run_disk"]+'/maize_irri/Soil_texture.df')
        elif bv =="ASA" :
                soil=pd.DataFrame()
                for j in np.arange(1,387):
                    a=pd.DataFrame({"id": j, "Clay": [np.nan],"Clay_std":[np.nan],"Sand":[np.nan], "Sand_std" : [np.nan]})
                    soil=soil.append(a)
                soil.to_pickle(d["path_run"]+'/maize_irri/Soil_texture.df')
        elif bv =="Adour_Tarn" :
                soil=pd.DataFrame()
                for j in list(set(FCOVER_Gers.id)):
                    a=pd.DataFrame({"id": j, "Clay": [np.nan],"Clay_std":[np.nan],"Sand":[np.nan], "Sand_std" : [np.nan]})
                    soil=soil.append(a)
                # soil=soil[soil.id!=96]
                # soil.to_pickle(d["path_run_disk"]+'/maize_irri/Soil_texture.df')
                    # soil=soil.loc[(soil.id<14.0)&(soil.id!=6.0) & (soil.id!=8.0) & (soil.id!=2) & (soil.id!=3)]
#                     soil=soil.loc[soil.id!=15]
# #                     # soil=pd.DataFrame({"id": [1], "Clay": [np.nan],"Clay_std":[np.nan],"Sand":[np.nan], "Sand_std" : [np.nan]})
# # # #             for i in ["Clay",'Sand']:
# # # #                 if i=="Clay":
# # # #                     soil[str(i)].loc[0]=0.5026
# # # #                     soil[str(i)].loc[1]=np.mean([0,0])
# # # #                 else:
# # # #                     soil[str(i)].loc[0]=0.5585
# # # #                     soil[str(i)].loc[1]=np.mean([0,0])
                soil.to_pickle(d["path_run_disk"]+'/maize_irri/Soil_texture.df') 
# #     # =============================================================================
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
    for bv in ZONE:
        if bv =="Fusion":
            print("pas pret")
        elif bv == "PARCELLE_CESBIO":
            df=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CESBIO/FCOVER_parcelles_"+str(years)+".csv",sep=";")
            # df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CESBIO/FCOVER_parcelles_"+str(years)+".csv",sep=",")
            # df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
            df.date=pd.to_datetime(df.date,format="%d/%m/%Y")
            df.set_index('date',inplace=True)
            df=df.resample("D").interpolate()
            df.reset_index(inplace=True)
            FCOVER=df
            FCOVER.columns=["date","FCov"]
            FCOVER["id"]=1
            FCOVER.to_pickle(d["path_run"]+"/maize_irri/FCOVER.df")
        elif bv =="PARCELLE_GRIGNON":
            df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_Grignon/FCOVER_Grignon_"+str(years)+".csv")
            df.date=pd.to_datetime(df.date,format="%Y-%m-%d")
            df.set_index('date',inplace=True)
            df=df[df.index !='2019-08-24'] # data nuageuse sur la parcelle de Grignon
            df=df.resample("D").interpolate()
            df.reset_index(inplace=True)
            FCOVER=df
            FCOVER.columns=["date","FCov"]
            FCOVER["id"]=2
            FCOVER.to_pickle(d["path_run"]+"/maize_rain/FCOVER.df")
        elif bv=="CACG":
            dfnames=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CACG/data_Raw/list_FCOVER_"+str(years)+"_TYP.txt",sep=',', header=None)
            dates=dfnames[0].apply(lambda x:x[11:19])
            dates=pd.to_datetime(dates,format="%Y%m%d")
            df=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CACG/data_Raw/PARCELLE_CACG_TYP_"+str(years)+".csv",decimal=".")
            tmp=df[["ID"]]
            tmp1=pd.DataFrame()
            for i in np.arange(0,49,2): #♣ 2018 : 49 :  2017 : 41
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a)
            Fcover=tmp1.T
            Fcover.columns=list(dates)
            # Fcover=Fcover.T
            Fcover.T.sort_index(inplace=True)
            Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
            Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
            Fcover=Fcover.append(df.ID)
            Fcover=Fcover.T
            Fcover.set_index("ID",inplace=True)
            FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
            FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'}, inplace=True)
            FCOVER=FCOVER.loc[(FCOVER.id<14.0)&(FCOVER.id!=6.0) & (FCOVER.id!=8.0) & (FCOVER.id!=2) & (FCOVER.id!=3)]
        elif bv =="PKGC" :
            dfnames=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CACG/data_Raw/list_FCOVER_2017_TYP.txt",sep=',', header=None)
            dates=dfnames[0].apply(lambda x:x[11:19])
            dates=pd.to_datetime(dates,format="%Y%m%d")
            df=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_PKGC/PARCELLE_PKGC_TYP.csv",decimal=".")
            tmp=df[["ID"]]
            tmp1=pd.DataFrame()
            for i in np.arange(0,41,2): #♣ 2018 : 49 :  2017 : 41
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a)
            Fcover=tmp1.T
            Fcover.columns=list(dates)
            # Fcover=Fcover.T
            Fcover.T.sort_index(inplace=True)
            Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
            Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
            Fcover=Fcover.append(df.ID)
            Fcover=Fcover.T
            Fcover.set_index("ID",inplace=True)
            FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
            FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'}, inplace=True)
            # FCOVER=FCOVER.loc[(FCOVER.id<14.0)&(FCOVER.id!=7.0)]
        elif bv =="ASA" :
            dfnames=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CACG/data_Raw/list_FCOVER_2018_TYP.txt",sep=',', header=None)
            dates=dfnames[0].apply(lambda x:x[11:19])
            dates=pd.to_datetime(dates,format="%Y%m%d")
            df=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_ASA/PARCELLE_ASA_2018_TYP.csv",decimal=".")
            tmp=df[["ID"]]
            tmp1=pd.DataFrame()
            for i in np.arange(0,49,2): #♣ 2018 : 49 :  2017 : 41
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a)
            Fcover=tmp1.T
            Fcover.columns=list(dates)
            # Fcover=Fcover.T
            Fcover.T.sort_index(inplace=True)
            Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
            Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
            Fcover=Fcover.append(df.ID)
            Fcover=Fcover.T
            Fcover.set_index("ID",inplace=True)
            FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
            FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'}, inplace=True)
        elif bv =="ADOUR":
            for t in ["TYN","TYP"]:
                dfnames=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_TARN/list_FCOVER_2018_"+t+".txt",sep=',', header=None)
                dates=dfnames[0].apply(lambda x:x[11:19])
                dates=pd.to_datetime(dates,format="%Y%m%d")
                df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_ADOUR/PARCELLE_ADOUR_2018_"+t+"_FCOVER.csv",decimal=".")
                tmp=df[["ID"]]
                tmp1=pd.DataFrame()
                if t =="TYN":
                    for i in np.arange(0,26,2): #♣ 2018 : 49 :  2017 : 41
                        a=df["mean_"+str(i)]
                        tmp1=tmp1.append(a)
                else:
                    for i in np.arange(0,49,2): #♣ 2018 : 49 :  2017 : 41
                        a=df["mean_"+str(i)]
                        tmp1=tmp1.append(a)
                Fcover=tmp1.T
                Fcover.columns=list(dates)
                # Fcover=Fcover.T
                Fcover.T.sort_index(inplace=True)
                Fcover.T.sort_index(ascending=True,inplace=True)
                Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
                Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
                Fcover=Fcover.append(df.ID)
                Fcover=Fcover.T
                Fcover.set_index("ID",inplace=True)
                FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
                if t =='TYN':
                    FCOVERTYN=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
                else:
                    FCOVERTYP=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
            FCOVER=pd.concat([FCOVERTYN,FCOVERTYP])
            ADOUR_FCOVER=FCOVER.drop_duplicates(subset=["id","date"])
        elif bv =="TARN":
            for t in ["TDJ","TCJ"]:
                dfnames=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_TARN/list_FCOVER_2018_"+t+".txt",sep=',', header=None)
                # if t == 'TCJ':
                #     dates=dfnames[0].apply(lambda x:x[0:8])
                # else:
                dates=dfnames[0].apply(lambda x:x[11:19])
                dates=pd.to_datetime(dates,format="%Y%m%d")
                df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_TARN/PARCELLE_TARN_2018_"+t+"_FCOVER.csv",decimal=".")
                tmp=df[["id"]]
                tmp1=pd.DataFrame()
                if t =="TCJ":
                    for i in np.arange(0,199,2): #♣ 2018 : 49 :  2017 : 41
                        a=df["mean_"+str(i)]
                        tmp1=tmp1.append(a)
                else:
                    for i in np.arange(0,49,2): #♣ 2018 : 49 :  2017 : 41
                        a=df["mean_"+str(i)]
                        tmp1=tmp1.append(a)
                Fcover=tmp1.T
                Fcover.columns=list(dates)
                Fcover=Fcover.T
                Fcover.sort_index(ascending=True,inplace=True)
                Fcover=Fcover.loc[~Fcover.index.duplicated(), :]
                Fcover=Fcover.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
                Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
                Fcover=Fcover.append(df.id)
                Fcover=Fcover.T
                Fcover.set_index("id",inplace=True)
                FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
                if t =='TCJ':
                    FCOVERTCJ=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
                else:
                    FCOVERTDJ=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
            FCOVER=pd.concat([FCOVERTCJ,FCOVERTDJ])
            FCOVER_TARN=FCOVER.drop_duplicates(subset=["id","date"])
            FCOVER=pd.concat([ADOUR_FCOVER,FCOVER_TARN])
            FCOVER.to_pickle(d["path_run_disk"]+"/maize_irri/Fcover.df")


# =============================================================================
# Pour le PKGC GErs
# =============================================================================
    for t in ["TYP","TCJ"]:
        # dfnames=pd.read_csv(d["PC_disk_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_2017.txt",sep=',', header=None)
        # dfs=pd.DataFrame(dfnames)
        # dates=pd.to_datetime(dfnames[0],format="%Y%m%d")
        dfnames=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_TARN/list_FCOVER_2017_"+t+".txt",sep=',', header=None)
        if t =='TCJ':
            dates=dfnames[0].apply(lambda x:x[0:8])
        else:
            dates=dfnames[0].apply(lambda x:x[11:19])
        dates=pd.to_datetime(dates,format="%Y%m%d")
        if t == 'TCJ':
            dates.drop(49,inplace=True)
        # df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref//PARCELLE_PKGC/GERS/NDVI_2017_PKGC_GERS_"+t+".csv",decimal=".")
        df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_PKGC/GERS/FCOVER_2017_PKGC_GERS_"+t+".csv",decimal=".")

        tmp=df[["ID"]]
        tmp1=pd.DataFrame()
        if t =="TYP":
            for i in np.arange(0,41,2): #♣ 2018 : 49 :  2017 : 41
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a)
        else:
            for i in np.arange(0,167,2): #♣ 2018 : 49 :  2017 : 41
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a)
        if t == "TCJ":
            tmp1.drop('mean_98',inplace=True)
        Fcover=tmp1.T
        Fcover.columns=list(dates)
        # Fcover=Fcover.T
        Fcover.T.sort_index(inplace=True)
        Fcover.T.sort_index(ascending=True,inplace=True)
        Fcover=Fcover.T.reindex(pd.date_range(start="2017-01-01",end="2017-12-31",freq='1D'))
        Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
        Fcover=Fcover.append(df.ID)
        Fcover=Fcover.T
        Fcover.set_index("ID",inplace=True)
        FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
        if t =='TYP':
            FCOVERTYN=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
        else:
            FCOVERTYP=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
    FCOVER=pd.concat([FCOVERTYN,FCOVERTYP])
    FCOVER_Gers=FCOVER.drop_duplicates(subset=["id","date"])
    FCOVER_Gers.to_pickle(d["path_run_disk"]+"/maize_irri/Fcover.df")
# =============================================================================
# Pour le PKGC HP
# =============================================================================
    Parcellaire= geo.read_file(d["PC_disk_labo"]+"/DONNEES_RAW/DONNEES_MAIS_CLASSIF/Classif_Adour_2017_maïs_all.shp")    
    Parcellaire["id"]=Parcellaire.ID
    for t in ["TYP","TYN"]:
        # dfnames=pd.read_csv(d["PC_disk_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_2017.txt",sep=',', header=None)
        # dfs=pd.DataFrame(dfnames)
        # dates=pd.to_datetime(dfnames[0],format="%Y%m%d")
        dfnames=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_TARN/list_FCOVER_2017_"+t+".txt",sep=',', header=None)
        dates=dfnames[0].apply(lambda x:x[11:19])
        dates=pd.to_datetime(dates,format="%Y%m%d")
        # df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref//PARCELLE_PKGC/HP/NDVI_2017_PKGC_HP_"+t+".csv",decimal=".")
        # df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_PKGC/HP/FCOVER_2017_PKGC_HP_"+t+".csv",decimal=".")
        # df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref//PARCELLE_CLASSIF/NDVI_"+t+"_Classif_all_maize_2017.csv",decimal=".")

        df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/PARCELLE_CLASSIF/FCOVER_"+t+"_CLASSIF_ADOUR_2017_MAIS_all.csv",decimal=".")

        
        tmp=df[["ID"]]
        tmp1=pd.DataFrame()
        if t =="TYP":
            for i in np.arange(0,42,2): #♣ 2018 : 49 :  2017 : 42
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a)
        else:
            for i in np.arange(0,46,2): #♣ 2018 : 49 :  2017 : 46
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a)
        Fcover=tmp1.T
        Fcover.columns=list(dates)
        # Fcover=Fcover.T
        Fcover.T.sort_index(inplace=True)
        Fcover.T.sort_index(ascending=True,inplace=True)
        Fcover=Fcover.T.reindex(pd.date_range(start="2017-01-01",end="2017-12-31",freq='1D'))
        Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
        Fcover=Fcover.append(df.ID)
        Fcover=Fcover.T
        Fcover.set_index("ID",inplace=True)
        FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
        if t =='TYP':
            FCOVERTYN=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
        else:
            FCOVERTYP=FCOVER.rename(columns={'ID':'id', 'level_1':'date',0: 'FCov'})
    #  Si NDVI supprimer les valuer dans TYN soit keep =fisrt
    FCOVER=pd.concat([FCOVERTYN,FCOVERTYP])
    FCOVER_Gers=FCOVER.drop_duplicates(subset=["id","date"],keep='first')
    # FCOVER_Gers.to_pickle(d["path_run_disk"]+"/maize_irri/Fcover.df")
    
    #  POUR LE NDVI Classif
    for t in ["TYP","TYN"]:
        dfnames=pd.read_csv(d["PC_disk_labo"]+"TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T31TCJ_interpolation_dates_2017.txt",sep=',', header=None)
        dfs=pd.DataFrame(dfnames)
        dates=pd.to_datetime(dfnames[0],format="%Y%m%d")
        df=pd.read_csv(d["PC_disk_labo"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Parcelle_ref//PARCELLE_CLASSIF/NDVI_"+t+"_Classif_all_maize_2017.csv",decimal=".")

        tmp=df[["ID"]]
        tmp1=pd.DataFrame()
        if t =="TYP":
            for i in np.arange(0,36,1): #♣ 2018 : 49 :  2017 : 42
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a/1000)
        else:
            for i in np.arange(0,36,1): #♣ 2018 : 49 :  2017 : 46
                a=df["mean_"+str(i)]
                tmp1=tmp1.append(a/1000)
        NDVI=tmp1.T
        NDVI.columns=list(dates)
        # Fcover=Fcover.T
        NDVI.T.sort_index(inplace=True)
        NDVI.T.sort_index(ascending=True,inplace=True)
        NDVI=NDVI.T.reindex(pd.date_range(start="2017-01-01",end="2017-12-31",freq='1D'))
        NDVI=NDVI.resample("D").interpolate(method='time',limit_direction='both')
        NDVI=NDVI.append(df.ID)
        NDVI=NDVI.T
        NDVI.set_index("ID",inplace=True)
        NDVI=pd.DataFrame(NDVI.T.unstack()).reset_index()
        if t =='TYP':
            NDVITYN=NDVI.rename(columns={'ID':'id', 'level_1':'date',0: 'NDVI'})
        else:
            NDVITYP=NDVI.rename(columns={'ID':'id', 'level_1':'date',0: 'NDVI'})
    #  Si NDVI supprimer les valuer dans TYN soit keep =fisrt
    NDVI=pd.concat([NDVITYN,NDVITYP])
    # NDVI_Gers=NDVI.drop_duplicates(subset=["id","date"],keep='first')
    NDVI_Gers=NDVI_Gers[NDVI_Gers.id.isin(FCOVER_Gers.id)]
    NDVI_Gers.to_pickle(d["path_run_disk"]+"/maize_irri/NDVI2017.df")
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
