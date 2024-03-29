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
import pickle

if __name__ == '__main__':
#     Parcellaire=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/Parcelle_labo/PARCELLE_CESBIO_L93.shp")
#     years='2017'
#     list_bd_drop=['ogc_fid', 'centroidx', 'centroidy', 'shape_leng', 'shape_area', 'id']
    
#     if years =="2017":
#         dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt",sep=',', header=None) 
#     else:
#         dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_TYN2018.txt",sep=',', header=None) 
#     df1=dfnames.T
#     df1.columns=["band_name"]
#     colnames=list(df1.band_name.apply(lambda s: s[2:-1]))
    
#     sql=sqlite3.connect('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SAR_parcelle/SampleExtractionVH_VV0112_ASC_'+str(years)+'.tif.sqlite')
#     df=pd.read_sql_query("SELECT * FROM output", sql)
#     df2017=df.groupby("originfid").mean()
#     lab=df2017["id"]
#     df2017.drop(columns=list_bd_drop,inplace=True)
#     df2017=df2017.T
#     if years == '2017':
#         df2017["band_names"]=colnames[146:183]
#     else:
#         df2017["band_names"]=colnames[148:185]
#     df2017["date"] = df2017.band_names.apply(lambda s: s[-8:])
#     df2017.set_index("date",inplace=True)
#     df2017.index=pd.to_datetime(df2017.index,format="%Y%m%d")
#     df2017.columns=['LAM','AUR','band']

#     # recupere la valeurs min et la ligne 
#     datemin_VH_VV=df2017.iloc[np.where(df2017.LAM[0:20]==df2017.LAM[0:20].min())]
#     if years == '2017':
#         meteo=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/DATA_SAFRAN_2017_EMPIRSE_ALL.shp")
#     else:
#         meteo=geo.read_file("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/DONNEES_METEO/SAFRAN_2018_EMPRISE_L93.shp")
#     meteo.drop(columns=['field_1', 'Unnamed_ 0', 'LAMBX', 'LAMBY', 'PRENEI_Q',
#         'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q', 'PE_Q', 'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE',
#        'RESR_NEI_1', 'HTEURNEIGE', 'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_',
#        'ECOULEMENT', 'WG_RACINE_', 'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q',
#        'lambX_1', 'lambY_1'],inplace=True)
#     dfmeteo=meteo.buffer(4000).envelope # Création d'un buffer carée de rayon 4 km
#     meteo.geometry=dfmeteo
#     # meteo.set_index("DATE",inplace=True)
#     meteo.DATE=pd.to_datetime(meteo.DATE,format='%Y%m%d')   
    
#     Parcel_LABO=geo.overlay(Parcellaire,meteo,how='intersection')
#     Parcel_LABO.set_index('DATE',inplace=True)
#     min_signal_labo=Parcel_LABO.iloc[np.where(Parcel_LABO.index.values==datemin_VH_VV.index.values)]
    
#     #  Multipilier la température par 500 pour otbeir
#     # min_signal_labo.iloc[1].T_Q*500 #=3150
#     # Cumsum_T=Parcel_LABO.T_Q.cumsum()
#     # res=Parcel_LABO.loc[np.where((Cumsum_T>=min_signal_labo.iloc[0].T_Q*500) & (Cumsum_T<=min_signal_labo.iloc[0].T_Q*500))]
    



# #  test
#     LAM=Parcel_LABO[Parcel_LABO.NOM_PARCEL=="Lamothe"]
#     # LAM.set_index("DATE",inplace=True)
#     LAM.sort_index(ascending=True,inplace=True)
#     Tcum_stat=LAM.loc[LAM.index>=min_signal_labo.index[0]].T_Q.cumsum()
#     # Tcum=LAM.T_Q.cumsum()
#     res=Tcum_stat.where(Tcum_stat >min_signal_labo.iloc[0].T_Q+500)
#     res=res.dropna()
    
#     min_signal_labo.index-res.index[0]
    
# =============================================================================
#     Parcelle PKGC
# =============================================================================
    years='2017'
    Prec=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_GSM_irri_auto/2017/Inputdata/maize_irri/meteo.df","rb"))
    Prec=Prec.rename(columns={'id':'ID', 'date':'date','ET0': 'ET0','Prec':'Prec','Irrig':'Irrig'})
    temp=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PKGC_temp.csv")
    temp=temp[["ID","DATE","T_Q","IRRDATDEB"]]
    temp["date"]=pd.to_datetime(temp["DATE"],format="%Y-%m-%d")
    PKGC=geo.read_file("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/INPUT_DATA/SAR_parcelle/PKGC/VH_VV_PKGC_32.shp")
    # add NDVI  
    NDVI=pickle.load(open("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_GSM_irri_auto/2017/Inputdata/maize_irri/NDVI2017.df",'rb'))
    PKGC.drop_duplicates(inplace=True,subset=["ID"])
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    PKGC=PKGC[~PKGC["ID"].isin(list_drop)]
    NDVI=NDVI[~NDVI["id"].isin(list_drop)]
    NDVI=NDVI.rename(columns={'id':'ID', 'date':'date','NDVI': 'NDVI'})
    dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt",sep=',', header=None)
    df1=dfnames.T
    df1.columns=["band_name"]
    colnames=list(df1.band_name.apply(lambda s: s[-9:-1]))
    dates=colnames[146:183]
    dates=pd.to_datetime(dates,format="%Y%m%d")
    tmp=PKGC[["ID"]]
    tmp1=pd.DataFrame()
    for i in np.arange(0,37,1): #♣ 2018 : 49 :  2017 : 41
        a=PKGC["mean_"+str(i)]
        tmp1=tmp1.append(a)
    Fcover=tmp1.T
    Fcover.columns=list(dates)
    # Fcover=Fcover.T
    Fcover.T.sort_index(inplace=True)
    Fcover.T.sort_index(ascending=True,inplace=True)
    # Fcover=Fcover.T
    Fcover=Fcover.T.reindex(pd.date_range(start=str(years)+"-01-01",end=str(years)+"-12-31",freq='1D'))
    Fcover=Fcover.resample("D").interpolate(method='time',limit_direction='both')
    Fcover=Fcover.append(PKGC.ID)
    Fcover=Fcover.T
    Fcover.set_index("ID",inplace=True)
    FCOVER=pd.DataFrame(Fcover.T.unstack()).reset_index()
    data_SAR=FCOVER.rename(columns={'ID':'ID', 'level_1':'date',0: 'VV_VH'})
    data_SAR=data_SAR.loc[(data_SAR.date>='2017-02-01')&(data_SAR.date<='2017-10-01')]
    data_SAR=data_SAR.merge(Prec,on=["ID","date"])
    DATA_all=pd.merge(data_SAR,temp,on=["ID","date"])
    DATA_all=pd.merge(DATA_all,NDVI,on=["ID",'date'])
    
    DATA_all['T_cum']=DATA_all.groupby("ID").cumsum()["T_Q"]
    test=[]
    
    for i in list(set(DATA_all.ID)):
        plt.figure(figsize=(7,7))
        id1=DATA_all[DATA_all.ID==i]
        datemin_VV_VH=id1[id1.VV_VH==id1.VV_VH.min()]["date"]
        temp_date=id1[id1.date==datemin_VV_VH.values[0]]["T_cum"]
        a=plt.plot(id1.date,id1.VV_VH,label="VV/VH", c='black')
        plt.ylim(0,0.5)
        plt.ylabel("VV/VH")
        plt.axvline(datemin_VV_VH.iloc[0])
        ax2 = plt.twinx()
        b=ax2.plot(id1.date,id1.NDVI,label='NDVI',c='black',linestyle="--")
        ax2.set_ylim(0,1)
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        plt.ylabel("NDVI")
        plt.text(datemin_VV_VH,0.20,s=datemin_VV_VH.astype(str).values[0])
        plt.legend([a[0],b[0]],["VV/VH",'NVDI'])
        # plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/Growth_data/plot_sar_NDVI_date_semis%s.png"%i)
        test.append([i,datemin_VV_VH.astype(str).values[0],temp_date.values[0]])
        
        
    
    df=pd.DataFrame(test,columns=['ID','date','Temp_cum'])
    df["date"]=pd.to_datetime(df["date"],format="%Y-%m-%d")
    df["Temp_lever"]=df.eval("Temp_cum+500")
    
    # Récupere date VV_VH min et ajouter + 500 °C et localiser date 
    date_leve=[]
    for iden,t in zip(df.ID,df.Temp_lever):
        a=DATA_all.loc[(DATA_all.ID==iden) & (DATA_all.T_cum>=t)].iloc[0]["date"]
        date_leve.append(a)
    df["date_leve"]=date_leve
    
    for i in list(set(DATA_all.ID)):
        plt.figure(figsize=(7,7))
        id1=DATA_all[DATA_all.ID==i]
        datemin_VV_VH=id1[id1.VV_VH==id1.VV_VH.min()]["date"]
        temp_date=id1[id1.date==datemin_VV_VH.values[0]]["T_cum"]
        a=plt.plot(id1.date,id1.VV_VH, c='black')
        plt.ylim(0,0.5)
        plt.ylabel("VV/VH")
        d=plt.axvline(datemin_VV_VH.iloc[0],label="Date de semis")
        c=plt.axvline(df[df.ID==i]["date_leve"].values,linestyle='--',color='red',label='Date de levée')
        plt.legend(loc='upper left')
        ax2 = plt.twinx()
        b=ax2.plot(id1.date,id1.NDVI,c='black',linestyle="--")
        ax2.set_ylim(0,1)
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        plt.ylabel("NDVI")
        plt.text(datemin_VV_VH,0.5,s=datemin_VV_VH.astype(str).values[0],color='blue')
        plt.text(df[df.ID==i]["date_leve"],0.8,s=df[df.ID==i]["date_leve"].iloc[0],color='red')
        plt.legend([a[0],b[0]],["VV/VH",'NVDI'])
        
        plt.savefig("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/Plot_resu_all_parcelle_partenaires/Growth_data/plot_sar_NDVI_date_semis%s.png"%i)

        