#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:21:12 2019

@author: pageot
"""

import os
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
from STAT_ZONAL_SPECRTE import *
from scipy import stats
from TEST_ANALYSE_SIGNATURE import *
from Cumul_spectre_main import *

if __name__ == '__main__':
#    Travail sur les deux zones NORD et SUD
#    colnames=[]
#    for y in ['2018','2017']:
#        globals()["dfallSUD%s"% (y)]=pd.DataFrame()
#        globals()["dfallNORD%s"% (y)]=pd.DataFrame()
#        for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/"):
#            if "NORD" in i and y[-2:] in i :
#                print (r" ici: %s" %i)
#                sql=sqlite3.connect("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/"+i)
#                df=pd.read_sql_query("SELECT * FROM output", sql)
#                df=df.groupby("originfid").mean()
#                labnord=df.labcroirr
#                pluiNORD=df.value_0
#                colnames.append(i[27:31])
#                globals()["dfallNORD%s"% y]=globals()["dfallNORD%s"% y].append(pluiNORD)
#            elif "SUD" in i and y[-2:] in i:
#                print (i)
#                sql=sqlite3.connect("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/"+i)
#                df=pd.read_sql_query("SELECT * FROM output", sql)
#                df=df.groupby("originfid").mean()
#                labsud=df.labcroirr
#                pluiSUD=df.value_0
#                colnames.append(i[27:31])
#                globals()["dfallSUD%s"% y]=globals()["dfallSUD%s"% y].append(pluiSUD)
#                    
#        for z in ["NORD","SUD"]:
#            globals()["dfall%s%s"% (z,y)]=globals()["dfall%s%s"% (z,y)].T
#            globals()["dfall%s%s"% (z,y)].columns=set(colnames)
#            if z =="NORD":
#                globals()["dfall%s%s"% (z,y)]["lab"]=labnord.astype(int)
#            else:
#                 globals()["dfall%s%s"% (z,y)]["lab"]=labsud.astype(int)
#            globals()["dfall%s%s"% (z,y)]=globals()["dfall%s%s"% (z,y)].reindex(columns = ['Apri', 'May1', 'June','July','Augu',"Sept","Octo",'lab'])
#            globals()["cropslab1%s%s"%(z,y)]=globals()["dfall%s%s"% (z,y)][globals()["dfall%s%s"% (z,y)].lab==1]
#            
#        globals()["cum_SAFRAN1NORD%s"%y]=globals()["cropslab1NORD%s"%(y)].iloc[:,:-1].T.cumsum()
#        globals()["cum_SAFRAN1SUD%s"%y]=globals()["cropslab1SUD%s"%(y)].iloc[:,:-1].T.cumsum()    
##        Calcule des intervalles de confiances 
#        globals()["_NORD%s"%y],globals()["b_NORD%s"%y]=stats.t.interval(0.95, globals()["cum_SAFRAN1NORD%s"%y].shape[1]-1,loc= globals()["cum_SAFRAN1NORD%s"%y].T,scale=stats.sem( globals()["cum_SAFRAN1NORD%s"%y].T))
#        globals() ["_SUD%s"%y],globals()["b_SUD%s"%y]=stats.t.interval(0.95,globals()["cum_SAFRAN1SUD%s"%y].shape[1]-1,loc=globals()["cum_SAFRAN1SUD%s"%y].T,scale=stats.sem(globals()["cum_SAFRAN1SUD%s"%y].T))
#    
#        globals()["_NORD%s"%y]=pd.DataFrame(globals()["_NORD%s"%y])
#        globals()["b_NORD%s"%y]=pd.DataFrame(globals()["b_NORD%s"%y])
#        globals()["_SUD%s"%y]=pd.DataFrame(globals() ["_SUD%s"%y])
#        globals()["b_SUD%s"%y]=pd.DataFrame(globals()["b_SUD%s"%y])
#    
#    fig, ax = plt.subplots(figsize=(10, 7))
#    ax1=plt.subplot(221)
#    sns.set(style="darkgrid")
#    sns.set_context('paper')
#    plt.title("Watershed : Adour")
#    p1=plt.plot(cum_SAFRAN1NORD2018.T.mean(),color='blue')
#    p2=plt.plot(cum_SAFRAN1SUD2018.T.mean(),color="red")
#    plt.fill_between(cum_SAFRAN1NORD2018.index, b_NORD2018.mean(), _NORD2018.mean(), facecolor='blue', alpha=0.2)
#    plt.fill_between(cum_SAFRAN1SUD2018.index,  b_SUD2018.mean(), _SUD2018.mean(), facecolor='red', alpha=0.2)
#    plt.legend((p1[0],p2[0]),("North","South"))
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.ylim(0,700)
#    plt.ylabel('Cumul %s mean in 2018'%"rainfall")
#    ax1=plt.subplot(222)
#    plt.title("Watershed : Adour")
#    p1=plt.plot(cum_SAFRAN1NORD2017.T.mean(),color='blue')
#    p2=plt.plot(cum_SAFRAN1SUD2017.T.mean(),color="red")
#    plt.fill_between(cum_SAFRAN1NORD2017.index, b_NORD2017.mean(), _NORD2017.mean(), facecolor='blue', alpha=0.2)
#    plt.fill_between(cum_SAFRAN1SUD2017.index,  b_SUD2017.mean(), _SUD2017.mean(), facecolor='red', alpha=0.2)
#    plt.legend((p1[0],p2[0]),("North","South"))
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.ylim(0,700)
#    plt.ylabel('Cumul %s mean in 2017'%"rainfall")
#    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/Culum_plui_2017_2018_Adour.png")
    
    
# =============================================================================
#     BV Years 
# =============================================================================
    # for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/CUMUL_BV_YEARS/"):
    #     if "TARN"in i:
    #         bv='TARN'
    #         if "2017" in i :
    #             globals()["%s2017"%bv]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/CUMUL_BV_YEARS/"+str(i),"dfz")
    #             globals()["%s2017"%bv].drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr', 'labcroirr'],inplace=True)
    #             globals()["%s2017"%bv].columns = ["January","February","March","April","May","June","July","August","September","October","November"]
    #             globals()["%s2017_cum"%bv]=globals()["%s2017"%bv].T.cumsum()
    #         else:
    #             globals()["%s2018"%bv]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/CUMUL_BV_YEARS/"+str(i),"dfz")
    #             globals()["%s2018"%bv].drop(columns=['ogc_fid', 'idparcelle', 'millesime', 'labcroirr', 'surfha', 'class','num_ilot', 'num_parcel', 'surf_adm'],inplace=True)
    #             globals()["%s2018"%bv].columns = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    #             globals()["%s2018_cum"%bv]=globals()["%s2018"%bv].T.cumsum()
    #     else:
    #         bv="ADOUR"
    #         if "2017" in i :
    #             globals()["%s2017"%bv]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/CUMUL_BV_YEARS/"+str(i),"dfz")
    #             globals()["%s2017"%bv].drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr', 'labcroirr'],inplace=True)
    #             globals()["%s2017"%bv].columns = ["January","February","March","April","May","June","July","August","September","October","November"]
    #             globals()["%s2017_cum"%bv]=globals()["%s2017"%bv].T.cumsum()
    #         else:
    #             globals()["%s2018"%bv]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_METEO_MONTH/CUMUL_BV_YEARS/"+str(i),"dfz")
    #             globals()["%s2018"%bv].drop(columns=['ogc_fid', 'idparcelle', 'millesime', 'labcroirr', 'surfha', 'class','num_ilot', 'num_parcel', 'surf_adm'],inplace=True)
    #             globals()["%s2018"%bv].columns = ["January","February","March","April","May","June","July","August","September","October","November","December"]
    #             globals()["%s2018_cum"%bv]=globals()["%s2018"%bv].T.cumsum()

    meteo2017=pd.read_csv("F:/THESE/CLASSIFICATION/TRAITEMENT/DATA_METEO/SAFRAN_ADOUR_ZONE_CENTRALE_NORD_17_18.csv")
    meteo2017["DATE"]=pd.to_datetime(meteo2017["DATE"],format="%Y%m%d")

    meteo2017.drop(columns=['field_1', 'LAMBX', 'LAMBY', 'PRENEI_Q', 'T_Q',
       'FF_Q', 'Q_Q', 'DLI_Q', 'SSI_Q', 'HU_Q', 'EVAP_Q', 'ETP_Q', 'PE_Q',
       'SWI_Q', 'DRAINC_Q', 'RUNC_Q', 'RESR_NEIGE', 'RESR_NEI_1', 'HTEURNEIGE',
       'HTEURNEI_1', 'HTEURNEI_2', 'SNOW_FRAC_', 'ECOULEMENT', 'WG_RACINE_',
       'WGI_RACINE', 'TINF_H_Q', 'TSUP_H_Q', 'X', 'Y'],inplace=True)
    ADOUR2018=meteo2017.loc[(meteo2017["DATE"] >= "2018-05-01") & (meteo2017["DATE"] <= "2018-09-30")]
    ADOUR2017=meteo2017.loc[(meteo2017["DATE"] >= "2017-05-01") & (meteo2017["DATE"] <= "2017-09-30")]
    AD17=ADOUR2017.groupby("DATE").mean()
    AD18=ADOUR2018.groupby("DATE").mean()
    
    ADOUR17 = pd.DataFrame()
    ADOUR17['Prec_2017'] = AD17.PRELIQ_Q.resample('W').sum()
    ADOUR17["cum_2017"]=ADOUR17.Prec_2017.cumsum()
    ADOUR18=pd.DataFrame()
    ADOUR18['Prec_2018'] = AD18.PRELIQ_Q.resample('W').sum()
    ADOUR18["cum_2018"]=ADOUR18.Prec_2018.cumsum()


    x=ADOUR18.index.strftime('%m-%d')
    # x_day=AD18.index.strftime("%m-%d")

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title("Watershed : Adour central area")
    plt.plot(x,ADOUR17["cum_2017"],color='red',label='2017')
    plt.plot(x,ADOUR18["cum_2018"],color="blue",label='2018')
    plt.xticks(size='large',rotation=90)
    plt.yticks(size='large')
    plt.ylim(0,600)
    plt.xlabel("month and day")
    plt.ylabel('Cumul mean rainfall')
    plt.legend()
    plt.savefig("F:/THESE/CLASSIFICATION/RESULT/PLOT/Culum_plui_2017_2018_BV_ADOUR_Centrale_Nord_may_sept.png",dpi=500,bbox_inches='tight', pad_inches=0.5)



