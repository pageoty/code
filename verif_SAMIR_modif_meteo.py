#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:47:26 2020

@author: pageot
"""

import pickle
import os
import pandas as pd
from RUN_SAMIR_nn_opti import *

if __name__ == '__main__':
    name_run_verif="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/run_verif_meteo/2019/"
    name_run_test='RUNS_SAMIR/RUN_MULTI_SITE_ICOS/run_test_meteo/2019/'
    y='2019'
    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/R12/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
# =============================================================================
# Evaluer la modification 
# =============================================================================
    df_test=pickle.load(open(d["path_labo"]+name_run_test+"/Output/output_valid_modif_meteo_itenti.df",'rb'))
    meteo=pickle.load(open("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/run_verif_meteo/2019/Inputdata/meteo.df",'rb'))
    meteo_vege=meteo.loc[(meteo.date >= str(y)+"-05-01") &(meteo.date <= str(y)+"-10-31")]
    df_test_lc=df_test.groupby('LC')
    for lc in list(set(df_test.LC)):
        if "rain" in lc:
            df_crop_test_rain=df_test_lc.get_group(lc)
            df_crop_test_rain=df_crop_test_rain.loc[(df_crop_test_rain.date >= str(y)+"-05-01") &(df_crop_test_rain.date <= str(y)+"-10-31")]
            df_crop_test_rain.date=pd.to_datetime(df_crop_test_rain.date,format="%Y-%m-%d")
        else:
            df_crop_test_irr=df_test_lc.get_group(lc)
            df_crop_test_irr=df_crop_test_irr.loc[(df_crop_test_irr.date >= str(y)+"-05-01") &(df_crop_test_irr.date <= str(y)+"-10-31")]
            df_crop_test_irr.date=pd.to_datetime(df_crop_test_irr.date,format="%Y-%m-%d")
    df_verif=pickle.load(open(d["path_labo"]+name_run_verif+"/Output/output_valid_modif.df",'rb'))
    df_verif_lc=df_verif.groupby('LC')
    for lc in list(set(df_verif.LC)):
        if "rain" in lc:
            df_crop_verif_rain=df_verif_lc.get_group(lc)
            df_crop_verif_rain=df_crop_verif_rain.loc[(df_crop_verif_rain.date >= str(y)+"-05-01") &(df_crop_verif_rain.date <= str(y)+"-10-31")]
            df_crop_verif_rain.date=pd.to_datetime(df_crop_verif_rain.date,format="%Y-%m-%d")
            df_crop_verif_rain.drop(columns=['FCstd',"WPstd"],inplace=True)
        else:
            df_crop_verif_irr=df_verif_lc.get_group(lc)
            df_crop_verif_irr=df_crop_verif_irr.loc[(df_crop_verif_irr.date >= str(y)+"-05-01") &(df_crop_verif_irr.date <= str(y)+"-10-31")]
            df_crop_verif_irr.date=pd.to_datetime(df_crop_verif_irr.date,format="%Y-%m-%d")
            df_crop_verif_irr.drop(columns=['FCstd',"WPstd"],inplace=True)
# =============================================================================
#   plot phase 
# =============================================================================
    print('************IRR*************')
    for var in ['DP', 'Dd', 'Dei', 'Dep', 'Dr', 'ET','Ir_auto', 'Kcb', 'Ks','RAW', 'RUE', 'SWC1', 'SWC2', 'SWC3', 'SWCvol1', 'SWCvol2', 'SWCvol3','TAW', 'TDW', 'TEW', 'Zd', 'Zr', 'fewi', 'fewp']:
        plt.figure(figsize=(10,7))
        plt.plot(df_crop_test_irr.date,df_crop_test_irr[str(var)],label='test')
        # plt.plot(meteo_vege.date,meteo_vege.ET0)
        plt.plot(df_crop_verif_irr.date,df_crop_verif_irr[str(var)],label='verif')
        plt.title(str(var))
        plt.legend()
        plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/run_test_meteo/2019/Output/Plot_verif/Crop_irr/%s.png"%str(var))
        plt.figure(figsize=(10,7))
        plt.scatter(df_crop_test_irr[str(var)],df_crop_verif_irr[str(var)])
        plt.xlabel("%s test"%str(var))
        plt.ylabel("%s verif"%str(var))
        val=mean_squared_error(df_crop_test_irr[str(var)],df_crop_verif_irr[str(var)])
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_crop_test_irr[str(var)],df_crop_verif_irr[str(var)])
        bias=1/df_crop_verif_irr[str(var)].shape[0]*sum(np.mean(df_crop_test_irr[str(var)])-df_crop_verif_irr[str(var)]) 
        print("==================")
        print (str(var))
        print('rmse : %s | bias : %s | r_value :%s '%(val,bias,r_value))
        print("==================")
        
    print("***************rain *****************")
    for var in ['DP', 'Dd', 'Dei', 'Dep', 'Dr', 'ET','Ir_auto', 'Kcb', 'Ks','RAW', 'RUE', 'SWC1', 'SWC2', 'SWC3', 'SWCvol1', 'SWCvol2', 'SWCvol3','TAW', 'TDW', 'TEW', 'WP', 'Zd', 'Zr', 'fewi', 'fewp']:
        plt.figure(figsize=(10,7))
        plt.plot(df_crop_test_rain.date,df_crop_test_rain[str(var)],label='test')
        plt.plot(df_crop_verif_rain.date,df_crop_verif_rain[str(var)],label='verif')
        plt.title(str(var))
        plt.legend()
        plt.savefig("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/RUN_MULTI_SITE_ICOS/run_test_meteo/2019/Output/Plot_verif/Crop_rain/%s.png"%str(var))
        plt.figure(figsize=(10,7))
        plt.scatter(df_crop_test_rain[str(var)],df_crop_verif_rain[str(var)])
        plt.xlabel("%s test"%str(var))
        plt.ylabel("%s verif"%str(var))
        val=mean_squared_error(df_crop_test_rain[str(var)],df_crop_verif_rain[str(var)])
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_crop_test_rain[str(var)],df_crop_verif_rain[str(var)])
        bias=1/df_crop_verif_rain[str(var)].shape[0]*sum(np.mean(df_crop_test_rain[str(var)])-df_crop_verif_rain[str(var)]) 
        print("==================")
        print (str(var))
        print('rmse : %s | bias : %s | r_value :%s '%(val,bias,r_value))
        print("==================")
