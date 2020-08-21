# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:10:18 2020

@author: yanou
"""


import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
#from scipy.optimize import minimize
#from sklearn.metrics import *
import seaborn as sns
import pickle
#from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
#from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    
    # recupérer les résultats 
    d={}
    name_run="RUNS_optim_LUT_LAM_ETR/Run_with_optim_params_init_RU_0_irr_man_Ze_125/"
    d['Output_model_PC_home']='D:/THESE_TMP/RUNS_SAMIR/'
    d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/'
    # for y in ["2006","2008","2010","2012"]:
    #     print (y)
    #     res=pickle.load(open( d['Output_model_PC_labo']+name_run+str(y)+"/output_T1.df",'rb'))
    #     meteo=pickle.load(open( d['Output_model_PC_labo']+name_run+str(y)+"/Inputdata/meteo.df",'rb'))
    #     meteo_periode=meteo.loc[(meteo.date >= res.date[0])&(meteo.date <= res.date.iloc[-1] )]
 
    #     entre=pd.merge(res[["date",'Ir_auto']],meteo_periode[["Prec","date"]],on=["date"])
    #     entre.fillna(0,inplace=True)
    #     entre["sum"]=entre.eval("Ir_auto + Prec")
    #     sortie=res[['date','SWC1', 'SWC2', 'SWC3', 'SWCvol1', 'SWCvol2', 'SWCvol3', 'Dd','ET']]
    #     sortie.fillna(0,inplace=True)
    #     sortie["sum"]=sortie.eval("ET+Dd")
    #     # verifier la sortie Dd
    #     plt.plot(sortie.Dd,label=(y))
    #     plt.legend()
    #     sortie['reserve']=sortie.eval("SWC1+SWC2+SWC3")
    #     print(sum(sortie["reserve"]))
    #     print(sum(sortie["sum"]))
        
    #     cloture=(entre["sum"]-sortie['sum']+sortie["reserve"])
    #     print("bilan hydrique : %s" %sum(cloture))
        
    #     # regader fermeture uniquement sur la préiode de végétation pour supprimer les artéfacts en début de saison
    #     entre_vege=entre.loc[(entre.date >= str(y)+"-03-02") &(entre.date <= str(y)+"-08-31")]
    #     sortie_vege=sortie.loc[(sortie.date >= str(y)+"-03-02") & (sortie.date <= str(y)+"-08-31")]
    #     cloture_vege=(entre_vege["sum"]-sortie_vege['sum']+sortie_vege["reserve"])
    #     print("bilan hydrique : %s" %sum(cloture_vege))
    #     plt.title("période de végétation")
    #     plt.plot(sortie_vege.Dd,label=(y))
    #     plt.legend()


# =============================================================================
# Version excel bilan hydrique 
# =============================================================================
    for y in ["2006","2008","2010","2012","2014","2015"]:
        print (y)
        res=pickle.load(open( d['Output_model_PC_labo']+name_run+str(y)+"/output_T1.df",'rb'))
        res.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/Bilan_hydrique/simu_opti_%s_init_0_ze_125.csv"%y)
        res_vege=res.loc[(res.date >= str(y)+"-05-01") &(res.date <= str(y)+"-10-31")]
        meteo=pickle.load(open( d['Output_model_PC_labo']+name_run+str(y)+"/Inputdata/meteo.df",'rb'))
        meteo_vege=meteo.loc[(meteo.date >= str(y)+"-05-01") &(meteo.date <= str(y)+"-10-31")]
        stop_init=res_vege.TAW.iloc[0]+res_vege.Dr.iloc[0]+res_vege.TDW.iloc[0]-res_vege.Dd.iloc[0]
        stop_fin=res_vege.TAW.iloc[-1]-res_vege.Dr.iloc[-1]+res_vege.TDW.iloc[-1]-res_vege.Dd.iloc[-1]
        if "irr_man" in name_run:
            bilan=stop_init+sum(meteo_vege.Prec)+sum(meteo_vege.Irrig)-sum(res_vege.ET)
        else:
            bilan=stop_init+sum(meteo_vege.Prec)+sum(res_vege.Ir_auto)-sum(res_vege.ET)
        print('bilan hydrique: %s' %round(bilan,2))
        print("bilan fin de simu :%s" %round(stop_fin,2))