# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:46:02 2021

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
import numpy as np
import pandas as pd
import seaborn as sns 
import TEST_ANALYSE_SIGNATURE
import shapely.geometry as geom
import descartes
import pickle
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression

def predict(x):
   return slope * x + intercept

def select_color_NDVI(x):
    couleurs=[]
    for i in range(len(x)):
        if x.iloc[i].NDVI<= 0.5 : 
            couleurs.append("r")
        else : 
            couleurs.append("b")
    return couleurs

if __name__ == '__main__':
#  lecture fichier 
    d={}
    name_run1="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/Merlin_init_ru_optim_Fcover_maxzr_lame_30/"
    name_run2="RUNS_SAMIR/RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_OPTIMI_LAM/FAO_init_ru_optim_Fcover_REWmaxzr_lame_30/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    years=['2008','2010','2012','2014','2015']
# =============================================================================
# Validation Flux ETR ICOS non Multi_sie run
# =============================================================================
# modif pour groupby lc

    for y in years:
        for lc in ["maize_irri"]: # maize_rain
            # d['Output_model_PC_labo']='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/'+name_run+"/"+y+"/"
            # d["Output_model_PC_home"]="/mnt/d/THESE_TMP/TRAITEMENT/"+name_run+"/"+y+"/"
            d["Output_model_PC_home"]="D:/THESE_TMP/TRAITEMENT/"+name_run1+"/"+y+"/"
            d["Output_model_PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run1+"/"+y+"/"
            d["Output_model_PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run1+"/"+y+"/"
            d["Output_model_PC_home_disk_run1"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run1+"/"+y+"/"
            d["Output_model_PC_home_disk_run2"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run2+"/"+y+"/"

# =============================================================================
# Fcover sat vs Fcov mod
# =============================================================================
            Fcoversat=pd.read_csv(d["Output_model_PC_home_disk_run1"][:-5]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1])
            Fcoversat.columns=pd.to_datetime(Fcoversat.columns,format="%Y-%m-%d")
            Fcoversat=Fcoversat.loc[:,(Fcoversat.columns >= str(y)+"-04-01") &(Fcoversat.columns <= str(y)+"-09-30")]
            Fcover_mod=pd.read_csv(d["Output_model_PC_home_disk_run2"][:-5]+"/LUT_ETR"+str(y)+".csv",index_col=[0,1,2])
            Fcover_mod.columns=pd.to_datetime(Fcover_mod.columns,format="%Y-%m-%d")
            Fcover_mod=Fcover_mod.loc[:,(Fcover_mod.columns >= str(y)+"-04-01") &(Fcover_mod.columns <= str(y)+"-09-30")]
            slope, intercept, r_value, p_value, std_err = stats.linregress(Fcover_mod.mean().to_list(),Fcoversat.mean().to_list())
            bias=1/Fcover_mod.shape[0]*sum(np.mean(Fcoversat.mean())-Fcover_mod.mean()) 
            fitLine = predict(Fcover_mod.mean())
            # Creation plot
            plt.figure(figsize=(7,7))
            plt.plot([0.0, 10], [0.0,10], 'black', lw=1,linestyle='--')
            plt.plot(Fcover_mod.mean(),fitLine,linestyle="--")
            # plt.scatter(dfETR_obs.LE,dfETR_obs.ET,s=9,c=select_color_NDVI(dfETR_obs))
            plt.scatter(Fcover_mod.mean(),Fcoversat.mean(),s=10,c=Fcoversat.std())
            plt.colorbar()
            plt.xlabel("ETR OBS")
            plt.ylabel("ETR model")
            plt.xlim(0,10)
            plt.ylim(0,10)
            # plt.legend(('Soil nu','Vege'))
            plt.title("Scatter ETR obs et ETR mod %s en %s"%(lc,y))
            rms = mean_squared_error(Fcover_mod.mean(),Fcoversat.mean())
            plt.text(8,min(Fcoversat.mean())+0.1,"RMSE = "+str(round(rms,2))) 
            plt.text(8,min(Fcoversat.mean())+0.4,"R² = "+str(round(r_value,2)))
            plt.text(8,min(Fcoversat.mean())+0.7,"Pente = "+str(round(slope,2)))
            plt.text(8,min(Fcoversat.mean())+1,"Biais = "+str(round(bias,2)))
            # plt.savefig( d["Output_model_PC_home"]+"/plt_scatter_ETR_%s_%s.png"%(lc,y))