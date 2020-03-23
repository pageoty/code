# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:59:32 2020

@author: Yann Pageot
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



if __name__ == '__main__':
    """ Relation NDVI-KC issu de Sentinel 2 et SWC Lamothe """
    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/R1/Inputdata/"
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    
#Data SWC lamothe
    df=pd.read_csv(d["PC_disk"]+"PARCELLE_LABO/FLUX_SWC/ICOS-FR-LAM_SWC_N2-UTCplus1_2016-nov2018_rep.csv")
    df["Date/Time"]=df["Date/Time"].apply(lambda x:x[0:10])
    df["Date/Time"]=pd.to_datetime(df["Date/Time"],format="%d/%m/%Y")
    SWC2017=df.loc[(df["Date/Time"] > "2016-12-31") & (df["Date/Time"] < "2017-12-31")]
    #   Selection 75 %  de la RU 
    RU75= (0.363-0.17)-((0.363-0.17)*75/100)
# Data NDVI sur mais Irr soir 2017
    NDVI2017=pd.read_csv(d["PC_disk"]+"TRAITEMENT/NDVI_parcelle/Parcelle_ref/PARCELLE_CESBIO/LAMOTHE_NDVI_2017.csv",header=None)
    NDVI2017[0]=pd.to_datetime(NDVI2017[0],format="%Y-%m-%d")
#    Rainfall lamothe
    ET0=pd.read_csv(d["PC_disk"]+"Calibration_SAMIR/DONNEES_CALIBRATION/Meteo_lam_2017.csv")
    ET0["date"]=pd.to_datetime(ET0["date"],format="%Y-%m-%d")
    ET0_Saf=ET0.sort_values(by="date",ascending=True)
# Plot 
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(411)
    plt.plot(NDVI2017[0],NDVI2017[1])
    plt.ylabel("NDVI")
    ax2 = plt.subplot(412)
    plt.plot(SWC2017["Date/Time"],SWC2017.SWC_0_moy/100)
    plt.plot(SWC2017["Date/Time"],np.repeat(0.172,len(SWC2017["Date/Time"])),c="r",label="WP")
    plt.plot(SWC2017["Date/Time"],np.repeat(0.363,len(SWC2017["Date/Time"])),c="b", label="FC")
    plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label='RU 75 %')
    plt.ylabel('SWC en surface')
    ax2 = plt.subplot(413)
    plt.plot(SWC2017["Date/Time"],SWC2017.SWC_5_moy/100)
    plt.plot(SWC2017["Date/Time"],np.repeat(0.172,len(SWC2017["Date/Time"])),c="r",label="WP")
    plt.plot(SWC2017["Date/Time"],np.repeat(0.363,len(SWC2017["Date/Time"])),c="b",label="FC")
    plt.plot(SWC2017["Date/Time"],np.repeat(0.363-RU75,len(SWC2017["Date/Time"])),c="b",linestyle='--',label="RU 75%")
    plt.legend()
    plt.ylabel('SWC en profondeur')
    ax34=plt.subplot(414)
    plt.plot(ET0_Saf.date,ET0_Saf.ET0)
    plt.ylabel("ET0 Safran")
#    plt.savefig(d["PC_disk"]+"RESULT/plt_dyna_relat.png")
    


    
#    ETR 2019 ce fichier eddypro_FR-Lam_biomet_2020-01-28T012345_adv
    