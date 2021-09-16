#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:31:19 2019

@author: pageot

Code permet de comparer les modes d'aquisition des données SAR, ainsi que l'angle d'incidence local (LIA)
"""

import os
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import geopandas as geo
from scipy import stats
from STAT_ZONAL_SPECRTE import *

if __name__ == "__main__":
    ASC2017=pd.DataFrame()
    DES2017=pd.DataFrame()
    ASC2018=pd.DataFrame()
    DES2018=pd.DataFrame()

    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_Evalu_mode_acqu_S1/2017/"):
        if "ASC" in i:
            print (i)
            print (i[27:-15])
            globals()['%s'% i[27:-15]]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_Evalu_mode_acqu_S1/2017/"+str(i),"dfASC_confidance2018")
            globals()['%s'% i[27:-15]].drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr'],inplace=True)
            label=globals()['%s'% i[27:-15]].labcroirr
            ASC2017=ASC2017.append(globals()['%s'% i[27:-15]].value_0)
        else:
            print (i)
            print (i[27:-15])
            globals()['%s'% i[27:-15]]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_Evalu_mode_acqu_S1/2017/"+str(i),"dfASC_confidance2018")
            globals()['%s'% i[27:-15]].drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr'],inplace=True)
            DES2017=DES2017.append(globals()['%s'% i[27:-15]].value_0)

    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_Evalu_mode_acqu_S1/2018/"):
        if "ASC" in i:
            print (i[27:-15])
            globals()['%s'% i[27:-15]]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_Evalu_mode_acqu_S1/2018/"+str(i),"dfASC_confidance2018")
            globals()['%s'% i[27:-15]].drop(columns=['ogc_fid','idparcelle', 'millesime','surfha','class','num_ilot', 'num_parcel', 'surf_adm'],inplace=True)
            label2018=globals()['%s'% i[27:-15]].labcroirr
            ASC2018=ASC2018.append(globals()['%s'% i[27:-15]].value_0)
        else:
            print (i[27:-15])
            globals()['%s'% i[27:-15]]=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_Evalu_mode_acqu_S1/2018/"+str(i),"dfASC_confidance2018")
            globals()['%s'% i[27:-15]].drop(columns=['ogc_fid','idparcelle', 'millesime','surfha','class','num_ilot', 'num_parcel', 'surf_adm'],inplace=True)
            DES2018=DES2018.append(globals()['%s'% i[27:-15]].value_0)

    
    
    
    plt.figure(figsize=(12,10))
    ax1 = plt.subplot(221)
    sns.barplot(label2018,ASC2018.mean())
    plt.title("confidance 2018 mode ASC")
    plt.ylim(0,100)
    plt.ylabel("Confidence")

    ax2 = plt.subplot(222)
    sns.barplot(label2018,DES2018.mean())
    plt.title("confidance 2018 mode DES")
    plt.ylim(0,100)

    
    ax3 = plt.subplot(223)
    sns.barplot(label,ASC2017.mean())
    plt.title("confidance 2017 mode ASC")
    plt.ylim(0,100)

    ax4 = plt.subplot(224)
    sns.barplot(label,DES2017.mean())
    plt.title("confidance 2017 mode DES")
    plt.ylim(0,100)
   
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/ANALYSE_LIA_SAR/histo_ADOUR_confidance_mode.png")


    # add les valeurs et les intervalle de confaince issu des 5 runs 
    
# =============================================================================
    #Scatter les LIA sur Adour 
# =============================================================================
    plt.figure(figsize=(12,10))
    test_zone_DES=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_LIA/SampleExtractionS1B_DES_ADOUR.tif.sqlite","dftest")
    test_zone_DES.drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr'],inplace=True)
    p1=plt.hist(test_zone_DES.value_3,label="DES")
    
    test_zone_ASC=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_LIA/SampleExtractionSIA_ASC_ADOUR.tif.sqlite","dftest")
    test_zone_ASC.drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr'],inplace=True)
    p2=plt.hist(test_zone_ASC.value_3,label="ASC")
    plt.xlabel("local incidance angle")
    plt.title("LIA Adour")
    plt.legend()
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/ANALYSE_LIA_SAR/histo_LIA_ADOUR.png")

    
    plt.figure(figsize=(12,10))
    plt.scatter(test_zone_ASC.value_3,test_zone_DES.value_3)
    plt.plot([50,25],[25,50], 'r-', lw=2)
    plt.xlim(25,50)
    plt.ylim(25,50)
    plt.xlabel("local incidance angle ASC" )
    plt.ylabel("local incidance anlge DES")
    
    plt.figure(figsize=(12,10))
    plt.hist(test_zone_DES[test_zone_DES.labcroirr==1].value_3,label='DES')
    plt.hist(test_zone_ASC[test_zone_ASC.labcroirr==1].value_3,label="ASC")
    plt.xlabel("local incidance angle")
    plt.title("LIA Adour, Maize_Irr")
    plt.legend()
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/ANALYSE_LIA_SAR/histo_LIA_ADOUR_MAIZE_IRR.png")

    
    plt.figure(figsize=(12,10))
    plt.hist(test_zone_DES[test_zone_DES.labcroirr==11].value_3,label='DES')
    plt.hist(test_zone_ASC[test_zone_ASC.labcroirr==11].value_3,label="ASC")
    plt.xlabel("local incidance angle")
    plt.title("LIA Adour, Maize_NIrr")
    plt.legend()
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/ANALYSE_LIA_SAR/histo_LIA_ADOUR_MAIZE_NIRR.png")

# =============================================================================
#     LIA TARN 
# =============================================================================
    plt.figure(figsize=(12,10))
    test_zone_ASC_TA=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_LIA/SampleExtractionLIA_S1A_ASC_TARN.tif.sqlite","dftest")
    test_zone_ASC_TA.drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr'],inplace=True)
    test_Z_ASC_T_nn_0=test_zone_ASC_TA[test_zone_ASC_TA.value_0 !=0]
    p1=plt.hist(test_Z_ASC_T_nn_0.value_0,label="ASC")

    test_zone_DES_TA=sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_LIA/SampleExtractionS1A_DES_TARN.tif.sqlite","dftest")
    test_zone_DES_TA.drop(columns=['ogc_fid', 'surf_parc', 'summer', 'id', 'gid', 'area', 'labelirr','labelcrirr'],inplace=True)
    p1=plt.hist(test_zone_DES_TA.value_0,label="DES")
    plt.xlabel("local incidance angle")
    plt.title("LIA TARN")
    plt.legend()
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/ANALYSE_LIA_SAR/histo_LIA_TARN.png")
    
    
    plt.figure(figsize=(12,10))
    plt.hist(test_zone_DES_TA[test_zone_DES_TA.labcroirr==1].value_0,label='DES')
    plt.hist(test_zone_ASC_TA[test_zone_ASC_TA.labcroirr==1].value_0,label="ASC")
    plt.xlabel("local incidance angle")
    plt.title("LIA TARN, Maize_Irr")
    plt.legend()
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/ANALYSE_LIA_SAR/histo_LIA_TARN_MAIZE_IRR.png")

    
    plt.figure(figsize=(12,10))
    plt.hist(test_zone_DES_TA[test_zone_DES_TA.labcroirr==11].value_0,label='DES')
    plt.hist(test_zone_ASC_TA[test_zone_ASC_TA.labcroirr==11].value_0,label="ASC")
    plt.xlabel("local incidance angle")
    plt.title("LIA TARN, Maize_NIrr")
    plt.legend()
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/ANALYSE_LIA_SAR/histo_LIA_TARN_MAIZE_NIRR.png")
