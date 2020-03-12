#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:37:53 2019

@author: pageot
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import seaborn as sns
import csv
from scipy import *
from pylab import *
from sklearn.metrics import *
import STAT_ZONAL_SPECRTE

    
def pltbox(x,y,data):
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    sns.boxplot(x=x,y=y,data=data)
    

    
    
if __name__ == "__main__":
    
    alldf=pd.DataFrame()
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_CLASSIF/R14_VV_VH/learningSamples/"):
        if "T31TCJ" not in i:
            print (i)
            STATR14 = sqlite3.connect('/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_CLASSIF/R14_VV_VH/learningSamples/'+i)
            df=pd.read_sql_query("SELECT * FROM output", STATR14)
            dfpar=df.groupby("originfid").mean()
            label=dfpar.labcroirr.astype(int)
            lab=list(set(label))
            alldf=alldf.append(dfpar,ignore_index=True)
    SAR_process_db(lab,alldf,"vv")
    
#    for j in lab:
#        globals()['cropslab%s' % j] = pd.DataFrame(alldf[alldf.labcroirr==j]).T
#        globals()['NDVI%s' % j]=[]
#        allndvi=pd.DataFrame()
#        for index,row in globals()['cropslab%s' % j].iterrows():
#            if "ndvi" in index:
#                globals()['NDVI%s' % j].append (row)
#                globals()['dfndvi%s' % j]=pd.DataFrame(globals()['NDVI%s' % j]) 
#    alldf30=pd.DataFrame()
 
#    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_CLASSIF/R15_GAP30j/learningSamples"):
#        if "T31TCJ" not in i:
#            print (i)
#            STATR14 = sqlite3.connect('/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_CLASSIF/R15_GAP30j/learningSamples/'+i)
#            df=pd.read_sql_query("SELECT * FROM output", STATR14)
#            dfpar=df.groupby("originfid").mean()
#            alldf30=alldf30.append(dfpar,ignore_index=True)
#    for j in lab:
#        globals()['cropslab%s' % j] = pd.DataFrame(alldf30[alldf30.labcroirr==j]).T
#        globals()['NDVI30%s' % j]=[]
#        for index,row in globals()['cropslab%s' % j].iterrows():
#            if "ndvi" in index:
#                globals()['NDVI30%s' % j].append (row)
#                globals()['dfndvi30%s' % j]=pd.DataFrame(globals()['NDVI30%s' % j])    
       
    
    
#    parcellendvi1=dfndvi1.columns
#    parcellendvi301=dfndvi301.columns
#    idenpar=list(set(parcellendvi1).intersection(set(parcellendvi301)))
#    idemdate=list(set(dfndvi1.index).intersection(set(dfndvi301.index)))
    
           
    #    for i in lab[:-1]:
    #        idenpar=list(set(globals()['dfndvi%s' % i].columns).intersection(set(globals()['dfndvi30%s' % i].columns)))
    #        plt.figure(figsize=(10,10))
    #        plt.scatter((globals()['dfndvi%s' % i]/1000).loc[idemdate][idenpar].sum(),(globals()['dfndvi30%s' % i]/1000)[idenpar].sum())
    #        plt.plot([0,8],[0,8], 'r-', lw=2)
    #        plt.title("Code culture_" +str(i))
    #        plt.xlabel("somme des NDVI gap 10j")
    #        plt.ylabel("some des NDVI gap 30j")
    #        rms = sqrt(mean_squared_error((globals()['dfndvi%s' % i]/1000).loc[idemdate][idenpar].sum(),(globals()['dfndvi30%s' % i]/1000)[idenpar].sum()))
    #        plt.text(0.5,8,"RMSE = "+str(round(rms,2)))
    #        R2=r2_score((globals()['dfndvi%s' % i]/1000).loc[idemdate][idenpar].sum(),(globals()['dfndvi30%s' % i]/1000)[idenpar].sum())
    #        plt.text(0.5,7.5,"R² = "+str(round(R2,2)))
    #        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/SCATTER_NDVI_GAP/SCATTER_%s.png"%(i))
    #    
    #    plt.boxplot(dfndvi1.sum()/1000,)
    #    plt.ylim(5,25)
    #    ax2=plt.twinx(ax=None)
    #    ax2.boxplot(dfndvi11.sum()/1000)
    #    plt.ylim(5,25)
    #    plt.show()
    
    # =============================================================================
    # boxplot cumul     
    # =============================================================================
    
    Alldf=dbdfvv1
    allndvi=pd.DataFrame()
    allndwi=pd.DataFrame()
    for index,row in Alldf.iterrows():
           if "asc_vv" in index:
               allndvi=allndvi.append(row)
           if "des_vv" in index:
               allndwi = allndwi.append(row)
    sumallndwi=allndwi.sum()           
    sumallndvi1=allndvi.iloc[7:13].sum() 
    sumallndvi2=allndvi.iloc[14:21].sum()  
    sumallndvi1=pd.DataFrame(sumallndvi1)
    sumallndvi2=pd.DataFrame(sumallndvi2)
    sumallndwi=pd.DataFrame(sumallndwi)
    sumallndvi1["lab"]=label.astype(str)
    sumallndvi2["lab"]=label.astype(str)
    sumallndwi["lab"]=label.astype(str)
    pltbox('lab',0,sumallndvi1)
    plt.ylabel("Cumul VV")
    plt.title("periode semis")
    pltbox('lab',0,sumallndvi2) 
    plt.ylabel("Cumul VV")
    plt.title("période Croissance")
    
#    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/BOXPLOT_CUMUL/BOXPLOT_CUMUL_NDVI.png")
    pltbox('lab',0,sumallndwi)
    plt.ylabel("Cumul VH")
#    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/BOXPLOT_CUMUL/BOXPLOT_CUMUL_NDWI.png")
    
    
    allss=pd.DataFrame()
    allasc=pd.DataFrame()
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_CLASSIF/R20_SS_GAP/learningSamples/"):
        if "T31TCJ" not in i:
            print (i)
            STATR14 = sqlite3.connect('/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_CLASSIF/R10_FIXE_DATE/learningSamples/'+i)
            df=pd.read_sql_query("SELECT * FROM output", STATR14)
            dfpar=df.groupby("originfid").mean()
            label=dfpar.labcroirr.astype(int)
            lab=list(set(label))
            allss=allss.append(dfpar,ignore_index=True)
            for index,row in allss.T.iterrows():
                if "asc" in index:
                    allasc=allasc.append(row)

    # =============================================================================
    # Boxplot_TST NIrr/IR    
    # =============================================================================

   
    
    timeTST=pd.DataFrame()
    Years=[]
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/STAT_IRR/"):
        print (i)
        tile=i[35:41]
        print (tile)
        date=i[42:50]
        print (date)
        sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/STAT_IRR/" +i,"dfTST")
        lab=dfTST.labcroirr.astype(int)
        globals()["meanTST%s"%(date)]=round(dfTST.value_0/100-273.15,2)
        globals()["meanTST%s"%date].rename("TST_"+date,inplace=True)
        timeTST=timeTST.append(globals()["meanTST%s"%date])
        Years.append(date)
    timeTST.sort_index(inplace=True)
    timeTST[timeTST<= -1]=pd.NaT
    timeTST["date"]=sorted(Years)
    timeTST.date=pd.to_datetime(timeTST.date,format="%Y%m%d")
    TimeTST=timeTST.groupby(timeTST.date).mean()


    timeTSTNIRR=pd.DataFrame()
    Years=[]
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/STAT_NIRR/"):
        print (i)
        tile=i[35:41]
        print (tile)
        date=i[42:50]
        print (date)
        sqlite_df("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_TST/STAT_NIRR/" +i,"dfTST")
        lab=dfTST.labcroirr.astype(int)
        globals()["meanTST%s"%(date)]=round(dfTST.value_0/100-273.15,2)
        globals()["meanTST%s"%date].rename("TST_"+date,inplace=True)
        timeTSTNIRR=timeTSTNIRR.append(globals()["meanTST%s"%date])
        Years.append(date)
    timeTSTNIRR.sort_index(inplace=True)
    timeTSTNIRR[timeTSTNIRR<= -1]=pd.NaT
    timeTSTNIRR["date"]=sorted(Years)
    timeTSTNIRR.date=pd.to_datetime(timeTSTNIRR.date,format="%Y%m%d")
    TimeTSTNIRR=timeTSTNIRR.groupby(timeTSTNIRR.date).mean()
    TimeTSTNIRR=TimeTSTNIRR.iloc[11:]

    # Fusion data 
    a= list(set(TimeTST.index).intersection(set(TimeTSTNIRR.index)))
    a=sorted(a)
    ti=a[3:7]
    IR=pd.DataFrame(TimeTST.loc[ti].mean())
    IR['lab']=1
    NIR=pd.DataFrame(TimeTSTNIRR.loc[ti].mean())
    NIR['lab']=2
    TSTdata=pd.concat([IR,NIR])
    pltbox('lab',0,TSTdata)
    
