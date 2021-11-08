#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:02:17 2019

@author: pageot

Analyse du nombre d'images en focntion des tuiles et des années
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import numpy as np
import pandas as pd
import seaborn as sns
if __name__ == "__main__":

# =============================================================================
#   nb acquisition sur 1 tuiles
# =============================================================================
    for file in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/NB_IMG/optic/"):
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/NB_IMG/optic/"+(file),sep=',', header=None)
        df_S2_tuile=df[0].apply(lambda s: s[-13:-7])
        df_S2_date=df[0].apply(lambda s: s[11:19])
        df_S2_date=pd.to_datetime(df_S2_date,format="%Y%m%d")
        id=pd.DataFrame(np.where(df_S2_tuile=='T31TCJ',1,2))
        dfS2=pd.concat([df_S2_tuile,df_S2_date,id],axis=1)
        dfS2.columns=["tuile","date","id"]
        print(dfS2.groupby('id').count())
# =============================================================================
#  Nb acquisitions claires by tiles
# =============================================================================
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/CloudPercent_tile/"):
        print (i)
        if "T31TDJ" in i :
            df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/CloudPercent_tile/{}".format(i),sep=":",header=None)
            date=pd.DataFrame(df[0].apply(lambda x: x[11:19]))
            tile=pd.DataFrame(df[0].apply(lambda x: x[35:41]))
            cloudPercent=pd.DataFrame(df[2].apply(lambda x: x))
            dfCloud=pd.concat([date,tile,cloudPercent],axis=1)
            dfCloud.columns=["date","tile","CloudPercent"]
            dfCloud.sort_values(by="date",ascending=True,inplace=True)
            if "2018" in i :
                dfCloudseason2018=dfCloud.loc[(dfCloud["date"] >= '20180401') & (dfCloud["date"] <= '20181131')]
                Month=dfCloudseason2018.date.apply(lambda x:x[4:6])
                dfCloudseason2018["Month"]=Month
                dfCloudseason2018.date=pd.to_datetime(dfCloudseason2018.date,format="%Y%m%d")
                dfCloudseason2018["classe"]="75-100"
                dfCloudseason2018.loc[(dfCloudseason2018.CloudPercent >= 0) & (dfCloudseason2018.CloudPercent <= 24),'classe']= "0-24"
                dfCloudseason2018.loc[(dfCloudseason2018.CloudPercent >= 25) & (dfCloudseason2018.CloudPercent <= 49),'classe']= "25-49"
                dfCloudseason2018.loc[(dfCloudseason2018.CloudPercent >= 50) & (dfCloudseason2018.CloudPercent <= 74),'classe']= "50-74"
                globals()["df%s" % i[13:-4]]=dfCloudseason2018.groupby(["classe","Month"]).count()
                globals()["data%s" % i[13:-4]]=dfCloudseason2018[["Month",'classe','date']].groupby(["Month"]).count()
                Classe2018=dfCloudseason2018[["Month",'classe','date']].groupby(["Month","classe"]).count()
                print (dfCloudseason2018.count())
                print("2018 + %s"%i)
                print(dfCloudseason2018.CloudPercent.mean())
    #            img_clair=Classe.loc(axis=0)[:, ['0-24']]
    #            print (r"  years : {} tile : {} : result : {}".format(i[13:17],list(tile.iloc[1]),img_clair))
            else:
                dfCloudseason=dfCloud.loc[(dfCloud["date"] >= '20170401') & (dfCloud["date"] <= '20171131')]
                Month=dfCloudseason.date.apply(lambda x:x[4:6])
                dfCloudseason["Month"]=Month
                dfCloudseason.date=pd.to_datetime(dfCloudseason.date,format="%Y%m%d")
                dfCloudseason["classe"]="75-100"
                dfCloudseason.loc[(dfCloudseason.CloudPercent >= 0) & (dfCloudseason.CloudPercent <= 24),'classe']= "0-24"
                dfCloudseason.loc[(dfCloudseason.CloudPercent >= 25) & (dfCloudseason.CloudPercent <= 49),'classe']= "25-49"
                dfCloudseason.loc[(dfCloudseason.CloudPercent >= 50) & (dfCloudseason.CloudPercent <= 74),'classe']= "50-74"
                MonthPercent=dfCloudseason.groupby(["Month","CloudPercent"]).count()
                globals()["df%s" %  i[13:-4]]=dfCloudseason.groupby(["classe","Month"]).count()
                globals()["data%s" % i[13:-4]]=dfCloudseason[["Month",'classe','date']].groupby(["Month"]).count()
                Classe=dfCloudseason[["Month",'classe','date']].groupby(["Month","classe"]).count()
                print( "2017")
                print (dfCloudseason.count())
                print(dfCloudseason.CloudPercent.mean())

# =============================================================================
#  Plot acquisitions claire by tuiles 
# =============================================================================
    

    for tile in ["2018_T30TYN","2018_T30TYP","2017_T30TYN","2017_T30TYP"]:
        globals()["df%s"% tile].unstack(level=0).date.plot(kind='bar')
        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/CloudPercent_S2/acquisition_S2_{}.png".format(tile))
   
    
    plt.figure(figsize=(7,5))
    plt.bar(data2018_T30TYN.index,data2018_T30TYN.date,label='TYN')
    plt.bar(data2018_T30TYP.index,data2018_T30TYP.date,bottom=data2018_T30TYN.date,label="TYP")
    plt.bar(data2018_T31TCJ.index,data2018_T31TCJ.date,bottom=np.add(data2018_T30TYN.date,data2018_T30TYP.date).tolist(),label="TCJ")
    plt.bar(data2018_T31TDJ.index,data2018_T31TDJ.date,bottom=np.add(data2018_T30TYN.date,data2018_T30TYP.date).to_list()+data2018_T31TCJ.date,label='TDJ')
    plt.legend()
    plt.xlabel("mois")
    plt.ylabel("Nombre d'acquisitions")
    plt.title("2018")
    plt.ylim(0,50)
    # plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/CloudPercent_S2/acquisition_S2_2018.png")

    plt.figure(figsize=(7,5))
    plt.bar(data2017_T30TYN.index,data2017_T30TYN.date,label='TYN')
    plt.bar(data2017_T30TYP.index,data2017_T30TYP.date,bottom=data2017_T30TYN.date,label="TYP")
    plt.bar(data2017_T31TCJ.index,data2017_T31TCJ.date,bottom=np.add(data2017_T30TYN.date,data2017_T30TYP.date).tolist(),label="TCJ")
    plt.bar(data2017_T31TDJ.index,data2017_T31TDJ.date,bottom=np.add(data2017_T30TYN.date,data2017_T30TYP.date).to_list()+data2017_T31TCJ.date,label='TDJ')
    plt.legend()
    plt.xlabel("mois")
    plt.ylabel("Nombre d'acquisitions")
    plt.title("2017")
    plt.ylim(0,50)
    # plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/CloudPercent_S2/acquisition_S2_2017.png")
