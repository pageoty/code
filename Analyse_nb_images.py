#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:02:17 2019

@author: pageot
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
#     Frise nb acquisition sur 1 tuiles
# =============================================================================
    # dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt",sep=',', header=None)
    # df1=dfnames.T
    # df1.columns=["band_name"]
    # date_interpol=list(df1.band_name.apply(lambda s: s[-9:-1]))
    # df1["date_inter"]=date_interpol
    
    # dfinterdate=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/list_2017dates_enterier.txt",header=None)
    # dfinterdate.columns=["band_name"]
    # date_nninter=dfinterdate.band_name.apply(lambda s: s[11:19]).astype(int)
    # date_nninter=pd.DataFrame(sorted(date_nninter))
    # tile=dfinterdate.band_name.apply(lambda s: s[-13:-7])
    # pd.DataFrame(np.repeat(1,tile.shape[0]))
    # df_S2=pd.concat([date_nninter,tile,pd.DataFrame(np.repeat(1,tile.shape[0]))],axis =1)
    # df_S2.columns=["date","tile","N"]
    # df_S2.date=pd.to_datetime(df_S2.date,format="%Y%m%d")
    
    # dfinterdateS1=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/S1_vh_ASC_dates_input.txt",header=None) 
    # #    df_S1=pd.concat([dfinterdateS1, pd.DataFrame(np.repeat('ASC',dfinterdateS1.shape[0])),pd.DataFrame(np.repeat(2,dfinterdateS1.shape[0]))],axis =1)
    # #    df_S1.columns=["date","mode","N"]
    # #    df_S1.date=pd.to_datetime(df_S1.date,format="%Y%m%d")
    # dfinterdate_S1_2018=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/S1_vv_DES_dates_input.txt",header=None)
    # dfinterdateS1=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/S1_vh_DES_dates_input.txt",header=None) 
    # df_S1_DES=pd.concat([dfinterdateS1, pd.DataFrame(np.repeat('DES',dfinterdateS1.shape[0])),pd.DataFrame(np.repeat(1.25,dfinterdateS1.shape[0]))],axis =1)
    # df_S1_DES.columns=["date","mode","N"]
    # df_S1_DES.date=pd.to_datetime(df_S1_DES.date,format="%Y%m%d")
    
    # df_S1_DES_2018=pd.concat([dfinterdate_S1_2018, pd.DataFrame(np.repeat('DES',dfinterdate_S1_2018.shape[0])),pd.DataFrame(np.repeat(1.25,dfinterdate_S1_2018.shape[0]))],axis =1)
    # df_S1_DES_2018.columns=["date","mode","N"]
    # df_S1_DES_2018.date=pd.to_datetime(df_S1_DES_2018.date,format="%Y%m%d")

    for file in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/NB_IMG/optic/"):
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/NB_IMG/optic/"+(file),sep=',', header=None)
        df_S2_tuile=df[0].apply(lambda s: s[-13:-7])
        df_S2_date=df[0].apply(lambda s: s[11:19])
        df_S2_date=pd.to_datetime(df_S2_date,format="%Y%m%d")
        id=pd.DataFrame(np.where(df_S2_tuile=='T31TCJ',1,2))
        dfS2=pd.concat([df_S2_tuile,df_S2_date,id],axis=1)
        dfS2.columns=["tuile","date","id"]
        print(dfS2.groupby('id').count())
        # plt.figure(figsize=(10,10))
        # ax1 = plt.subplot(111)
        # sns.set(style="darkgrid")
        # sns.set_context('paper')
        # plt.setp(ax1.get_yticklabels(), visible=False)
        # p1=plt.plot(dfS2.date,dfS2.id,marker='o',linestyle='')
        # p3=plt.plot(df_S1_DES_2018.date[30:108],df_S1_DES_2018.N[30:108],marker='v',linestyle='')
        # plt.xticks(rotation=0)
        # plt.ylim(0.75,1.5)
        # plt.legend((p1[0],p3[0]),("Sentinel-2","SAR-des"),loc='best')
#    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/imagesdate_entier_2018.png")


# =============================================================================
#  Nb acquisitions claires by tiles
# =============================================================================
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/CloudPercent_tile/"):
        print (i)
        if "T30TYN" in i :
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
                Classe2018=dfCloudseason2018[["Month",'classe','date']].groupby(["Month","classe"]).count()
                print (dfCloudseason2018.count())
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
                Classe=dfCloudseason[["Month",'classe','date']].groupby(["Month","classe"]).count()
                print( "2017")
                print (dfCloudseason.count())
    #            img_clair=Classe.loc(axis=0)[:, ['0-24']]
    #            print (r"  years : {} tile : {} : result : {}".format(i[13:17],list(tile.iloc[1]),img_clair))
                Acqui_claire_2018_TCJ=dfCloudseason[dfCloudseason.classe=="0-24"]
                Acqui_claire_2018_TCJ["nb"]=1
# =============================================================================
#  Plot acquisitions claire by tuiles 
    for tile in ["2018_T30TYN","2018_T30TYP","2017_T30TYN","2017_T30TYP"]:
         sns.set(style="darkgrid")
         sns.set_context('paper')
#         globals()["df%s"% tile].unstack(level=0).date.plot(kind='bar')
#         plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/CloudPercent_S2/acquisition_S2_{}.png".format(tile))