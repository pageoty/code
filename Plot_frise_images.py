# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:26:55 2019

@author: pageot
"""

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import seaborn as sns
import os



#Create fonction plotenbarre
def plotenbar(x,y):
    y_pos= np.arange(len(list(x)))
    pal=sns.set_palette("RdYlBu",len(y_pos))
    plt.figure(figsize=(10,15))
    sns.barplot(x,y,palette=pal)
    name_axe_x=input("Name axe X :")
    name_axe_y=input("Name axe Y :")
    plt.xlabel(name_axe_x)
    plt.ylabel(name_axe_y)
    plt.xticks(y_pos, list(x),rotation=45)
#    plt.text(x=x.shape[0]-2,y=max(y/sum(y)),s=round(float(sum(y)),2))
#    plt.text(x=x.shape[0]-4.5,y=max(y/sum(y)),s="superfice totale (ha):")
#    #label=round(y,2)
#    #for j in range(len(df["sum"])):
#        #plt.text(x = y_pos[j] , y = y[j] +0.01, s = list(label)[j], size = 11)





#
if __name__ == '__main__':
    d = {}
    d["PATH_DT"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_ECOREG/CSV/"
    d["PATH_RPG"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_ECO_REG/CSV/"
    # =============================================================================
    # Ajoute de la fonction add avec ponderation
    # =============================================================================
    list_csv=os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAY_PRAGRI/CSV")
    #names_crop=["Maize_Nirr","Soybean_Nirr","Peas_Nirr","Sunflow_Nirr","Maize_Irr","Sorghum_Nirr","Sorghum_Irr","Soybean_Irr","Peas_Irr","Sunflower_Irr","Others"]
    names_crop=["Maize_Irr","Soybean_Irr","Sorghum_Irr","Sunflower_Irr","Peas_Irr","Others","Maize_Nirr","Soybean_Nirr","Sorghum_Nirr","Sunflower_Nirr","Peas_Nirr"]
    names_RPG=["Maize","Soybean","Sorghum","Sunflower","Peas"]
    code_crops_RPG=[1,2,3,4,5]
    code_crops=[1,2,3,4,5,6,11,22,33,44,55]
    #for i in list_csv:
    #    df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAY_PRAGRI/CSV/"+ i)
    #    zone=i[8:-4]
    #    if df.shape[0] != len(code_crops_RPG):
    #        print ("Error dimension")
    #        result=set(list(df.category)).union(set(code_crops_RPG)) - set(list(df.category)).intersection(set(code_crops_RPG))
    #        miss=pd.DataFrame(result)
    #        val=pd.DataFrame(np.repeat(0,miss.shape[0])).values
    #        add=pd.DataFrame({"category":list(miss.values),"min":list(val),"max":list(val),"mean":list(val),
    #                                  "stddev":list(val),"sum":list(val),"count":list(val)})
    #        df2=pd.concat([df,add],ignore_index=True)
    #        df2=df2.sort_values(by='category',asending=True)
    #        df2["name"]=names_RPG
    #        df3=df2.sort_values(by='sum', ascending=False)
    #        df3=df3.reset_index()
    #        y_pos = np.arange(len(list(df3.name)))
    #        fig = plt.figure(figsize=(10,15))
    #        pal=sns.set_palette("RdYlBu",len(y_pos))
    #        #sns.barplot(df1.name,df1["mean"],palette=pal,yerr = df1["stddev"]) # permet d'ajoute les ecart types
    #        sns.barplot(df3.name,(df3["sum"]/sum(df3["sum"])),palette=pal)
    #        plt.xticks(y_pos, list(df3.name),rotation=45)
    #        plt.xlabel("crops")
    #        plt.ylabel("répartitions des OS")
    #        plt.title(zone)
    #        plt.text(x=df3.shape[0]-1,y=max(df3["sum"]/sum(df3["sum"])),s=round(float(sum(df3["sum"])),2))
    #        plt.text(x=df3.shape[0]-2.5,y=max(df3["sum"]/sum(df3["sum"])),s="superfice totale (ha):")
    ##        label=round(df3["sum"]/sum(df3["sum"]),2)
    ##        # Tet on the top of each barplot
    ##        for j in range(len(df3["sum"])):
    ##            plt.text(x = y_pos[j] , y = (df3["sum"]/sum(df3["sum"]))[j] +0.01, s = list(label)[j], size = 11)
    ##        # Show graphic
    #        plt.show()
    #        fig.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAY_PRAGRI/PLOT/PLOT_SUM_POND/%s.png"%(zone))
    #    else:
    #        print ("Create plot")
    #        df=df.sort_values(by="category",ascending=True)
    #        df["name"]=names_RPG
    #        df1=df.sort_values(by='sum', ascending=False)
    #        df1=df1.reset_index()
    #        y_pos = np.arange(len(list(df1.name)))
    #        fig = plt.figure(figsize=(10,15))
    #        pal=sns.set_palette("RdYlBu",len(y_pos))
    #        #sns.barplot(df1.name,df1["mean"],palette=pal,yerr = df1["stddev"]) # permet d'ajoute les ecart types
    #        sns.barplot(df1.name,(df1["sum"]/sum(df1["sum"])),palette=pal)
    #        plt.xticks(y_pos, list(df1.name),rotation=45)
    #        plt.xlabel("crops")
    #        plt.ylabel("Répartitions des OS")
    #        plt.title(zone)
    #        plt.text(x=df1.shape[0]-1,y=max(df["sum"]/sum(df1["sum"])),s=round(sum(df["sum"]),2))
    #        plt.text(x=df1.shape[0]-2.5,y=max(df["sum"]/sum(df1["sum"])),s="superfice totale (ha):")
    #        label=round(df1["sum"]/sum(df1["sum"]),2)
    #        # Tet on the top of each barplot
    #        for j in range(len(df1["sum"]/sum(df1["sum"]))):
    #            plt.text(x = y_pos[j] , y = (df1["sum"]/sum(df1["sum"]))[j] +0.01, s = list(label)[j], size = 11)
    #        # Show graphic
    #        plt.show()
    #        fig.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAY_PRAGRI/PLOT/PLOT_SUM_POND/%s.png"%(zone))
            
            
    # =============================================================================
    # Catplot with concat_data        
    # =============================================================================
    list_RPG=os.listdir(d["PATH_RPG"])
    list_DT=os.listdir(d["PATH_DT"])
    dfall=pd.DataFrame()
    for i in list_RPG:
        df=pd.read_csv(d["PATH_RPG"] +i)
        add=pd.DataFrame({'zone':i[17:-4],'source':'RPG'},index=[0,1])
        df[["source","zone"]]=add 
        df.sort_values(by='category',ascending=True)
        df["name"]=names_RPG
        dfall=dfall.append(df)
        
    dfdt=pd.DataFrame()
    for j in list_DT:
        df=pd.read_csv(d["PATH_DT"] +j)
        if df.shape[0] != len(code_crops):
            print ("Error dimension")
            result=set(list(df.category)).union(set(code_crops)) - set(list(df.category)).intersection(set(code_crops))
            miss=pd.DataFrame(result)
            val=pd.DataFrame(np.repeat(0,miss.shape[0])).values
            add=pd.DataFrame({"category":list(miss.values),"min":list(val),"max":list(val),"mean":list(val),
                                      "stddev":list(val),"sum":list(val),"count":list(val)})
            df=pd.concat([df,add],ignore_index=True)
            add2=pd.DataFrame({'zone':j[16:-4],'source':'DT'},index=[0])
            df[["source","zone"]]=add2 
            df.sort_values(by='category',ascending=True)
            df["name"]=names_crop
            dfdt=dfdt.append(df)
    

    # =============================================================================
    #  Plot frise data
    # =============================================================================
    dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt",sep=',', header=None)
    df1=dfnames.T
    df1.columns=["band_name"]
    date_interpol=list(df1.band_name.apply(lambda s: s[-9:-1]))
    df1["date_inter"]=date_interpol

    dfinterdate=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/list_2017dates_enterier.txt",header=None)
    dfinterdate.columns=["band_name"]
    date_nninter=dfinterdate.band_name.apply(lambda s: s[11:19]).astype(int)
    date_nninter=pd.DataFrame(sorted(date_nninter))
    tile=dfinterdate.band_name.apply(lambda s: s[-13:-7])
    pd.DataFrame(np.repeat(1,tile.shape[0]))
    df_S2=pd.concat([date_nninter,tile,pd.DataFrame(np.repeat(1,tile.shape[0]))],axis =1)
    df_S2.columns=["date","tile","N"]
    df_S2.date=pd.to_datetime(df_S2.date,format="%Y%m%d")

#    dfinterdateS1=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/S1_vh_ASC_dates_input.txt",header=None) 
#    df_S1=pd.concat([dfinterdateS1, pd.DataFrame(np.repeat('ASC',dfinterdateS1.shape[0])),pd.DataFrame(np.repeat(2,dfinterdateS1.shape[0]))],axis =1)
#    df_S1.columns=["date","mode","N"]
#    df_S1.date=pd.to_datetime(df_S1.date,format="%Y%m%d")
#    
    dfinterdateS1=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/S1_vh_DES_dates_input.txt",header=None) 
    df_S1_DES=pd.concat([dfinterdateS1, pd.DataFrame(np.repeat('DES',dfinterdateS1.shape[0])),pd.DataFrame(np.repeat(1.25,dfinterdateS1.shape[0]))],axis =1)
    df_S1_DES.columns=["date","mode","N"]
    df_S1_DES.date=pd.to_datetime(df_S1_DES.date,format="%Y%m%d")

    plt.figure(figsize=(15,4))
    ax1 = plt.subplot(111)
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.setp(ax1.get_yticklabels(), visible=False)
    p1=plt.plot(df_S2.date,df_S2.N,marker='o',linestyle='')
    p3=plt.plot(df_S1_DES.date,df_S1_DES.N,marker='v',linestyle='')
    plt.xticks(rotation=0)
    plt.ylim(0.75,1.5)
    plt.legend((p1[0],p3[0]),("Sentinel-2","SAR-des"),loc='best')
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/imagesdate_entier.png")
