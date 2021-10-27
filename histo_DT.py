#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:42:00 2019

@author: pageot
"""


import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import seaborn as sns
import os


if __name__ == '__main__':
    
    names_crop=["Maize_Irr","Soybean_Irr","Others","Maize_Nirr","Soybean_Nirr","Sorghum","Sunflower","Peas"]
    names_RPG=["Maize","Soybean","Others","Maize","Soybean","Sorghum","Sunflower","Peas"]
    names_crop_fr=["Mais_Irr","Soja_Irr","Autres","Mais_Nirr","Soja_Nirr","Sorgho","Tournesol","Pois"]
    names_RPG_fr=["Mais","Soja","autres","Mais","Soja","Sorgho","Tournesol","Pois"]
    code_crops_RPG=[1,2,3,4,5]
    code_crops=[1,2,6,11,22,33,44,55]
    pal=sns.set_palette("RdYlBu")
    
    # =============================================================================
    # # Analye DT/RPG en nb hectares par tuils
    # =============================================================================
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_TUILES/"):
        print (i)
        tile=i[8:11]
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_TUILES/%s"%i)
        df=df.sort_values(by='category', ascending=True)
        df["name"]=names_crop
        df=df.reset_index()
        y_pos= np.arange(len(list(df.name)))
        plt.figure(figsize=(10,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        sns.barplot(df.name,df["sum"],palette=pal)
        plt.xticks(y_pos, names_crop,rotation=45)
        plt.xlabel("OS")
        plt.title("Tile %s" %tile)
        plt.ylim(0,3500)
        plt.ylabel("Superficie en ha")

        label=df["sum"]
                # Tet on the top of each barplot
        for j in range(len(df["sum"])):
            plt.text(x = y_pos[j] -0.3 , y = df["sum"][j], s = list(label)[j], size = 11)
#        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_DT_%s.png" %tile) 
    
    
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_TUILES/"):
        print (i)
        tile =i[9:12]
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_TUILES/%s" %i)
        df=df.sort_values(by='category', ascending=True)
        df["name"]=names_RPG
        df=df.reset_index()
        y_pos= np.arange(len(list(df.name)))
        plt.figure(figsize=(10,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        sns.barplot(df.name,df["sum"],palette=pal)
        plt.xticks(y_pos, names_RPG,rotation=45)
        plt.xlabel("OS")
        plt.ylim(0,180000)
        plt.title("Tile %s" %tile)
        plt.ylabel("superfice en ha")
        label=df["sum"]
                # Tet on the top of each barplot
        for j in range(len(df["sum"])):
            plt.text(x = y_pos[j] -0.3 , y = df["sum"][j], s = list(label)[j], size = 11)
#        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_RPG_%s.png" %tile) 
    
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_BV/"):
        print(i)
        BV=i[-9:-4]
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_BV/%s" %i)
        df=df.sort_values(by='category', ascending=True)
        df["name"]=names_RPG
        df=df.reset_index()
        y_pos= np.arange(len(list(df.name)))
        plt.figure(figsize=(10,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        sns.barplot(df.name,df["sum"],palette=pal)
        plt.xticks(y_pos, names_RPG,rotation=45)
        plt.xlabel("OS")
        plt.ylim(0,140000)
        plt.title("Bassin versant %s" %BV)
        plt.ylabel("superfice en ha")
        label=df["sum"]
                # Tet on the top of each barplot
        for j in range(len(df["sum"])):
            plt.text(x = y_pos[j] -0.3 , y = df["sum"][j], s = list(label)[j], size = 11)
#        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_RPG_%s.png" %BV) 
        
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/"):
        print (i)
        zone=i[-9:-4]
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/%s" %i)
        df=df.sort_values(by='category', ascending=True)
        df["name"]=names_crop
        df=df.reset_index()
        y_pos= np.arange(len(list(df.name)))
        plt.figure(figsize=(10,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        sns.barplot(df.name,df["sum"]/0.01,palette=pal)
        plt.xticks(y_pos, names_crop,rotation=45)
        plt.xlabel("OS")
        plt.ylim(0,120000)
        plt.title("zone %s" %zone)
        plt.ylabel("nombre de pixels")
#        label=df["sum"]
#                # Tet on the top of each barplot
#        for j in range(len(df["sum"])):
#            plt.text(x = y_pos[j] -0.3 , y = df["sum"][j], s = list(label)[j], size = 11)
#        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_RPG_%s.png" %zone) 
        
# =============================================================================
#         Test data même plot
# =============================================================================
#    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/"):
#        for j in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_BV/"):
#            print (i,j)
#            if i[-9:-4] == j[-9:-4]: 
#                df1=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/%s" %i)
#                df2=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_BV/%s" %j)
#                df1=df1.sort_values(by='category', ascending=True)
#                df1["name"]=names_crop
#                df1=df1.reset_index()
#                df1["source"]=np.repeat("DT",len(names_crop))
#                df2=df2.sort_values(by='category', ascending=True)
#                df2["name"]=names_RPG
#                df2=df2.reset_index()
#                df2["source"]=np.repeat("RPG",len(names_RPG))
#                DF=pd.concat([df1,df2])
#                y_pos= np.arange(len(list(df1.name)))
#                plt.figure(figsize=(15,10))
#                sns.set(style="darkgrid")
#                sns.set_context('paper')
#                sns.barplot(x="name",y="sum",hue="source",data=DF)
#                plt.title("Bassin versant %s" %i[-9:-4])
#                
#            else : 
#                print("pas même zone")
#                
    # =============================================================================
    # Diagramme secteur 
    # =============================================================================
    a=pd.DataFrame()
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_BV/"):
        print(i)
        if '2018' in i:
            BV=i[-14:-4]
            df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_RPG_BV/%s" %i)
            df=df.sort_values(by='category', ascending=True)
            df["name"]=names_RPG_fr
            df=df.reset_index()
            y_pos= np.arange(len(list(df.name)))
            plt.figure(figsize=(10,10))
            sns.set(style="darkgrid")
            sns.set_context('paper')
            colors=["b","cadetblue","peru","orange","salmon"]
            plt.pie(df["sum"].iloc[[0,1,5,6,7]],labels=df["name"].iloc[[0,1,5,6,7]],autopct='%1.1f%%',colors=colors,textprops={'fontsize': 25})
            # sns.barplot(df.name.iloc[[0,1,3,4,5,6]],df["sum"].iloc[[0,1,3,4,5,6]],ci=None,colors=colors,textprops={'fontsize': 20})
            plt.title("Bassin versant %s" %BV)
            print(df)
            plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_hist_RPG_%s_%s.png" %BV) 
#    plt.pie(df2["sum"].iloc[[0,1,5,6,7]],labels=df2["name"].iloc[[0,1,5,6,7]],autopct='%1.1f%%')
#    plt.axis('equal')
    
    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/"):
        if "ADOUR" in i or "TARN" in i:
            if "2018" not in i:
                print (i)
                zone=i[-11:-4]
                print(zone)
                df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/%s" %i)
                df=df.sort_values(by='category', ascending=True)
                df["name"]=names_crop_fr
                df=df.reset_index()
                y_pos= np.arange(len(list(df.name)))
                plt.figure(figsize=(10,10))
                sns.set(style="darkgrid")
                sns.set_context('paper')
                plt.title("Bassin versant 2017 %s" %zone)
                colors=["b","cadetblue","olive","r","pink","peru","orange",'green']
    #            plt.pie(df["sum"],labels=df["name"],autopct='%1.1f%%',colors=colors,textprops={'fontsize': 20})
                sns.barplot(df.name.iloc[[0,1,3,4,5,6]],df["sum"].iloc[[0,1,3,4,5,6]],ci=None)
                plt.ylabel("surface en ha")
                plt.ylim(0,1600)
                plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_hist_DT_%s.png" %zone) 
                
            else:
                print (i)
                zone2=i[-14:-9]
                df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/%s" %i)
                df=df.sort_values(by='category', ascending=True)
                df["name"]=names_crop_fr
                df=df.reset_index()
                y_pos= np.arange(len(list(df.name)))
                plt.figure(figsize=(10,10))
                sns.set(style="darkgrid")
                sns.set_context('paper')
                plt.title("Bassin versant 2018 %s" %zone2)
                colors=["b","cadetblue","olive","r","pink","peru","orange",'green']
    #            plt.pie(df["sum"],labels=df["name"],autopct='%1.1f%%',colors=colors,textprops={'fontsize': 20})
                sns.barplot(df.name.iloc[[0,1,3,4,5,6]],df["sum"].iloc[[0,1,3,4,5,6]],ci=None)
                plt.ylabel("surface en ha")
                plt.ylim(0,1600)
                plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_histDT_2018%s.png" %zone2)
    
# =============================================================================
#     Stat_all 2017 ss CACG
# =============================================================================
    df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_ALL_2018.csv") 
    df=df.sort_values(by='category', ascending=True)
    df["name"]=names_crop_fr[:-1]
    df=df.reset_index()
    y_pos= np.arange(len(list(df.name)))
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.title("Bassin versant %s" %zone2)
    colors=["b","cadetblue","olive","r","pink","peru","orange",'green']
    plt.pie(df["sum"],labels=df["name"],autopct='%1.1f%%',colors=colors,textprops={'fontsize': 20})

    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_BV_DT_all/"):
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_BV_DT_all/%s" %i)
        df=df.sort_values(by='category', ascending=True)
        df["name"]=names_crop_fr[:-1]
        df=df.reset_index()
        y_pos= np.arange(len(list(df.name)))
        plt.figure(figsize=(10,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        plt.title("Bassin versant %s" %i[:-4])
        colors=["b","cadetblue","olive","r","pink","peru","orange",'green']
        plt.pie(df["sum"],labels=df["name"],autopct='%1.1f%%',colors=colors,textprops={'fontsize': 20})
        plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_secteur_DT_%s.png" %i[:-4])

# =============================================================================
#  Polygon_Validation/run
# =============================================================================
    all=pd.DataFrame()
    for csv in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_LEARN_VAL_CLASSIF_MT/2017/RUN_POLA_DES/Stat_polygon_vali/") :
        df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_LEARN_VAL_CLASSIF_MT/2017/RUN_POLA_DES/Stat_polygon_vali/"+str(csv))
        df=df.sort_values(by='category', ascending=True)
        df["name"]=[names_crop_fr[index] for index in [0,1,3,4,5,6]]
        df=df.reset_index()
        all=all.append(df)
    df_mean_classe=all.groupby("name").mean()
    y_pos= np.arange(len(list(df.name)))
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    colors=["b","r","cadetblue","pink","peru","orange"]
    plt.pie(df_mean_classe["sum"],labels=df_mean_classe.index,autopct='%1.1f%%',colors=colors,textprops={'fontsize': 20})


# =============================================================================
# NESTE ASA
# =============================================================================
    df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/STAT_DT_NESTE_ASA_RPG2017_v2.csv")
    df=df.sort_values(by='category', ascending=True)
    df["name"]=["Maize_Irr","Soybean_Irr","Others","Sorghum",'Sunflower']
    df=df.reset_index()
    y_pos= np.arange(len(list(df.name)))
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
#    colors=["b","cadetblue","peru","orange","salmon"]
    plt.pie(df["sum"].iloc[[0,1,3,4]],labels=df["name"].iloc[[0,1,3,4]],autopct='%1.1f%%',colors=colors,textprops={'fontsize': 20})
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_secteur_DT_NESTE_ASA_2017.png")
    
    df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/STAT_POLY/STAT_DT_PARTENAIRES/STAT_DT_NESTE_ASA_2018.csv")
    df=df.sort_values(by='category', ascending=True)
    df["name"]=names_crop_fr[:-1]
    df=df.reset_index()
    y_pos= np.arange(len(list(df.name)))
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    colors=["b","cadetblue","peru","orange","salmon"]
    plt.pie(df["sum"].iloc[[0,1,5,6]],labels=df["name"].iloc[[0,1,5,6]],autopct='%1.1f%%',colors=colors,textprops={'fontsize': 20})
    plt.savefig("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/HISTO_REP_DT/REPARITION_secteur_DT_NESTE_ASA_2018.png")


