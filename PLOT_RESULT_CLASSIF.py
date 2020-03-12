#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:44:18 2019

@author: pageot
"""



# =============================================================================
# Parametrisation des PATH
# =============================================================================
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import seaborn as sns
import os
import matplotlib.patches as patches
import csv

def plt_classif(df,var1,var2,var3):
    plt.figure(figsize=(10,10))
    y_pos=np.arange(df.shape[0])
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.bar(y_pos,df["mean_"+var1],yerr=df["std_"+var1],capsize=3,width = 1,label=var1)
    plt.bar(y_pos+df.shape[0]+1,df["mean_"+var2],yerr=df["std_"+var2],capsize=3,width = 1,label=var2)
    plt.bar(y_pos+df.shape[0]*2+2,df["mean_"+var3],yerr=df["std_"+var3],capsize=3,width = 1,label=var3)
    plt.xticks(y_pos, tuple(df.index),rotation=90,size=9)
    y_pos2=y_pos+df.shape[0]+1
    y_pos3=y_pos+df.shape[0]*2+2
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos2[j]-0.25 , y = -0.03, s = list(df.index)[j],size=9,rotation=90)
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos3[j]-0.25 , y = -0.03, s = list(df.index)[j],size=9,rotation=90)
    plt.legend()
    
def plt_classif_kappa(df,var1,var2):
    plt.figure(figsize=(10,5))
    y_pos=np.arange(df.shape[0])
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.grid(axis='x')
    plt.bar(y_pos,df["mean_"+var1],yerr=df["std_"+var1],capsize=3,width = 1,label=var1)
    plt.bar(y_pos+df.shape[0]+0.5,df["mean_"+var2],yerr=df["std_"+var2],capsize=3,width = 1,label=var2)
    plt.ylabel("score")
    plt.xlabel("step")
    plt.xticks(y_pos, tuple(df.index),rotation=90)
    y_pos3=y_pos+df.shape[0]+0.5
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos3[j]-0.25 , y = -0.05, s = list(df.index)[j],size=9,rotation=90)
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos[j]-0.3, y=df["mean_"+var1][j]+0.05,s = list(round(df["mean_"+var1],2))[j],size=9)
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos3[j]-0.3, y=df["mean_"+var2][j]+0.04,s = list(round(df["mean_"+var2],2))[j],size=9)
    plt.legend()

def errplot(x, y, yerr, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, yerr=yerr, kind="bar", ax=ax, **kwargs)
    
def pltax2(y,x1,x2):
    plt.figure(figsize=(10,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    plt.plot(y,x1,color="blue")
    ax2 = plt.twinx()
    ax2.plot(y,x2)


        
if __name__ == "__main__":
    d={}
    d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/RUN_2017/"
    d["SAVE"]="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/PLOT_SYNTH_CLASSIF/"
    # =============================================================================
    # Recuperation KAPPA & OA 
    # =============================================================================
    KAPPA=pd.DataFrame()
    run=pd.DataFrame()
    matrix=pd.DataFrame()
    for j in os.listdir(d["data_file"]):
        run=run.append([j],ignore_index=True)
        for i in os.listdir(d["data_file"] +j):
            with open(d["data_file"]+j+"/"+i+"/"+"RESULTS.txt", "r") as res_file:
                for line in res_file:
                    if "KAPPA" in line.rstrip():
                        print (line.rstrip()[7:])
                        KAPPA=KAPPA.append([line.rstrip()[7:]],ignore_index=True)
                        
    OA=pd.DataFrame()
    for j in os.listdir(d["data_file"]):
        for i in os.listdir(d["data_file"] +j):
            with open(d["data_file"]+j+"/"+i+"/"+"RESULTS.txt", "r") as res_file:
                for line in res_file:
                    if "OA" in line.rstrip():
                        print (line.rstrip()[4:])
                        OA=OA.append([line.rstrip()[4:]],ignore_index=True)
    dfindice=pd.concat([run,KAPPA,OA],axis=1)
    names_indice=["step","kappa",'OA']
    dfindice.columns=names_indice
    kappa=dfindice["kappa"].str.split(expand=True)
    kappa.drop([1],axis=1,inplace=True)
    kappa=kappa.astype(float)
    oa=dfindice["OA"].str.split(expand=True)
    oa.drop([1],axis=1,inplace=True)
    oa=oa.astype(float)
    dfindice=pd.concat([dfindice,kappa,oa],axis=1,ignore_index=True)
    names_indice=["step","kappa",'OverA',"mean_Kappa","std_Kappa","mean_OA","std_OA"]
    dfindice.columns=names_indice
    dfindice.set_index("step",inplace=True)
    dfindice.sort_index(inplace=True)
    
    
    
    # =============================================================================
    #RECUPERATION DATA_MATRIX 
    # ===========================================================================
    
    dfall=pd.DataFrame()
    dfaccu=pd.DataFrame()
    dfrapp=pd.DataFrame()
    dffsc=pd.DataFrame()
    names_df=["Classe","Accur","Rappel","F_score","Confusion_classe"]
    names_dfs=["index","Classe","Precision","Rappel","F_score","Confusion_classe","step","mean_Precision","std_Precision","mean_rappel","std_rappel","mean_Fscore","std_Fscore"]
    nombre_de_classe=8
    
    for j in os.listdir(d["data_file"]):
        print (j)
        for i in os.listdir(d["data_file"] +j):
            df=pd.read_csv(d["data_file"]+j+"/"+i+"/"+"RESULTS.txt",sep='\|',skiprows=int(18),header=None)
            df.columns=names_df
#            df1=df.drop([0])
            df1=df.reset_index()
            origin=pd.DataFrame({'step':j},index=[0],dtype="category")
            origindup=pd.DataFrame(np.repeat(origin.values,df.shape[0]),dtype="category")
            df1["step"]=origindup
            accu=df1["Accur"].str.split(expand=True)
            accu.drop([1],axis=1,inplace=True)
            dfaccu=dfaccu.append(accu).astype(float)
            rap=df1["Rappel"].str.split(expand=True)
            rap.drop([1],axis=1,inplace=True)
            dfrapp=dfrapp.append(rap).astype(float)
            Fsc=df1["F_score"].str.split(expand=True)
            Fsc.drop([1],axis=1,inplace=True)
            dffsc=dffsc.append(Fsc).astype(float)
            dfall=dfall.append(df1)
    
    data=pd.concat([dfall,dfaccu,dfrapp,dffsc],axis=1)
    data.columns=names_dfs
    data.sort_index(by=["step","Classe"],ascending=True,inplace=True)
    data.reset_index(inplace=True)
    data.replace(to_replace ='nan' , value= pd.NaT, inplace =True)
    data1=data.replace(to_replace =pd.NaT , value= 0)
    Name_lab=[]
    for i in np.arange(data1.shape[0]):
        a=data1.Classe.iloc[i].strip()
        Name_lab.append(a)
    data1["Name_label"]=Name_lab
    dfstep=data1.groupby('step').mean()
    dfstep.drop(['level_0','index'],axis=1,inplace=True)
    dfstep.replace(to_replace ='nan' , value= pd.NaT, inplace =True)
    
    #dita=data.drop([2,12,17],axis=0)
    # =============================================================================
    #  plot comparer les run  
    # =============================================================================
#    plt_classif(dfstep,'Precision','Fscore','rappel')
#    plt.savefig(d["SAVE"]+"CLASSIF_RUN.png")
    plt_classif_kappa(dfindice,"Kappa","OA")
    plt.savefig(d["SAVE"]+"KAPPA_RUN_2018.png",dpi=600,bbox_inches='tight', pad_inches=0.5)
    # =============================================================================
    # Plot visualisation run[i] par classe a finir
    # =============================================================================
#    plt.figure(figsize=(10,10))
#    sns.set(style="darkgrid")
#    sns.set_context('paper')
#    sns.barplot(data["Classe"][0:nombre_de_classe],data["mean_Precision"][0:nombre_de_classe])
#    y_pos=np.arange(data.shape[0])
#    plt.xticks(np.arange(data.shape[0]), df1.Classe ,rotation=45)
#    label=round(data["mean_Precision"][0:nombre_de_classe],2)
#    for i in y_pos:
#        plt.text(x = y_pos[i]-0.2 , y = data["mean_Precision"][0:17][i], s = label[i], size = 11) 
#    # =============================================================================
    # PLOT Compare les esulats des run à la classe
    # =============================================================================
    data2=pd.concat([data1[data1['level_0'] == 0],data1[data1["level_0"]== 2],data1[data1["level_0"]== 1],data1[data1["level_0"]== 3]])
    for i in data1[["mean_Fscore","mean_Precision","mean_rappel"]]:
        print(i)
        var=i[5:]
        plt.figure(figsize=(20,20))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        g = sns.FacetGrid(data1, col="Name_label", col_wrap=9, palette="Set1",height=5)# Gerer la color par run et +3 a modifier en focntion du nb de run 
        g.map_dataframe(errplot, "step", "mean_"+var, "std_"+var)
        g.savefig(d["SAVE"]+var+"_plot_classe_run_MT2018.png",dpi=600,bbox_inches='tight', pad_inches=0.5)

    
    #pltax2(dfindice.index,dfindice.mean_Kappa,dfindice.mean_OA)
    plt.figure(figsize=(15,7))
    sns.heatmap(dfindice[["mean_Kappa","mean_OA"]],annot=True,cmap="coolwarm",annot_kws={"size": 15})
    plt.savefig(d["SAVE"]+"tab_mean.png",dpi=600,bbox_inches='tight', pad_inches=0.5)

#    
# =============================================================================
#    MAIZE
# =============================================================================
    data2=data.set_index("step")
    plt.figure(figsize=(15,7))
    sns.heatmap( data2[["mean_Fscore","mean_rappel","mean_Precision"]].loc[data2["index"].isin([0,2])],annot=True,cmap="coolwarm",annot_kws={"size": 15})
    label1=list(np.repeat(['Irr'],len(dfstep)*2))
    label=list(np.repeat(['No Irr'],len(dfstep)*2))
    for j in np.arange(0,len(dfstep)*2,2):
        plt.text(x = -0.25 , y = j+0.90, s = label1[j],size=9,fontweight = 'bold')
    for k in np.arange(1,len(dfstep)*2,2):
        plt.text(x = -0.25 , y = k+0.90, s = label[k],size=9,fontweight = 'bold')
    plt.title("Comparaison perf Maize")
    plt.savefig(d["SAVE"]+"tab_mean_Fscore_MaisMT2018.png",dpi=600,bbox_inches='tight', pad_inches=0.5)
   
# =============================================================================
# SOYBEAN
# =============================================================================
    plt.figure(figsize=(15,7))
    sns.heatmap(data2[["mean_Fscore","mean_rappel","mean_Precision"]].loc[data2["index"].isin([1,3])],annot=True,cmap="coolwarm",annot_kws={"size": 15})
    label1=list(np.repeat(['Irr'],len(dfstep)*2))
    label=list(np.repeat(['No Irr'],len(dfstep)*2))
    for j in np.arange(0,len(dfstep)*2,2):
        plt.text(x = -0.25 , y = j+0.90, s = label1[j],size=9,fontweight = 'bold')
    for k in np.arange(1,len(dfstep)*2,2):
        plt.text(x = -0.25 , y = k+0.90, s = label[k],size=9,fontweight = 'bold')
    plt.title("Comparaison perf Soybean")
    plt.savefig(d["SAVE"]+"tab_mean_Fscore_SojaMT2018.png",dpi=600,bbox_inches='tight', pad_inches=0.5)
