# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:50:47 2021

@author: yann 

Validation Irrigation automatique parcelle de référence
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


if __name__ == '__main__':
    d={}
    name_run="RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_Fcover_fewi_De_Kr_days10_p04_1500_modif_irri_auto_soil/"
    name_run_save_fig="RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_Fcover_fewi_De_Kr_days10_p04_1500_modif_irri_auto_soil/"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    label="Init ru année n-1 + Irrigation auto"
    years=["2017","2018"]
    lc="maize_irri"
    
    
# =============================================================================
#  Validation Irri_ préparation data 
# =============================================================================
    # all_quantity=[]
    # all_number=[]
    # all_id=[]
    # all_runs=[]
    # all_date=[]
    # all_date_jj=[]
    # for r in os.listdir('D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'):
    #     d["path_PC"]='D:/THESE_TMP/RUNS_SAMIR/RUN_RELATION/TARN_CACG/'+str(r)+'/Inputdata/'
    #     for s in os.listdir(d["path_PC"][:-10]):
    #         if "output" in s:
    #             print (s)
    #             df=pickle.load(open(d["path_PC"][:-10]+str(s),'rb'))
    #             for id in list(set(df.id)):
    #                 lam=df.loc[df.id==id]
    #                 date_irr=lam.loc[lam.Ir_auto!=0.0]["date"]
    #                 size_R=len(date_irr) # vecteur de répétiton de la variable RUNS
    #                 print(r'n° du runs : %s '% r)
    #                 print(r' n° parcelle : %s' %id)
    #                 print(r'sum irrigation in mm : %s'%lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
    #                 print(r' nb irrigation : %s' %lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
    #                 all_runs.append(r)
    #                 all_id.append(id)
    #                 all_quantity.append(lam.groupby(["LC","id"])["Ir_auto"].sum()[0])
    #                 all_number.append(lam.Ir_auto.where(df["Ir_auto"] != 0.0).dropna().count())
    #                 all_date.append(date_irr.values)
    #                 for i in date_irr:
    #                     a=i.strftime("%j") # date en jj sur l'année
    #                     all_date_jj.append(a)
    # all_resu=pd.DataFrame([all_runs,all_id,all_quantity,all_number,all_date]).T
    # all_resu.columns=["runs",'id','cumul_irr',"nb_irr","date_irr"]
# =============================================================================
#   Mean flux ETR -> Lam corrigées
# =============================================================================
    All_lam=[]
    for i in os.listdir(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/"):
        if "ETR_maize_irri" in i and "semi" not in i:
            ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/"+i,decimal='.',sep=",")
            ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
            All_lam.append(ETR.LE_Bowen)
    lam_et=pd.DataFrame(All_lam)
    mean_lam=pd.DataFrame(lam_et.mean())
    std_lam=pd.DataFrame(lam_et.std())
    # mean_lam["date"]=ETR.date
# =============================================================================
# Validation des Irr cumulées CACG
# =============================================================================
    

    for y in years: 
        Vol_tot=pd.DataFrame()
        df_date_aqui=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T30TYP_input_dates_"+str(y)+".txt",header=None)
        df_date_aqui[0]=pd.to_datetime(df_date_aqui[0],format='%Y%m%d')
        df_date_aqui.columns=["date"]
        vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/DATE_DOES_CACG_"+str(y)+".csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
        vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%Y')
        vali_cacg["Quantite"].astype(float)
        sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum()
        nb_irr=vali_cacg.groupby("ID")["Date_irrigation"].count()
        a=vali_cacg.groupby("ID")
        
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        Irri_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_"+str(y)+".csv",index_col=[0,1],skipinitialspace=True)
        gro=Irri_mod.groupby("ID")
        
        NDVI=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb"))
        NDVI=NDVI.loc[(NDVI.date >= str(y)+'-04-01')&(NDVI.date<=str(y)+"-09-30")]
        Prec=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        Prec=Prec.loc[(Prec.date >= str(y)+'-04-01')&(Prec.date<=str(y)+"-09-30")]
        
# =============================================================================
#          Recuperation ETR flux +SWC
# =============================================================================
#  Lecture file param 

        All_ETR=pd.DataFrame()
        param=pd.read_csv(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
        for r in os.listdir(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"):
            if ".df" in r:
                print(r)
                num_run=r[23:-3]
                df=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"+r,'rb'))
                df["param"]=np.repeat(param.loc[param[0]==int(num_run)][1].values,df.shape[0])
                All_ETR=All_ETR.append(df[["date","ET","id",'param']])
        et=All_ETR.groupby("id")
        # inser loop parcelle Id
        if y =="2017":
            id_CACG=[1,4,5,6,13]
        else:
            id_CACG=[1,5,9,10,12,13]
        for p in id_CACG:
            par1=gro.get_group(p)
            par1.reset_index(inplace=True)
            par1.num_run=pd.to_datetime(par1.num_run,format="%Y-%m-%d")
            df_aqui=pd.merge(df_date_aqui,NDVI.loc[NDVI.id==p],on="date")
            
            
            # # maxIr.replace(0.0,pd.NaT,inplace=True)
            # # minIr.replace(0.0,pd.NaT,inplace=True)
            # # print(mean_run.loc[mean_run['maxZr_1000']!=0.0])
            
            # # Pour ETR
            # paret1=et.get_group(p)
            # paret1.reset_index(inplace=True)
            # paret1.date=pd.to_datetime(paret1.date,format="%Y-%m-%d")
            # paret1=paret1.loc[(paret1.date >= str(y)+"-04-01") &(paret1.date <= str(y)+"-09-30")]
            # # validation
            # par1_val=a.get_group(p)
            # par1_val=par1_val[["Date_irrigation",'Quantite']]
            # par1_val.set_index("Date_irrigation",inplace=True)
            # par1_val_res=par1_val.resample("D").asfreq()
            # par1_val_res.fillna(0.0,inplace=True)
            # par1_val_res["date"]=par1_val_res.index
            # all_res=pd.merge(par1_val_res,mean_run,on=["date"]) # fusion des sim/obs
            # all_res_min=pd.merge(par1_val_res,minIr,on=["date"])
            # all_res_max=pd.merge(par1_val_res,maxIr,on=["date"])
            # all_res_min2=pd.merge(par1_val_res,min2Ir,on=["date"])
            # all_res_max2=pd.merge(par1_val_res,max2Ir,on=["date"])
            # all_res_max3=pd.merge(par1_val_res,max3Ir,on=["date"])
            # all_resu=all_res.replace(0.0,pd.NaT)
            # # print("============")
            # # print("parcelle :%s"%p)
            # # print(all_res.sum())
            # # print("============")
            # #### plot
            # plt.figure(figsize=(7,7))
            # plt.title(p)
            # plt.plot(all_resu.date,all_resu.maxZr_1000,marker="x",linestyle="",label="Simulée")
            # # plt.plot(minIr.date,minIr.maxZr_800,marker="x",linestyle="",label="Simulée_min",alpha=0.5)
            # # plt.plot(maxIr.date,maxIr.maxZr_1200,marker="x",linestyle="",label="Simulée_max",alpha=0.5)
            # plt.plot(all_resu.date,all_resu.Quantite,marker="o",linestyle="",label="Observée")
            # plt.ylim(0,50)
            # plt.ylabel("Irrigation en mm")
            # plt.legend()
            # ax2=plt.twinx(ax=None)
            # ax2.plot(NDVI.loc[NDVI.id==p].date,NDVI.loc[NDVI.id==p].NDVI,color="darkgreen",linestyle="--")
            # ax2.plot(df_aqui.date,df_aqui.NDVI,marker="o",linestyle="")
            # ax2.set_ylabel("NDVI")
            # ax2.set_ylim(0,1)
            # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_Irrigation_%s_%s.png"%(p,y))
            # # print(p)
            # # print(Prec.loc[Prec.id==p].Prec.sum())
            # #  Plot ETR comparer avec flux moyenne 6 years LAM 
            # # plt.figure(figsize=(7,7))
            # # plt.plot(paret1.loc[paret1.param==1000.0]["date"],mean_lam[0].rolling(5).mean(),color='black',label="ETR lam moyenne 6 years")
            # # plt.fill_between(paret1.loc[paret1.param==1000.0]["date"],mean_lam[0].rolling(5).mean()-std_lam[0].rolling(5).mean(),mean_lam[0].rolling(5).mean()+std_lam[0].rolling(5).mean(),facecolor="None",ec='black',linestyle="--",alpha=0.5)
            # # plt.plot(paret1.loc[paret1.param==1000.0]["date"],paret1.loc[paret1.param==1000.0]["ET"].rolling(5).mean(),label='ETR parcelle')
            # # plt.fill_between(paret1.loc[paret1.param==1000.0]["date"],paret1.loc[paret1.param==800.0]["ET"].rolling(5).mean(),paret1.loc[paret1.param==1200.0]["ET"].rolling(5).mean(),alpha=0.5)
            # # plt.legend()
            # # plt.title(p)
            # # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_ETR_dynamiuqe_%s_%s.png"%(p,y))
            
            Vol_tot=Vol_tot.append(par1)
# =============================================================================
# Volumes annuels
# =============================================================================
        Vol_tot.columns=["date"]+param[1].to_list()+["ID"]
        Vol_tot["datejj"]=Vol_tot["date"].dt.strftime("%j")
        Vol_tot["annee"]=Vol_tot["date"].dt.strftime("%Y")
        nb_irr=Vol_tot[Vol_tot!=0.0].groupby("ID").count()
        tot_ID=Vol_tot.groupby("ID").sum()
        tot_IRR=pd.merge(tot_ID,sum_irr_cacg_val,on=["ID"])
        table_RMSE_parcelle=[]
        tab_quant=[]
        for t in Vol_tot.columns[1:-3]:
            for p in tot_IRR.index:
                rmsep = np.sqrt(mean_squared_error(tot_IRR.loc[tot_IRR.index==p]["Quantite"],tot_IRR.loc[tot_IRR.index==p][t]))
                quant=tot_IRR.loc[tot_IRR.index==p][t]
                table_RMSE_parcelle.append([rmsep,t,p,quant])
        table_RMSE_parcelle=pd.DataFrame(table_RMSE_parcelle)
        table_RMSE_parcelle.columns=["RMSE",'p','ID',"Quant"]
        a=table_RMSE_parcelle.groupby(["p",'ID']).min()
        p_value_min=pd.DataFrame(a.unstack()["RMSE"].idxmin())
        value_RMSE=pd.DataFrame(a.unstack()["Quant"])
        for j in tot_IRR.index:
            b=value_RMSE.loc[value_RMSE.index==p_value_min.loc[p_value_min.index==j][0].values[0]][j]
            tab_quant.append([j,b.values[0]])
        vaRMSE=pd.DataFrame(tab_quant,columns=["ID","Quant"])
        tab_f=pd.merge(p_value_min,vaRMSE,on='ID')
        tab_f2=pd.merge(tab_f,sum_irr_cacg_val,on='ID')
        if y =="2018":
            tot_2018=pd.DataFrame(tab_f2.values,columns=["ID","param",'conso','Vali'])
            tot_2018.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_CAGC_mod_2018.csv")
        else:
            tot_2017=pd.DataFrame(tab_f2.values,columns=["ID","param",'conso','Vali'])
            tot_2017.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_CACG_mod_2017.csv")
  
    plt.figure(figsize=(7,7))
    for y in years :
        slope, intercept, r_value, p_value, std_err = stats.linregress(globals()["tot_"+y].Vali.to_list(), globals()["tot_"+y].conso.to_list())
        bias=1/len(globals()["tot_"+y].Vali.to_list())*sum(globals()["tot_"+y].conso-np.mean(globals()["tot_"+y].Vali.to_list())) 
        rms = np.sqrt(mean_squared_error(globals()["tot_"+y].Vali.to_list(), globals()["tot_"+y].conso))
        plt.scatter(globals()["tot_"+y].Vali.to_list(),globals()["tot_"+y].conso,label=y)
        plt.xlim(-10,300)
        plt.ylim(-10,300)
        plt.legend()
        plt.xlabel("Quantité annuelles observés en mm ")
        plt.ylabel("Quantité annuelles modélisés en mm ")
        plt.plot([-10.0, 300], [-10.0,300], 'black', lw=1,linestyle='--')
        if "2017" in y :
            rectangle = plt.Rectangle((95, 245),70,40, ec='blue',fc='blue',alpha=0.1)
            plt.gca().add_patch(rectangle)
            plt.text(100,280,"RMSE = "+str(round(rms,2))) 
            plt.text(100,270,"R² = "+str(round(r_value,2)))
            plt.text(100,260,"Pente = "+str(round(slope,2)))
            plt.text(100,250,"Biais = "+str(round(bias,2)))
        else:
            rectangle = plt.Rectangle((225, 117),70,40, ec='orange',fc='orange',alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.text(230,150,"RMSE = "+str(round(rms,2))) 
            plt.text(230,140,"R² = "+str(round(r_value,2)))
            plt.text(230,130,"Pente = "+str(round(slope,2)))
            plt.text(230,120,"Biais = "+str(round(bias,2)))
        for i in enumerate(set(globals()["tot_"+y].ID)):
            label = i[1]
            plt.annotate(label, # this is the text
                 (globals()["tot_"+y].Vali[i[0]],globals()["tot_"+y].conso[i[0]]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_%s_Irrigation.png"%t)

        
    
