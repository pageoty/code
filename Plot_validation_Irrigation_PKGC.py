# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:47:32 2021

@author: yann 

Validation Irrigation automatique parcelle de référence PKGC
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
    name_run="RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_GSM_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/"
    name_run_save_fig="RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_GSM_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"

    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # label="Init ru année n-1 + Irrigation auto"
    years=["2017"]
    lc="maize_irri"
    soil=["GSM","GSM"]#['SOIL_RIGOU',"RRP_Rigou"]["RRP","RRP_GERS"]
    optim_val="maxZrp"
    
# =============================================================================
#   Mean flux ETR -> Lam corrigées
# =============================================================================
    # All_lam=[]
    # for i in os.listdir(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/"):
    #     if "ETR_maize_irri" in i and "semi" not in i:
    #         ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/"+i,decimal='.',sep=",")
    #         ETR["date"]=pd.to_datetime(ETR["date"],format="%Y-%m-%d")
    #         All_lam.append(ETR.LE_Bowen)
    # lam_et=pd.DataFrame(All_lam)
    # mean_lam=pd.DataFrame(lam_et.mean())
    # std_lam=pd.DataFrame(lam_et.std())
    # mean_lam["date"]=ETR.date

    df_date_aqui=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T30TYP_input_dates_2017.txt",header=None)
    df_date_aqui[0]=pd.to_datetime(df_date_aqui[0],format='%Y%m%d')
    df_date_aqui.columns=["date"]
    Vol_tot=pd.DataFrame()
    Id=pd.DataFrame()
    for y in years: 
        vali_PKGC=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/VOL_TOT_PKGC_"+str(y)+".csv",encoding='latin-1',decimal='.',sep=',',na_values="nan")
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        if len(optim_val) <5 :
            param=pd.read_csv(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/"+optim_val+"/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";") #2
            param.loc[param.shape[0]]='ID' 
        else:
            param=pd.read_csv(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/"+optim_val+"/output_test_maize_irri_param.txt",header=None,skiprows=2,sep=";") #2
            param.loc[param.shape[0]]='ID' 
            param['p']=param[1]+param[2]
        Irri_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_"+str(y)+".csv",index_col=[0,1],skipinitialspace=True)
        gro=Irri_mod.groupby("ID")
        NDVI=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb"))
        NDVI=NDVI.loc[(NDVI.date >= str(y)+'-04-01')&(NDVI.date<=str(y)+"-09-30")]
        Prec=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        Prec=Prec.loc[(Prec.date >= str(y)+'-04-01')&(Prec.date<=str(y)+"-09-30")]
        uts_type=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/"+soil[0]+"/Extract_"+soil[1]+"_parcelle_PKCG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
        data_v=pd.merge(vali_PKGC,uts_type,on="ID")
    

        for p in vali_PKGC.ID:
            par1=gro.get_group(p)
            par1.reset_index(inplace=True)
            par1.num_run=pd.to_datetime(par1.num_run,format="%Y-%m-%d")
            df_aqui=pd.merge(df_date_aqui,NDVI.loc[NDVI.id==p],on="date")
            # maxIr.replace(0.0,pd.NaT,inplace=True)
            # minIr.replace(0.0,pd.NaT,inplace=True)
            # print(mean_run.loc[mean_run['maxZr_1000']!=0.0])
            
            # Pour ETR
            # paret1=et.get_group(p)
            # paret1.reset_index(inplace=True)
            # paret1.date=pd.to_datetime(paret1.date,format="%Y-%m-%d")
            # paret1=paret1.loc[(paret1.date >= str(y)+"-04-01") &(paret1.date <= str(y)+"-09-30")]
            # validation
            # all_res=pd.merge(vali_PKGC,mean_run,on=["ID"]) # fusion des sim/obs
            # all_res_min=pd.merge(vali_PKGC,minIr,on=["ID"])
            # all_res_max=pd.merge(vali_PKGC,maxIr,on=["ID"])
            # all_res_min2=pd.merge(vali_PKGC,min2Ir,on=["ID"])
            # all_res_max2=pd.merge(vali_PKGC,max2Ir,on=["ID"])
            # all_res_max3=pd.merge(vali_PKGC,max3Ir,on=["ID"])
            # all_resu=all_res.replace(0.0,pd.NaT)
            # print("============")
            # print("parcelle :%s"%p)
            # print(all_res.sum())
            # print("============")
            # #### plot
            # plt.figure(figsize=(7,7))
            # plt.title(p)
            # plt.plot(NDVI.loc[NDVI.id==p].date,NDVI.loc[NDVI.id==p].NDVI,color="darkgreen",linestyle="--")
            # plt.plot(df_aqui.date,df_aqui.NDVI,marker="x",linestyle="")
            # plt.ylabel("NDVI")
            # plt.ylim(0,1)
            # ax2=plt.twinx(ax=None)
            # ax2.bar(Prec.loc[Prec.id==p].date,Prec.loc[Prec.id==p].Prec,color="blue")
            # ax2.set_ylim(0,50)
            # ax2.set_ylabel("Irrigation en mm")
            # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_Irrigation_%s_%s.png"%(p,y))
            # #  Plot ETR comparer avec flux moyenne 6 years LAM 
            # plt.figure(figsize=(7,7))
            # plt.plot(paret1.loc[paret1.param==1000.0]["date"],mean_lam[0].rolling(5).mean(),color='black',label="ETR lam moyenne 6 years")
            # plt.fill_between(paret1.loc[paret1.param==1000.0]["date"],mean_lam[0].rolling(5).mean()-std_lam[0].rolling(5).mean(),mean_lam[0].rolling(5).mean()+std_lam[0].rolling(5).mean(),facecolor="None",ec='black',linestyle="--",alpha=0.5)
            # plt.plot(paret1.loc[paret1.param==1000.0]["date"],paret1.loc[paret1.param==1000.0]["ET"].rolling(5).mean(),label='ETR parcelle')
            # plt.fill_between(paret1.loc[paret1.param==1000.0]["date"],paret1.loc[paret1.param==800.0]["ET"].rolling(5).mean(),paret1.loc[paret1.param==1200.0]["ET"].rolling(5).mean(),alpha=0.5)
            # plt.legend()
            # plt.title(p)
            # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_ETR_dynamiuqe_%s_%s.png"%(p,y))
            Vol_tot=Vol_tot.append(par1)

        
# =============================================================================
# Volumes annuels
# =============================================================================
    if len(optim_val) >5:
        Vol_tot.columns=["date"]+param['p'][:-1].to_list()+["ID"]
    else :
        Vol_tot.columns=["date"]+param[1].to_list()
    tot_ID=Vol_tot.groupby("ID").sum()
    tot_IRR=pd.merge(tot_ID,data_v,on=["ID"])
    tot_IRR=tot_IRR[tot_IRR.ID!=10.0]
    tot_IRR.dropna(inplace=True)
    table_RMSE_parcelle=[]
    tab_quant=[]
    labels, index = np.unique(tot_IRR["Classe"], return_inverse=True)
    for t in Vol_tot.columns[1:-1]:
        for p in tot_IRR.ID:
            rmsep = np.sqrt(mean_squared_error(tot_IRR.loc[tot_IRR.ID==p]["MMEAU"],tot_IRR.loc[tot_IRR.ID==p][t]))
            quant=tot_IRR.loc[tot_IRR.ID==p][t]
            table_RMSE_parcelle.append([rmsep,t,p,quant])
    table_RMSE_parcelle=pd.DataFrame(table_RMSE_parcelle)
    table_RMSE_parcelle.columns=["RMSE",'p','ID',"Quant"]
    a=table_RMSE_parcelle.groupby(["p",'ID']).min()
    p_value_min=pd.DataFrame(a.unstack()["RMSE"].idxmin())
    value_RMSE=pd.DataFrame(a.unstack()["Quant"])
    vRMSE=pd.DataFrame(a.unstack()["RMSE"])
    for j in tot_IRR.ID:
        b=value_RMSE.loc[value_RMSE.index==p_value_min.loc[p_value_min.index==j][0].values[0]][j]
        vr=vRMSE.loc[vRMSE.index==p_value_min.loc[p_value_min.index==j][0].values[0]][j]
        tab_quant.append([j,b.values[0],vr.values[0]])
    vaRMSE=pd.DataFrame(tab_quant,columns=["ID","Quant","RMSE"])
    tab_f=pd.merge(p_value_min,vaRMSE,on='ID')
    if soil[0]=="SOIL_RIGOU":
        tab_f2=pd.merge(tab_f,data_v[["ID","MMEAU","ProfRacPot"]],on='ID')
        tab_f2.columns=["ID","maxZr","Quant","RMSE","MMEAU","Prof_rac_UTS"]
    else:
        tab_f2=pd.merge(tab_f,data_v[["ID","MMEAU"]],on='ID')
        tab_f2.columns=["ID","maxZr","Quant","RMSE","MMEAU"]
    tab_f2.drop_duplicates(inplace=True)
    tab_f2.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/Table_RMSE_parcelle_min.csv")
    slope, intercept, r_value, p_value, std_err = stats.linregress(tab_f2.MMEAU.to_list(),tab_f2.Quant.to_list())
    bias=1/tab_f2["MMEAU"].shape[0]*sum(tab_f2.Quant-np.mean(tab_f2.MMEAU)) 
    rms = np.sqrt(mean_squared_error(tab_f2.MMEAU,tab_f2.Quant))
    plt.figure(figsize=(7,7))
    a=plt.scatter(tab_f2.MMEAU,tab_f2.Quant,c=index,cmap='coolwarm')
    plt.legend(a.legend_elements()[0],labels)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité annuelles observées en mm ")
    plt.ylabel("Quantité annuelles modélisées en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 300),72,42, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,330,"RMSE = "+str(round(rms,2))) 
    plt.text(100,320,"R² = "+str(round(r_value,2)))
    plt.text(100,310,"Pente = "+str(round(slope,2)))
    plt.text(100,300,"Biais = "+str(round(bias,2)))
    for i,m in zip(enumerate(tab_f2.ID),tab_f2.maxZr):
        label = int(i[1])
        plt.annotate(label, # this is the text
              (tab_f2["MMEAU"].iloc[i[0]],tab_f2.Quant.iloc[i[0]]), # this is the point to label
              textcoords="offset points", # how to position the text
              xytext=(0,5), # distance from text to points (x,y)
              ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_Irrigation.png")
    
# ===============================================================================
#  Plot résultats optimisation P
# # =============================================================================
    # data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    # param=pd.read_csv(d["PC_disk"]+"//TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    # dfUTS=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    # IRR=[]
    # yerrmore=[]
    # yerrless=[]
    # for i in dfUTS.ID:
    #     maxUTS=data_prof.loc[data_prof.index==i]["ProfRacPot"].values[0]*10 # Si forcage 
    #     # maxUTS=param.loc[param[1].isin(dfUTS.loc[dfUTS.ID==i]["maxZr"])][1].values[0]
    #     # maxUTSFAO=param.loc[param[1].isin(dfUTSFAO.loc[dfUTSFAO.ID==i]["maxZr"])][1]
    #     param2=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/2017/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    #     dfUTSp=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    #     c=param2.loc[param2[1].isin(dfUTSp.loc[dfUTSp.ID==i]["maxZr"])][0]+1
    #     val=param2.loc[param2[1].isin(dfUTSp.loc[dfUTSp.ID==i]["maxZr"])][1]
    #     UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/2017/Output/p/output_test_maize_irri_"+str(int(c))+".df","rb"))
    #     data_id=UTS.groupby("id")
    #     ID_data=data_id.get_group(i)
    #     IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],maxUTS,ID_data.TAW.max()])
    #     # dfmore
    #     param2more=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC//Optim_P/PKGC_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    #     dfUTSpmore=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/Table_RMSE_parcelle_min.csv")
    #     cmore=param2more.loc[param2more[1].isin(dfUTSpmore.loc[dfUTSpmore.ID==i]["maxZr"])][0]+1
    #     valmore=param2more.loc[param2more[1].isin(dfUTSpmore.loc[dfUTSpmore.ID==i]["maxZr"])][1]
    #     UTSmore=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/p/output_test_maize_irri_"+str(int(cmore))+".df","rb"))
    #     data_idmore=UTSmore.groupby("id")
    #     ID_datamore=data_idmore.get_group(i)
    #     yerrmore.append(abs(ID_data.Ir_auto.sum()-ID_datamore.Ir_auto.sum()))
    #       # dfless
    #     param2less=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    #     dfUTSpless=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/Table_RMSE_parcelle_min.csv")
    #     cless=param2less.loc[param2less[1].isin(dfUTSpless.loc[dfUTSpless.ID==i]["maxZr"])][0]+1
    #     valless=param2less.loc[param2less[1].isin(dfUTSpless.loc[dfUTSpless.ID==i]["maxZr"])][1]
    #     UTSless=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/PKGC_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/p/output_test_maize_irri_"+str(int(cless))+".df","rb"))
    #     data_idless=UTSless.groupby("id")
    #     ID_dataless=data_idless.get_group(i)
    #     yerrless.append(abs(ID_dataless.Ir_auto.sum()-ID_data.Ir_auto.sum()))
    # yerr=[yerrless,yerrmore]
    # tab_irr=pd.DataFrame(IRR)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_f2.MMEAU.to_list(),tab_irr[1].to_list())
    # bias=1/tab_f2["MMEAU"].shape[0]*sum(tab_irr[1]-np.mean(tab_f2.MMEAU)) 
    # rms = np.sqrt(mean_squared_error(tab_f2.MMEAU,tab_irr[1]))
    # plt.figure(figsize=(7,7))
    # plt.legend(a.legend_elements()[0],labels)
    # plt.xlim(-10,350)
    # plt.ylim(-10,350)
    # plt.xlabel("Quantité annuelles observées en mm ")
    # plt.ylabel("Quantité annuelles modélisées en mm ")
    # plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # plt.errorbar(tab_f2.MMEAU,tab_irr[1],marker=",",yerr=yerr,fmt='o',elinewidth=0.7,capsize=4)
    # a=plt.scatter(tab_f2.MMEAU,tab_irr[1],c=index,cmap='coolwarm')
    # plt.legend(a.legend_elements()[0],labels)
    # rectangle = plt.Rectangle((95, 300),72,42, ec='blue',fc='blue',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(100,330,"RMSE = "+str(round(rms,2))) 
    # plt.text(100,320,"R² = "+str(round(r_value,2)))
    # plt.text(100,310,"Pente = "+str(round(slope,2)))
    # plt.text(100,300,"Biais = "+str(round(bias,2)))
    # for i,m in zip(enumerate(tab_f2.ID),tab_f2.maxZr):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (tab_f2["MMEAU"].iloc[i[0]],tab_irr[1].iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(0,5), # distance from text to points (x,y)
    #           ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_volumes_Irrigation_post_optim_forcagemaxZr_optim_p_RRP_maps.png")
    
    # ##### plot TAW /RUM
    # tab_irr.columns=["ID","conso","p","maxZr","TAWmax"]
    # test=pd.merge(tab_irr,data_prof,on='ID')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_f2.MMEAU.to_list(),tab_irr.TAWmax.to_list())
    # bias=1/tab_f2["MMEAU"].shape[0]*sum(tab_irr.TAWmax-np.mean(tab_f2.MMEAU)) 
    # rms = np.sqrt(mean_squared_error(tab_f2.MMEAU,tab_irr.TAWmax))
    # plt.figure(figsize=(7,7))
    # a=plt.scatter(test.RUM,test.TAWmax,c=index,cmap="coolwarm")
    # plt.legend(a.legend_elements()[0],labels)
    # plt.xlim(-10,200)
    # plt.ylim(-10,200)
    # plt.xlabel("RUM observée en cm ")
    # plt.ylabel("RUM modélisées en cm ")
    # plt.text(50,165,"RMSE = "+str(round(rms,2))) 
    # plt.text(50,160,"R² = "+str(round(r_value,2)))
    # plt.text(50,155,"Pente = "+str(round(slope,2)))
    # plt.text(50,150,"Biais = "+str(round(bias,2)))
    # plt.plot([-10.0, 200], [-10.0,200], 'black', lw=1,linestyle='--')
    # for i in enumerate(test.classe):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (test["RUM"].iloc[i[0]],test.TAWmax.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_RUM_post_optim_forcagemaxZr_optim_p_RRP_maps.png")
# =============================================================================
#     PFCC GSM pédotransfert
# =============================================================================
    # data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj2.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    # param=pd.read_csv(d["PC_disk"]+"//TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    # dfUTS=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    # IRR=[]
    # yerrmore=[]
    # yerrless=[]
    # for i in dfUTS.ID:
    #     maxUTS=data_prof.loc[data_prof.index==i]["ProfRacPot"].values[0]*10 # Si forcage 
    #     # maxUTS=param.loc[param[1].isin(dfUTS.loc[dfUTS.ID==i]["maxZr"])][1].values[0]
    #     # maxUTSFAO=param.loc[param[1].isin(dfUTSFAO.loc[dfUTSFAO.ID==i]["maxZr"])][1]
    #     param2=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_GSM/2017/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    #     dfUTSp=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_GSM/Table_RMSE_parcelle_min.csv")
    #     c=param2.loc[param2[1].isin(dfUTSp.loc[dfUTSp.ID==i]["maxZr"])][0]+1
    #     val=param2.loc[param2[1].isin(dfUTSp.loc[dfUTSp.ID==i]["maxZr"])][1]
    #     UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_GSM/2017/Output/p/output_test_maize_irri_"+str(int(c))+".df","rb"))
    #     data_id=UTS.groupby("id")
    #     ID_data=data_id.get_group(i)
    #     IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],maxUTS,ID_data.TAW.max()])
    #     # # dfmore
    #     # param2more=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC//Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    #     # dfUTSpmore=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/Table_RMSE_parcelle_min.csv")
    #     # cmore=param2more.loc[param2more[1].isin(dfUTSpmore.loc[dfUTSpmore.ID==i]["maxZr"])][0]+1
    #     # valmore=param2more.loc[param2more[1].isin(dfUTSpmore.loc[dfUTSpmore.ID==i]["maxZr"])][1]
    #     # UTSmore=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/p/output_test_maize_irri_"+str(int(cmore))+".df","rb"))
    #     # data_idmore=UTSmore.groupby("id")
    #     # ID_datamore=data_idmore.get_group(i)
    #     # yerrmore.append(abs(ID_data.Ir_auto.sum()-ID_datamore.Ir_auto.sum()))
    #     #   # dfless
    #     # param2less=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    #     # dfUTSpless=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/Table_RMSE_parcelle_min.csv")
    #     # cless=param2less.loc[param2less[1].isin(dfUTSpless.loc[dfUTSpless.ID==i]["maxZr"])][0]+1
    #     # valless=param2less.loc[param2less[1].isin(dfUTSpless.loc[dfUTSpless.ID==i]["maxZr"])][1]
    #     # UTSless=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Optim_P/GSM_PFCC/PKGC_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/p/output_test_maize_irri_"+str(int(cless))+".df","rb"))
    #     # data_idless=UTSless.groupby("id")
    #     # ID_dataless=data_idless.get_group(i)
    #     # yerrless.append(abs(ID_dataless.Ir_auto.sum()-ID_data.Ir_auto.sum()))
    # # yerr=[yerrless,yerrmore]
    # tab_irr_GSM=pd.DataFrame(IRR)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_f2.MMEAU.to_list(),tab_irr_GSM[1].to_list())
    # bias=1/tab_f2["MMEAU"].shape[0]*sum(tab_irr_GSM[1]-np.mean(tab_f2.MMEAU)) 
    # rms = np.sqrt(mean_squared_error(tab_f2.MMEAU,tab_irr_GSM[1]))
    # plt.figure(figsize=(7,7))
    # plt.legend(a.legend_elements()[0],labels)
    # plt.xlim(-10,350)
    # plt.ylim(-10,350)
    # plt.xlabel("Quantité annuelles observées en mm ")
    # plt.ylabel("Quantité annuelles modélisées en mm ")
    # plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # a=plt.scatter(tab_f2.MMEAU,tab_irr_GSM[1],c=index,cmap='coolwarm')
    # plt.legend(a.legend_elements()[0],labels)
    # rectangle = plt.Rectangle((95, 300),72,42, ec='blue',fc='blue',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(100,330,"RMSE = "+str(round(rms,2))) 
    # plt.text(100,320,"R² = "+str(round(r_value,2)))
    # plt.text(100,310,"Pente = "+str(round(slope,2)))
    # plt.text(100,300,"Biais = "+str(round(bias,2)))
    # for i,m in zip(enumerate(tab_f2.ID),tab_f2.maxZr):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (tab_f2["MMEAU"].iloc[i[0]],tab_irr_GSM[1].iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(0,5), # distance from text to points (x,y)
    #           ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_volumes_Irrigation_post_optim_forcagemaxZr_optim_p_UTS_maps_GSM_PFCC.png")
   
    # # Plot TAW/ RUM
    # tab_irr_GSM.columns=["ID","conso","p","maxZr","TAWmax"]
    # test=pd.merge(tab_irr_GSM,data_prof,on='ID')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_f2.MMEAU.to_list(),tab_irr_GSM.TAWmax.to_list())
    # bias=1/tab_f2["MMEAU"].shape[0]*sum(tab_irr_GSM.TAWmax-np.mean(tab_f2.MMEAU)) 
    # rms = np.sqrt(mean_squared_error(tab_f2.MMEAU,tab_irr_GSM.TAWmax))
    # plt.figure(figsize=(7,7))
    # a=plt.scatter(test.RUM,test.TAWmax,c=index,cmap="coolwarm")
    # plt.legend(a.legend_elements()[0],labels)
    # plt.xlim(-10,200)
    # plt.ylim(-10,200)
    # plt.xlabel("RUM observée en cm ")
    # plt.ylabel("RUM modélisées en cm ")
    # plt.plot([-10.0, 200], [-10.0,200], 'black', lw=1,linestyle='--')
    # plt.text(50,165,"RMSE = "+str(round(rms,2))) 
    # plt.text(50,160,"R² = "+str(round(r_value,2)))
    # plt.text(50,155,"Pente = "+str(round(slope,2)))
    # plt.text(50,150,"Biais = "+str(round(bias,2)))
    # for i in enumerate(test.classe):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (test["RUM"].iloc[i[0]],test.TAWmax.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_RUM_post_optim_forcagemaxZr_optim_p_UTS_maps_GSM_PFCC.png")
# =============================================================================
#     Forcer maxZr et p
# =============================================================================
    # data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    # dfUTS=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    # IRR=[]
    # yerrmin=[]
    # yerrmax=[]
    # for i in dfUTS.ID:
    #     maxUTS=data_prof.loc[data_prof.index==i]["ProfRacPot"].values[0]*10
    #     UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Sans_optim/PKGC_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/2017/output_test_2017.df","rb"))
    #     data_id=UTS.groupby("id")
    #     ID_data=data_id.get_group(i)
    #     print(r'ID == %s ==> TAW == %s'%(i,max(round(ID_data.TAW,2))))
    #     IRR.append([i,ID_data.Ir_auto.sum(),maxUTS,ID_data.TAW.max()])
    #     # dfmore#
    #     UTSmore=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Sans_optim/PKGC_init_ru_optim_P055_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/2017//output_test_2017.df","rb"))
    #     data_idmore=UTSmore.groupby("id")
    #     ID_datamore=data_idmore.get_group(i)
    #     yerrmax.append(abs(ID_data.Ir_auto.sum()-ID_datamore.Ir_auto.sum()))
    #     # dfless
    #     UTSless=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Sans_optim/PKGC_init_ru_optim_P055_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/2017/output_test_2017.df","rb"))
    #     data_idless=UTSless.groupby("id")
    #     ID_dataless=data_idless.get_group(i)
    #     yerrmin.append(abs(ID_dataless.Ir_auto.sum()-ID_data.Ir_auto.sum()))
    # yerr=[yerrmin,yerrmax]
    # tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWmax"])
    # tab_irr2=pd.merge(tab_irr,dfUTS[["ID","MMEAU"]],on='ID')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_irr2.MMEAU.to_list(),tab_irr2.conso.to_list())
    # bias=1/tab_irr2["MMEAU"].shape[0]*sum(tab_irr2.conso-np.mean(tab_irr2.MMEAU)) 
    # rms = np.sqrt(mean_squared_error(tab_irr2.MMEAU,tab_irr2.conso))
    # plt.figure(figsize=(7,7))
    # plt.xlim(-10,350)
    # plt.ylim(-10,350)
    # plt.xlabel("Quantité annuelles observées en mm ")
    # plt.ylabel("Quantité annuelles modélisées en mm ")
    # plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # plt.errorbar(tab_irr2.MMEAU,tab_irr2.conso,marker=',',yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    # rectangle = plt.Rectangle((95, 265),70,45, ec='blue',fc='blue',alpha=0.1)
    # a=plt.scatter(tab_irr2.MMEAU,tab_irr2.conso,c=index,cmap='coolwarm')
    # plt.legend(a.legend_elements()[0],labels)
    # plt.gca().add_patch(rectangle)
    # plt.text(100,300,"RMSE = "+str(round(rms,2))) 
    # plt.text(100,290,"R² = "+str(round(r_value,2)))
    # plt.text(100,280,"Pente = "+str(round(slope,2)))
    # plt.text(100,270,"Biais = "+str(round(bias,2)))
    # for i in enumerate(dfUTS.ID):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (tab_irr2["MMEAU"].iloc[i[0]],tab_irr2.conso.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    # # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_volumes_Irrigation_post_optim_forcagemaxZr_et_p_UTS_maps.png")

# =============================================================================
# résul pour maxzr 600 (median maize 2017 et 2018 sur UTS maps + 055 p)
# ============================================================================= 
    Valid=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/PKGC_init_ru_optim_Fcover_fewi_De_Kr_days10_dose30_500_800_irri_auto_soil/Table_RMSE_parcelle_min.csv")
    df_mod=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Sans_optim/PKGC_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_1000_irri_auto_soil/2017/output_test_2017.df","rb"))
    mod_irr=df_mod.groupby("id")["Ir_auto"].sum()
    mod_Irr=pd.DataFrame(mod_irr)
    mod_Irr = mod_Irr[mod_irr.index!=10]
    slope, intercept, r_value, p_value, std_err = stats.linregress(Valid.MMEAU.to_list(),mod_Irr.Ir_auto.to_list())
    bias=1/Valid["MMEAU"].shape[0]*sum(mod_Irr.Ir_auto-np.mean(Valid.MMEAU)) 
    rms = np.sqrt(mean_squared_error(Valid.MMEAU,mod_Irr.Ir_auto))
    plt.figure(figsize=(7,7))
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité annuelles observées en mm ")
    plt.ylabel("Quantité annuelles modélisées en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 265),70,45, ec='blue',fc='blue',alpha=0.1)
    a=plt.scatter(Valid.MMEAU,mod_Irr.Ir_auto,c=index,cmap='coolwarm')
    plt.legend(a.legend_elements()[0],labels)
    plt.gca().add_patch(rectangle)
    plt.text(100,300,"RMSE = "+str(round(rms,2))) 
    plt.text(100,290,"R² = "+str(round(r_value,2)))
    plt.text(100,280,"Pente = "+str(round(slope,2)))
    plt.text(100,270,"Biais = "+str(round(bias,2)))
    for i in enumerate(Valid.ID):
        label = int(i[1])
        plt.annotate(label, # this is the text
              (Valid["MMEAU"].iloc[i[0]],mod_Irr.Ir_auto.iloc[i[0]]), # this is the point to label
              textcoords="offset points", # how to position the text
              xytext=(-6,2), # distance from text to points (x,y)
              ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_volumes_Irrigation_post_optim_foracgemaxZr897_et_p062_RRP_maps.png")
