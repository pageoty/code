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
    name_run="RUNS_SAMIR/RUN_CACG/CACG_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/"
    name_run_save_fig="RUNS_SAMIR/RUN_CACG/CACG_GSM_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_1800_irri_auto_soil/"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # label="Init ru année n-1 + Irrigation auto"
    years=["2017"]
    lc="maize_irri"
    optim_val="maxZr"
    
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
        data_soil=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_2017_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
        Vol_tot=pd.DataFrame()
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        df_date_aqui=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T30TYP_input_dates_"+str(y)+".txt",header=None)
        df_date_aqui[0]=pd.to_datetime(df_date_aqui[0],format='%Y%m%d')
        df_date_aqui.columns=["date"]
        vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/DATE_DOES_CACG_"+str(y)+".csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
        vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%Y')
        vali_cacg["Quantite"].astype(float)
        sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum()
        nb_irr=vali_cacg.groupby("ID")["Date_irrigation"].count()
        a=vali_cacg.groupby("ID")
        if len(optim_val) <=5 :
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
        
# =============================================================================
#          Recuperation ETR flux +SWC
# =============================================================================
#  Lecture file param 

        # All_ETR=pd.DataFrame()
        # for r in os.listdir(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"):
        #     if ".df" in r:
        #         print(r)
        #         num_run=r[23:-3]
        #         df=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"+r,'rb'))
        #         df["param"]=np.repeat(param.loc[param[0]==int(num_run)][1].values,df.shape[0])
        #         All_ETR=All_ETR.append(df[["date","ET","id",'param']])
        # et=All_ETR.groupby("id")
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
        if len(optim_val) >5:
            Vol_tot.columns=["date"]+param['p'][:-1].to_list()+["ID"]
        else :
            Vol_tot.columns=["date"]+param[1].to_list()
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
        vRMSE=pd.DataFrame(a.unstack()["RMSE"])
        for j in tot_IRR.index:
            b=value_RMSE.loc[value_RMSE.index==p_value_min.loc[p_value_min.index==j][0].values[0]][j]
            vr=vRMSE.loc[vRMSE.index==p_value_min.loc[p_value_min.index==j][0].values[0]][j]
            tab_quant.append([j,b.values[0],vr.values[0]])
        vaRMSE=pd.DataFrame(tab_quant,columns=["ID","Quant","RMSE"])
        tab_f=pd.merge(p_value_min,vaRMSE,on='ID')
        tab_f2=pd.merge(tab_f,sum_irr_cacg_val,on='ID')
        if y =="2018":
            tot_2018=pd.DataFrame(tab_f2.values,columns=["ID","param",'conso','RMSE','Vali'])
            tot_2018.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_CACG_mod_2018.csv")
        else:
            tot_2017=pd.DataFrame(tab_f2.values,columns=["ID","param",'conso','RMSE','Vali'])
            tot_2017_v2=pd.merge(tot_2017,data_soil[["ProfRacPot","RUM","CC_mean",'PF_mean']],on='ID')
            tot_2017.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_CACG_mod_2017.csv")

    plt.figure(figsize=(7,7))
    for y in years :
          # read file var more and less value 
        # dfless=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run[:51]+"_m20"+name_run[51:-1]+"_varmo20/tab_CACG_mod_"+str(y)+".csv") # if UTS 51 , RRP et GSM 55 
        # dfmore=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run[:51]+"_pl20"+name_run[51:-1]+"_varplus20/tab_CACG_mod_"+str(y)+".csv")
        # yerrmin=abs(dfless.conso- globals()["tot_"+y].conso)
        # yerrmax=abs(dfmore.conso- globals()["tot_"+y].conso)
        # yerr=[yerrmin.to_list(),yerrmax.to_list()]
        #  calcul des statistiques
        slope, intercept, r_value, p_value, std_err = stats.linregress(globals()["tot_"+y].Vali.to_list(), globals()["tot_"+y].conso.to_list())
        bias=1/len(globals()["tot_"+y].Vali.to_list())*sum(globals()["tot_"+y].conso-np.mean(globals()["tot_"+y].Vali.to_list())) 
        rms = np.sqrt(mean_squared_error(globals()["tot_"+y].Vali.to_list(), globals()["tot_"+y].conso))
        plt.scatter(globals()["tot_"+y].Vali.to_list(),globals()["tot_"+y].conso,label=y)
        plt.xlim(-10,350)
        plt.ylim(-10,350)
        plt.legend()
        plt.xlabel("Quantité annuelles observés en mm ")
        plt.ylabel("Quantité annuelles modélisés en mm ")
        plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
        #  add error bar in scatter plot
        # plt.errorbar(globals()["tot_"+y].Vali,globals()["tot_"+y].conso,yerr=yerr,fmt='o',elinewidth=0.7,capsize =4)
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
            label = int(i[1])
            plt.annotate(label, # this is the text
                  (globals()["tot_"+y].Vali[i[0]],globals()["tot_"+y].conso[i[0]]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,5), # distance from text to points (x,y)
                  ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_optimiser_avec_incertitude_Irrigation.png")
    
    tot_2017_v2["TAWmax"]=tot_2017_v2.eval("(CC_mean-PF_mean)*param")
    tot_2017_v2.param=tot_2017_v2.param/10
    slope, intercept, r_value, p_value, std_err = stats.linregress(tot_2017_v2.RUM.to_list(),tot_2017_v2.TAWmax.to_list())
    bias=1/tot_2017_v2["RUM"].shape[0]*sum(tot_2017_v2.TAWmax-np.mean(tot_2017_v2.RUM)) 
    rms = np.sqrt(mean_squared_error(tot_2017_v2.RUM,tot_2017_v2.TAWmax))
    slopeZr, intercept, r_valueZr, p_value, std_err = stats.linregress(tot_2017_v2.ProfRacPot.to_list(),tot_2017_v2.param.to_list())
    biasZr=1/tot_2017_v2["ProfRacPot"].shape[0]*sum(tot_2017_v2.param-np.mean(tot_2017_v2.ProfRacPot)) 
    rmsZr = np.sqrt(mean_squared_error(tot_2017_v2.ProfRacPot,tot_2017_v2.param))
    plt.figure(figsize=(7,7))
    a=plt.scatter(tot_2017_v2.RUM,tot_2017_v2.TAWmax,color='r',label='RUM')
    plt.scatter(tot_2017_v2.ProfRacPot,tot_2017_v2.param,color='b',label="MaxZr")
    plt.xlim(-10,300)
    plt.ylim(-10,300)
    plt.xlabel("RUM et maxZr observée en cm ")
    plt.ylabel("RUM et maxZr modélisées en cm ")
    rectangle = plt.Rectangle((190, 145),70,45, ec='r',fc='r',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(200,180,"RMSE = "+str(round(rms,2))) 
    plt.text(200,170,"R² = "+str(round(r_value,2)))
    plt.text(200,160,"Pente = "+str(round(slope,2)))
    plt.text(200,150,"Biais = "+str(round(bias,2)))
    rectangle = plt.Rectangle((45, 195),70,45, ec='b',fc='b',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(50,230,"RMSE = "+str(round(rmsZr,2))) 
    plt.text(50,220,"R² = "+str(round(r_valueZr,2)))
    plt.text(50,210,"Pente = "+str(round(slopeZr,2)))
    plt.text(50,200,"Biais = "+str(round(biasZr,2)))
    plt.plot([-10.0, 300], [-10.0,300], 'black', lw=1,linestyle='--')
    plt.legend()
    for i in enumerate(tot_2017_v2.ID):
        label = int(i[1])
        plt.annotate(label, # this is the text
              (tot_2017_v2["RUM"].iloc[i[0]],tot_2017_v2.TAWmax.iloc[i[0]]), # this is the point to label
              textcoords="offset points", # how to position the text
              xytext=(-6,2), # distance from text to points (x,y)
              ha='center')
    for i in enumerate(tot_2017_v2.ID):
       label = int(i[1])
       plt.annotate(label, # this is the text
             (tot_2017_v2["ProfRacPot"].iloc[i[0]],tot_2017_v2.param.iloc[i[0]]), # this is the point to label
             textcoords="offset points", # how to position the text
             xytext=(-6,2), # distance from text to points (x,y)
             ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_RUm_TAW.png")

# =============================================================================
#     Optim P et forcer maxZr
# =============================================================================
   #  plt.figure(figsize=(7,7))
   #  for y in years :
   #      data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal='.')
   #      IRR=[]
   #      yerrmore=[]
   #      yerrless=[]
   #      if y =="2017":
   #          id_CACG=[1,4,5,6,13]
   #      else:
   #          id_CACG=[1,5,9,10,12,13]
   #      Vali_TAW=data_prof.loc[data_prof.index.isin(id_CACG)]["RUM"]
   #      for i in id_CACG:
            
   #          maxUTS=data_prof.loc[data_prof.index==i]["ProfRacPot"].values[0] # Si forcage 
   #          maxUTS=int(float(maxUTS)*10)
   #          param2=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/"+str(y)+"/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
   #          dfUTSp=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/tab_CACG_mod_"+str(y)+".csv")
   #          c=param2.loc[param2[1].isin(dfUTSp.loc[dfUTSp.ID==i]["param"])][0]+1
   #          val=param2.loc[param2[1].isin(dfUTSp.loc[dfUTSp.ID==i]["param"])][1]
   #          UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/"+str(y)+"/Output/p/output_test_maize_irri_"+str(int(c))+".df","rb"))
   #          data_id=UTS.groupby("id")
   #          ID_data=data_id.get_group(i)
   #          # print(r'ID == %s ==> RAW == %s'%(i,max(round(ID_data.TAW*val.values[0],2))))
   #          IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],maxUTS,ID_data.TAW.max()])
   #          # dfmore
   #          param2more=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
   #          dfUTSpmore=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/tab_CACG_mod_"+str(y)+".csv")
   #          cmore=param2more.loc[param2more[1].isin(dfUTSpmore.loc[dfUTSpmore.ID==i]["param"])][0]+1
   #          valmore=param2more.loc[param2more[1].isin(dfUTSpmore.loc[dfUTSpmore.ID==i]["param"])][1]
   #          UTSmore=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/p/output_test_maize_irri_"+str(int(cmore))+".df","rb"))
   #          data_idmore=UTSmore.groupby("id")
   #          ID_datamore=data_idmore.get_group(i)
   #          yerrmore.append(abs(ID_data.Ir_auto.sum()-ID_datamore.Ir_auto.sum()))
   #            # dfless
   #          param2less=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/p/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
   #          dfUTSpless=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/tab_CACG_mod_"+str(y)+".csv")
   #          cless=param2less.loc[param2less[1].isin(dfUTSpless.loc[dfUTSpless.ID==i]["param"])][0]+1
   #          valless=param2less.loc[param2less[1].isin(dfUTSpless.loc[dfUTSpless.ID==i]["param"])][1]
   #          UTSless=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P0407_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/p/output_test_maize_irri_"+str(int(cless))+".df","rb"))
   #          data_idless=UTSless.groupby("id")
   #          ID_dataless=data_idless.get_group(i)
   #          yerrless.append(abs(ID_dataless.Ir_auto.sum()-ID_data.Ir_auto.sum()))
   #      yerr=[yerrless,yerrmore]
   #      tab_irr=pd.DataFrame(IRR)
   #      slope, intercept, r_value, p_value, std_err = stats.linregress(dfUTSp.Vali.to_list(),tab_irr[1].to_list())
   #      bias=1/dfUTSp["Vali"].shape[0]*sum(tab_irr[1]-np.mean(dfUTSp.Vali)) 
   #      rms = np.sqrt(mean_squared_error(dfUTSp.Vali,tab_irr[1]))
   #      a=plt.scatter(dfUTSp.Vali,tab_irr[1],label=y)
   #      plt.legend()
   #      plt.xlim(-10,350)
   #      plt.ylim(-10,350)
   #      plt.xlabel("Quantité annuelles observées en mm ")
   #      plt.ylabel("Quantité annuelles modélisées en mm ")
   #      plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
   #      plt.errorbar(dfUTSp.Vali,tab_irr[1],yerr=yerr,fmt='o',elinewidth=0.7,capsize=4)
   #      if "2017" in y :
   #          rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
   #          plt.gca().add_patch(rectangle)
   #          plt.text(100,280,"RMSE = "+str(round(rms,2))) 
   #          plt.text(100,270,"R² = "+str(round(r_value,2)))
   #          plt.text(100,260,"Pente = "+str(round(slope,2)))
   #          plt.text(100,250,"Biais = "+str(round(bias,2)))
   #      else:
   #          rectangle = plt.Rectangle((225, 117),70,45, ec='orange',fc='orange',alpha=0.3)
   #          plt.gca().add_patch(rectangle)
   #          plt.text(230,150,"RMSE = "+str(round(rms,2))) 
   #          plt.text(230,140,"R² = "+str(round(r_value,2)))
   #          plt.text(230,130,"Pente = "+str(round(slope,2)))
   #          plt.text(230,120,"Biais = "+str(round(bias,2)))
   #      for i,m in zip(enumerate(dfUTSp.ID),dfUTSp.param):
   #          label = int(i[1])
   #          plt.annotate(label, # this is the text
   #                (dfUTSp["Vali"].iloc[i[0]],tab_irr[1].iloc[i[0]]), # this is the point to label
   #                textcoords="offset points", # how to position the text
   #                xytext=(0,5), # distance from text to points (x,y)
   #                ha='center')
   #  plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_post_optim_forcagemaxZr_optim_p.png")
    
   # #  Validation TAW RUM$
   #  plt.figure(figsize=(7,7))
   #  plt.scatter(Vali_TAW.astype(float),tab_irr[4])
   #  plt.xlim(0,200)
   #  plt.ylim(0,200)
   #  plt.xlabel("RUM observées en mm ")
   #  plt.ylabel("RUM modélisées en mm ")
   #  plt.plot([0.0, 200], [0.0,200], 'black', lw=1,linestyle='--')
   #  for i in enumerate(tab_irr[0]):
   #          label = int(i[1])
   #          plt.annotate(label, # this is the text
   #                (Vali_TAW.astype(float).iloc[i[0]],tab_irr[4].iloc[i[0]]), # this is the point to label
   #                textcoords="offset points", # how to position the text
   #                xytext=(0,5), # distance from text to points (x,y)
   #                ha='center')
   # =============================================================================
#    forcer maxZr avec Depth GSM
# =============================================================================
    plt.figure(figsize=(7,7))
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal='.')
    depth_GSM=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_CACG_2017_soil_depth.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    param=pd.read_csv(d["PC_disk"]+"//TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    dfUTS=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/tab_CACG_mod_2017.csv")
    IRR=[]
    IRR=[]
    yerrmore=[]
    yerrless=[]
    id_CACG=[1,4,5,6,13]
    Vali_TAW=data_prof.loc[data_prof.index.isin(id_CACG)]["RUM"]
    for i in id_CACG:
        c=param.loc[param[1].isin(depth_GSM.loc[depth_GSM.index==i]["mean_arrondi"])][0]
        val=param.loc[param[1].isin(depth_GSM.loc[depth_GSM.index==i]["mean_arrondi"])][1]
        UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_400_2500_irri_auto_soil/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        data_id=UTS.groupby("id")
        ID_data=data_id.get_group(i)
        # print(r'ID == %s ==> RAW == %s'%(i,max(round(ID_data.TAW*val.values[0],2))))
        IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],ID_data.TAW.max()])
        # dfmore
    tab_irr=pd.DataFrame(IRR)
    slope, intercept, r_value, p_value, std_err = stats.linregress(dfUTS.Vali.to_list(),tab_irr[1].to_list())
    bias=1/dfUTS["Vali"].shape[0]*sum(tab_irr[1]-np.mean(dfUTS.Vali)) 
    rms = np.sqrt(mean_squared_error(dfUTS.Vali,tab_irr[1]))
    a=plt.scatter(dfUTS.Vali,tab_irr[1],label=y)
    plt.legend()
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité annuelles observées en mm ")
    plt.ylabel("Quantité annuelles modélisées en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    for i in enumerate(dfUTS.ID):
        label = int(i[1])
        plt.annotate(label, # this is the text
              (dfUTS["Vali"].iloc[i[0]],tab_irr[1].iloc[i[0]]), # this is the point to label
              textcoords="offset points", # how to position the text
              xytext=(0,5), # distance from text to points (x,y)
              ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_post_optim_forcagemaxZr_p_depthGSM.png")
 
#  Validation TAW RUM$
    plt.figure(figsize=(7,7))
    plt.scatter(Vali_TAW.astype(float),tab_irr[3])
    plt.xlim(0,300)
    plt.ylim(0,300)
    plt.xlabel("RUM observées en mm ")
    plt.ylabel("RUM modélisées en mm ")
    plt.plot([0.0, 300], [0.0,300], 'black', lw=1,linestyle='--')
    for i in enumerate(tab_irr[0]):
            label = int(i[1])
            plt.annotate(label, # this is the text
                  (Vali_TAW.astype(float).iloc[i[0]],tab_irr[3].iloc[i[0]]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,5), # distance from text to points (x,y)
                  ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_RUM_optim_forcagemaxZr_p_depth_GSM.png")
    plt.figure(figsize=(7,7))
    plt.hist( tab_irr[3],label="RUM_modèle",ec="black",bins=10,linestyle="--")
    plt.hist(Vali_TAW.astype(float),label="RUM_obs",alpha=0.7,ec="black",bins=10)
    plt.xlabel("Valeur des RUM")
    plt.legend()
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_histo_RUM_forcagemaxZr_p_depth_GSM.png")
# =============================================================================
#     forcage P et forcer maxZr
# =============================================================================
    # plt.figure(figsize=(7,7))
    # for y in years :
    #     data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
    #     IRR=[]
    #     yerrmin=[]
    #     yerrmax=[]
    #     vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/DATE_DOES_CACG_"+str(y)+".csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
    #     vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%Y')
    #     vali_cacg["Quantite"].astype(float)
    #     sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum()
    #     if y =="2017":
    #         id_CACG=[1,4,5,6,13]
    #     else:
    #         id_CACG=[1,5,9,10,13,12]
    #     for i in id_CACG:
    #         maxUTS=data_prof.loc[data_prof.index==i]["ProfRacPot"].values[0] # Si forcage 
    #         maxUTS=int(float(maxUTS)*10)
    #         UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Sans_optim/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/"+str(y)+"/Output/output_test_"+str(y)+".df","rb"))
    #         data_id=UTS.groupby("id")
    #         ID_data=data_id.get_group(i)
    #         print(r'ID == %s ==> RAW == %s'%(i,max(round(ID_data.RAW,2))))
    #         IRR.append([i,ID_data.Ir_auto.sum(),maxUTS,ID_data.TAW.max()])
    #         # dfmore
    #         UTSmore=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Sans_optim/CACG_init_ru_optim_P055_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/output_test_"+str(y)+".df","rb"))
    #         data_idmore=UTSmore.groupby("id")
    #         ID_datamore=data_idmore.get_group(i)
    #         yerrmax.append(abs(ID_data.Ir_auto.sum()-ID_datamore.Ir_auto.sum()))
    #         # dfless
    #         UTSless=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Sans_optim/CACG_init_ru_optim_P055_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/output_test_"+str(y)+".df","rb"))
    #         data_idless=UTSless.groupby("id")
    #         ID_dataless=data_idless.get_group(i)
    #         yerrmin.append(abs(ID_dataless.Ir_auto.sum()-ID_data.Ir_auto.sum()))
    #     yerr=[yerrmin,yerrmax]
    #     tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
    #     vali_RUM=pd.merge(tab_irr,data_prof["RUM"],on="ID")
    #     tab_irr2=pd.merge(tab_irr,sum_irr_cacg_val,on='ID')
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(tab_irr2.Quantite.to_list(),tab_irr2.conso.to_list())
    #     bias=1/tab_irr2["Quantite"].shape[0]*sum(tab_irr2.conso-np.mean(tab_irr2.Quantite)) 
    #     rms = np.sqrt(mean_squared_error(tab_irr2.Quantite,tab_irr2.conso))
    #     plt.scatter(tab_irr2.Quantite,tab_irr2.conso,label=y)
    #     plt.legend()
    #     plt.xlim(-10,350)
    #     plt.ylim(-10,350)
    #     plt.xlabel("Quantité annuelles observées en mm ")
    #     plt.ylabel("Quantité annuelles modélisées en mm ")
    #     plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    #     plt.errorbar(tab_irr2.Quantite,tab_irr2.conso,yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    #     if "2017" in y :
    #         rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    #         plt.gca().add_patch(rectangle)
    #         plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    #         plt.text(100,270,"R² = "+str(round(r_value,2)))
    #         plt.text(100,260,"Pente = "+str(round(slope,2)))
    #         plt.text(100,250,"Biais = "+str(round(bias,2)))
    #     else:
    #         rectangle = plt.Rectangle((225, 117),70,45, ec='orange',fc='orange',alpha=0.3)
    #         plt.gca().add_patch(rectangle)
    #         plt.text(230,150,"RMSE = "+str(round(rms,2))) 
    #         plt.text(230,140,"R² = "+str(round(r_value,2)))
    #         plt.text(230,130,"Pente = "+str(round(slope,2)))
    #         plt.text(230,120,"Biais = "+str(round(bias,2)))
    #     for i in enumerate(id_CACG):
    #         label = int(i[1])
    #         plt.annotate(label, # this is the text
    #               (tab_irr2["Quantite"].iloc[i[0]],tab_irr2.conso.iloc[i[0]]), # this is the point to label
    #               textcoords="offset points", # how to position the text
    #               xytext=(-6,2), # distance from text to points (x,y)
    #               ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_post_forcagemaxZr_p.png")
    # # TAW et RUM
    # plt.figure(figsize=(7,7))
    # slope, intercept, r_value, p_value, std_err = stats.linregress(vali_RUM.RUM.to_list(),tab_irr.TAWMax.to_list())
    # bias=1/vali_RUM["RUM"].shape[0]*sum(tab_irr.TAWMax-np.mean(vali_RUM.RUM)) 
    # rms = np.sqrt(mean_squared_error(vali_RUM.RUM,tab_irr.TAWMax))
    # plt.scatter(vali_RUM.RUM,tab_irr["TAWMax"])
    # plt.xlim(0,200)
    # plt.ylim(0,200)
    # plt.xlabel("RUM observées en mm ")
    # plt.ylabel("RUM modélisées en mm ")
    # rectangle = plt.Rectangle((45, 145),45,30, ec='b',fc='b',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(50,165,"RMSE = "+str(round(rms,2))) 
    # plt.text(50,160,"R² = "+str(round(r_value,2)))
    # plt.text(50,155,"Pente = "+str(round(slope,2)))
    # plt.text(50,150,"Biais = "+str(round(bias,2)))
    # plt.plot([0.0, 200], [0.0,200], 'black', lw=1,linestyle='--')
    # for i in enumerate(tab_irr.ID):
    #         label = int(i[1])
    #         plt.annotate(label, # this is the text
    #               (vali_RUM.RUM.iloc[i[0]],tab_irr["TAWMax"].iloc[i[0]]), # this is the point to label
    #               textcoords="offset points", # how to position the text
    #               xytext=(0,5), # distance from text to points (x,y)
    #               ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_RUM_post_forcagemaxZr_p.png")
    
# =============================================================================
#     Forcage p et maxZr avec la RUM
# =============================================================================
    plt.figure(figsize=(7,7))
    for y in years :
        data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
        IRR=[]
        yerrmin=[]
        yerrmax=[]
        vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/DATE_DOES_CACG_"+str(y)+".csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
        vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%Y')
        vali_cacg["Quantite"].astype(float)
        sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum()
        if y =="2017":
            id_CACG=[1,4,5,6,13]
        else:
            id_CACG=[1,5,9,10,13,12]
        for i in id_CACG:
            maxUTS=data_prof.loc[data_prof.index==i]["Zrmax_RUM"].values[0] # Si forcage 
            maxUTS=int(float(maxUTS))
            UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/"+str(y)+"/output_test_"+str(y)+".df","rb"))
            data_id=UTS.groupby("id")
            ID_data=data_id.get_group(i)
            print(r'ID == %s ==> RAW == %s'%(i,max(round(ID_data.RAW,2))))
            IRR.append([i,ID_data.Ir_auto.sum(),maxUTS,ID_data.TAW.max()])
            # dfmore
        #     UTSmore=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/CACG_init_ru_optim_P055_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/output_test_"+str(y)+".df","rb"))
        #     data_idmore=UTSmore.groupby("id")
        #     ID_datamore=data_idmore.get_group(i)
        #     yerrmax.append(abs(ID_data.Ir_auto.sum()-ID_datamore.Ir_auto.sum()))
        #     # dfless
        #     UTSless=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/CACG_init_ru_optim_P055_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/output_test_"+str(y)+".df","rb"))
        #     data_idless=UTSless.groupby("id")
        #     ID_dataless=data_idless.get_group(i)
        #     yerrmin.append(abs(ID_dataless.Ir_auto.sum()-ID_data.Ir_auto.sum()))
        # yerr=[yerrmin,yerrmax]
        tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
        vali_RUM=pd.merge(tab_irr,data_prof[["RUM","CC_mean",'PF_mean']],on="ID")
        tab_irr2=pd.merge(tab_irr,sum_irr_cacg_val,on='ID')
        slope, intercept, r_value, p_value, std_err = stats.linregress(tab_irr2.Quantite.to_list(),tab_irr2.conso.to_list())
        bias=1/tab_irr2["Quantite"].shape[0]*sum(tab_irr2.conso-np.mean(tab_irr2.Quantite)) 
        rms = np.sqrt(mean_squared_error(tab_irr2.Quantite,tab_irr2.conso))
        plt.scatter(tab_irr2.Quantite,tab_irr2.conso,label=y)
        plt.legend()
        plt.xlim(-10,350)
        plt.ylim(-10,350)
        plt.xlabel("Quantité annuelles observées en mm ")
        plt.ylabel("Quantité annuelles modélisées en mm ")
        plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
        # plt.errorbar(tab_irr2.Quantite,tab_irr2.conso,yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
        if "2017" in y :
            rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
            plt.gca().add_patch(rectangle)
            plt.text(100,280,"RMSE = "+str(round(rms,2))) 
            plt.text(100,270,"R² = "+str(round(r_value,2)))
            plt.text(100,260,"Pente = "+str(round(slope,2)))
            plt.text(100,250,"Biais = "+str(round(bias,2)))
        else:
            rectangle = plt.Rectangle((225, 117),70,45, ec='orange',fc='orange',alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.text(230,150,"RMSE = "+str(round(rms,2))) 
            plt.text(230,140,"R² = "+str(round(r_value,2)))
            plt.text(230,130,"Pente = "+str(round(slope,2)))
            plt.text(230,120,"Biais = "+str(round(bias,2)))
        for i in enumerate(id_CACG):
            label = int(i[1])
            plt.annotate(label, # this is the text
                  (tab_irr2["Quantite"].iloc[i[0]],tab_irr2.conso.iloc[i[0]]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(-6,2), # distance from text to points (x,y)
                  ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_post_forcagemaxZr_RUMvalue_p.png")
    # TAW et RUM
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(vali_RUM.RUM.to_list(),tab_irr.TAWMax.to_list())
    bias=1/vali_RUM["RUM"].shape[0]*sum(tab_irr.TAWMax-np.mean(vali_RUM.RUM)) 
    rms = np.sqrt(mean_squared_error(vali_RUM.RUM,tab_irr.TAWMax))
    plt.scatter(vali_RUM.RUM,tab_irr["TAWMax"])
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.xlabel("RUM observées en mm ")
    plt.ylabel("RUM modélisées en mm ")
    rectangle = plt.Rectangle((45, 145),45,30, ec='b',fc='b',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(50,165,"RMSE = "+str(round(rms,2))) 
    plt.text(50,160,"R² = "+str(round(r_value,2)))
    plt.text(50,155,"Pente = "+str(round(slope,2)))
    plt.text(50,150,"Biais = "+str(round(bias,2)))
    plt.plot([0.0, 200], [0.0,200], 'black', lw=1,linestyle='--')
    for i in enumerate(tab_irr.ID):
            label = int(i[1])
            plt.annotate(label, # this is the text
                  (vali_RUM.RUM.iloc[i[0]],tab_irr["TAWMax"].iloc[i[0]]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,5), # distance from text to points (x,y)
                  ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_RUM_post_forcagemaxZr_RUMvalue_p.png")
    plt.hist( tab_irr["TAWMax"],label="RUM_modèle",ec="black",bins=10,linestyle="--")
    plt.hist(vali_RUM.RUM,label="RUM_obs",alpha=0.7,ec="black",bins=10)
    plt.xlabel("Valeur des RUM")
    plt.legend()
    
    # =============================================================================
#     Forcage p et maxZr aavec modif CC
# =============================================================================
    plt.figure(figsize=(7,7))
    for y in years :
        data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
        IRR=[]
        yerrmin=[]
        yerrmax=[]
        vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/DATE_DOES_CACG_"+str(y)+".csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
        vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%Y')
        vali_cacg["Quantite"].astype(float)
        sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum()
        if y =="2017":
            id_CACG=[1,4,5,6,13]
        else:
            id_CACG=[1,5,9,10,13,12]
        for i in id_CACG:
            maxUTS=data_prof.loc[data_prof.index==i]["ProfRacPot"].values[0] # Si forcage 
            maxUTS=int(float(maxUTS)*10)
            UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/modif_CC/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil/"+str(y)+"/output_test_"+str(y)+".df","rb"))
            data_id=UTS.groupby("id")
            ID_data=data_id.get_group(i)
            print(r'ID == %s ==> RAW == %s'%(i,max(round(ID_data.RAW,2))))
            IRR.append([i,ID_data.Ir_auto.sum(),maxUTS,ID_data.TAW.max()])
            # dfmore
        #     UTSmore=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/CACG_init_ru_optim_P055_Fcover_pl20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varplus20/"+str(y)+"/Output/output_test_"+str(y)+".df","rb"))
        #     data_idmore=UTSmore.groupby("id")
        #     ID_datamore=data_idmore.get_group(i)
        #     yerrmax.append(abs(ID_data.Ir_auto.sum()-ID_datamore.Ir_auto.sum()))
        #     # dfless
        #     UTSless=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/maxZr_rum/CACG_init_ru_optim_P055_Fcover_m20_fewi_De_Kr_days10_dose30_"+str(int(maxUTS))+"_irri_auto_soil_varmo20/"+str(y)+"/Output/output_test_"+str(y)+".df","rb"))
        #     data_idless=UTSless.groupby("id")
        #     ID_dataless=data_idless.get_group(i)
        #     yerrmin.append(abs(ID_dataless.Ir_auto.sum()-ID_data.Ir_auto.sum()))
        # yerr=[yerrmin,yerrmax]
        tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
        vali_RUM=pd.merge(tab_irr,data_prof[["RUM","CC_mean",'PF_mean']],on="ID")
        tab_irr2=pd.merge(tab_irr,sum_irr_cacg_val,on='ID')
        slope, intercept, r_value, p_value, std_err = stats.linregress(tab_irr2.Quantite.to_list(),tab_irr2.conso.to_list())
        bias=1/tab_irr2["Quantite"].shape[0]*sum(tab_irr2.conso-np.mean(tab_irr2.Quantite)) 
        rms = np.sqrt(mean_squared_error(tab_irr2.Quantite,tab_irr2.conso))
        plt.scatter(tab_irr2.Quantite,tab_irr2.conso,label=y)
        plt.legend()
        plt.xlim(-10,350)
        plt.ylim(-10,350)
        plt.xlabel("Quantité annuelles observées en mm ")
        plt.ylabel("Quantité annuelles modélisées en mm ")
        plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
        # plt.errorbar(tab_irr2.Quantite,tab_irr2.conso,yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
        if "2017" in y :
            rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
            plt.gca().add_patch(rectangle)
            plt.text(100,280,"RMSE = "+str(round(rms,2))) 
            plt.text(100,270,"R² = "+str(round(r_value,2)))
            plt.text(100,260,"Pente = "+str(round(slope,2)))
            plt.text(100,250,"Biais = "+str(round(bias,2)))
        else:
            rectangle = plt.Rectangle((225, 117),70,45, ec='orange',fc='orange',alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.text(230,150,"RMSE = "+str(round(rms,2))) 
            plt.text(230,140,"R² = "+str(round(r_value,2)))
            plt.text(230,130,"Pente = "+str(round(slope,2)))
            plt.text(230,120,"Biais = "+str(round(bias,2)))
        for i in enumerate(id_CACG):
            label = int(i[1])
            plt.annotate(label, # this is the text
                  (tab_irr2["Quantite"].iloc[i[0]],tab_irr2.conso.iloc[i[0]]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(-6,2), # distance from text to points (x,y)
                  ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_post_forcagemaxZr_p_modif_CC.png")
    # TAW et RUM
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(vali_RUM.RUM.to_list(),tab_irr.TAWMax.to_list())
    bias=1/vali_RUM["RUM"].shape[0]*sum(tab_irr.TAWMax-np.mean(vali_RUM.RUM)) 
    rms = np.sqrt(mean_squared_error(vali_RUM.RUM,tab_irr.TAWMax))
    plt.scatter(vali_RUM.RUM,tab_irr["TAWMax"])
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.xlabel("RUM observées en mm ")
    plt.ylabel("RUM modélisées en mm ")
    rectangle = plt.Rectangle((45, 145),45,30, ec='b',fc='b',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(50,165,"RMSE = "+str(round(rms,2))) 
    plt.text(50,160,"R² = "+str(round(r_value,2)))
    plt.text(50,155,"Pente = "+str(round(slope,2)))
    plt.text(50,150,"Biais = "+str(round(bias,2)))
    plt.plot([0.0, 200], [0.0,200], 'black', lw=1,linestyle='--')
    for i in enumerate(tab_irr.ID):
            label = int(i[1])
            plt.annotate(label, # this is the text
                  (vali_RUM.RUM.iloc[i[0]],tab_irr["TAWMax"].iloc[i[0]]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,5), # distance from text to points (x,y)
                  ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_RUM_post_forcagemaxZr_p_modif_CC.png")
# =============================================================================
#     Robuste du paramètrage 2017 sur 2018
# =============================================================================
    # data1=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P07_Fcover_fewi_De_Kr_days10_dose30_600_irri_auto_soil/2018/output_test_2018.df","rb"))
    # dataid1=data1.loc[data1.id==1]
    # sum_id1=dataid1.Ir_auto.sum()
    # data2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_500_irri_auto_soil/2018/output_test_2018.df","rb"))
    # dataid5=data2.loc[data2.id==5]
    # sum_id5=dataid5.Ir_auto.sum()
    # data3=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_850_irri_auto_soil/2018/output_test_2018.df","rb"))
    # dataid13=data3.loc[data3.id==13]
    # sum_id13=dataid13.Ir_auto.sum()
    # data_2018_for2017=pd.DataFrame(np.array([[1.0,sum_id1 , 172.0], [5.0, sum_id5, 134.0], [13, sum_id13,195.0 ]]),columns=['ID', 'conso', 'Vali'])
    # data1=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P07_Fcover_fewi_De_Kr_days10_dose30_600_irri_auto_soil/2017/output_test_2017.df","rb"))
    # dataid1=data1.loc[data1.id==1]
    # sum_id1=dataid1.Ir_auto.sum()
    # data2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_500_irri_auto_soil/2017/output_test_2017.df","rb"))
    # dataid5=data2.loc[data2.id==5]
    # sum_id5=dataid5.Ir_auto.sum()
    # data3=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_850_irri_auto_soil/2017/output_test_2017.df","rb"))
    # dataid13=data3.loc[data3.id==13]
    # sum_id13=dataid13.Ir_auto.sum()
    # tot_2017=pd.DataFrame(np.array([[1.0,sum_id1 , 169.8], [5.0, sum_id5, 170.0], [13, sum_id13,149.0 ]]),columns=['ID', 'conso', 'Vali'])
    
    
    
    # plt.figure(figsize=(7,7))
    # plt.scatter(tot_2017.Vali,tot_2017.conso,label="2017")
    # plt.scatter(data_2018_for2017.Vali,data_2018_for2017.conso,label="2018")
    # plt.legend()
    # plt.xlim(-10,350)
    # plt.ylim(-10,350)
    # plt.xlabel("Quantité annuelles observées en mm ")
    # plt.ylabel("Quantité annuelles modélisées en mm ")
    # plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # plt.title("transposition optimisation p 2017 sur 2018")
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tot_2017.Vali.to_list(),tot_2017.conso.to_list())
    # bias=1/tot_2017["Vali"].shape[0]*sum(tot_2017.conso-np.mean(tot_2017.Vali)) 
    # rms = np.sqrt(mean_squared_error(tot_2017.Vali,tot_2017.conso))
    # rectangle = plt.Rectangle((95, 245),70,40, ec='blue',fc='blue',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    # plt.text(100,270,"R² = "+str(round(r_value,2)))
    # plt.text(100,260,"Pente = "+str(round(slope,2)))
    # plt.text(100,250,"Biais = "+str(round(bias,2)))
    # slope2, intercept, r_value2, p_value, std_err = stats.linregress(data_2018_for2017.Vali.to_list(),data_2018_for2017.conso.to_list())
    # bias2=1/data_2018_for2017["Vali"].shape[0]*sum(data_2018_for2017.conso-np.mean(data_2018_for2017.Vali)) 
    # rms2 = np.sqrt(mean_squared_error(data_2018_for2017.Vali,data_2018_for2017.conso))
    # rectangle = plt.Rectangle((225, 117),70,40, ec='orange',fc='orange',alpha=0.3)
    # plt.gca().add_patch(rectangle)
    # plt.text(230,150,"RMSE = "+str(round(rms2,2))) 
    # plt.text(230,140,"R² = "+str(round(r_value2,2)))
    # plt.text(230,130,"Pente = "+str(round(slope2,2)))
    # plt.text(230,120,"Biais = "+str(round(bias2,2)))
    # for i in enumerate(data_2018_for2017.ID):
    #         label = int(i[1])
    #         plt.annotate(label, # this is the text
    #               (data_2018_for2017["Vali"].iloc[i[0]],data_2018_for2017.conso.iloc[i[0]]), # this is the point to label
    #               textcoords="offset points", # how to position the text
    #               xytext=(0,5), # distance from text to points (x,y)
    #               ha='center')
    #         plt.annotate(label, # this is the text
    #               (tot_2017["Vali"].iloc[i[0]],tot_2017.conso.iloc[i[0]]), # this is the point to label
    #               textcoords="offset points", # how to position the text
    #               xytext=(0,5), # distance from text to points (x,y)
    #               ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_transposition_optimp2017sur2018.png")
# # =============================================================================
# #     Robuste du paramètrage 2018 sur 2017
# # =============================================================================
    # data1=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P04_Fcover_fewi_De_Kr_days10_dose30_600_irri_auto_soil/2017/output_test_2017.df","rb"))
    # dataid1=data1.loc[data1.id==1]
    # sum_id1=dataid1.Ir_auto.sum()
    # data2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P04_Fcover_fewi_De_Kr_days10_dose30_500_irri_auto_soil/2017/output_test_2017.df","rb"))
    # dataid5=data2.loc[data2.id==5]
    # sum_id5=dataid5.Ir_auto.sum()
    # data3=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P04_Fcover_fewi_De_Kr_days10_dose30_850_irri_auto_soil/2017/output_test_2017.df","rb"))
    # dataid13=data3.loc[data3.id==13]
    # sum_id13=dataid13.Ir_auto.sum()
    # data_2017_for2018=pd.DataFrame(np.array([[1.0,sum_id1 , 169.8], [5.0, sum_id5, 170.0], [13, sum_id13,149.0 ]]),columns=['ID', 'conso', 'Vali'])
    # data1=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P04_Fcover_fewi_De_Kr_days10_dose30_600_irri_auto_soil/2018/output_test_2018.df","rb"))
    # dataid1=data1.loc[data1.id==1]
    # sum_id1=dataid1.Ir_auto.sum()
    # data2=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P04_Fcover_fewi_De_Kr_days10_dose30_500_irri_auto_soil/2018/output_test_2018.df","rb"))
    # dataid5=data2.loc[data2.id==5]
    # sum_id5=dataid5.Ir_auto.sum()
    # data3=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/CACG_init_ru_optim_P04_Fcover_fewi_De_Kr_days10_dose30_850_irri_auto_soil/2018/output_test_2018.df","rb"))
    # dataid13=data3.loc[data3.id==13]
    # sum_id13=dataid13.Ir_auto.sum()
    # tot_2018=pd.DataFrame(np.array([[1.0,sum_id1 , 172.0], [5.0, sum_id5, 134.0], [13, sum_id13,195.0]]),columns=['ID', 'conso', 'Vali'])
    
    
    # plt.figure(figsize=(7,7))
    # plt.scatter(data_2017_for2018.Vali,data_2017_for2018.conso,label="2017")
    # plt.scatter(tot_2018.Vali,tot_2018.conso,label="2018")
    # plt.legend()
    # plt.xlim(-10,350)
    # plt.ylim(-10,350)
    # plt.xlabel("Quantité annuelles observées en mm ")
    # plt.ylabel("Quantité annuelles modélisées en mm ")
    # plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # plt.title("transposition optimisation p 2018 sur 2017")
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tot_2018.Vali.to_list(),tot_2018.conso.to_list())
    # bias=1/tot_2018["Vali"].shape[0]*sum(tot_2018.conso-np.mean(tot_2018.Vali)) 
    # rms = np.sqrt(mean_squared_error(tot_2018.Vali,tot_2018.conso))
    # rectangle = plt.Rectangle((95, 245),70,40, ec='orange',fc='orange',alpha=0.3)
    # plt.gca().add_patch(rectangle)
    # plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    # plt.text(100,270,"R² = "+str(round(r_value,2)))
    # plt.text(100,260,"Pente = "+str(round(slope,2)))
    # plt.text(100,250,"Biais = "+str(round(bias,2)))
    # slope2, intercept, r_value2, p_value, std_err = stats.linregress(data_2017_for2018.Vali.to_list(),data_2017_for2018.conso.to_list())
    # bias2=1/data_2017_for2018["Vali"].shape[0]*sum(data_2017_for2018.conso-np.mean(data_2017_for2018.Vali)) 
    # rms2 = np.sqrt(mean_squared_error(data_2017_for2018.Vali,data_2017_for2018.conso))
    # rectangle = plt.Rectangle((225, 117),70,40, ec='blue',fc='blue',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(230,150,"RMSE = "+str(round(rms2,2))) 
    # plt.text(230,140,"R² = "+str(round(r_value2,2)))
    # plt.text(230,130,"Pente = "+str(round(slope2sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum(),2)))
    # plt.text(230,120,"Biais = "+str(round(bias2,2)))
    # for i in enumerate(data_2017_for2018.ID):
    #         label = int(i[1])
    #         plt.annotate(label, # this is the text
    #               (data_2017_for2018["Vali"].iloc[i[0]],data_2017_for2018.conso.iloc[i[0]]), # this is the point to label
    #               textcoords="offset points", # how to position the text
    #               xytext=(0,5), # distance from text to points (x,y)
    #               ha='center')
    #         plt.annotate(label, # this is the text
    #               (tot_2018["Vali"].iloc[i[0]],tot_2018.conso.iloc[i[0]]), # this is the point to label
    #               textcoords="offset points", # how to position the text
    #               xytext=(0,5), # distance from text to points (x,y)
    #               ha='center')
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_transposition_optimp2018sur2017.png")
# =============================================================================
#  Plot résul maxZr 1000,1500,3000
# =============================================================================
  
    df1000=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Sans_optim/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_1000_irri_auto_soil/2017/output_test_2017.df","rb"))
    df1000_irr=df1000.groupby("id")["Ir_auto"].sum()
    df1500=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Sans_optim/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_1500_irri_auto_soil/2017/output_test_2017.df","rb"))
    df1500_irr=df1500.groupby("id")["Ir_auto"].sum()
    df3000=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Sans_optim/CACG_init_ru_optim_P055_Fcover_fewi_De_Kr_days10_dose30_3000_irri_auto_soil/2017/output_test_2017.df","rb"))
    df3000_irr=df3000.groupby("id")["Ir_auto"].sum()
    vali_cacg=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/DATE_DOES_CACG_2017.csv",encoding='latin-1',decimal=',',sep=';',na_values="nan")
    vali_cacg.Date_irrigation=pd.to_datetime(vali_cacg.Date_irrigation,format='%d/%m/%Y')
    vali_cacg["Quantite"].astype(float)
    sum_irr_cacg_val=vali_cacg.groupby("ID")["Quantite"].sum()
    plt.figure(figsize=(7,7))
    plt.scatter(sum_irr_cacg_val.loc[[1,4,5,6,13]],df1000_irr.loc[[1,4,5,6,13]],marker="+",label="MaxZr 1000",c='orange')
    plt.scatter(sum_irr_cacg_val.loc[[1,4,5,6,13]],df1500_irr.loc[[1,4,5,6,13]],marker='x',label="MaxZr 1500",c='b')
    plt.scatter(sum_irr_cacg_val.loc[[1,4,5,6,13]],df3000_irr.loc[[1,4,5,6,13]],marker="1",label="MaxZr 3000",c='red')
    plt.legend()
    plt.xlim(-10,300)
    plt.ylim(-10,300)
    plt.plot([-10.0, 300], [-10.0,300], 'black', lw=1,linestyle='--')
    slope, intercept, r_value, p_value, std_err = stats.linregress(sum_irr_cacg_val.loc[[1,4,5,6,13]].to_list(),df1000_irr.loc[[1,4,5,6,13]].to_list())
    bias=1/sum_irr_cacg_val.loc[[1,4,5,6,13]].shape[0]*sum(df1000_irr.loc[[1,4,5,6,13]]-np.mean(sum_irr_cacg_val.loc[[1,4,5,6,13]])) 
    rms = np.sqrt(mean_squared_error(sum_irr_cacg_val.loc[[1,4,5,6,13]],df1000_irr.loc[[1,4,5,6,13]]))
    rectangle = plt.Rectangle((95, 245),70,40, ec='orange',fc='orange',alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R² = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    slope2, intercept, r_value2, p_value, std_err = stats.linregress(sum_irr_cacg_val.loc[[1,4,5,6,13]].to_list(),df1500_irr.loc[[1,4,5,6,13]].to_list())
    bias2=1/sum_irr_cacg_val.loc[[1,4,5,6,13]].shape[0]*sum(df1500_irr.loc[[1,4,5,6,13]]-np.mean(sum_irr_cacg_val.loc[[1,4,5,6,13]])) 
    rms2 = np.sqrt(mean_squared_error(sum_irr_cacg_val.loc[[1,4,5,6,13]],df1500_irr.loc[[1,4,5,6,13]]))
    rectangle = plt.Rectangle((225, 117),70,40, ec='blue',fc='blue',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(230,150,"RMSE = "+str(round(rms2,2))) 
    plt.text(230,140,"R² = "+str(round(r_value2,2)))
    plt.text(230,130,"Pente = "+str(round(slope2,2)))
    plt.text(230,120,"Biais = "+str(round(bias2,2)))
    slope3, intercept, r_value3, p_value, std_err = stats.linregress(sum_irr_cacg_val.loc[[1,4,5,6,13]].to_list(),df3000_irr.loc[[1,4,5,6,13]].to_list())
    bias3=1/sum_irr_cacg_val.loc[[1,4,5,6,13]].shape[0]*sum(df3000_irr.loc[[1,4,5,6,13]]-np.mean(sum_irr_cacg_val.loc[[1,4,5,6,13]])) 
    rms3 = np.sqrt(mean_squared_error(sum_irr_cacg_val.loc[[1,4,5,6,13]],df3000_irr.loc[[1,4,5,6,13]]))
    rectangle = plt.Rectangle((225, 17),70,40, ec='r',fc='r',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(230,50,"RMSE = "+str(round(rms3,2))) 
    plt.text(230,40,"R² = "+str(round(r_value3,2)))
    plt.text(230,30,"Pente = "+str(round(slope3,2)))
    plt.text(230,20,"Biais = "+str(round(bias3,2)))
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_CACG/Plot_result/plot_scatter_volumes_Irrigation_maxZr_1000_3000.png")