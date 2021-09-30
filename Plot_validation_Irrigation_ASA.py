# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:05:24 2021

@author: yann 

Validation Irrigation automatique parcelle de référence ASA
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
    name_run="RUNS_SAMIR/RUN_ASA/ASA_GSM_Fcover_400_2500_irri_auto_soil/"
    name_run_save_fig="RUNS_SAMIR/RUN_ASA/ASA_GSM_Fcover_400_2500_irri_auto_soil/"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"

    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # label="Init ru année n-1 + Irrigation auto"
    years=["2017",'2018']
    lc="maize_irri"


# =============================================================================
# Validation des Irr cumulées CACG
# =============================================================================

    for y in years: 
        Vol_tot=pd.DataFrame()
        Id=pd.DataFrame()
        vali_ASA=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/ConsoASA_NESTE_"+str(y)+".csv",encoding='latin-1',decimal='.',sep=';',na_values="nan")
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        Irri_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_"+str(y)+".csv",index_col=[0,1],skipinitialspace=True)
        gro=Irri_mod.groupby("ID")
        param=pd.read_csv(d["Output_model_PC_home_disk"]+"/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
        param.loc[param.shape[0]]='ID' 
        NDVI=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb"))
        NDVI=NDVI.loc[(NDVI.date >= str(y)+'-04-01')&(NDVI.date<=str(y)+"-09-30")]
        Prec=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Inputdata/maize_irri/meteo.df","rb"))
        Prec=Prec.loc[(Prec.date >= str(y)+'-04-01')&(Prec.date<=str(y)+"-09-30")]
        df_date_aqui=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/INPUT_DATA/NDVI_parcelle/Sentinel2_T30TYP_input_dates_"+str(y)+".txt",header=None)
        df_date_aqui[0]=pd.to_datetime(df_date_aqui[0],format='%Y%m%d')
        df_date_aqui.columns=["date"]
# =============================================================================
#          Recuperation ETR flux +SWC
# =============================================================================
#  Lecture file param 

        # All_ETR=pd.DataFrame()
        # param=pd.read_csv(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
        # for r in os.listdir(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"):
        #     if ".df" in r:
        #         print(r)
        #         num_run=r[23:-3]
        #         df=pickle.load(open(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/"+r,'rb'))
        #         df["param"]=np.repeat(param.loc[param[0]==int(num_run)][1].values,df.shape[0])
        #         All_ETR=All_ETR.append(df[["date","ET","id",'param']])
        # et=All_ETR.groupby("id")
        # # inser loop parcelle Id
        
        for p in list(set(Irri_mod.ID))[2:]:
            par1=gro.get_group(p)
            par1.reset_index(inplace=True)
            par1.num_run=pd.to_datetime(par1.num_run,format="%Y-%m-%d")
            df_aqui=pd.merge(df_date_aqui,NDVI.loc[NDVI.id==p],on="date")
            Vol_tot=Vol_tot.append(par1)
        
# =============================================================================
# Volumes annuels
# =============================================================================
    
        Vol_tot.columns=["date"]+param[1].to_list()
        tot_ID=Vol_tot.groupby("ID").sum()
        tot_IRR=pd.merge(tot_ID,vali_ASA,on=["ID"])
        tot_IRR.dropna(inplace=True)
        asagroup=tot_IRR.groupby("nomasa")
        conso=[]
        asa_n=[]
        data_v=[]
        run=[]
        for t in Vol_tot.columns[1:-1]:
            for asa in list(set(tot_IRR.nomasa)):
                tet=asagroup.get_group(asa)
                tet["conso"]=tet.eval("%s*area*10"%t)
                conso_asa=sum(tet["conso"])
                conso.append(conso_asa)
                asa_n.append(asa)
                data_v.append(tet.data_Conso.iloc[0])
                run.append(t)
        g=[conso,data_v,run]
        TT=pd.DataFrame(g).T
        if y =="2018":
            tot_2018=pd.DataFrame(TT.values,index=asa_n,columns=["conso","Vali",'param'])
            tot_2018=tot_2018/100000
            tot_2018=tot_2018[tot_2018.index!= "LASSERRE"]
            tot_2018.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_ASA_mod_2018.csv")
        else:
            tot_2017=pd.DataFrame(TT.values,index=asa_n,columns=["conso","Vali",'param'])
            tot_2017=tot_2017/100000
            tot_2017=tot_2017[tot_2017.index!= "LASSERRE"]
            tot_2017.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_ASA_mod_2017.csv")
            
    for t in [0.01,0.011,0.012,0.013,0.017]:
        plt.figure(figsize=(7,7))
        for y in years: 
            slope, intercept, r_value, p_value, std_err = stats.linregress(globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].Vali.to_list(), globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].conso.to_list())
            bias=1/len(globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].Vali.to_list())*sum(globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].conso-np.mean(globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].Vali.to_list())) 
            rms = np.sqrt(mean_squared_error(globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].Vali.to_list(), globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].conso))
            plt.scatter(globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].Vali.to_list(),globals()["tot_"+y].loc[globals()["tot_"+y]["param"]==t].conso,label=y)
            plt.xlim(-1,10)
            plt.ylim(-1,10)
            plt.xlabel("Volumes annuels observés en millions de m3 ")
            plt.ylabel("Volumes annuels modélisés en million de m3 ")
            plt.plot([-1, 10], [-1,10], 'black', lw=1,linestyle='--')
            if "2017" in y :
                rectangle = plt.Rectangle((3.9, 7),2.1,1.3, ec='blue',fc='blue',alpha=0.1)
                plt.gca().add_patch(rectangle)
                plt.text(4,7.9,"RMSE = "+str(round(rms,2))) 
                plt.text(4,7.6,"R² = "+str(round(r_value,2)))
                plt.text(4,7.3,"Pente = "+str(round(slope,2)))
                plt.text(4,7,"Biais = "+str(round(bias,2)))
            else:
                rectangle = plt.Rectangle((7.90, 5),2.1,1.30, ec='orange',fc='orange',alpha=0.3)
                plt.gca().add_patch(rectangle)
                plt.text(8,5.9,"RMSE = "+str(round(rms,2))) 
                plt.text(8,5.6,"R² = "+str(round(r_value,2)))
                plt.text(8,5.3,"Pente = "+str(round(slope,2)))
                plt.text(8,5,"Biais = "+str(round(bias,2)))
        # for i in enumerate(globals()["tot_"+y].index):
        #     if "2017" in y :
        #         label = i[1].lower()
        #         plt.annotate(label, # this is the text
        #               (globals()["tot_"+y].Vali[i[0]],globals()["tot_"+y].conso[i[0]]), # this is the point to label
        #               textcoords="offset points", # how to position the text
        #               xytext=(0,5), # distance from text to points (x,y)
        #               ha='center')
        #     else:
        #         label = i[1].lower()
        #         plt.annotate(label, # this is the text
        #               (globals()["tot_"+y].Vali[i[0]],globals()["tot_"+y].conso[i[0]]), # this is the point to label
        #               textcoords="offset points", # how to position the text
        #               xytext=(0,-10), # distance from text to points (x,y)
        #               ha='center')
        plt.legend()
        plt.title(t)
        plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_%s_Irrigation.png"%(t))
        
        
# =============================================================================
#          Plot avec RUM inversion
# =============================================================================
    for y in years: 
        data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_ASA_"+y+"_UTS_maj_all_crops.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
        Vol_tot=pd.DataFrame()
        Id=pd.DataFrame()
        vali_ASA=pd.read_csv(d["PC_disk"]+"TRAITEMENT/DATA_VALIDATION/DATA_VOL_IRRIGATION/ConsoASA_NESTE_"+str(y)+"_all_crops.csv",encoding='latin-1',decimal='.',sep=',',na_values="nan")
        d["Output_model_PC_home_disk"]=d["PC_disk"]+"/TRAITEMENT/"+name_run
        Irri_mod=pd.read_csv(d["Output_model_PC_home_disk"]+"/LUT_"+str(y)+".csv",index_col=[0,1],skipinitialspace=True)
        gro=Irri_mod.groupby("ID")
        param=pd.read_csv(d["Output_model_PC_home_disk"]+"/"+str(y)+"/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
        param.loc[param.shape[0]]='ID' 
    #  Isloer pour chaque parcelle maxZr arondi 
        for p in list(set(Irri_mod.ID))[2:]:
            par1=gro.get_group(p)
            par1.reset_index(inplace=True)
            par1.num_run=pd.to_datetime(par1.num_run,format="%Y-%m-%d")
            df_aqui=pd.merge(df_date_aqui,NDVI.loc[NDVI.id==p],on="date")
            Vol_tot=Vol_tot.append(par1)
        Vol_tot.columns=["date"]+param[1].to_list()
        tot_ID=Vol_tot.groupby("ID").sum()
        tot_IRR=pd.merge(tot_ID,vali_ASA,on=["ID"])
        tot_IRR.dropna(inplace=True)
        tot_IRR.drop_duplicates(inplace=True)
        tot_IRR_ss_sunfl=tot_IRR[tot_IRR.RPG_CODE_C!='TRN']
        # tot_IRR_only_mais=tot_IRR_ss_sunfl[tot_IRR_ss_sunfl.RPG_CODE_C!='SOJ']
        # tot_IRR_ss_sunfl=tot_IRR
        #  Isloer pour chaque parcelle maxZr arondi 
        conso_par=[]
        idpar=[]
        n_asa=[]
        # for i in tot_IRR.ID:
        #     maxzr=data_prof[data_prof.ID==i]["Zrmax_RUM_arrondi"].values
        #     irri=tot_IRR.loc[tot_IRR.ID==i][maxzr].iloc[0].values
        #     conso_par.append(irri[0]*tot_IRR.loc[tot_IRR.ID==i]["area"].iloc[0]*10)
        #     idpar.append(i)
        #     n_asa.append(tot_IRR.loc[tot_IRR.ID==i]['nomasa'].values[0])
        # tab_par_irri=pd.DataFrame([idpar,conso_par,n_asa]).T
        # tab_par_irri.columns=["ID","conso","ASA"]
        # tab_asa_irr=tab_par_irri.groupby("ASA").sum()
        #  SS sunflower
        for i in tot_IRR_ss_sunfl.ID:
            maxzr=data_prof[data_prof.ID==i]["Zrmax_RUM_arrondi"].values
            if tot_IRR_ss_sunfl.loc[tot_IRR_ss_sunfl.ID==i]["RPG_CODE_C"].values[0] == "SOJ":
                irri=tot_IRR_ss_sunfl.loc[tot_IRR_ss_sunfl.ID==i][maxzr].iloc[0].values+10
                print(irri)
            else:
                irri=tot_IRR_ss_sunfl.loc[tot_IRR_ss_sunfl.ID==i][maxzr].iloc[0].values
            conso_par.append(irri[0]*tot_IRR_ss_sunfl.loc[tot_IRR_ss_sunfl.ID==i]["area"].iloc[0]*10)
            idpar.append(i)
            n_asa.append(tot_IRR_ss_sunfl.loc[tot_IRR_ss_sunfl.ID==i]['nomasa'].values[0])
        tab_par_irri=pd.DataFrame([idpar,conso_par,n_asa]).T
        tab_par_irri.columns=["ID","conso","ASA"]
        tab_asa_irr=tab_par_irri.groupby("ASA").sum()
        
        if y == "2017":
            Parcellaire= geo.read_file(d["PC_disk"]+"/TRAITEMENT/DONNEES_ASA/PACRELLE_ASA_2017_RPG_all_crops_summer_Gers.shp")
        else:
            Parcellaire= geo.read_file(d["PC_disk"]+"/TRAITEMENT/DONNEES_ASA/PACRELLE_ASA_2018_RPG_all_crops_summer_Gers_er10.shp")
        ET= {}
        gdf = {}
        data=[]
        Parcellaire["id"]=Parcellaire.ID
    
        
        for i in tot_IRR_ss_sunfl.ID:
          maxzr=data_prof[data_prof.ID==i]["Zrmax_RUM_arrondi"].values
          irri=tot_IRR_ss_sunfl.loc[tot_IRR_ss_sunfl.ID==i][maxzr].iloc[0].values
          data.append([i,irri[0]])
          
        data=pd.DataFrame(data,columns=['id','conso'])
        gdf = Parcellaire
        gdf = gdf.merge(data, on='id')
        gdf.to_file(d["PC_disk_water"]+"/TRAITEMENT/RUNS_ASA/ASA_GSM_Fcover_400_2500_irri_auto_soil/carte_surface_irriguée_"+str(y)+"_SAMIR.shp")
   
     
        # 
        asa_n=[]
        data_v=[]
        asagroup=tot_IRR.groupby("nomasa")
        for asa in list(set(tot_IRR.nomasa)):
            tet=asagroup.get_group(asa)
            asa_n.append(asa)
            data_v.append(tet.data_Conso_fuite.iloc[0])
        TT=pd.DataFrame([data_v,asa_n]).T
        TT.columns=["Vali","ASA"]
        tab_irri_asa=pd.merge(tab_asa_irr,TT,on="ASA")
        
        if y =="2018":
            tab_irri_asa_2018=tab_irri_asa
            tab_irri_asa_2018.set_index("ASA",inplace=True)
            tab_irri_asa_2018=tab_irri_asa_2018/100000
            tab_irri_asa_2018=tab_irri_asa_2018[tab_irri_asa_2018.index!= "LASSERRE"]
            tab_irri_asa_2018.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_ASA_mod_2018_maxZr_parcelle.csv")
        else:
            tab_irri_asa_2017=tab_irri_asa
            tab_irri_asa_2017.set_index("ASA",inplace=True)
            tab_irri_asa_2017=tab_irri_asa_2017/100000
            tab_irri_asa_2017=tab_irri_asa_2017[tab_irri_asa_2017.index!= "LASSERRE"]
            tab_irri_asa_2017.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/tab_ASA_mod_2017_maxZr_parcelle.csv")
            
        
    plt.figure(figsize=(7,7))
    for y in years: 
        slope, intercept, r_value, p_value, std_err = stats.linregress(globals()["tab_irri_asa_"+y].Vali.to_list(), globals()["tab_irri_asa_"+y].conso.to_list())
        bias=1/len(globals()["tab_irri_asa_"+y].Vali.to_list())*sum(globals()["tab_irri_asa_"+y].conso-np.mean(globals()["tab_irri_asa_"+y].Vali.to_list())) 
        rms = np.sqrt(mean_squared_error(globals()["tab_irri_asa_"+y].Vali.to_list(), globals()["tab_irri_asa_"+y].conso))
        plt.scatter(globals()["tab_irri_asa_"+y].Vali.to_list(),globals()["tab_irri_asa_"+y].conso,label=y)
        plt.xlim(-1,10)
        plt.ylim(-1,10)
        plt.xlabel("Volumes annuels observés en millions de m3 ")
        plt.ylabel("Volumes annuels modélisés en million de m3 ")
        plt.plot([-1, 10], [-1,10], 'black', lw=1,linestyle='--')
        if "2017" in y :
            rectangle = plt.Rectangle((3.9, 7),2.1,1.3, ec='blue',fc='blue',alpha=0.1)
            plt.gca().add_patch(rectangle)
            plt.text(4,7.9,"RMSE = "+str(round(rms,2))) 
            plt.text(4,7.6,"R² = "+str(round(r_value,2)))
            plt.text(4,7.3,"Pente = "+str(round(slope,2)))
            plt.text(4,7,"Biais = "+str(round(bias,2)))
        else:
            rectangle = plt.Rectangle((7.90, 5),2.1,1.30, ec='orange',fc='orange',alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.text(8,5.9,"RMSE = "+str(round(rms,2))) 
            plt.text(8,5.6,"R² = "+str(round(r_value,2)))
            plt.text(8,5.3,"Pente = "+str(round(slope,2)))
            plt.text(8,5,"Biais = "+str(round(bias,2)))
        for i in enumerate(globals()["tab_irri_asa_"+y].index):
            if "2017" in y :
                label = i[1].lower()
                plt.annotate(label, # this is the text
                      (globals()["tab_irri_asa_"+y].Vali[i[0]],globals()["tab_irri_asa_"+y].conso[i[0]]), # this is the point to label
                      textcoords="offset points", # how to position the text
                      xytext=(0,5), # distance from text to points (x,y)
                      ha='center')
            else:
                label = i[1].lower()
                plt.annotate(label, # this is the text
                      (globals()["tab_irri_asa_"+y].Vali[i[0]],globals()["tab_irri_asa_"+y].conso[i[0]]), # this is the point to label
                      textcoords="offset points", # how to position the text
                      xytext=(0,-10), # distance from text to points (x,y)
                      ha='center')
    plt.legend()
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run_save_fig+"/plot_scatter_volumes_maxZr_all_summer_crops_parcelle_Irrigation_ss_sunflower_soj_20mm.png")
    
    
    # Récupération shp

   