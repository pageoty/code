#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:20:20 2021

@author: pageot

Validation SAMIR sur PKGC avec les données pédologiques GSM 
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
from sklearn.linear_model import LinearRegression
from pylab import *
from sklearn.metrics import *



def predict(x):
   return slope * x + intercept




if __name__ == '__main__':
    d={}
    name_run="RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto_v2/"
    name_run_save_fig="RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto_v2/"
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    # d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"

    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    # label="Init ru année n-1 + Irrigation auto"
    years=["2017"]
    lc="maize_irri"
    soil=['SOIL_RIGOU',"RRP_Rigou"]#['SOIL_RIGOU',"RRP_Rigou"]["RRP","RRP_GERS"]
    optim_val="maxZr"

    # parcellaire = geo.read_file(d["PC_disk"]+"/DONNEES_RAW/data_SSP/PARCIRR_2017_32_avecRES_with_ID.shp")
    # parcellaire["ID"]=parcellaire["_ID"]
    # data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKGC_GERS_2017_GSM_PF_CC_class_name.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    # list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    # data_prof =data_prof[-data_prof["ID"].isin(list_drop)]
    # param=pd.read_csv(d["PC_disk"]+"//TRAITEMENT/"+name_run+"/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    # IRR=[]
    # for i in data_prof.ID:
    #     c=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["mean_arrondi"])][0]
    #     val=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["mean_arrondi"])][1]
    #     UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
    #     data_id=UTS.groupby("id")
    #     ID_data=data_id.get_group(i)
    #     IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],ID_data.TAW.max()])
    # tab_irr_GSM=pd.DataFrame(IRR,columns=["ID","Quant","maxZr","TAWmax"])
    # tab_f2=pd.merge(tab_irr_GSM,data_prof[["MMEAU","ID","Classe_Bruand"]],on='ID')
    # tab_f2.to_csv(d["PC_disk"]+"//TRAITEMENT/"+name_run+"/tab_resu_PKGC_depth_PF_CC_GSM.csv")
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_f2.MMEAU.to_list(),tab_f2["Quant"].to_list())
    # bias=1/tab_f2["MMEAU"].shape[0]*sum(tab_f2["Quant"]-np.mean(tab_f2.MMEAU)) 
    # rms = np.sqrt(mean_squared_error(tab_f2.MMEAU,tab_f2["Quant"]))
    # plt.figure(figsize=(7,7))
    # plt.xlim(-10,350)
    # plt.ylim(-10,350)
    # plt.xlabel("Quantité annuelles observées en mm ")
    # plt.ylabel("Quantité annuelles modélisées en mm ")
    # plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # a=plt.scatter(tab_f2.MMEAU,tab_f2["Quant"])
    # rectangle = plt.Rectangle((95, 300),72,42, ec='blue',fc='blue',alpha=0.1)
    # plt.gca().add_patch(rectangle)
    # plt.text(100,330,"RMSE = "+str(round(rms,2))) 
    # plt.text(100,320,"R² = "+str(round(r_value,2)))
    # plt.text(100,310,"Pente = "+str(round(slope,2)))
    # plt.text(100,300,"Biais = "+str(round(bias,2)))
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/plot_scatter_volumes_Irrigation_forcagemaxZr_soil_depth_GSM_PFCC.png")
    
# =============================================================================
#      Plot ETR soutenance exemple
# =============================================================================
    # plt.figure(figsize=(7,7))
    # plt.plot(UTS[UTS.id==11]["date"],UTS[UTS.id==11]["ET"].rolling(5).mean(),label='Evapotranspiration',linewidth="2")
    # plt.plot(UTS.loc[(UTS.id==11)&(UTS.Ir_auto!=0)]["date"],UTS.loc[(UTS.id==11)&(UTS.Ir_auto!=0)]["Ir_auto"]/5,marker="o",linestyle="",label="Irrigation")
    # plt.legend()
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT//plot_soutenance_resu_SAMIR.png")
    # # regrouper les classes de sol
    # tab_f2["TEX"]="L"
    # tab_f2.loc[(tab_f2.Classe_Bruand == "A"),'TEX']= "A"
    # tab_f2.loc[(tab_f2.Classe_Bruand == "AL"),'TEX']="A"
    
    # plt.figure(figsize=(7,7))
    # for c in list(set(tab_f2.TEX)):
    #     a=tab_f2.groupby("TEX")
    #     texture=a.get_group(c)
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(texture.MMEAU.to_list(),texture["Quant"].to_list())
    #     bias=1/texture["MMEAU"].shape[0]*sum(texture["Quant"]-np.mean(texture.MMEAU)) 
    #     rms = np.sqrt(mean_squared_error(texture.MMEAU,texture["Quant"]))
    #     # plt.legend(a.legend_elements()[0],labels)
    #     plt.xlim(-10,350)
    #     plt.ylim(-10,350)
    #     plt.xlabel("Quantité annuelles observées en mm ")
    #     plt.ylabel("Quantité annuelles modélisées en mm ")
    #     plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    #     # plt.errorbar(tab_f2.MMEAU,tab_irr[1],marker=",",yerr=yerr,fmt='o',linewidth=0.7,capsize=4)
    #     plt.scatter(texture.MMEAU,texture["Quant"],label=c)
    #     plt.legend()
    #     if c == "A":
    #         rectangle = plt.Rectangle((245, 215),72,42, ec='orange',fc='orange',alpha=0.1)
    #         plt.gca().add_patch(rectangle)
    #         plt.text(250,250,"RMSE = "+str(round(rms,2))) 
    #         plt.text(250,240,"R² = "+str(round(r_value,2)))
    #         plt.text(250,230,"Pente = "+str(round(slope,2)))
    #         plt.text(250,220,"Biais = "+str(round(bias,2)))
    #     else:
    #         rectangle = plt.Rectangle((245, 25),72,42, ec='blue',fc='blue',alpha=0.1)
    #         plt.gca().add_patch(rectangle)
    #         plt.text(250,60,"RMSE = "+str(round(rms,2))) 
    #         plt.text(250,50,"R² = "+str(round(r_value,2)))
    #         plt.text(250,40,"Pente = "+str(round(slope,2)))
    #         plt.text(250,30,"Biais = "+str(round(bias,2)))
    # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_Irrigation_forcagemaxZr_RUM_sep_texture_PF_CC_GSM.png")
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
# Analyse des partiques agricoles
# =============================================================================   
    # tab_pratique=pd.merge(tab_f2,parcellaire,on='ID')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_pratique.MMEAU_x.to_list(),tab_pratique.Quant.to_list())
    # bias=1/tab_pratique["MMEAU_x"].shape[0]*sum(tab_pratique.Quant-np.mean(tab_pratique.MMEAU_x)) 
    # rms = np.sqrt(mean_squared_error(tab_pratique.MMEAU_x,tab_pratique.Quant))
    # valid_sol_classe=pd.merge(data_prof[["Classe_Bruand",'ID']],tab_pratique,on='ID')
    # labels, index = np.unique(valid_sol_classe["Classe_Bruand_x"], return_inverse=True)
    
    # #  plot avec classe de texture GSM
    # plt.figure(figsize=(7,7))
    # plt.xlim(-10,350)
    # plt.ylim(-10,350)
    # plt.xlabel("Quantité annuelles observées en mm ")
    # plt.ylabel("Quantité annuelles modélisées en mm ")
    # plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # # plt.errorbar(tab_irr2.MMEAU,tab_irr2.conso,marker=',',yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    # rectangle = plt.Rectangle((95, 265),70,45, ec='blue',fc='blue',alpha=0.1)
    # a=plt.scatter(tab_pratique.MMEAU_x,tab_pratique.Quant,c=index,cmap='coolwarm')
    # plt.legend(a.legend_elements()[0],labels)
    # plt.gca().add_patch(rectangle)
    # plt.text(100,300,"RMSE = "+str(round(rms,2))) 
    # plt.text(100,290,"R² = "+str(round(r_value,2)))
    # plt.text(100,280,"Pente = "+str(round(slope,2)))
    # plt.text(100,270,"Biais = "+str(round(bias,2)))

    # # res=tab_pratique.groupby("RESIDUS")
    # # dist_resu={"Pas de résidus":0.0,"broyés":1.0,"non_broyés":2.0,"broyés_enfouis":3.0,"non_broyés_enfouis":4.0,"Exportés":5.0}
    # # for i,z in zip(dist_resu.values(),dist_resu.keys()):
    # #     devenir=res.get_group(i)
    # #     slope, intercept, r_value, p_value, std_err = stats.linregress(devenir.MMEAU_x.to_list(),devenir.Quant.to_list())
    # #     bias=1/devenir["MMEAU_x"].shape[0]*sum(devenir.Quant-np.mean(devenir.MMEAU_x)) 
    # #     rms = np.sqrt(mean_squared_error(devenir.MMEAU_x,devenir.Quant))
    # #     valid_sol_classe=pd.merge(data_prof[["Classe_Bruand",'ID']],devenir,on='ID')
    # #     labels, index = np.unique(valid_sol_classe["Classe_Bruand_x"], return_inverse=True)
    # #     plt.figure(figsize=(7,7))
    # #     plt.xlim(-10,350)
    # #     plt.ylim(-10,350)
    # #     plt.xlabel("Quantité annuelles observées en mm ")
    # #     plt.ylabel("Quantité annuelles modélisées en mm ")
    # #     plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    # #     # plt.errorbar(tab_irr2.MMEAU,tab_irr2.conso,marker=',',yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    # #     rectangle = plt.Rectangle((95, 265),70,45, ec='blue',fc='blue',alpha=0.1)
    # #     a=plt.scatter(devenir.MMEAU_x,devenir.Quant,c=index,cmap='coolwarm')
    # #     plt.legend(a.legend_elements()[0],labels)
    # #     plt.gca().add_patch(rectangle)
    # #     plt.text(100,300,"RMSE = "+str(round(rms,2))) 
    # #     plt.text(100,290,"R² = "+str(round(r_value,2)))
    # #     plt.text(100,280,"Pente = "+str(round(slope,2)))
    # #     plt.text(100,270,"Biais = "+str(round(bias,2)))
    # #     plt.title(z)
    # # plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_Irri_RUM_inversion_RESIDUS_PF_CC_GSM.png")


    # dist_resu={"Pas de résidus":0.0,"broyés":1.0,"non_broyés":2.0,"broyés_enfouis":3.0,"non_broyés_enfouis":4.0,"Exportés":5.0}
    # TEX=tab_pratique.groupby("TEX")
    # for tex in ["L",'A']:
    #     plt.figure(figsize=(7,7))
    #     TEX_resu=TEX.get_group(tex)
    #     res=TEX_resu.groupby("RESIDUS")
    #     for i,z in zip(dist_resu.values(),dist_resu.keys()):
    #         if i not in res.RESIDUS.count().index :
    #             continue 
    #         devenir=res.get_group(i)
    #         if i == 0.0:
    #             coloris='blue'
    #         elif i == 1.0 :
    #             coloris='orange'
    #         elif i == 2.0 :
    #             coloris='green'
    #         elif i == 3.0 :
    #             coloris='red'
    #         elif i == 4.0 :
    #             coloris='purple'
    #         elif i == 5.0 :
    #             coloris='black'
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(devenir.MMEAU_x.to_list(),devenir.Quant.to_list())
    #         bias=1/devenir["MMEAU_x"].shape[0]*sum(devenir.Quant-np.mean(devenir.MMEAU_x)) 
    #         rms = np.sqrt(mean_squared_error(devenir.MMEAU_x,devenir.Quant))
    #         # valid_sol_classe=pd.merge(data_prof[["Class_Bruand","RUM",'ID']],devenir,on='ID')
    #         # labels, index = np.unique(valid_sol_classe["Class_Bruand_x"], return_inverse=True)
    #         plt.xlim(-10,350)
    #         plt.ylim(-10,350)
    #         plt.xlabel("Quantité annuelles observées en mm ")
    #         plt.ylabel("Quantité annuelles modélisées en mm ")
    #         plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    #         # plt.errorbar(tab_irr2.MMEAU,tab_irr2.conso,marker=',',yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    #         a=plt.scatter(devenir.MMEAU_x,devenir.Quant,label=z,color=coloris)
    #         plt.legend()
    #         plt.title(tex)
    #         if i ==0.0:
    #             rectangle = plt.Rectangle((10, 295),72,52, ec='blue',fc='blue',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(15,340,"RMSE = "+str(round(rms,2))) 
    #             plt.text(15,330,"R² = "+str(round(r_value,2)))
    #             plt.text(15,320,"Pente = "+str(round(slope,2)))
    #             plt.text(15,310,"Biais = "+str(round(bias,2)))
    #             plt.text(15,300,"Nb. = "+str(devenir.shape[0]))
    #         elif i == 1.0:
    #             rectangle = plt.Rectangle((95, 295),72,52, ec='orange',fc='orange',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(100,340,"RMSE = "+str(round(rms,2))) 
    #             plt.text(100,330,"R² = "+str(round(r_value,2)))
    #             plt.text(100,320,"Pente = "+str(round(slope,2)))
    #             plt.text(100,310,"Biais = "+str(round(bias,2)))
    #             plt.text(100,300,"Nb. = "+str(devenir.shape[0]))
    #         elif i == 3.0:
    #             rectangle = plt.Rectangle((275, 245),72,52, ec='red',fc='red',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(280,290,"RMSE = "+str(round(rms,2))) 
    #             plt.text(280,280,"R² = "+str(round(r_value,2)))
    #             plt.text(280,270,"Pente = "+str(round(slope,2)))
    #             plt.text(280,260,"Biais = "+str(round(bias,2)))
    #             plt.text(280,250,"Nb. = "+str(devenir.shape[0]))
    #         elif i == 2.0:
    #             rectangle = plt.Rectangle((195, 295),72,52, ec='green',fc='green',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(200,340,"RMSE = "+str(round(rms,2))) 
    #             plt.text(200,330,"R² = "+str(round(r_value,2)))
    #             plt.text(200,320,"Pente = "+str(round(slope,2)))
    #             plt.text(200,310,"Biais = "+str(round(bias,2)))
    #             plt.text(200,300,"Nb. = "+str(devenir.shape[0]))
    #         elif i == 4.0:
    #             rectangle = plt.Rectangle((275, 175),72,52, ec='purple',fc='purple',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(280,220,"RMSE = "+str(round(rms,2))) 
    #             plt.text(280,210,"R² = "+str(round(r_value,2)))
    #             plt.text(280,200,"Pente = "+str(round(slope,2)))
    #             plt.text(280,190,"Biais = "+str(round(bias,2)))
    #             plt.text(280,180,"Nb. = "+str(devenir.shape[0]))
    #         else:
    #             rectangle = plt.Rectangle((275,115),72,52, ec='black',fc='black',alpha=0.2)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(280,160,"RMSE = "+str(round(rms,2))) 
    #             plt.text(280,150,"R² = "+str(round(r_value,2)))
    #             plt.text(280,140,"Pente = "+str(round(slope,2)))
    #             plt.text(280,130,"Biais = "+str(round(bias,2)))
    #             plt.text(280,120,"Nb. = "+str(devenir.shape[0]))
    #     plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_Irri_RUM_inversion_RESIDUS_texture_sol_%s_PF_CC_GSM.png"%tex)
# =============================================================================
# regouper les champ résidus 
# =============================================================================
    # tab_pratique["residus"]=1.0
    # tab_pratique.loc[(tab_pratique.RESIDUS == 0.0),'residus']= 0.0
    # tab_pratique.loc[(tab_pratique.RESIDUS == 5.0),'residus']= 0.0
    # dist_resu={"Pas de résidus":0.0,"résidus":1.0}
    # TEX=tab_pratique.groupby("TEX")
    # for tex in ['A','L']:
    #     plt.figure(figsize=(7,7))
    #     TEX_resu=TEX.get_group(tex)
    #     res=TEX_resu.groupby("residus")
    #     for i,z in zip(dist_resu.values(),dist_resu.keys()):
    #         if i not in res.residus.count().index :
    #             continue 
    #         devenir=res.get_group(i)
    #         if i == 0.0:
    #             coloris='blue'
    #         elif i == 1.0 :
    #             coloris='orange'
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(devenir.MMEAU_x.to_list(),devenir.Quant.to_list())
    #         bias=1/devenir["MMEAU_x"].shape[0]*sum(devenir.Quant-np.mean(devenir.MMEAU_x)) 
    #         rms = np.sqrt(mean_squared_error(devenir.MMEAU_x,devenir.Quant))
    #         plt.xlim(-10,350)
    #         plt.ylim(-10,350)
    #         plt.xlabel("Quantité annuelles observées en mm ")
    #         plt.ylabel("Quantité annuelles modélisées en mm ")
    #         plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    #         # plt.errorbar(tab_irr2.MMEAU,tab_irr2.conso,marker=',',yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    #         a=plt.scatter(devenir.MMEAU_x,devenir.Quant,label=z,color=coloris)
    #         plt.legend()
    #         plt.title(tex)
    #         if i ==0.0:
    #             rectangle = plt.Rectangle((10, 295),72,52, ec='blue',fc='blue',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(15,340,"RMSE = "+str(round(rms,2))) 
    #             plt.text(15,330,"R² = "+str(round(r_value,2)))
    #             plt.text(15,320,"Pente = "+str(round(slope,2)))
    #             plt.text(15,310,"Biais = "+str(round(bias,2)))
    #             plt.text(15,300,"Nb. = "+str(devenir.shape[0]))
    #         elif i == 1.0:
    #             rectangle = plt.Rectangle((95, 295),72,52, ec='orange',fc='orange',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(100,340,"RMSE = "+str(round(rms,2))) 
    #             plt.text(100,330,"R² = "+str(round(r_value,2)))
    #             plt.text(100,320,"Pente = "+str(round(slope,2)))
    #             plt.text(100,310,"Biais = "+str(round(bias,2)))
    #             plt.text(100,300,"Nb. = "+str(devenir.shape[0]))
    #     plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_Irri_RESIDUS_texture_sol_simpli_%s_PF_CC_depth_GSM.png"%tex)


# =============================================================================
#  valeur des tables de la FAO soit 700 / 1700
# =============================================================================
    # data_mod=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto/LUT_2017.csv")
    # list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    # data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKGC_GERS_2017_GSM_PF_CC_class_name.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    # data_prof=data_prof[~data_prof["ID"].isin(list_drop)]["MMEAU"]
    # data_id=data_mod.groupby("ID").sum()
    # data_id.columns=data_mod.iloc[0][1:-1] # Attention bug verifier avec 1 au lieu de 2 
    # data_mod_PKGC=data_id[~data_id.index.isin(list_drop)]
    # data_mod_PKGC.to_csv(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto/tab_resu_GSM_PF_CC_FAO_maxZr.csv")
    # for col in data_mod_PKGC.columns[8:27]:
    #     plt.figure(figsize=(7,7))
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(data_prof.to_list(),data_mod_PKGC[col].to_list())
    #     bias=1/data_prof.shape[0]*sum(data_mod_PKGC[col]-np.mean(data_prof)) 
    #     rms = np.sqrt(mean_squared_error(data_prof,data_mod_PKGC[col]))
    #     plt.scatter(data_prof,data_mod_PKGC[col],label=col)
    #     plt.legend()
    #     plt.xlim(-10,350)
    #     plt.ylim(-10,350)
    #     plt.xlabel("Quantité annuelles observées en mm ")
    #     plt.ylabel("Quantité annuelles modélisées en mm ")
    #     plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    #     # plt.errorbar(tab_irr2.Quantite,tab_irr2.conso,yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    #     rectangle = plt.Rectangle((95, 245),70,45, ec='blue',fc='blue',alpha=0.1)
    #     plt.gca().add_patch(rectangle)
    #     plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    #     plt.text(100,270,"R² = "+str(round(r_value,2)))
    #     plt.text(100,260,"Pente = "+str(round(slope,2)))
    #     plt.text(100,250,"Biais = "+str(round(bias,2)))
    #     plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/RUN_FAO_TABLE/plot_scatter_volumes_Irrigation_forcagemaxZr_p_table_22_GSM_PFCC_Fcover%s.png"%(col))

# =============================================================================
# Utilisé PF_CC GSM et RUM dim UTS 
# =============================================================================
    param=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/2017/Output/maxZr/output_test_maize_irri_param.txt",header=None,skiprows=1,sep=";")
    IRR=[]
    ids=[]
    data_prof=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKGC_GERS_2017_GSM_PF_CC_class_name.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
    list_drop=[7,9,10,13,25,29,34,50,54,61,83,90,98]
    data_prof =data_prof[-data_prof["ID"].isin(list_drop)]
    for i in data_prof.ID:
        c=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["Zrmax_UTS"])][0]
        val=param.loc[param[1].isin(data_prof.loc[data_prof.ID==i]["Zrmax_UTS"])][1]
        UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/2017/Output/maxZr/output_test_maize_irri_"+str(int(c))+".df","rb"))
        data_id=UTS.groupby("id")
        ID_data=data_id.get_group(i)
        IRR.append([i,ID_data.Ir_auto.sum(),val.values[0],ID_data.TAW.max()])
    tab_irr=pd.DataFrame(IRR,columns=["ID","conso","maxzr","TAWMax"])
    tab_irr2=pd.merge(tab_irr,data_prof[["ID","MMEAU","Classe_Bruand"]],on='ID')
    tab_irr2.drop_duplicates(inplace=True)
    #  Isoler les parcelles en fonction du Sol
    tab_irr2["TEX"]="L"
    tab_irr2.loc[(tab_irr2.Classe_Bruand == "A"),'TEX']= "A"
    tab_irr2.loc[(tab_irr2.Classe_Bruand == "AL"),'TEX']="A"
    tab_irr2.loc[(tab_irr2.Classe_Bruand == "SL"),'TEX']="S"
    tab_irr2.loc[(tab_irr2.Classe_Bruand == "SA"),'TEX']="S"
    # tab_irr2.to_csv(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/tab_resu_GSM_PF_CC_maxzrUTS.csv")
    plt.figure(figsize=(7,7))
    slope, intercept, r_value, p_value, std_err = stats.linregress(tab_irr2.MMEAU.to_list(),tab_irr2.conso.to_list())
    bias=1/tab_irr2["MMEAU"].shape[0]*sum(tab_irr2.conso-np.mean(tab_irr2.MMEAU)) 
    rms = np.sqrt(mean_squared_error(tab_irr2.MMEAU,tab_irr2.conso))
    # labels, index = np.unique(tab_irr2["TEX"], return_inverse=True)
    plt.scatter(tab_irr2.MMEAU,tab_irr2.conso)
    # plt.legend(a.legend_elements()[0],labels)
    plt.xlim(-10,350)
    plt.ylim(-10,350)
    plt.xlabel("Quantité annuelles observées en mm ")
    plt.ylabel("Quantité annuelles modélisées en mm ")
    plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    
    rectangle = plt.Rectangle((95, 245),70,45, ec='gray',fc='gray',alpha=0.1)
    plt.gca().add_patch(rectangle)
    plt.text(100,280,"RMSE = "+str(round(rms,2))) 
    plt.text(100,270,"R = "+str(round(r_value,2)))
    plt.text(100,260,"Pente = "+str(round(slope,2)))
    plt.text(100,250,"Biais = "+str(round(bias,2)))
    # for i in enumerate(tab_irr2.ID):
    #     label = int(i[1])
    #     plt.annotate(label, # this is the text
    #           (tab_irr2["MMEAU"].iloc[i[0]],tab_irr2.conso.iloc[i[0]]), # this is the point to label
    #           textcoords="offset points", # how to position the text
    #           xytext=(-6,2), # distance from text to points (x,y)
    #           ha='center')
    plt.savefig(d["PC_disk"]+"/TRAITEMENT/"+name_run+"/plot_scatter_Irrigation_forcagemaxZr_RUMvalue_GSM_PF_CC_mai.png")
    
# =============================================================================
#  Analyse des partique avec RUM issue de UTS maps
# =============================================================================
    # tab_pratique=pd.merge(tab_irr2,parcellaire,on='ID')
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tab_pratique.MMEAU_x.to_list(),tab_pratique.conso.to_list())
    # bias=1/tab_pratique["MMEAU_x"].shape[0]*sum(tab_pratique.conso-np.mean(tab_pratique.MMEAU_x)) 
    # rms = np.sqrt(mean_squared_error(tab_pratique.MMEAU_x,tab_pratique.conso))
    # valid_sol_classe=pd.merge(data_prof[["Classe_Bruand",'ID']],tab_pratique,on='ID')
    # labels, index = np.unique(valid_sol_classe["Classe_Bruand_x"], return_inverse=True)
    
    # tab_pratique["residus"]=1.0
    # tab_pratique.loc[(tab_pratique.RESIDUS == 0.0),'residus']= 0.0
    # tab_pratique.loc[(tab_pratique.RESIDUS == 5.0),'residus']= 0.0
    # dist_resu={"Pas de résidus":0.0,"résidus":1.0}
    # TEX=tab_pratique.groupby("TEX")
    # for tex in ['A','L']:
    #     plt.figure(figsize=(7,7))
    #     TEX_resu=TEX.get_group(tex)
    #     res=TEX_resu.groupby("residus")
    #     for i,z in zip(dist_resu.values(),dist_resu.keys()):
    #         if i not in res.residus.count().index :
    #             continue 
    #         devenir=res.get_group(i)
    #         if i == 0.0:
    #             coloris='blue'
    #         elif i == 1.0 :
    #             coloris='orange'
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(devenir.MMEAU_x.to_list(),devenir.conso.to_list())
    #         bias=1/devenir["MMEAU_x"].shape[0]*sum(devenir.conso-np.mean(devenir.MMEAU_x)) 
    #         rms = np.sqrt(mean_squared_error(devenir.MMEAU_x,devenir.conso))
    #         plt.xlim(-10,350)
    #         plt.ylim(-10,350)
    #         plt.xlabel("Quantité annuelles observées en mm ")
    #         plt.ylabel("Quantité annuelles modélisées en mm ")
    #         plt.plot([-10.0, 350], [-10.0,350], 'black', lw=1,linestyle='--')
    #         # plt.errorbar(tab_irr2.MMEAU,tab_irr2.conso,marker=',',yerr=yerr,fmt='o',elinewidth=0.7,capsize = 4)
    #         a=plt.scatter(devenir.MMEAU_x,devenir.conso,label=z,color=coloris)
    #         plt.legend()
    #         plt.title(tex)
    #         if i ==0.0:
    #             rectangle = plt.Rectangle((10, 295),72,52, ec='blue',fc='blue',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(15,340,"RMSE = "+str(round(rms,2))) 
    #             plt.text(15,330,"R² = "+str(round(r_value,2)))
    #             plt.text(15,320,"Pente = "+str(round(slope,2)))
    #             plt.text(15,310,"Biais = "+str(round(bias,2)))
    #             plt.text(15,300,"Nb. = "+str(devenir.shape[0]))
    #         elif i == 1.0:
    #             rectangle = plt.Rectangle((95, 295),72,52, ec='orange',fc='orange',alpha=0.1)
    #             plt.gca().add_patch(rectangle)
    #             plt.text(100,340,"RMSE = "+str(round(rms,2))) 
    #             plt.text(100,330,"R² = "+str(round(r_value,2)))
    #             plt.text(100,320,"Pente = "+str(round(slope,2)))
    #             plt.text(100,310,"Biais = "+str(round(bias,2)))
    #             plt.text(100,300,"Nb. = "+str(devenir.shape[0]))
    #     plt.savefig(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/Plot_result/plot_scatter_Irri_RUM_inversion_RESIDUS_texture_sol_simpli_%s_PF_CC_GSM.png"%tex)


    