# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:45:14 2020

@author: Yann Pageot
"""


import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
from scipy.optimize import minimize
from sklearn.metrics import *
from scipy.optimize import linprog
from scipy import optimize
import random
import seaborn as sns
import pickle
import time
from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
# import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression
import argparse
from RUN_SAMIR_opti import params_update,params_opti ,select_color_date

def predict(x):
    return slope * x + intercept

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimisation SAMIR ')
    parser.add_argument('-path', dest='path',nargs='+',help="path file ",required = True)
    parser.add_argument('-optim',dest='optim',nargs='+',help='optimisation value',required = True)
    parser.add_argument('-optim2',dest='optim2',nargs='+',help='optimisation value')
    parser.add_argument('-name',dest='name_run',nargs='+',help='name run',required = True)
    parser.add_argument('-REW',dest='REW',nargs='+',help='Value REW ',required = True)
    parser.add_argument('-RU_start',dest='IniRU',nargs='+',help='Value init ru',required = True)
    parser.add_argument('-meteo',dest='meteo',nargs='+',help='source meteo data')
    parser.add_argument('-REW2',dest='REW2',nargs='+',help='source meteo data')
    parser.add_argument('-A_kcb',dest='akcb',nargs='+',help='slope_relation_NDVI/Kcb')
    parser.add_argument('-B_kcb',dest='bkcb',nargs='+',help='offset_relation_NDVI/Kcb')
    parser.add_argument('-PC',dest='Pc',nargs='+',help='PC_localisation', choices=('home','labo'))
    args = parser.parse_args()
    # years=["2008","2010","2012","2014","2015","2019"]
    years=["2017"]
    
    #  Add args User PC home/ PC labo
    result=[]
    for y in years:# 
        print (y)
        # name_run="RUN_MULTI_SITE_ICOS/RUN_OPTIMISATION_ICOS/SAMIR_calibr_Init_ru/Merlin_init_ru_optim_Fcover_maxzr_test_2019"
        name_run=str(args.name_run).strip("['']")
        print(name_run)
        # optimis_val="REW"
        optimis_val=str(args.optim).strip("['']")
        if len(optimis_val) < 6:
            optimis_val=optimis_val
        else:
            optimis_val=optimis_val
        print(r'===============')
        print(optimis_val)
        print(r'===============')
        d={}
        print(args.Pc)
        if args.Pc == ["home"] :
            d["data2"]='/mnt/d/THESE_TMP/'
            d["data"]="/mnt/h/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
            d["disk"]="/mnt/h/"
            user="yann"
        else:
            d["data"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
            user="pageot"
            d["disk"]="/run/media/pageot/Transcend/"
            
        d['SAMIR_run']="/mnt/d/THESE_TMP/TRAITEMENT/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d['SAMIR_run_Wind']="D:/THESE_TMP/TRAITEMENT/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d["PC_disk_Wind"]="D:/THESE_TMP/RUNS_SAMIR/DATA_Validation/"
        d['PC_disk_unix']="/mnt/d/THESE_TMP/RUNS_SAMIR/"
        d["SAMIR_run"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/"+name_run+"/"+str(y)+"/"
        d["SAMIR_run"]=str(args.path).strip("['']")+"/"+name_run+"/"+str(y)+"/"
        print(d["SAMIR_run"])
        print(str(args.meteo))
        #  Creation file du run 
        if name_run not in os.listdir(str(args.path).strip("['']")):
            os.mkdir ("%s/%s"%(str(args.path).strip("['']"),str(args.name_run).strip("['']")))
        # os.mkdir('%s/%s/'%(d["PC_labo"],name_run))
        if str(y) not in os.listdir(d["SAMIR_run"][:-5]):
            os.mkdir ('%s/%s/%s'%(str(args.path).strip("['']"),str(args.name_run).strip("['']"),str(y)))
        #  Déplacement all file 
        if str(args.meteo).strip("['']")=="SAFRAN":
            if "CACG" in name_run:
                os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/CACG_SAFRAN/"+str(y)+"/* %s"%(d['SAMIR_run']))
            elif "PKGC" in name_run:
                print("ici")
                # os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/PKGC/"+str(y)+"/* %s"%(d['SAMIR_run']))
                os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/PKGC_GERS/"+str(y)+"/* %s"%(d['SAMIR_run'])) # pour la préparation des données ADOUR_TARN
            elif "ASA" in name_run:
                os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/ASA/"+str(y)+"/* %s"%(d['SAMIR_run']))
            else:
                os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/SAFRAN_Irri_man/"+str(y)+"/* %s"%(d['SAMIR_run']))
        else:
            os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/ICOS_STAT_ss_Irri/"+str(y)+"/* %s"%(d['SAMIR_run']))
# =============================================================================
#   PFT burand estimation PF-CC .df  
# =============================================================================
        if "Merlin" in name_run:
            print('Merlin')
        #  Lecture file PF_CC
            PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/PF_CC_Bruand_parcelle.csv",index_col=[0],sep=',')
            if "max" in name_run:
                FC_Bru=float(PF_CC.Lamothe_pond_max["CC_Bruand"])
                WP_Bru=float(PF_CC.Lamothe_pond_max["PF_Bruand"])
                Sand_Ainse=float(PF_CC.Lamothe_pond_max["Sable"])
                Clay_Ainse=float(PF_CC.Lamothe_pond_max["Argile"])
            elif "min" in name_run:
                FC_Bru=float(PF_CC.Lamothe_pond_mod["CC_Bruand"])
                WP_Bru=float(PF_CC.Lamothe_pond_mod["PF_Bruand"])
                Sand_Ainse=float(PF_CC.Lamothe_pond_min["Sable"])
                Clay_Ainse=float(PF_CC.Lamothe_pond_min["Argile"])
            else:
                FC_Bru=float(PF_CC.Lamothe_pond_mod["CC_Bruand"])
                WP_Bru=float(PF_CC.Lamothe_pond_mod["PF_Bruand"])
                Sand_Ainse=float(PF_CC.Lamothe_pond_mod["Sable"])
                Clay_Ainse=float(PF_CC.Lamothe_pond_mod["Argile"])
            # modification df soil FC et WP 
            for p in ["WP_Bru","FC_Bru"]:
                tmp=open(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df","rb")
                data=pickle.load(tmp)
                tmp.close()
                valeur=globals()['%s'%p]
                data[p[:-4]]=valeur
                data.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df")
            #  Modifcation Soil texture 
            tmp1=open(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df","rb")
            data_tex=pickle.load(tmp1)
            tmp1.close()
            for tex in ["Sand_Ainse",'Clay_Ainse']:
                val=globals()['%s'%tex]
                data_tex[tex[:-6]]=val
            data_tex.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df")
        
            if Sand_Ainse >= 0.80 :
                REW = 20 - 0.15 * Sand_Ainse
            elif Clay_Ainse >= 0.50 : 
                REW = 11 - 0.06 * Clay_Ainse 
            elif (Sand_Ainse < 0.80) and (Clay_Ainse < 0.50):
                REW = 8 + 0.008 * Clay_Ainse
        elif "CACG" in name_run : 
            print('CACG parcelle')
        #  Lecture file PF_CC
            if "GSM" in name_run:
                PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
            elif "RRP" in name_run: 
                PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/RRP/Extract_RRP_GERS_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
            else:
                print("soil_UTS")
                if "varplus20" in name_run :
                    PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj_varplus20.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
                elif "varmo20" in name_run :
                    PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj_varmo20.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
                else:
                    PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_CACG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
            # PF_CC.dropna(inplace=True)
            FC_Bru=PF_CC["CC_mean"]
            WP_Bru=PF_CC["PF_mean"]
            Sand_Ainse=PF_CC["Sable"]/100
            Clay_Ainse=PF_CC["Argile"]/100
            # modification df soil FC et WP 
            for p in ["WP_Bru","FC_Bru"]:
                tmp=open(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df","rb")
                data=pickle.load(tmp)
                tmp.close()
                valeur=globals()['%s'%p]
                data[p[:-4]]=valeur.values
                data.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df")
            #  Modifcation Soil texture 
            tmp1=open(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df","rb")
            data_tex=pickle.load(tmp1)
            tmp1.close()
            for tex in ["Sand_Ainse",'Clay_Ainse']:
                val=globals()['%s'%tex]
                data_tex[tex[:-6]]=val.values
            data_tex.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df")
            
            
        elif "PKGC" in name_run : 
            print('PKGC parcelle')
        #  Lecture file PF_CC
            if "GSM" in name_run:
                # PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKCG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
                print("data soil GSM use")
                PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_Adour_Tarn_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
            elif "RRP" in name_run: 
                PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/RRP/Extract_RRP_GERS_parcelle_PKCG_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')
                print("data soil RRP use")
            else:
                if "varmo20" in name_run:
                    PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_2017_UTS_maj_varmo20.csv",index_col=[0],sep=',',encoding='latin-1',decimal='.')
                    print("incertitude ---")
                elif "varplus20" in name_run:
                    PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_2017_UTS_maj_varplus20.csv",index_col=[0],sep=',',encoding='latin-1',decimal='.')
                    print("incertitude +++")
                else:
                    PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=',',encoding='latin-1',decimal=',')
                    print('ICI')
            # PF_CC.dropna(inplace=True)Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj
            FC_Bru=PF_CC["CC_mean"] ## ATTENTION modif 
            WP_Bru=PF_CC["PF_mean"]
            Sand_Ainse=PF_CC["Sable"]/100
            Clay_Ainse=PF_CC["Argile"]/100
            # modification df soil FC et WP 
            for p in ["WP_Bru","FC_Bru"]:
                tmp=open(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df","rb")
                data=pickle.load(tmp)
                tmp.close()
                valeur=globals()['%s'%p]
                data[p[:-4]]=valeur.values
                data.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df")
            #  Modifcation Soil texture 
            tmp1=open(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df","rb")
            data_tex=pickle.load(tmp1)
            tmp1.close()
            for tex in ["Sand_Ainse",'Clay_Ainse']:
                val=globals()['%s'%tex]
                data_tex[tex[:-6]]=val.values
            data_tex.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df")
        elif "ASA" in name_run : 
            print('ASA parcelle')
        #  Lecture file PF_CC
            PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_ASA_"+str(y)+"_UTS_maj.csv",index_col=[0],sep=';',encoding='latin-1',decimal=',')

            # PF_CC.dropna(inplace=True)Extract_RRP_Rigou_parcelle_PKCG_2017_UTS_maj
            FC_Bru=PF_CC["CC_mean"]
            WP_Bru=PF_CC["PF_mean"]
            Sand_Ainse=PF_CC["Sable"]/100
            Clay_Ainse=PF_CC["Argile"]/100
            # modification df soil FC et WP 
            for p in ["WP_Bru","FC_Bru"]:
                tmp=open(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df","rb")
                data=pickle.load(tmp)
                tmp.close()
                valeur=globals()['%s'%p]
                data[p[:-4]]=valeur.values
                data.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df")
            #  Modifcation Soil texture 
            tmp1=open(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df","rb")
            data_tex=pickle.load(tmp1)
            tmp1.close()
            for tex in ["Sand_Ainse",'Clay_Ainse']:
                val=globals()['%s'%tex]
                data_tex[tex[:-6]]=val.values
            data_tex.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/Soil_texture.df")
            # print(Sand_Ainse)
            # if Sand_Ainse >= 0.80 :
            #     REW = 20 - 0.15 * Sand_Ainse
            # elif Clay_Ainse >= 0.50 : 
            #     REW = 11 - 0.06 * Clay_Ainse 
            # elif (Sand_Ainse < 0.80) and (Clay_Ainse < 0.50):
            #     REW = 8 + 0.008 * Clay_Ainse
# =============================================================================
#         Incertitude sur le Fcover
# =============================================================================
        # Lecture du Fcover 
#         Fco=open(d["SAMIR_run"]+"Inputdata/maize_irri/Fcover.df","rb")
#         Fcover=pickle.load(Fco)
#         Fco.close()
#         if "Fcover_pl20" in name_run:
#             print("add +20 %")
#             Fcover.FCov=Fcover.FCov+(20*Fcover.FCov/100)
#         elif "Fcover_m20" in name_run:
#             print("add -20 %")
#             Fcover.FCov=Fcover.FCov-(20*Fcover.FCov/100)
#         Fcover.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/Fcover.df")
# # =============================================================================
      # Calcule REW allen 2005 
# =============================================================================
        # Sand = mean([0.471,0.646]) # ensemble de la colonne sol 
        # Clay = 0.5026
        if "FAO" in name_run:
            PF_CC=pd.read_csv(d["disk"]+"/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/PF_CC_Bruand_parcelle.csv",index_col=[0],sep=',')
            #  Lecture file PF_CC
            if "max" in name_run:
                FC_Bru=float(PF_CC.Lamothe_pond_max["CC_Bruand"])
                WP_Bru=float(PF_CC.Lamothe_pond_max["PF_Bruand"])
                Sand=float(PF_CC.Lamothe_pond_max["Sable"])
                Clay=float(PF_CC.Lamothe_pond_max["Argile"])
            elif "min" in name_run:
                FC_Bru=float(PF_CC.Lamothe_pond_mod["CC_Bruand"])
                WP_Bru=float(PF_CC.Lamothe_pond_mod["PF_Bruand"])
                Sand=float(PF_CC.Lamothe_pond_min["Sable"])
                Clay=float(PF_CC.Lamothe_pond_min["Argile"])
            else:
                FC_Bru=float(PF_CC.Lamothe_pond_mod["CC_Bruand"])
                WP_Bru=float(PF_CC.Lamothe_pond_mod["PF_Bruand"])
                Sand=float(PF_CC.Lamothe_pond_mod["Sable"])
                Clay=float(PF_CC.Lamothe_pond_mod["Argile"])
                
            for p in ["WP_Bru","FC_Bru"]:
                tmp=open(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df","rb")
                data=pickle.load(tmp)
                tmp.close()
                valeur=globals()['%s'%p]
                data[p[:-4]]=valeur
                data.to_pickle(d["SAMIR_run"]+"Inputdata/maize_irri/"+str(p)[:-4]+".df")
    
            if Sand >= 0.80 :
                REW = 20 - 0.15 * Sand
            elif Clay >= 0.50 : 
                REW = 11 - 0.06 * Clay 
            elif (Sand < 0.80) and (Clay < 0.50):
                REW = 8 + 0.008 * Clay
# =============================================================================
#         # Calibration Init_Ru, année n-1
# =============================================================================
        if "CACG" not in name_run and "PKGC" not in name_run and "ASA" not in name_run:
            y1=int(y)-1
    
            if str(args.meteo).strip("['']")=="SAFRAN":
                # print("scp -r /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/SAFRAN/"+str(y1)+"/* %s"%(d['SAMIR_run']))
                os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/SAFRAN_Irri_man/"+str(y1)+"/ %s"%(d["SAMIR_run"]))
            else:
                os.system("scp -r "+d["data"]+"/TRAITEMENT/RUNS_SAMIR/DATA_SCP_ICOS/ICOS_STAT_ss_Irri/"+str(y1)+"/ %s"%(d['SAMIR_run']))
            d['SAMIR_run_RU']=str(args.path).strip("['']")+"/"+name_run+"/"+str(y)+"/"+str(y1)+"/"
            params_update(d['SAMIR_run_RU']+"/Inputdata/param_SAMIR12_13.csv",
                          d['SAMIR_run_RU']+"/Inputdata/param_modif.csv",date_start=str(y1)+str('0101'),date_end=str(y1)+str('1231'),
                          Ze=150,REW=REW,minZr=150,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(str(args.IniRU).strip("['']")),Irrig_auto=1,Irrig_man=0,A_kcb=1.50,m=1, Koffset=-0.23)
            if "Output" in os.listdir(d['SAMIR_run_RU']):
                print ("existing file")
            else :
                os.mkdir ("%s/Output"%d['SAMIR_run_RU']) # allows create file via call system
            if optimis_val in os.listdir(d['SAMIR_run_RU']+"/Output/"):
                print ("existing file")
            else :
                os.mkdir ("%s/Output/%s"%(d['SAMIR_run_RU'],optimis_val)) 
    
            if "LAI" in name_run :
                os.system('python /home/'+user+'/sources/modspa2_LAI/modspa2/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run_RU']+' -dd '+d['SAMIR_run_RU']+'/Inputdata/ -m /*/meteo.df -n /*/LAI'+str(y1)+'.df -fcover /*/FCOVER.df -fc /*/FC.df -wp /*/WP.df  --fc_input  -o Output/'+optimis_val+'/output_test -p param_modif.csv ')
            else:
                os.system('python /home/'+user+'/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run_RU']+' -dd '+d['SAMIR_run_RU']+'/Inputdata/ -m meteo.df -n NDVI'+str(y1)+'.df -fc FC.df -wp WP.df -o Output/'+optimis_val+'/output_test.df -p param_modif.csv --formaREW Merlin -soiltext Soil_texture.df ')
               
              # Extraction du résultat
            df=pickle.load(open(d["SAMIR_run_RU"]+'/Output/'+optimis_val+'/output_test.df','rb'))
            result_init_cops=df.groupby("LC")
            result_init=result_init_cops.get_group("maize_irri")
            result_init=result_init[["date","SWC1i","SWC1p","SWC2","SWC3"]]
            ru_init=result_init[["date","SWC1i","SWC1p","SWC2","SWC3"]].loc[result_init.date==str(y1)+"-12-31"]
            # RUn1=ru_init.SWC2.values
            # print(r'===============')
            # print(RUn1)
            # print(r'===============')
            RUNsoln1 = ru_init[["SWC1i","SWC1p","SWC2","SWC3"]].values.mean()
            # moyenne pondére à l'épaisseur 
            # print(ru_init["SWC1i"].values[0]) # probleme SWC1p ou SWC1i si irrigation ou non
            RUSOL_ponde=np.average([ru_init["SWC1i"].values[0],ru_init["SWC2"].values[0],ru_init["SWC3"].values[0]], weights=[150,df["Zr"].loc[df.date==str(y1)+"-12-31"].values[0] , df["Zd"].loc[df.date==str(y1)+"-12-31"].values[0]])
            print(r'===============')
            print(RUSOL_ponde)
            print(r'===============')

        classes=["maize_irri"]
        if "LAI" in name_run:
            b=open(d["SAMIR_run"]+"/Inputdata/maize_irri/LAI"+str(y)+".df","rb")
            LAI=pickle.load(b)
            b.close()
            if optimis_val =="REW" :
                timestart=str(y)+"-07-01"
                solnu=LAI.loc[(LAI.LAI>0.1)&(LAI.date<timestart)]
                lastdate=solnu.iloc[-1]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['PC_labo']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                              Ze=150,REW='optim',minZr=150,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                               ligne_OS=7,Ze=150,REW='optim',minZr=150,maxZr=1500,Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin",ligne_OS=2)
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str("lastdate"),
                              Ze=150,REW='optim',minZr=150,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-100/100/10/lin")
            elif optimis_val =="maxZr" :
                timestart=str(y)+"-05-01"
                vege=LAI.loc[(LAI.LAI>0.2)&(LAI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                              Ze=150,REW=float(str(args.REW).strip("['']")),minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                                  ligne_OS=7,Ze=150,REW=float(str(args.REW2).strip("['']")),minZr=150,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/500/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/100/lin",ligne_OS=2)
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                             d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                             Ze=150,REW=float(str(args.REW).strip("['']")),minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=1,Irrig_auto=1,plateau=30,Irrig_man=0,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2999/100/lin")
            else:
                print('two optimisation')
                timestart=str(y)+"-05-01"
                vege=LAI.loc[(LAI.LAI>1)&(LAI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                              Ze=150,REW="optim",minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                                  ligne_OS=7,Ze=150,REW="optim",minZr=150,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=ffloat(RUn1),Irrig_auto=0,Irrig_man=1,plateau=30,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",param2="REW",value_P2="-50/40/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",ligne_OS=2,param2="REW",value_P2="-50/40/10/lin")
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                             d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                             Ze=150,REW="optim",minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUn1),Irrig_auto=1,Irrig_man=0,A_kcb=float(str(args.akcb).strip("['']")),Lame_max=30, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",param2="REW",value_P2="-50/40/10/lin")
        else:
            # Définir période sol nu
            b=open(d["SAMIR_run"]+"/Inputdata/maize_irri/NDVI"+str(y)+".df","rb")
            NDVI=pickle.load(b)
            b.close()
            if optimis_val =="REW" :
                timestart=str(y)+"-07-01"
                solnu=NDVI.loc[(NDVI.NDVI>0.25)&(NDVI.date<timestart)]
                lastdate=solnu.iloc[-1]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['PC_labo']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                              Ze=150,REW='optim',minZr=150,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUSOL_ponde),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(lastdate),
                               ligne_OS=7,Ze=150,REW='optim',minZr=150,maxZr=1500,Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(RUSOL_ponde),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-50/40/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-50/40/10/lin",ligne_OS=2)
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0302'),date_end=str(y)+str(1026),
                              Ze=150,REW='optim',minZr=150,maxZr=1500,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUSOL_ponde),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")),m=0.15, Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="REW",value_P1="-50/40/10/lin")
            elif optimis_val =="maxZr" :
                timestart=str(y)+"-05-01"
                vege=NDVI.loc[(NDVI.NDVI>0.25)&(NDVI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                              Ze=150,REW=float(str(args.REW).strip("['']")),minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                                  ligne_OS=7,Ze=150,REW=float(str(args.REW2).strip("['']")),minZr=150,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2500/250/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2500/250/lin",ligne_OS=2)
                else:
                    if 'irri_auto' in name_run:
                        params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                                 d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0101'),date_end=str(y)+str('1231'),#
                                 Ze=150,REW=8,minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=1,Irrig_auto=1,Irrig_man=0,Plateau=0,Lame_max=30,m=1,minDays=10,p=0.55,Start_date_Irr=str(y)+str('0501'),A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                        params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="400/1800/50/lin")
                    else:
                        print("Irri manuel activé")
                        params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                                 d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0101'),date_end=str(y)+str('1231'),
                                 Ze=150,REW=REW,minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUSOL_ponde),Irrig_auto=0,Irrig_man=1,Plateau=0,Lame_max=30,m=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                        params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="800/1200/50/lin")
            elif optimis_val =="p" :
                timestart=str(y)+"-05-01"
                vege=NDVI.loc[(NDVI.NDVI>0.25)&(NDVI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                              Ze=150,REW=float(str(args.REW).strip("['']")),minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1031'),
                                  ligne_OS=7,Ze=150,REW=float(str(args.REW2).strip("['']")),minZr=150,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(RUn1),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2500/250/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2500/250/lin",ligne_OS=2)
                else:
                    if 'irri_auto' in name_run:
                        params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                                 d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0101'),date_end=str(y)+str('1231'),#
                                 Ze=150,REW=8,minZr=150,maxZr=600,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=1,Irrig_auto=1,Irrig_man=0,Plateau=0,Lame_max=30,m=1,minDays=10,p='optim',Start_date_Irr=str(y)+str('0501'),A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                        params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="p",value_P1="0.4/0.7/0.05/lin")
                    else:
                        print("Irri manuel activé")
                        params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                                 d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0101'),date_end=str(y)+str('1231'),
                                 Ze=150,REW=REW,minZr=150,maxZr=1000,Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUSOL_ponde),Irrig_auto=0,Irrig_man=1,Plateau=0,Lame_max=30,m=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                        params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="800/1200/50/lin")

            else:
                print('two optimisation')
                timestart=str(y)+"-05-01"
                vege=NDVI.loc[(NDVI.NDVI>0.25)&(NDVI.date>timestart)]
                lastdate=vege.iloc[0]["date"].strftime('%m-%d').replace("-", "")
                if len(classes)==2:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                              d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                              Ze=150,REW="optim",minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=float(RUSOL_ponde),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_update(d['SAMIR_run']+"/Inputdata/param_modif.csv",
                                  d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str(lastdate),date_end=str(y)+str('1026'),
                                  ligne_OS=7,Ze=150,REW="optim",minZr=150,maxZr='optim',Zsoil=3000,DiffE=5,DiffR=5,Init_RU=float(RUSOL_ponde),Irrig_auto=0,Irrig_man=1,A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",param2="REW",value_P2="-50/40/10/lin")
                    params_opti(d["SAMIR_run"]+"/Inputdata/test_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="500/2900/250/lin",ligne_OS=2,param2="REW",value_P2="-50/40/10/lin")
                else:
                    params_update(d['SAMIR_run']+"/Inputdata/param_SAMIR12_13.csv",
                             d['SAMIR_run']+"/Inputdata/param_modif.csv",date_start=str(y)+str('0101'),date_end=str(y)+str('1231'),
                             Ze=150,REW=8,minZr=150,maxZr='optim',Zsoil=3000,DiffE=0.00001,DiffR=0.00001,Init_RU=1,Irrig_auto=1,Irrig_man=0,m=1,Plateau=0,minDays=10,p="optim",Lame_max=30,Start_date_Irr=str(y)+str('0501'),A_kcb=float(str(args.akcb).strip("['']")), Koffset=float(str(args.bkcb).strip("['']")))
                    params_opti(d["SAMIR_run"]+"/Inputdata/param_SAMIR12_13_optim.csv",output_path=d["SAMIR_run"]+"/Inputdata/test_optim.csv",param1="maxZr",value_P1="400/1800/50/lin",param2="p",value_P2="0.40/0.7/0.05/lin")
        #  Lancement du code
        if "Output" in os.listdir(d['SAMIR_run']):
            print ("existing file")
        else :
            os.mkdir ("%s/Output"%d['SAMIR_run']) # allows create file via call system
        #  création file Output
        if optimis_val in os.listdir(d['SAMIR_run']+"/Output/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s"%(d['SAMIR_run'],optimis_val)) 
            
        if "Plot" in os.listdir(d['SAMIR_run']+"/Output/"+optimis_val+"/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s/Plot"%(d['SAMIR_run'],optimis_val)) 
            
        if "Plot_dyna" in os.listdir(d['SAMIR_run']+"/Output/"+optimis_val+"/Plot/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s/Plot/Plot_dyna"%(d['SAMIR_run'],optimis_val)) 
            
        if "CSV" in os.listdir(d['SAMIR_run']+"/Output/"+optimis_val+"/"):
            print ("existing file")
        else :
            os.mkdir ("%s/Output/%s/CSV"%(d['SAMIR_run'],optimis_val)) 
        # os.environ["PYTHONPATH"] = "/home/pageot/sources/modspa2/Code/models/main/:$PYTHONPATH"
        if "LAI" in name_run :
            os.system('python /home/'+user+'/sources/modspa2_LAI/modspa2/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m /*/meteo.df -n /*/LAI'+str(y)+'.df -fcover /*/FCOVER.df -fc /*/FC.df -wp /*/WP.df  --fc_input  -o Output/'+optimis_val+'/output_test -p param_modif.csv  -optim test_optim.csv --cal ET ')
        else:
            if "Merlin" in name_run :
                if "Fcover" in name_run:
                    os.system('python /home/'+user+'/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m meteo.df -n NDVI'+str(y)+'.df -fcover FCOVER.df -fc FC.df -wp WP.df  --fc_input  -o Output/'+optimis_val+'/output_test.df -p param_modif.csv  -optim test_optim.csv --cal ET Ir_auto NDVI Ks Kei Kep Irrig fewi fewp FCov Dei Dr DP Dd SWC1 Kri TEW TAW --cpu 6 --formaREW Merlin -soiltext Soil_texture.df')
                else:
                    os.system('python /home/'+user+'/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m meteo.df -n NDVI'+str(y)+'.df  -fc FC.df -wp WP.df  -o Output/'+optimis_val+'/output_test.df -p param_modif.csv  -optim test_optim.csv --cal ET Ir_auto NDVI Ks Kei Kep Irrig fewi fewp FCov Dei Dr DP Dd SWC1 Kri TEW TAW --cpu 6 --formaREW Merlin -soiltext Soil_texture.df')
            elif "CACG" in name_run or "PKGC" in name_run or "ASA" in name_run:
                if "Fcover" in name_run:
                    os.system('python /home/'+user+'/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m meteo.df -n NDVI'+str(y)+'.df -fcover Fcover.df -fc FC.df -wp WP.df  --fc_input  -o Output/'+optimis_val+'/output_test.df -p param_modif.csv  -optim test_optim.csv --cal ET Ir_auto NDVI Ks Kei Kep Irrig fewi fewp FCov Dei Dr DP Dd SWC1 Kri TEW TAW Zr --cpu 1 --formaREW Merlin -soiltext Soil_texture.df')
                else:
                    os.system('python /home/'+user+'/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m meteo.df -n NDVI'+str(y)+'.df  -fc FC.df -wp WP.df  -o Output/'+optimis_val+'/output_test.df -p param_modif.csv  -optim test_optim.csv --cal ET Ir_auto NDVI Ks Kei Kep Irrig fewi fewp FCov Dei Dr DP Dd SWC1 Kri TEW TAW Zr --cpu 3 --formaREW Merlin -soiltext Soil_texture.df')
            elif "FAO" in name_run:
                print("FAO use")
                if "Fcover" in name_run:
                    os.system('python /home/'+user+'/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m meteo.df -n NDVI'+str(y)+'.df  -fc FC.df -wp WP.df --fc_input -fcover FCOVER.df -o Output/'+optimis_val+'/output_test.df -p param_modif.csv  -optim test_optim.csv --cal ET Ir_auto NDVI Ks Kei Kep Irrig fewi fewp FCov Dei Dr DP Dd SWC1 Kri TEW TAW Zr --cpu 3')
                else:
                    print('ici sans Fcover')
                    os.system('python /home/'+user+'/sources/modspa_SAMIR/modspa/Code/models/main/runSAMIR.py -wd '+d['SAMIR_run']+' -dd '+d['SAMIR_run']+'/Inputdata/ -m meteo.df -n NDVI'+str(y)+'.df  -fc FC.df -wp WP.df  -o Output/'+optimis_val+'/output_test.df -p param_modif.csv  -optim test_optim.csv --cal ET Ir_auto NDVI Ks Kei Kep Irrig fewi fewp FCov Dei Dr DP Dd SWC1 Kri TEW TAW --cpu 3')
        for classe in classes:
            if len(optimis_val) > 5:
                param=pd.read_csv(d["SAMIR_run"]+"Output/"+optimis_val+"/output_test_"+classe+"_param.txt",header=None,skiprows=2,sep=";")
                param.set_index(0,inplace=True)
            else:
                param=pd.read_csv(d["SAMIR_run"]+"Output/"+optimis_val+"/output_test_"+classe+"_param.txt",header=None,skiprows=1,sep=";")
                param.set_index(0,inplace=True)
              # Récuparation data_validation ETR
            if "CACG" not in name_run and "PKGC" not in name_run and "ASA" not in name_run: 
                if classe =='maize_irri':
                    # ETR_lam=pd.read_csv("D:/THESE_TMP/DATA_ETR_CESBIO/DATA_ETR_LAM/DATA_ETR_LAM_ICOS/ETR_LAM"+str(y)+".csv",decimal='.')
                    ETR_lam=pd.read_csv(d["data"]+"/DATA_ETR_CESBIO/DATA_ETR_LAM/DATA_ETR_LAM_ICOS/ETR_LAM"+str(y)+".csv",decimal='.') # flux corrigées a mettre 
                    ETR_lam.date=pd.to_datetime(ETR_lam["date"],format='%Y-%m-%d')
                else:
                    ETR_gri=pd.read_csv(d["data"]+"/DATA_ETR_CESBIO/DATA_ETR_GRIGNON/DATA_ETR_GRIGNON_ICOS/ETR_GRIGNON"+str(y)+".csv",decimal='.')
                    ETR_gri.date=pd.to_datetime(ETR_gri["date"],format='%Y-%m-%d')
                # Récupération des output de la simulation 
                concat_ETR=[]
                params=[]
                for run in os.listdir(d["SAMIR_run"]+"Output/"+optimis_val+"/"):
                    print(run)
                    if classe in run and "txt" not in run:
                        num_run=run[23:-3]
                        a=open(d["SAMIR_run"]+"Output/"+optimis_val+"/"+run,"rb")
                        output_sim=pickle.load(a)
                        ETRmod=output_sim[["ET","date"]]
                        a.close()
                        #  Récuper le couple de paramètre que je vais varier
                        if len(optimis_val) < 6:
                            parametre=param.iloc[int(num_run)]
                            parametre1=parametre[1]
                            # print (r'para %s;'%(parametre1)) 
                        else: 
                            parametre=param.iloc[int(num_run)]
                            parametre1=parametre[1]
                            parametre2=parametre[2]
                            # print (r'para %s; para2 %s'%(parametre1,parametre2))    
                        # localiser les nan dans ETR, les supprimer ainsi que les dates pour ensuite comparer 
                        if classe =='maize_irri':
                            # dfETR=pd.concat([ETR_lam,ETRmod],axis=1)
                            dfETR=pd.merge(ETR_lam,ETRmod,on=["date"])
                        else:
                            # dfETR=pd.concat([ETR_gri,ETRmod],axis=1)
                            dfETR=pd.merge(ETR_gri,ETRmod,on=["date"])
                        dfETR.columns=["date",'LE','ET']
                        dfETR=dfETR.set_index('date').resample("D").asfreq()
                        # dfETR=dfETR.set_index('date').resample("W").asfreq()
                        # dfETR.to_csv(d["SAMIR_run"]+"Output/"+optimis_val+"/CSV/ETR_%s_%s.csv"%(classe,num_run))
                        dfETR.dropna(inplace=True)
                        if dfETR.shape[0]==0:
                            print("%s non utilisable " %y) # pas de date similaire entre modélisation et ETRobs
                            continue
                        else:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(dfETR.LE.to_list(),dfETR.ET.to_list())
                            bias=1/dfETR.shape[0]*sum(np.mean(dfETR.ET)-dfETR.LE) 
                            fitLine = predict(dfETR.LE)
                            # plt.figure(figsize=(7,7))
                            # plt.plot([0.0, 10], [0.0,10], 'r-', lw=2)
                            # plt.plot(dfETR.LE,fitLine,linestyle="-")
                            # plt.scatter(dfETR.LE,dfETR.ET,s=9)
                            # plt.xlabel("ETR OBS")
                            # plt.ylabel("ETR model")
                            # plt.xlim(0,10)
                            # plt.ylim(0,10)
                            rms = mean_squared_error(dfETR.LE,dfETR.ET)
                        #     plt.text(8,min(dfETR.ET)+0.1,"RMSE = "+str(round(rms,2)))
                        #     plt.text(8,min(dfETR.ET)+0.3,"R² = "+str(round(r_value,2)))
                        #     plt.text(8,min(dfETR.ET)+0.5,"Pente = "+str(round(slope,2)))
                        #     plt.text(8,min(dfETR.ET)+0.7,"Biais = "+str(round(bias,2)))
                        #     plt.savefig(d["SAMIR_run"]+"Output/"+optimis_val+"/Plot/plt_scatter_ETR_%s_%s_%s.png"%(classe,optimis_val,str(int(parametre1))))
                        #     plt.figure(figsize=(7,7))
                        #     plt.plot(dfETR.index,dfETR.LE,label='ETR_obs',color="black")
                        #     plt.plot(dfETR.index,dfETR.ET,label='ETR_mod',color='red')
                        #     plt.ylabel("ETR")
                        #     plt.ylim(0,10)
                        #     plt.legend()
                        #     plt.savefig(d["SAMIR_run"]+"Output/"+optimis_val+"/Plot/Plot_dyna/plt_dynamique_ETR_%s_%s_%s.png"%(classe,optimis_val,str(int(parametre1))))
                            if len(optimis_val) < 6:
                                result.append([num_run,parametre1,rms,bias,r_value,y,classe])
                                concat_ETR.append(ETRmod["ET"])
                                params.append([num_run,parametre1])
                            else: 
                                result.append([num_run,parametre1,parametre2,rms,bias,r_value,y,classe])
                                concat_ETR.append(ETRmod["ET"])
                                params.append([num_run,parametre1,parametre2])
                if len(optimis_val) < 6:
                    resultat=pd.DataFrame(result,columns=["Num_run","Param1","RMSE",'bias','R','years','OS'])
                    # conca=pd.DataFrame(concat_ETR)
                    # para=pd.DataFrame(params)
                    # conca.columns=ETRmod.date
                    # conca=conca.T
                    # conca.columns=para[1].values
                    conca=pd.DataFrame(concat_ETR)
                    para=pd.DataFrame(params)
                    a=pd.MultiIndex.from_frame(para,names=['num_run',"maxZr"])
                    RESU=pd.DataFrame(conca.values,index=a,columns=ETRmod.date)
                    RESU.sort_index(inplace=True)
                else:
                    resultat=pd.DataFrame(result,columns=["Num_run","Param1","Param2","RMSE",'bias','R','years','OS'])
                    # Utiliser le multi_index de pandas avec Date en columns et params1 /2/3 en index 
                    conca=pd.DataFrame(concat_ETR)
                    para=pd.DataFrame(params)
                    a=pd.MultiIndex.from_frame(para,names=['num_run',"REW","maxZr"])
                    RESU=pd.DataFrame(conca.values,index=a,columns=ETRmod.date)
                    RESU.sort_index(inplace=True)
                    # RESU.[-50.0][1000.0] # selection les ET REW -50 et maxZr = 1000
                RESU.to_csv(d["SAMIR_run"][:-5]+"LUT_ETR%s.csv"%(y))
                resultat.to_csv(d["SAMIR_run"][:-5]+"param_RMSE%s.csv"%(optimis_val))
            else:
                 params=[]
                 concat_ir=[]
                 for run in os.listdir(d["SAMIR_run"]+"Output/"+optimis_val+"/"):
                    print(run)
                    if classe in run and "txt" not in run:
                        num_run=run[23:-3]
                        print(num_run)
                        a=open(d["SAMIR_run"]+"Output/"+optimis_val+"/"+run,"rb")
                        output_sim=pickle.load(a)
                        a.close()
                        Irrigation_mod=output_sim[["Ir_auto","date","id"]]
                        num_run=int(num_run)-1
                        parametre=param.iloc[int(num_run)]
                        parametre1=parametre[1]
                        params.append([num_run,parametre1])
                        para=pd.DataFrame(params)
                        a=pd.MultiIndex.from_frame(para,names=['num_run',"maxZr"])
                        concat_ir.append(Irrigation_mod["Ir_auto"])
                        conca=pd.DataFrame(concat_ir)
                        RESU=pd.DataFrame(conca.values,index=a,columns=Irrigation_mod.date)
                        RESU.sort_index(inplace=True)
                        RESU=RESU.T
                        RESU["ID"]=Irrigation_mod["id"].values
                 RESU.to_csv(d["SAMIR_run"][:-5]+"LUT_%s.csv"%(y))
