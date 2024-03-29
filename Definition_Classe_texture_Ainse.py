#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:10:50 2021

@author: pageot

trage de texture Ainse + Table PF burand 2003
"""

import os
import pandas as pd
import geopandas as geo
import numpy as np
import math

def Classe_texture (df):
    Argile=df["Argile"]
    Limon=df["Limon"]
    Sable=df["Sable"]
    if (Argile.values[0]) <=8 and (Limon.values[0]) >= 80  and  (Sable.values[0]) <= 15:
        return "LL"
    elif (Argile.values[0]) <=18 and (Limon.values[0]) >= 70 and  (Sable.values[0]) <= 15 :
        return "LM"
    elif (Argile.values[0]) <=30 and (Limon.values[0]) >= 55 and  (Sable.values[0]) <= 15 :
        return "LA"
    elif (Argile.values[0]) <=45 and (Limon.values[0]) >= 35 and  (Sable.values[0]) <= 20 :
        return "AL"
    elif (Argile.values[0]) >=45 :
        return "ALO"
    elif (Argile.values[0]) <=45 and (Argile.values[0]) >=30  and (Limon.values[0]) >= 10 and (Limon.values[0]) <= 50 and  (Sable.values[0]) <= 45 and (Sable.values[0]) >= 20 :
        return "A"
    elif (Argile.values[0]) <=30 and (Argile.values[0]) >=18  and (Limon.values[0]) >= 35 and (Limon.values[0]) <= 65 and  (Sable.values[0]) <= 35 and (Sable.values[0]) >= 15 :
        return "LAS"
    elif (Argile.values[0]) <=18 and (Argile.values[0]) >=8  and (Limon.values[0]) >= 48 and (Limon.values[0]) <= 68 and  (Sable.values[0]) <= 35 and (Sable.values[0]) >= 15 :
        return "LMS"
    elif (Argile.values[0]) <=8 and (Limon.values[0]) >= 40 and (Limon.values[0]) <= 78 and  (Sable.values[0]) <= 55 and (Sable.values[0]) >= 15 :
        return "LLS"
    elif (Argile.values[0]) <=18 and (Argile.values[0]) >= 8 and (Limon.values[0]) >= 28 and (Limon.values[0]) <= 58 and  (Sable.values[0]) <= 55 and (Sable.values[0]) >= 35 :
        return "LS"
    elif (Argile.values[0]) <=30 and (Argile.values[0]) >= 18 and (Limon.values[0]) >= 20 and (Limon.values[0]) <= 35 and  (Sable.values[0]) <= 55 and (Sable.values[0]) >= 35 :
        return "LSA"
    elif (Argile.values[0]) <=45 and (Argile.values[0]) >= 25 and (Limon.values[0]) >= 0 and (Limon.values[0]) <= 25 and  (Sable.values[0]) <= 75 and (Sable.values[0]) >= 45 :
        return "AS"
    elif (Argile.values[0]) <=25 and (Argile.values[0]) >= 15 and (Limon.values[0]) >= 0 and (Limon.values[0]) <= 31 and  (Sable.values[0]) <= 86 and (Sable.values[0]) >= 75 :
        return "SA"
    elif (Argile.values[0]) <=15 and (Limon.values[0]) <= 20 and (Limon.values[0]) >= 0 and  (Sable.values[0]) <= 86 and (Sable.values[0]) >= 100 :
        return "S"
    else:
        return "SL"
def PF_bruand (df):
    # Prof=df["Prof"]
    Classe=df["Classe_Bruand"]
    # if Prof >= 30 :
    if Classe.values[0] == "LS":
        return [0,274,0,102]
    elif Classe.values[0] == 'ALO':
        return [0.408,0.297]
    elif Classe.values[0] == 'AL':
        return [0.335,0.222]
    elif Classe.values[0] == 'AS':
        return [0.296,0.201]
    elif Classe.values[0] == 'A':
        return [0.315,0.221]
    elif Classe.values[0] == 'LA':
        return [0.312,0.163]
    elif Classe.values[0] == 'LAS':
        return [0.304,0.156]
    elif Classe.values[0] == 'LSA':
        return [0.262,0.158]
    elif Classe.values[0] == 'LM':
        return [0.321,0.114]
    elif Classe.values[0] == 'LMS':
        return [0.330,0.129]
    elif Classe.values[0] == 'LLS':
        return  [np.nan,np.nan]
    elif Classe.values[0] == 'LL':
        return  [np.nan,np.nan]
    elif Classe.values[0] == 'SA':
        return [0.239,0.136]
    elif Classe.values[0] == 'SL':
        return [0.201,0.085]
    else:
        return [0.110,0.037]
if __name__ == '__main__':
    d={}
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    
    #  ouverture du fichier avec les taux d'argile/limon et sable 
    df=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_CACG_2017.csv",decimal=',')
    a=[]
    for i in df.ID:
        a.append(Classe_texture(df[df.ID==i]))
    df["Classe_Bruand"]=a
    pf2=[]
    pf4=[]
    for i in df.ID:
        pf2.append(PF_bruand(df[df.ID==i])[0])
        pf4.append(PF_bruand(df[df.ID==i])[1])
    df["CC_mean"]=pf2
    df["PF_mean"]=pf4

    # Ecriture du csv avec les CC et PF pour chaque UTS.
    df.to_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/GSM/Extract_GSM_parcelle_PKGC_GERS_2017_GSM_PF_CC_class_name.csv")
    
    
   