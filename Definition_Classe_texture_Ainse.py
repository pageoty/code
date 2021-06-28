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
    elif (Argile.values[0]) <=18 and (Argile.values[0]) >= 8 and (Limon.values[0]) >= 26 and (Limon.values[0]) <= 58 and  (Sable.values[0]) <= 55 and (Sable.values[0]) >= 35 :
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
if __name__ == '__main__':
    d={}
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"

def Classe_texture (df):
    Argile=df["Argile"]
    Limon=df["Limon"]
    Classe=[]
    if Argile >=7.5 and Limon <= 85 :
        Classe.append("LL")
    


    df=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/SOIL_RIGOU/Extract_RRP_Rigou_parcelle_PKCG_GERS_2017_UTS_maj.csv")
    a=[]
    for i in df.ID:
        a.append(Classe_texture(df[df.ID==i]))
    df["Classe_Bruand"]=a
