#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:00:45 2021

@author: pageot
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    d={}
    d["SAVE_disk"]="/run/media/pageot/Transcend/Yann_THESE/RESULTAT_CLASSIFICATION/PLOT/"
    
    df=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/RESULTAT_CLASSIFICATION/Vote_majoritaire_cours_saison_2017_Adour_data_ref.csv")
    lab_ref=df.groupby("labcroirr")
    ref_irri_maize=lab_ref.get_group(1)
    ref_rain_maize=lab_ref.get_group(11)

    index_tab=arrays = [["maize_irri", "maize_rain"],["mai", "juin", "juillet", "aout", "septembre"],]
    index=pd.MultiIndex.from_product(index_tab, names=["verite", "mois"])
    
    data=pd.DataFrame(data=None,index=index,columns=["Maize_irri","Maize_rain","autres_cultures"])
#  calucl 
    mai_maize=ref_irri_maize.where(ref_irri_maize.mai_majori==1.0,0)["mai_majori"]
    mai_maize_rain=ref_irri_maize.where(ref_irri_maize.mai_majori==11.0,0)["mai_majori"]
    # mai_maize_rain[mai_maize_rain!=0.0] # 17%
    # mai_maize[mai_maize!=0.0]
    # completer le tableau
    for m,n in zip(["mai","juin",'juillet',"aout",'septembre'],ref_irri_maize.columns[3:]):
        maize=ref_irri_maize.where(ref_irri_maize[n]==1.0,0)[n]
        maize_rain_i=ref_irri_maize.where(ref_irri_maize[n]==11.0,0)[n]
        data.loc[("maize_irri",str(m)),"Maize_irri"]=maize[maize!=0.0].shape[0]/maize.shape[0]
        data.loc[("maize_irri",str(m)),"Maize_rain"]=maize_rain_i[maize_rain_i!=0.0].shape[0]/maize.shape[0]
        data.loc[("maize_irri",str(m)),"autres_cultures"]=1-(data.loc[("maize_irri",str(m)),"Maize_irri"]+data.loc[("maize_irri",str(m)),"Maize_rain"])
        maize_irri=ref_rain_maize.where(ref_rain_maize[n]==1.0,0)[n]
        maize_rain=ref_rain_maize.where(ref_rain_maize[n]==11.0,0)[n]
        data.loc[("maize_rain",str(m)),"Maize_irri"]=maize_irri[maize_irri!=0.0].shape[0]/maize_rain.shape[0]
        data.loc[("maize_rain",str(m)),"Maize_rain"]=maize_rain[maize_rain!=0.0].shape[0]/maize_rain.shape[0]
        data.loc[("maize_rain",str(m)),"autres_cultures"]=1-(data.loc[("maize_rain",str(m)),"Maize_irri"]+data.loc[("maize_rain",str(m)),"Maize_rain"])
    data=data.astype(float)
    plt.figure(figsize=(7,5))
    plt.bar(data.loc["maize_irri"].index,data.loc["maize_irri"]["Maize_irri"],label='Maïs irrigué')
    plt.bar(data.loc["maize_irri"].index,data.loc["maize_irri"]["Maize_rain"],bottom=data.loc["maize_irri"]["Maize_irri"],label='Maïs pluvial')
    plt.bar(data.loc["maize_irri"].index,data.loc["maize_irri"]["autres_cultures"],bottom=data.loc["maize_irri"]["Maize_irri"]+data.loc["maize_irri"]["Maize_rain"],label="autres cultures")
    plt.ylim(0,1)
    plt.title("Maïs irrigué")
    plt.legend()
    plt.savefig(d["SAVE_disk"]+"/Cours_de_saison_confusion_parcelle_mais_irri.png")

    plt.figure(figsize=(7,5))
    plt.bar(data.loc["maize_rain"].index,data.loc["maize_rain"]["Maize_irri"],label='Maïs irrigué')
    plt.bar(data.loc["maize_rain"].index,data.loc["maize_rain"]["Maize_rain"],bottom=data.loc["maize_rain"]["Maize_irri"],label='Maïs pluvial')
    plt.bar(data.loc["maize_rain"].index,data.loc["maize_rain"]["autres_cultures"],bottom=data.loc["maize_rain"]["Maize_irri"]+data.loc["maize_rain"]["Maize_rain"],label="autres cultures")
    plt.ylim(0,1)
    plt.title("Maïs pluvial")
    plt.legend()
    plt.savefig(d["SAVE_disk"]+"/Cours_de_saison_confusion_parcelle_mais_pluvial.png")