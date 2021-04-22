# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:17:17 2021

@author: yann
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


if __name__ == '__main__':
    d={}
    # d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    drpt="32"

    RRP_shp=geo.read_file(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/Cartographie_RRP"+drpt+"/RRP_"+drpt+"_v5.shp")
    parcelle=geo.read_file(d["PC_labo_disk"]+"/DONNEES_RAW/data_SSP/ParcellesPKGC_MAIS_2017_32_valid_TYP_only.shp")
    # parcelle=geo.read_file(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/DATA_Validation/Parcelle_2017.shp")
    Inter=geo.overlay(parcelle,RRP_shp,how='intersection')
    No_ucs=Inter.NO_UCS
    
    
    name32="donesol3_661_20190516_1147_E3RU"
    name31="donesol3_661_20191011_0915_C2DI"
    # Lecture file table UCS
    tab_UCS=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_ucs.csv",sep=";",encoding="latin-1")
    id_UCS=tab_UCS.loc[tab_UCS.no_ucs==float(No_ucs)]["id_ucs"]
    # Lecture file Table UTS
    tab_UTS=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_l_ucs_uts.csv",sep=";")
    UTS=tab_UTS.loc[tab_UTS.id_ucs==int(id_UCS)]
    UTS_maj=UTS.loc[UTS.pourcent==UTS.pourcent.max()]["id_uts"]
    tab_UTS_prof=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_uts.csv",sep=";",encoding="latin-1")
    UTS_maj_prof=tab_UTS_prof.loc[tab_UTS_prof.id_uts==int(UTS_maj)][["prof_rac_mod","prof_rac_min","prof_rac_max"]]
    print(UTS_maj_prof)
    # LEcture file table strat
    tab_strat=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_strate.csv",sep=";",encoding="latin-1")
    Strat=tab_strat.groupby("id_uts")
    Strat_uts=Strat.get_group(int(UTS_maj))
    Strat_uts_maj=Strat_uts.loc[Strat_uts.epais_moy==Strat_uts.epais_moy.max()]["no_strate"]
    
    #  Lecture file table strate qual
    tab_strat_quat=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_strate_quant.csv",sep=";",encoding="latin-1")
    tab_strat_quant_uts=tab_strat_quat.loc[(tab_strat_quat.id_uts==int(UTS_maj)) & (tab_strat_quat.no_strate==int(Strat_uts_maj))]
    Argile=tab_strat_quant_uts.loc[(tab_strat_quant_uts.nom_var=="TAUX ARGILE")]["val_min"]/1000
    Sable=tab_strat_quant_uts.loc[(tab_strat_quant_uts.nom_var=="TAUX SABLE")]['val_min']/1000
    Limon=tab_strat_quant_uts.loc[(tab_strat_quant_uts.nom_var=="TAUX LIMON")]['val_min']/1000
    print(Limon)
    print(Sable)
    print(Argile)


    #other méthode pondération 
    RRP_shp=geo.read_file(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/Cartographie_RRP"+drpt+"/RRP_"+drpt+"_v5.shp")
    parcelle=geo.read_file(d["PC_labo_disk"]+"/DONNEES_RAW/data_SSP/ParcellesPKGC_MAIS_2017_32_valid_TYP_only.shp")
    # parcelle=geo.read_file(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/DATA_Validation/Parcelle_2017.shp")
    Inter=geo.overlay(parcelle,RRP_shp,how='intersection')
    No_ucs=Inter.NO_UCS
    
    
    name32="donesol3_661_20190516_1147_E3RU"
    name31="donesol3_661_20191011_0915_C2DI"
    # Lecture file table UCS
    tab_UCS=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_ucs.csv",sep=";",encoding="latin-1")
    id_UCS=tab_UCS.loc[tab_UCS.no_ucs==float(No_ucs)]["id_ucs"]
    tab_UTS=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_l_ucs_uts.csv",sep=";")
    UTS=tab_UTS.loc[tab_UTS.id_ucs==int(id_UCS)]
    UTS_maj=UTS.loc[UTS.pourcent==UTS.pourcent.max()]["id_uts"]
    tab_UTS_prof=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_uts.csv",sep=";",encoding="latin-1")
    UTS_maj_prof=tab_UTS_prof.loc[tab_UTS_prof.id_uts==int(UTS_maj)][["prof_rac_mod","prof_rac_min","prof_rac_max"]]
    
    tab_strat=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_strate.csv",sep=";",encoding="latin-1")
    Strat=tab_strat.groupby("id_uts")
    Strat_uts=Strat.get_group(int(UTS_maj))[['no_strate', 'prof_appar_min', 'prof_appar_moy', 'prof_appar_max',
       'epais_min', 'epais_max', 'epais_moy', 'id_uts']]


    tab_strat_quat=pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_strate_quant.csv",sep=";",encoding="latin-1")
    tab_strat_quant_uts=tab_strat_quat.loc[(tab_strat_quat.id_uts==int(UTS_maj))][['nom_var', 'val_min', 'val_mod', 'val_max','no_strate']]
    
    All= pd.merge(tab_strat_quant_uts,Strat_uts, on="no_strate")
    All.dropna(inplace=True)
    Argile_moy=np.average(All.loc[(All.nom_var=="TAUX ARGILE")]["val_max"]/1000, weights=All.loc[(All.nom_var=="TAUX ARGILE")]["epais_moy"].to_list())
    Sable_moy=np.average(All.loc[(All.nom_var=="TAUX SABLE")]["val_max"]/1000, weights=All.loc[(All.nom_var=="TAUX SABLE")]["epais_moy"].to_list())
    Limon_moy=np.average(All.loc[(All.nom_var=="TAUX LIMON")]["val_max"]/1000, weights=All.loc[(All.nom_var=="TAUX LIMON")]["epais_moy"].to_list())
    
    
    print(Argile_moy)
    print(Sable_moy)
    print(Limon_moy)