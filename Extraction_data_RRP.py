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


    RRP_shp=geo.read_file(d["PC_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP31/Cartographie_RRP31/RRP_31_VF.shp")
    parcelle=geo.read_file(d["PC_disk"]+"/DONNEES_RAW/PARCELLE_LABO/PARCELLE_LABO_LAM_L93.shp")
    Inter=geo.overlay(parcelle,RRP_shp,how='intersection')
    No_ucs=Inter.NO_UCS
    
    # Lecture file table UCS
    tab_UCS=pd.read_csv(d["PC_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP31/BDDonesol_RRP31/donesol3_661_20191011_0915_C2DI_table_ucs.csv",sep=";",encoding="latin-1")
    id_UCS=tab_UCS.loc[tab_UCS.no_ucs==float(No_ucs)]["id_ucs"]
    # Lecture file Table UTS
    tab_UTS=pd.read_csv(d["PC_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP31/BDDonesol_RRP31/donesol3_661_20191011_0915_C2DI_table_l_ucs_uts.csv",sep=";")
    UTS=tab_UTS.loc[tab_UTS.id_ucs==int(id_UCS)]
    UTS_maj=UTS.loc[UTS.pourcent==UTS.pourcent.max()]["id_uts"]
    # LEcture file table strat
    tab_strat=pd.read_csv(d["PC_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP31/BDDonesol_RRP31/donesol3_661_20191011_0915_C2DI_table_strate.csv",sep=";",encoding="latin-1")
    Strat=tab_strat.groupby("id_uts")
    Strat_uts=Strat.get_group(int(UTS_maj))
    Strat_uts_maj=Strat_uts.loc[Strat_uts.epais_moy==Strat_uts.epais_moy.max()]["no_strate"]
    
    #  Lecture file table strate qual
    tab_strat_quat=pd.read_csv(d["PC_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP31/BDDonesol_RRP31/donesol3_661_20191011_0915_C2DI_table_strate_quant.csv",sep=";",encoding="latin-1")
    tab_strat_quant_uts=tab_strat_quat.loc[(tab_strat_quat.id_uts==int(UTS_maj)) & (tab_strat_quat.no_strate==int(Strat_uts_maj))]
    Argile=tab_strat_quant_uts.loc[(tab_strat_quant_uts.nom_var=="TAUX ARGILE")]["val_mod"]/1000
    Sable=tab_strat_quant_uts.loc[(tab_strat_quant_uts.nom_var=="TAUX SABLE")]['val_mod']/1000
    
    # PTF Bruand
