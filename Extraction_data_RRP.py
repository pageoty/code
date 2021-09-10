# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:17:17 2021

@author: yann
"""


import os
import sqlite3
import geopandas as geo 
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import numpy as np
import pandas as pd
import shapely.geometry as geom
import pickle





if __name__ == '__main__':
    d={}
    d["PC_home"] = "/mnt/d/THESE_TMP/"
    d["PC_home_Wind"] = "D:/THESE_TMP/"
    d["PC_disk"] = "H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"] = "/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_labo_disk"] = "/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["SAVE_path"] = "/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/SOIL/RRP/"
    drpt = "32" # Code du département: soit 31 pour Haute-Garonne
    Pondere = False # Choix de la méthode

    # méthode pondération  
    RRP_shp = geo.read_file(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/Cartographie_RRP"+drpt+"/RRP_"+drpt+"_v5.shp") # Ouverture des données RRP
    parcelle = geo.read_file(d["PC_labo_disk"]+"/DONNEES_RAW/DONNEES_CACG_PARCELLE_REF/Parcelle_CAGC.shp") # Ouverture du parcellaire
    Inter = geo.overlay(parcelle,RRP_shp,how = 'intersection') # intersection entre le parcellaire  et la carte du RRP.
    No_ucs = Inter[["ID",'NO_UCS','Surface']] # attention récupérer l'UCS majoritaire pour chaque id
    UCS_no = pd.DataFrame()
    for j in parcelle.ID: # boucle sur les parcelles, afin de récupérer l'UCS majoritaire au sein de la parcelle, à partir de la superficie. 
        id_UCS_maj = No_ucs.loc[No_ucs.ID == float(j)][["ID","Surface","NO_UCS"]].max()
        UCS_no = UCS_no.append([id_UCS_maj])
    
    name32 = "donesol3_661_20190516_1147_E3RU"
    name31 = "donesol3_661_20191011_0915_C2DI"
    tab_UCS = pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_ucs.csv",sep = ";",encoding="latin-1")
    tab_UTS = pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_l_ucs_uts.csv",sep = ";")
    tab_UTS_prof = pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_uts.csv",sep = ";",encoding="latin-1")
    tab_strat = pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_strate.csv",sep = ";",encoding="latin-1")
    tab_strat_quat = pd.read_csv(d["PC_labo_disk"]+"DONNEES_RAW/DONNES_SOIL/RRP_Midi_pyrénées/RRP"+drpt+"/BDDonesol_RRP"+drpt+"/"+name32+"_table_strate_quant.csv",sep = ";",encoding="latin-1")

    data_parcelle_Texture=pd.DataFrame()
    ids = []
    entete = []
    variable = ["Argile",'Sable',"Limon","epais"]
    variable2 = ["Argile",'Sable',"Limon"]
    for i in enumerate(UCS_no.values):
        print("id : %s" %i[1][0])
        print("no_ucs :%s" %i[1][2])
        id_UCS = tab_UCS.loc[tab_UCS.no_ucs == float(i[1][2])]["id_ucs"] # Récupération de l'ID UCS, car no_UCS est différent de l'ID ucs. 
        print(id_UCS)
        UTS = tab_UTS.loc[tab_UTS.id_ucs == int(id_UCS)] # Récupération des UTS au sein d'un UCS
        UTS_maj = UTS.loc[UTS.pourcent == UTS.pourcent.max()]["id_uts"] # Isoler l"UTS majoritaire
        UTS_maj_prof = tab_UTS_prof.loc[tab_UTS_prof.id_uts == int(UTS_maj)][["prof_rac_mod","prof_rac_min","prof_rac_max"]]# Extraction des données par UTS
        
        Strat = tab_strat.groupby("id_uts") # regrouper les variables en focntion de l'ID UTS. 
        Strat_uts = Strat.get_group(int(UTS_maj))[['no_strate', 'prof_appar_min', 'prof_appar_moy', 'prof_appar_max',
           'epais_min', 'epais_max', 'epais_moy', 'id_uts']]# Extraction des données par Startes
        Strat_uts["id_parcelle"] = i[1][0] # Ajouter l'Id parcelle à pour chaque strate

        tab_strat_quant_uts = tab_strat_quat.loc[(tab_strat_quat.id_uts == int(UTS_maj))][['nom_var', 'val_min', 'val_mod', 'val_max','no_strate']] # Extraction des données par strates
        All = pd.merge(tab_strat_quant_uts,Strat_uts, on = "no_strate") # Jointure des deux tableaux
        All.dropna(inplace = True)
        Argile_moy = np.average(All.loc[(All.nom_var == "TAUX ARGILE")]["val_mod"]/1000, weights = All.loc[(All.nom_var == "TAUX ARGILE")]["epais_moy"].to_list()) # Calcul de la moyenne pondérée en focntino de l'épaisseur des strates
        Sable_moy = np.average(All.loc[(All.nom_var == "TAUX SABLE")]["val_mod"]/1000, weights = All.loc[(All.nom_var == "TAUX SABLE")]["epais_moy"].to_list())
        Limon_moy = np.average(All.loc[(All.nom_var == "TAUX LIMON")]["val_mod"]/1000, weights = All.loc[(All.nom_var == "TAUX LIMON")]["epais_moy"].to_list())
        
        # Extraction des variables sans pondération 
        Argile = All.loc[(All.nom_var == "TAUX ARGILE")]["val_mod"]/1000 
        Argile = Argile.to_list()
        Sable = All.loc[(All.nom_var == "TAUX SABLE")]["val_mod"]/1000
        Sable = Sable.to_list()
        Limon = All.loc[(All.nom_var == "TAUX LIMON")]["val_mod"]/1000
        Limon = Limon.to_list()
        epais = All.loc[(All.nom_var == "TAUX LIMON")]["epais_moy"].to_list()

        if Pondere == True :
            ids.append(np.repeat(i[1][0],4)) # Dimensionner la variable ID parcelle en fonction du nombre de strate
            entete.append(variable)
            data_parcelle_Texture = data_parcelle_Texture.append([Argile,Sable,Limon,epais])
        else:
            ids.append(np.repeat(i[1][0],3))
            entete.append(variable2)
            data_parcelle_Texture = data_parcelle_Texture.append([Argile_moy,Sable_moy,Limon_moy])
    if Pondere == True :
        data_parcelle_Texture.columns = ["Strat1","Strat2","Strat3"] # modifier les noms de colonnes. 
        
    data_parcelle_Texture["ID"] = pd.DataFrame(ids).T.unstack().values # Ajout des ID parcelle dans le tableau contenant les variables texturales
    data_parcelle_Texture["Variable_modale"] = pd.DataFrame(entete).T.unstack().values # Ajout du nom des variables
    data_parcelle_Texture.to_csv(d["SAVE_path"]+"/Extract_RRP_GERS_parcelle_CACG_2017.csv") # Enregister le résultat dans un tableur en CSV.
