#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:28:42 2021

@author: pageot
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:13:23 2021

@author: pageot
Extraction données SAFRAN à partir d'un ficher parcelle est SHP. La méthode permet de récupérer le centroîde des polygones
Les centroides permet de calcul la distance minimal enetre le polygone et la maille la plus proche. 
"""

import os
import pandas as pd
import geopandas as geo
import numpy as np
from osgeo import osr

if __name__ == '__main__':
    d={}
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["PC_labo_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    years="2017"
    name_run="RUNS_SAMIR/DATA_SCP_ICOS/ASA_all_crops_summer/"+str(years)+"/Inputdata/"
    d["path_run"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    d["path_run_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    # d["path_run_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"+name_run+"/"
    
# =============================================================================
# Conversion csv en shp (geopandas)
# =============================================================================
    # df=pd.read_csv(/file/meteo.csv)
    # LAMBX=df.LAMBX*100
    # LAMBY=df.LAMBY*100
    # df["lambX"]=LAMBX
    # df['lambY']=LAMBY
    # df=df.loc[(df.DATE >= 20090101) &(df.DATE <= 20091231)]
    # geometry = df['geometry'].map(shapely.wkt.loads)
    # meteo_spa=geo.df(df, crs="EPSG:", geometry=geometry)
    

# =============================================================================
#     Extaction de la donnée METEO par rapport centoide parcelle (methode NN)
# =============================================================================
    
    meteo=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/DONNES_METEO/SAFRAN_ZONE_2017_L93.shp")
    parcelle= geo.read_file(d["PC_labo_disk"]+"/DONNEES_RAW/data_SSP/PARCIRR_2017_32_avecRES_with_ID.shp")    
    parcelle["ID"]=parcelle._ID
    # parcelle=geo.read_file("/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/DONNEES_ASA/PACRELLE_ASA_2018_RPG_all_crops_summer_Gers_er10.shp")
    # parcelle=geo.read_file("H:/Yann_THESE/BESOIN_EAU//BESOIN_EAU/DONNEES_RAW/data_SSP/ParcellesPKGC_MAIS_2017_32_valid_TYP_only.shp")
    
    # MAIS_IRR=parcelle[parcelle.classifmaj==1.0]
    # MAIS_NIRR=parcelle[parcelle.classifmaj==11.0]
    
    
    meteo.DATE=meteo.DATE.astype(int)
    meteo.DATE=pd.to_datetime(meteo.DATE,format="%Y%m%d")
    meteo.set_index("field_1",inplace=True)
    resu=pd.DataFrame()
    idgeom=[]
    # parcelle.reset_index(inplace=True)
    for par in parcelle.index:
          print("ID parcelle en cours de traitement :%s"%par)
          extart_meteo=meteo.loc[meteo["geometry"].distance(parcelle["geometry"].iloc[par])==meteo["geometry"].distance(parcelle["geometry"].iloc[par]).min()][['DATE',"PRELIQ_Q","T_Q","ETP_Q"]]
          idgeom.append(np.repeat(parcelle["_ID"].iloc[par],extart_meteo.shape[0]))
          resu=resu.append(extart_meteo)
    idpar=pd.DataFrame(idgeom).stack().to_list()
    resu["ID"]=idpar
    test=pd.merge(parcelle,resu[["DATE","ETP_Q","PRELIQ_Q","T_Q",'ID']],on="ID")
    test.set_index("ID",inplace=True)
    test.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PKGC_temp.csv")
# mettre cela en df SAMIR    
    meteo=test.filter(['ID',"DATE","ETP_Q", 'PRELIQ_Q'])
    meteo.reset_index(inplace=True)
    meteo.columns=["id",'date',"ET0",'Prec']
    meteo["Irrig"]=0.0
    # meteo_irr=meteo[meteo.id.isin(MAIS_IRR.ID)]
    # meteo_irr.to_pickle("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/meteo_Adour_mais_IRR_2018.df")
    meteo_nirr=meteo[meteo.id.isin(MAIS_NIRR.ID)]
    meteo_nirr.to_pickle("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/meteo_Adour_mais_NIRR_2018.df")
