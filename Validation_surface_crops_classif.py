#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:52:09 2019

@author: pageot
"""

#import gdal
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Plot_frise_images import plotenbar
import seaborn as sns
#import geopandas as gp
import pandas as pd
from rasterstats import zonal_stats, point_query
import matplotlib.pyplot as plt
import collections
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

#def loadRaster(inRaster):
#    gdalRaster = gdal.Open(inRaster, gdal.GA_ReadOnly)
#    if gdalRaster is None:
#        raise ReferenceError('Impossible to open ' + inRaster)
#    # Get the geoinformation
#    GeoTransform = gdalRaster.GetGeoTransform()
#    Projection = gdalRaster.GetProjection()
#
#    return gdalRaster,GeoTransform,Projection

if __name__ == "__main__":
    names_crop=["Maize_Irr","Soybean_Irr","Maize_Nirr","Soybean_Nirr","Sorghum","Sunflower"]
#    path_vector = "/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/RPG/RPG_BV/RPG_SUMMER_2017_ADOUR_AMONT.shp"
    path_vector="F:/THESE/CLASSIFICATION/TRAITEMENT/RPG/RPG_ADOUR_2017_supp_others_crops.shp"
#    Data_Val=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/DATA_VALIDATION_PARTENAIRE/ADOUR/DONNEES_VALIDATION_SURFACE_IRRIG_ADOUR_2018_2019.csv",sep=",")
#    Data_Val["label"]=[1,1,1,2,44]
#    Data_Val_regroup=Data_Val.groupby("label").sum()
#    Data_Val_regroup.loc[33]=0
#    Data_Val_regroup.loc[11]=0
#    Data_Val_regroup.loc[22]=0
    step=[]
    total_classe_ha=[]
    for classif in os.listdir("G:/Yann_THESE/RESULTAT_CLASSIFICATION/2017/RUN_fixe_seed/SHARK/"):
        if "ASC" in classif and 'NESTE' not in classif or  "3ind" in classif and "DES" not in classif : 
            print (classif)
#            step.append(classif)
            for i in np.arange(0,5):
                print(i)
                step.append(classif)
        #        path_raster = "/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/2017/RUN_fixe_seed/SHARK/DES_F_3ind_SAFRAN/final/Classif_ADOUR_"+str(i)+"_regularized.tif"
                path_raster="G:/Yann_THESE/RESULTAT_CLASSIFICATION/2017/RUN_fixe_seed/SHARK/"+str(classif)+"/final/Classif_paper_ADOUR_"+str(i)+"_regularized.tif"
                # path_raster="F:/2017.tif"
                Stat_zonal = zonal_stats(path_vector, 
                                path_raster, 
                                categorical=True, all_touched=True) #surestimation
                df_dic = pd.DataFrame(Stat_zonal)
            
                classe_pixel = df_dic.sum()
                classe_ha=classe_pixel*0.01
                print(classe_ha)
                total_classe_ha.append(classe_ha)
                Area_classif=pd.DataFrame(total_classe_ha)
                Area_classif_mean=Area_classif.mean()
                Area_classif_std=Area_classif.std()
    Area_classif["scenario"]=step
    all_area_mean=Area_classif.groupby("scenario").mean()
    all_area_mean.to_csv("G:/Yann_THESE/RESULTAT_CLASSIFICATION/surface_ha_classif_2017.csv")
                
        #    Area_classif["name"]=names_crop
    
    # =============================================================================
    #   Data validation od partenaire 
    # =============================================================================

    #    Data_Val_regroup["name"]=['Maize_Irr',"Soybean_Irr",'Sunflower',"Sorghum","Maize_Nirr","Soybean_Nirr"]
            
    
    # =============================================================================
    # Plot Comparatif
    # =============================================================================
        
#        df=pd.concat([Area_classif_mean,Data_Val_regroup],axis=1)
#        df.columns=("Classif","surface2018","surface2019")
#        df.astype(float)
#        df=df.T
#        df["origin"]=["Classif","Parten","Parten"]
# =============================================================================
#     Visualisation via python 
# =============================================================================
#    plt.figure(figsize=(25,25))
#
#    classe=["Maize_Irr","Soybean_Irr","Maize_Nirr","Soybean_Nirr","Sorghum","Sunflower"]
#    colors = ['blue', 'lightgreen', 'maroon',"linen","pink","yellow"]
#    cmap = ListedColormap(colors)
#    legend_patches = [Patch(color=icolor, label=label)
#                  for icolor, label in zip(colors, classe)]
#    src_classif, geoTransf, proj = loadRaster(path_raster)
#    rasterArray_classif = src_classif.ReadAsArray()
#    plt.imshow(rasterArray_classif,cmap=cmap)
#    plt.legend(handles=legend_patches,
#         facecolor ="white",
#         edgecolor = "white",
#         bbox_to_anchor = (1.5,1)) # Place legend to the RIGHT of the map
