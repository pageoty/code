#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:37:06 2019

@author: dahingerv
"""
import gdal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Plot import plotenbar
import seaborn as sns
import geopandas as gp
import pandas as pd
from rasterstats import zonal_stats, point_query
import matplotlib.pyplot as plt
import collections
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

def loadRaster(inRaster):
    gdalRaster = gdal.Open(inRaster, gdal.GA_ReadOnly)
    if gdalRaster is None:
        raise ReferenceError('Impossible to open ' + inRaster)
    # Get the geoinformation
    GeoTransform = gdalRaster.GetGeoTransform()
    Projection = gdalRaster.GetProjection()

    return gdalRaster,GeoTransform,Projection

if __name__ == "__main__":
     path_classif = '/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/LIA_ASC_2017.tif'
    
    path_classif = '/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp//file_Mask_classif/CLASSIF_CUMUL_GSM_3in_T31.tif'
    path_masque_crops='/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/RPG/MASK_T31_UINT.tif'
    path_BV = '/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/EMPRISE/EMPRISE_BV_TARN_AVAL/RASTER_BV_TARN.tif'
    
    src_classif, geoTransf, proj = loadRaster(path_classif)
    src_masque, geoTransf, proj = loadRaster(path_masque_crops)
    src_bv, geoTransf, proj = loadRaster(path_BV)
    
    rasterArray_classif = src_classif.ReadAsArray()
    
    
    rasterArray_masque = src_masque.ReadAsArray()
    rasterArray_BV = src_bv.ReadAsArray()
    
    classif=rasterArray_BV*rasterArray_classif
    bv=classif*rasterArray_masque
    # Distribution par label
    from collections import Counter
    
    lst_label=[1,2,11,22,33,44,55,6]
    Label=[1]
    lst_df=[]
    for label in lst_label:
        Sum=np.sum(classif==label)*0.01
        lst_df.append(Sum)
        
        
        
        
        coords = np.where(bv_mask==label)
        my_tbl = []
        for x, y in zip(coords[1], coords[0]):
             my_tbl.append(bv_mask[y][x])
        
        tbl_counter = Counter(my_tbl)   
        print ("number of pixel - Label %s :"%label)
        print(tbl_counter)
        
        df_counter = pd.DataFrame.from_dict(tbl_counter, orient='index').reset_index()
        df_counter['label']= label
        lst_df.append(df_counter)
        
    df_distrib = pd.concat(lst_df)
    df_distrib[0] = df_distrib[0]*0.01
#    df_distrib["index"]=["Maize_Irr","Soybean_Irr","Maize_Nirr","Soybean_Nirr","Sunflower","Sorghum","others","peas"]
    print (df_distrib)
    
# =============================================================================
#     Map classif 
# =============================================================================
    bv_mask = np.ma.masked_where(bv == 0 , 
                              bv, 
                              copy=True)
    plt.figure(figsize=(25,25))

    classe=["Maize_Irr","Soybean_Irr","Others","Maize_Nirr","Soybean_Nirr","Sorghum","Sunflower"]
    colors = ['blue', 'lightgreen', 'darkgreen', 'maroon',"linen","pink","yellow"]
    cmap = ListedColormap(colors)
    legend_patches = [Patch(color=icolor, label=label)
                  for icolor, label in zip(colors, classe)]
    plt.imshow(bv_mask,cmap=cmap)
    plt.legend(handles=legend_patches,
         facecolor ="white",
         edgecolor = "white",
         bbox_to_anchor = (1.5,1)) # Place legend to the RIGHT of the map
    plt.figure(figsize=(15,15))
    plt.imshow(bv)


# =============================================================================
#  test
# =============================================================================
    path_vector = "/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/RPG/RPG_BV/RPG_SUMMER_2017_ADOUR_AMONT.shp"
    path_raster = "/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/file_Mask_classif/CLASSIF_T30_IND.TIF"
#
#    stats_ture = zonal_stats(path_vector, 
#                    path_raster, 
#                    categorical=True, all_touched=True) surestimation
#    
#    df_dic_T = pd.DataFrame(stats_ture)
##    col = ["Maize_Irr","Soybean_Irr","Others","Maize_Nirr","Soybean_Nirr","Sorghum","Sunflower"]
##    df_dic.columns = col
#    df_dic.head(5)
#
#    classe_pixel_T = df_dic_T.sum()
#    classe_ha_T=classe_pixel_T*0.01

    stats = zonal_stats(path_vector, 
                    path_raster, 
                    categorical=True, all_touched=False)
   
    df_dic = pd.DataFrame(stats)
#    col = ["Maize_Irr","Soybean_Irr","Others","Maize_Nirr","Soybean_Nirr","Sorghum","Sunflower"]
#    df_dic.columns = col
    df_dic.head(5)

    classe_pixel = df_dic.sum()
    classe_ha=classe_pixel*0.01