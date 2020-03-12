#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:15:17 2019

@author: pageot
"""

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np 
import os
import otbApplication




d={}
d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/SRTM/SRTM/SRTM_TILE/"
list_img=os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/SRTM/SRTM/TILE/")

for j in os.listdir(d["data_file"]):   
    path_img=os.path.basename(j)[0:-7]
    tile=path_img[20:26]
    print(tile)
    d["image_name"]=path_img[0:-9]
    d["ELEV"]=d["image_name"]+ "_0001_ALT_R1.TIF"
    os.system("gdaldem aspect -trigonometric -zero_for_flat /datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/SRTM/SRTM/SRTM_TILE/%s %s_exposition.tif "%(d["ELEV"],path_img))
    os.system("rm /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SCRIPT/python/S2__TEST_AUX_REFDE2_%s_0001_SLP_exposition.tif"%(tile))
os.system(" mv *exposition.tif /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SRTM/EXPO_TILE_SRTM/")

for i in list_img:
    path_img=os.path.basename(i)[0:-11]
    tile=path_img[20:26]
    print(tile)
    d["image_name"]=path_img
    d["ELEV"]=d["image_name"]+ "_ALT_R1.TIF"
    d["PENTE"]=d["image_name"]+"_SLP_R1.TIF"
    d["EXPO"]=d["image_name"]+ "_ALT_exposition.tif"
    
    
    App = otbApplication.Registry.CreateApplication("ConcatenateImages")
    App.SetParameterStringList("il",[str(d["data_file"]+d["ELEV"])])
    App.SetParameterString("out","STRM.tif")
    App.Execute()

    App1= otbApplication.Registry.CreateApplication("ConcatenateImages")
    App1.SetParameterStringList("il",[str(d["data_file"]+d["PENTE"])])
    App1.SetParameterString("out","PENTE.tif")
    App1.Execute()
    
    App2= otbApplication.Registry.CreateApplication("ConcatenateImages")
    App2.SetParameterStringList("il",[str("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SRTM/EXPO_TILE_SRTM/"+d["EXPO"])])
    App2.SetParameterString("out","EXPO.tif")
    App2.Execute()
    
    application1 = otbApplication.Registry.CreateApplication("BandMath")
    application1.AddImageToParameterInputImageList("il",App1.GetParameterOutputImage("out"))
    application1.SetParameterString("out", "Mask_pente_%s.tif"%(tile))
    application1.SetParameterString("exp", 'im1b1 <= 5 ? im1b1= 0 : im1b1=1')
    print("Create Mask\n")
    application1.Execute()
    
    application2 = otbApplication.Registry.CreateApplication("BandMath")
    application2.AddImageToParameterInputImageList("il",application1.GetParameterOutputImage("out"))
    application2.AddImageToParameterInputImageList("il",App2.GetParameterOutputImage("out"))
    application2.SetParameterString("out", "EXPO_MASK_%s.tif"%(tile))
    application2.SetParameterString("exp", 'im1b1*im2b1')
    print("Application of Mask\n")
    application2.ExecuteAndWriteOutput()

#
    os.system ("mv EXPO_MASK* /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SRTM/MASK_SRTM_TILE/")
    
#for j in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SRTM/MASK_SRTM_TILE/"):   
#    print (j)
#    tile=os.path.basename(j)[10:-4]
#    print (tile)
#    os.system("gdaldem aspect -trigonometric -zero_for_flat /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SRTM/MASK_SRTM_TILE/SRTM_MASK_%s.tif exposition_%s.tif "%(tile,j))
#
#os.system(" mv exposition* /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SRTM/EXPO_TILE_SRTM/")
##

