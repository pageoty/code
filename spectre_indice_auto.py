#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:43:31 2019

@author: pageot
Scprit permettant de calcule de manière automatique les signature temporelle des indices suivants :' NDVI,NDRE,NDWI'

"""


import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
#import seaborn as sns
import csv
import zipfile
from scipy import stats
from datetime import datetime
import otbApplication

d={}
d["data_file"]="/run/media/pageot/ADATAHD650/PROJET_M2_ATT/IMG_TEST/"# Path where your sentinel images are located
d["output_file"]="/run/media/pageot/ADATAHD650/PROJET_M2_ATT/test_code/" 
d["path_vector"]="/run/media/pageot/ADATAHD650/PROJET_M2_ATT/DONNEES_SIG/DONNEES_TERRAIN/donnees_terrain_select_50%_code_culture.shp" 
label="Summer" # Name of the field containing the labels in shapefile (coded in int)
ram= 8096 # expressed in MB 
if __name__ == "__main__":
    # =============================================================================
    #     Calcule des indices à partir des images satellitaires 
    # =============================================================================    
    start=datetime.now()
    date=[]
    for i in os.listdir(d["data_file"]):
        path_img=os.path.basename(i)
        print (path_img)
        date.append(path_img[11:19])
   
#   storage of other images 
        ConcatenateImages = otbApplication.Registry.CreateApplication("ConcatenateImages") # Create Otb Application 
        ConcatenateImages.SetParameterStringList("il",[str(d["data_file"]+path_img+"/"+path_img+"_FRE_B4.tif")])
        ConcatenateImages.SetParameterString("out", "B4_image.tif")
        ConcatenateImages.Execute()
#   Resampling of bands at 10 meters
        Superimpose_B8A = otbApplication.Registry.CreateApplication("Superimpose")
        Superimpose_B8A.SetParameterString("inr",str(d["data_file"]+path_img+"/"+path_img+"_FRE_B4.tif")) # reference image (10m)
        Superimpose_B8A.SetParameterString("inm",str(d["data_file"]+path_img+"/"+path_img+"_FRE_B8A.tif")) # resampled image (20  or 60 m)
        Superimpose_B8A.SetParameterString("out", "B8A_10m.tif")
        Superimpose_B8A.Execute()
        
        Superimpose_B11 = otbApplication.Registry.CreateApplication("Superimpose")
        Superimpose_B11.SetParameterString("inr",str(d["data_file"]+path_img+"/"+path_img+"_FRE_B4.tif")) # reference image (10m)
        Superimpose_B11.SetParameterString("inm",str(d["data_file"]+path_img+"/"+path_img+"_FRE_B11.tif")) # resampled image (20  or 60 m)
        Superimpose_B11.SetParameterString("out", "B11_10m.tif")
        Superimpose_B11.Execute()
        
        Superimpose_B5 = otbApplication.Registry.CreateApplication("Superimpose")
        Superimpose_B5.SetParameterString("inr",str(d["data_file"]+path_img+"/"+path_img+"_FRE_B4.tif")) # reference image (10m)
        Superimpose_B5.SetParameterString("inm",str(d["data_file"]+path_img+"/"+path_img+"_FRE_B5.tif")) # resampled image (20  or 60 m)
        Superimpose_B5.SetParameterString("out", "B5_10m.tif")
        Superimpose_B5.Execute()
        
#   Create otb_Application to calculate the NDVI index
        BandMath = otbApplication.Registry.CreateApplication("BandMath")
        BandMath.AddImageToParameterInputImageList("il",Superimpose_B8A.GetParameterOutputImage("out"))
        BandMath.AddImageToParameterInputImageList("il",ConcatenateImages.GetParameterOutputImage("out"))
        BandMath.SetParameterString("out", d["output_file"]+"%s_NDVI.tif"%(path_img))
        BandMath.SetParameterString("exp", "(im1b1-im2b1)/(im1b1+im2b1)*1000")
        BandMath.SetParameterString("ram",str(ram))
        BandMath.SetParameterOutputImagePixelType("out",otbApplication.ImagePixelType_uint16) # Allows you to define the size of the image 
        BandMath.ExecuteAndWriteOutput()
        
#   Create otb_Application to calculate the NDWI index
        BandMath_NDWI = otbApplication.Registry.CreateApplication("BandMath")
        BandMath_NDWI.AddImageToParameterInputImageList("il",Superimpose_B8A.GetParameterOutputImage("out"))
        BandMath_NDWI.AddImageToParameterInputImageList("il",Superimpose_B11.GetParameterOutputImage("out"))
        BandMath_NDWI.SetParameterString("out", d["output_file"]+"%s_NDWI.tif"%(path_img))
        BandMath_NDWI.SetParameterString("exp", "(im1b1-im2b1)/(im1b1+im2b1)*1000")
        BandMath_NDWI.SetParameterString("ram",str(ram))
        BandMath_NDWI.SetParameterOutputImagePixelType("out",otbApplication.ImagePixelType_uint16) 
        BandMath_NDWI.ExecuteAndWriteOutput()
        
#   Create otb_Application to calculate the NDRE index
        BandMath_NDRE = otbApplication.Registry.CreateApplication("BandMath")
        BandMath_NDRE.AddImageToParameterInputImageList("il",Superimpose_B8A.GetParameterOutputImage("out"))
        BandMath_NDRE.AddImageToParameterInputImageList("il",Superimpose_B5.GetParameterOutputImage("out"))
        BandMath_NDRE.SetParameterString("out", d["output_file"]+"%s_NDRE.tif"%(path_img))
        BandMath_NDRE.SetParameterString("exp", "(im1b1-im2b1)/(im1b1+im2b1)*1000")
        BandMath_NDRE.SetParameterString("ram",str(ram))
        BandMath_NDRE.SetParameterOutputImagePixelType("out",otbApplication.ImagePixelType_uint16)
        BandMath_NDRE.ExecuteAndWriteOutput()
##        
#    # =============================================================================
#    # Conctatenation of images 
#    # =============================================================================
    if "Concatenate_index" in os.listdir(d["output_file"]):
        print ("existing file")
    else :
         os.mkdir ("%s/Concatenate_index"%d["output_file"]) # allows create file via call system
    date=sorted(date)  
    list_NDVI = []
    list_NDWI = []
    list_NDRE = []
    for j in os.listdir(d["output_file"]) :
        if "NDVI" in j :
            list_NDVI.append(d["output_file"]+j)
        elif "NDRE" in j :
            list_NDRE.append(d["output_file"]+j)
        elif "NDWI" in j:
            list_NDWI.append(d["output_file"]+j)
    print("NDVI image:\n %s \n NDWI image:\n %s \n NDRE image:\n %s" %(sorted(list_NDVI),sorted(list_NDWI),sorted(list_NDRE)))
    for l in [sorted(list_NDVI),sorted(list_NDWI),sorted(list_NDRE)] :
        print ("Index : %s"%l[0][-8:-4])
        print ("List of processed images : %s" %l)
        ConcatenateImages = otbApplication.Registry.CreateApplication("ConcatenateImages")
        ConcatenateImages.SetParameterStringList('il',l)
        ConcatenateImages.SetParameterString("out",d["output_file"]+"Concatenate_index/Concatenate_%s.tif"%l[0][-8:-4])
        ConcatenateImages.SetParameterString("ram",str(ram))
        ConcatenateImages.SetParameterOutputImagePixelType("out",otbApplication.ImagePixelType_uint16)
        ConcatenateImages.ExecuteAndWriteOutput()
        
    # =============================================================================
    # Extraction of statistics 
    # =============================================================================
    if "Statistics_Extraction" in os.listdir(d['output_file']):
        print("existing file")
    else:
        os.mkdir("%s/Statistics_Extraction"%d['output_file'])
    for stat in os.listdir(d["output_file"]+"Concatenate_index"):
        print (stat)
        os.system("otbcli_PolygonClassStatistics -in %s%s -vec %s -field %s -ram %s -out stat.xml"%(d["output_file"]+"/Concatenate_index/",stat,d["path_vector"],label,ram))
        os.system("otbcli_SampleSelection -in %s%s -vec %s -field %s -instats stat.xml -strategy all -ram %s -out SampleSelection.sqlite"%(d["output_file"]+"/Concatenate_index/",stat,d["path_vector"],label,ram))
        os.system("otbcli_SampleExtraction -in %s%s -vec SampleSelection.sqlite -field %s -ram %s -out %s/SampleExtraction%s.sqlite"%(d["output_file"]+"Concatenate_index/",stat,label.lower(),ram,d["output_file"]+"/Statistics_Extraction/",stat[:-4]))
        os.system("rm stat.xml")
        os.system("rm SampleSelection.sqlite")
       
    list_bd_drop=['ogc_fid', 'surf_parc']
#    date=["01","02","03"]
    labe=[label]
    colnames=labe+date
    if "Plot_result" in os.listdir(d['output_file']):
        print("existing file")
    else:
        os.mkdir("%s/Plot_result"%d['output_file'])
    for k in os.listdir(d["output_file"]+"/Statistics_Extraction"):
        print (k)
        if ".sqlite" in k:
                plt.figure(figsize=(10,10))
                conn = sqlite3.connect(d["output_file"]+"/Statistics_Extraction/"+k)
                df=pd.read_sql_query("SELECT * FROM output", conn)
                globals()["dfpar%s"%k[-11:-7]]=df.groupby("originfid").mean()
                lab=globals()["dfpar%s"%k[-11:-7]][label.lower()]
                globals()["dfpar%s"%k[-11:-7]].drop(columns=list_bd_drop,inplace=True)
                globals()["dfpar%s"%k[-11:-7]]=globals()["dfpar%s"%k[-11:-7]].T
                globals()["dfpar%s"%k[-11:-7]]["band_names"]=colnames
                globals()["dfpar%s"%k[-11:-7]].set_index("band_names",inplace=True)
                globals()["dfpar%s"%k[-11:-7]].sort_index(inplace=True)
#                globals()["dfpar%s"%k[-11:-7]].index=pd.to_datetime(globals()["dfpar%s"%k[-11:-7]].iloc[:-1].index,format="%Y%m%d")
                globals()["dfpar%s"%k[-11:-7]]=globals()["dfpar%s"%k[-11:-7]].T
                for i in set(lab):
                    globals()['df%s%s' % (k[-11:-7],int(i))] = pd.DataFrame( globals()["dfpar%s"%k[-11:-7]][ globals()["dfpar%s"%k[-11:-7]][label]==int(i)]).T
#                    plt.figure(figsize=(10,10))
                    plt.title("%s%s"%(k[-11:-7],int(i)))
                    plt.plot( globals()['df%s%s' % (k[-11:-7],int(i))].iloc[:-1].T.mean())
#                    plt.fill_between(globals()['df%s%s' % (k[-11:-7],int(i))].iloc[:-1].index, globals()['df%s%s' % (k[-11:-7],int(i))].iloc[:-1].T.mean()-  globals()['df%s%s' % (k[-11:-7],int(i))].iloc[:-1].T.std() , globals()['df%s%s' % (k[-11:-7],int(i))].iloc[:-1].T.mean() + globals()['df%s%s' % (k[-11:-7],int(i))].iloc[:-1].T.std(), alpha=0.2)
                    plt.xticks(rotation=90,size=9)
#                    plt.legend([i])
                plt.savefig(d["output_file"]+"Plot_result/Plot_%s_%s.png"%(k[-11:-7],int(i)))
    ending=datetime.now() - start
#    print ("===============================")
#    print ("time process : %s" % ending)
#    print ("===============================")
#    i=4
#    k='NDRE'
#    plt.figure(figsize=(10,10))
#    sns.set(style="darkgrid")
#    sns.set_context('paper')
#    plt.title("%s%s"%(k[-11:-7],int(i)))
#    plt.plot( globals()['df%s%s' % (k,int(i))].iloc[:-1])
