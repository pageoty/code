# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:31:27 2019

@author: pageot
"""

import os
import otbApplication
import time


d={}
d["dataset_folder"] = "/datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_L8_LST"
d["data_mask"]="/datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_L8/file_unzip/"


list_l8_lst=os.listdir(d["dataset_folder"])
list_mask=os.listdir(d["data_mask"])

#for j in list_mask:
#    path_img=os.path.basename(i)
#    print path_img
#    d["image_name"] = path_img 
#    d["mask"] = d["data_mask"] + d["image_name"] + "_BQA.TIF"
#    print d["mask"]
    
    
for i in list_l8_lst:
    path_img=os.path.basename(i)
    print path_img
    d["dataset_folder"] = "/datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_L8_LST/"
    d["image_name"] = path_img  
    d["input_path"] = d["dataset_folder"]+d["image_name"] +"/"
    d["input_path_could"]=d["data_mask"]+d["image_name"] +"/"
    d["tst"] =  d["image_name"] + "_tst.tif"  
    d["mask"] = d["image_name"]+ "_BQA.TIF"
    
    App= otbApplication.Registry.CreateApplication("ConcatenateImages")
    App.SetParameterStringList("il",[str(d["input_path"] + d["tst"])])
    App.SetParameterString("out","tst.tif")
    App.Execute()
    
    application1 = otbApplication.Registry.CreateApplication("BandMath")
    application1.SetParameterStringList("il",[str(d["input_path_could"] + d["mask"])])
    application1.SetParameterString("out", "maskname_L8.tif")
    application1.SetParameterString("exp", 'im1b1 > 2720 ? im1b1 = 0 : im1b1 = 1')
    
    application1.Execute()
    print("conversion Mask\n")
    
    application2 = otbApplication.Registry.CreateApplication("BandMath")

    application2.AddImageToParameterInputImageList("il",application1.GetParameterOutputImage("out")) 
    application2.AddImageToParameterInputImageList("il",App.GetParameterOutputImage("out"))
    application2.SetParameterString("out", "%s_TST_Mask.tif"%(path_img))
    application2.SetParameterString("exp", "(im1b1*im2b1)")
    
    application2.ExecuteAndWriteOutput()
    
os.system("mv *TST_Mask.tif /datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_L8_LST/")
    
    