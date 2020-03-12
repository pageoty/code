#!/usr/bin/python
# coding: utf-8
# source /datalocal/vboxshare/OTB-6.6.0-Linux64/otbenv.profile
# 

import os
import otbApplication
import zipfile
import csv




images=os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_S2/S2_TILES_2018/S2_zip/")
for i in images:
	zip_ref = zipfile.ZipFile("/datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_S2/S2_TILES_2018/S2_zip/"+i, 'r')
	zip_ref.extractall()
	zip_ref.close()
	print (i)
os.system ("mv *_V1* /datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_S2/S2_TILES_2018/S2_unzip/")

Images_nan_zip= os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_S2/S2_TILES_2018/S2_unzip/")	
print (Images_nan_zip)

#list (ensembe des ficiers sans l'extension.zip)	
for j in Images_nan_zip: 
	path_img=os.path.basename(j)
	print (path_img)

	d={}
	d["dataset_folder"] = "/datalocal/vboxshare/THESE/CLASSIFICATION/IMG_SAT/DONNEES_S2/S2_TILES_2018/S2_unzip/"

	d["image_name"] = path_img  
	d["input_path"] = d["dataset_folder"]+d["image_name"] +"/"
	d["B4_image"] =  d["image_name"] + "_FRE_B4.tif"  
	d["B8_image"] = d["image_name"] + "_FRE_B8.tif" 
	d["B3_image"] = d["image_name"] + "_FRE_B3.tif" 
	d["coulds_mask"] = "MASKS/" + d["image_name"] + "_CLM_R1.tif"
	print  (d["coulds_mask"])
	print (d["input_path"] + d["coulds_mask"])
	print (d["input_path"] + d["B4_image"])
	###Aplliquer mask 

    #cahine en mémore les autres bandes avec concat
	App= otbApplication.Registry.CreateApplication("ConcatenateImages")
	App.SetParameterStringList("il",[str(d["input_path"] + d["B4_image"])])
	App.SetParameterString("out","B4_image.tif")
	App.Execute()

	App2 = otbApplication.Registry.CreateApplication("ConcatenateImages")
	App2.SetParameterStringList("il",[str(d["input_path"] + d["B8_image"])])
	App2.SetParameterString("out","B8_image.tif")
	App2.Execute()

	App3 = otbApplication.Registry.CreateApplication("ConcatenateImages")
	App3.SetParameterStringList("il",[str (d["input_path"] + d["B3_image"])])
	App3.SetParameterString("out", "B3_image.tif")
	App3.Execute()

   #Création du mask nuage en binaire
	application1 = otbApplication.Registry.CreateApplication("BandMath")
	application1.SetParameterStringList("il",[str(d["input_path"] + d["coulds_mask"])])
	application1.SetParameterString("out", "maskname.tif")
   	#application1.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint8)
	application1.SetParameterString("exp", 'im1b1 > 1 ? im1b1 = 0 : im1b1 = 1')
  
	
 
   # The following line execute the application1
	application1.Execute()
	print("conversion Mask\n")

   	#Application mask B4 

	application2 = otbApplication.Registry.CreateApplication("BandMath")

	application2.AddImageToParameterInputImageList("il",application1.GetParameterOutputImage("out")) 
	application2.AddImageToParameterInputImageList("il",App.GetParameterOutputImage("out"))
	application2.SetParameterString("out", "B4_image_M.tif")
   	#application2.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint8)
	application2.SetParameterString("exp", "(im1b1*im2b1)")
   	
	application2.Execute()

   	#Application mask sur B8

	application3 = otbApplication.Registry.CreateApplication("BandMath")

	application3.AddImageToParameterInputImageList("il",application1.GetParameterOutputImage("out"))
	application3.AddImageToParameterInputImageList("il",App2.GetParameterOutputImage("out"))
	application3.SetParameterString("out","B8_image_M.tif")
	application3.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint8)
	application3.SetParameterString("exp", "(im1b1*im2b1)")
	application3.Execute()

	#Application mask sur B3

	application5 = otbApplication.Registry.CreateApplication("BandMath")

	application5.AddImageToParameterInputImageList("il",application1.GetParameterOutputImage("out"))
	application5.AddImageToParameterInputImageList("il",App3.GetParameterOutputImage("out"))
	application5.SetParameterString("out","B3_image_M.tif")
	#application5.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint8)
	application5.SetParameterString("exp", "(im1b1*im2b1)")
	application5.Execute()

   	########### Application 2 : NDVI Calculation
    #

	application4 = otbApplication.Registry.CreateApplication("BandMath")

	application4.AddImageToParameterInputImageList("il",application2.GetParameterOutputImage("out"))
	application4.AddImageToParameterInputImageList("il",application3.GetParameterOutputImage("out"))

	application4.SetParameterString("out", "%s_NDVI.tif"%(path_img))

	application4.SetParameterString("exp", "(im2b1-im1b1)/(im2b1+im1b1)")

	application4.ExecuteAndWriteOutput()
	


	###CREATION IMG_FAUSSE COLOR


	App4 = otbApplication.Registry.CreateApplication("ConcatenateImages")
	App4.AddImageToParameterInputImageList("il",application3.GetParameterOutputImage("out"))
	App4.AddImageToParameterInputImageList("il",application2.GetParameterOutputImage("out"))
	App4.AddImageToParameterInputImageList("il",application5.GetParameterOutputImage("out"))
	App4.SetParameterString("out", "IMG_F_COLOR_%s.tif"%(path_img))
	App4.ExecuteAndWriteOutput()

#	os.system("mv *_NDVI.tif /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_GROUND_2018/DPI_2018/S2_NDVI_FAUSSE_COLOR")
#	os.system("mv *IMG_F* /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_GROUND_2018/DPI_2018/S2_NDVI_FAUSSE_COLOR")