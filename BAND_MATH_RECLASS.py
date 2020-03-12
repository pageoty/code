#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:12:43 2019

@author: pageot
"""

import os
import otbApplication

d={}
d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SRTM/MASK_SRTM_TILE/"

for i in os.listdir(d["data_file"]): 
    print (i)
    name=i[0:-4]
    App1= otbApplication.Registry.CreateApplication("ConcatenateImages")
    App1.SetParameterStringList("il",[str(d["data_file"]+i)])
    App1.SetParameterString("out","EXPO_MASK.tif")
    App1.Execute()
    
    application2 = otbApplication.Registry.CreateApplication("BandMath")
    application2.AddImageToParameterInputImageList("il",App1.GetParameterOutputImage("out"))
    application2.SetParameterString("out", "RECLASS_%s.tif"%(name))
    application2.SetParameterString("exp", 'im1b1 >=1 && im1b1 <=45 ? im1b1 =1 : im1b1 >= 46 &&  im1b1 <= 135 ? im1b1 =2 : im1b1 >= 136 &&   im1b1 <= 225 ? im1b1 =3 : im1b1 >= 226 &&  im1b1 <=  315 ? im1b1 = 4 : im1b1 >= 316 &&  im1b1  <=  360 ? im1b1 =1 : im1b1=0')
    print("Reclass the values \n")
    application2.ExecuteAndWriteOutput()
