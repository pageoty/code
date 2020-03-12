#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:13:33 2019

@author: pageoty
"""

import otbApplication
import os 
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fusion classif with crops mask ')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inp', dest='image',nargs='+',help="path of image",required = True)
    parser.add_argument('-maskcrops', dest='maskcrops',help="path of image crops")
    parser.add_argument('-namerun', dest='run',help="name of run")
    args = parser.parse_args()
    print (args.image)
    
    path_masque = "{}".format(str(args.maskcrops).strip("['']"))

    App= otbApplication.Registry.CreateApplication("ConcatenateImages")
    App.SetParameterStringList("il",[path_masque])
    App.SetParameterString("out","mask.tif")
    App.Execute()
    
    for i in range(0,5):
        print (i)
        App0= otbApplication.Registry.CreateApplication("ConcatenateImages")
        App0.SetParameterStringList("il",["{}".format(str(args.image).strip("['']"))+"Classif_Seed_"+str(i)+"_ColorIndexed.tif"])
        App0.SetParameterString("out","classif.tif")
        App0.Execute()
    
        App1= otbApplication.Registry.CreateApplication("BandMath")
        App1.AddImageToParameterInputImageList("il",App.GetParameterOutputImage("out"))
        App1.AddImageToParameterInputImageList("il",App0.GetParameterOutputImage('out'))
        App1.SetParameterString("out","{}".format(str(args.out).strip("['']"))+"/mask_classif%s.tif"%i)
        App1.SetParameterString("exp","im1b1 * im2b1")
        App1.ExecuteAndWriteOutput()
    
    list_img=["{}".format(str(args.out).strip("['']"))+"/mask_classif0.tif","{}".format(str(args.out).strip("['']"))+"/mask_classif1.tif","{}".format(str(args.out).strip("['']"))+"/mask_classif2.tif"
              ,"{}".format(str(args.out).strip("['']"))+"/mask_classif3.tif","{}".format(str(args.out).strip("['']"))+"/mask_classif4.tif"]
    print (list_img)

    FusionClassif = otbApplication.Registry.CreateApplication("FusionOfClassifications")
    FusionClassif.SetParameterStringList('il',list_img)
    FusionClassif.SetParameterString("out","{}".format(str(args.out).strip("['']")) +"/Classif_msk_Merged%s.tif"%"{}".format(str(args.run).strip("['']")))
    FusionClassif.ExecuteAndWriteOutput()

    for i in range(0,5):
        os.system('rm %s/mask_classif%s.tif' % ("{}".format(str(args.out).strip("['']")),i))
        
