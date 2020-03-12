#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:49:09 2019

@author: pageot
"""

import os
import otbApplication
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Band_Math_programme ')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inp', dest='image',nargs='+',help="path of image",required = True)
    parser.add_argument('-exp', dest='exp',nargs='+',help="Expression of the BM",required = True)
    args = parser.parse_args()
    print (args.image)

    
    

for i in os.listdir("{}".format(str(args.image).strip("['']"))): 
    print (str("{}".format(str(args.image).strip("['']"))+i))
    
    App= otbApplication.Registry.CreateApplication("ConcatenateImages")
    App.SetParameterStringList("il",["{}".format(str(args.image).strip("['']"))+i])
    App.SetParameterString("out","TST_L8.tif")
    App.Execute()
    
    App1= otbApplication.Registry.CreateApplication("BandMath")
    App1.AddImageToParameterInputImageList("il",App.GetParameterOutputImage("out"))
    App1.SetParameterString("out","{}".format(str(args.out).strip("['']"))+"Ss_border"+i)
    App1.SetParameterString("exp","{}".format(str(args.exp).strip("['']")))
    App1.ExecuteAndWriteOutput()
