#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:27:33 2019

@author: dahingerv
"""

from osgeo import gdal
import otbApplication
import numpy as np
import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cumul Indice')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inl', dest='inlist')
    parser.add_argument('-NIR', dest='NIR') # exemple : 'B8A'
    parser.add_argument('-rededge', dest='rededge') # exemple : 'B5'
    parser.add_argument('-tile', dest='tile')
    parser.add_argument('-inp', dest='path',nargs='+',required = True)
    parser.add_argument('-time_vege',dest='vege', action='store_true') 
    args = parser.parse_args()

    print (args.vege)

	#Mettre en forme le df
    df=pd.read_csv("{}".format(str(args.inlist).strip("['']")),sep=',', header=None)
    df1=df.T
    df1.columns=["band_name"]
    df1.index = np.arange(1, len(df1)+1)
    df1["band"] = df1.index
    df1["indice"] = df1.band_name.apply(lambda s: s[12:15])

    #Tableaux et liste avec les bandes NDVI
    NIR = "{}".format(str(args.NIR).strip("[]"))
    rededge = "{}".format(str(args.rededge).strip("[]"))
    df_NIR = df1[df1['indice'] == "{}".format(str(args.NIR).strip("[]"))]
    df_rededge = df1[df1['indice'] == "{}".format(str(args.rededge).strip("[]"))]
    band_NIR = df_NIR["band"].tolist()
    band_rededge = df_rededge["band"].tolist()

    path_folder = "{}".format(str(args.path).strip("['']"))
    
    if args.vege == False:
		expres = []
		for i, j in zip(band_NIR, band_rededge):
			num = "(im1b"+str(i)+"-im1b"+str(j)+")"
			denum = "(im1b"+str(i)+"+im1b"+str(j)+")"
			x = num+"/"+denum
			condition = denum+"<=0 ? 1: "
			compute = condition + x
			expres.append(compute)
		expres = ','.join(str(x) for x in expres)
			
		tile="{}".format(str(args.tile).strip("['']"))
		print (tile)
		print (path_folder)
		print (df_NIR)
		print (df_rededge)
		print(NIR)
		print(rededge)

		BMapp1 = otbApplication.Registry.CreateApplication("BandMath")
		BMapp1.SetParameterStringList("il",[path_folder +"Sentinel2_%s_Features.tif"%tile])
		BMapp1.SetParameterString("out",path_folder+"CUMUL_%s_%s_%s.tif"% (tile,NIR,rededge))
		BMapp1.SetParameterString("exp", expres)
		BMapp1.ExecuteAndWriteOutput()  

    else:
		expres = []
		for i, j in zip(band_NIR[8:-3], band_rededge[8:-3]):
			num = "(im1b"+str(i)+"-im1b"+str(j)+")"
			denum = "(im1b"+str(i)+"+im1b"+str(j)+")"
			x = num+"/"+denum
			condition = denum+"<=0 ? 1: "
			compute = condition + x
			expres.append(compute)

		expres = ','.join(str(x) for x in expres)
		print(expres)
	
		tile="{}".format(str(args.tile).strip("['']"))
		print (df_NIR)
		print (df_rededge)
		print(NIR)
		print(rededge)
		BMapp1 = otbApplication.Registry.CreateApplication("BandMath")
		BMapp1.SetParameterStringList("il",[path_folder +"Sentinel2_%s_Features.tif"%tile])
		BMapp1.SetParameterString("out",path_folder+"CUMUL_%s_%s_%s_time_vege.tif"% (tile,NIR,rededge))
		BMapp1.SetParameterString("exp", expres)
		BMapp1.ExecuteAndWriteOutput() 
