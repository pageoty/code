#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:27:33 2019

@author: dahingerv
"""

#from osgeo import gdal
import otbApplication
import numpy as np
import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cumul Indice')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inl', dest='inlist')
    parser.add_argument('-sigma', dest='indice') # exemple : 'userfeatures'
    parser.add_argument('-tile', dest='tile')
    parser.add_argument('-inp', dest='path',nargs='+',required = True)
    parser.add_argument('-time_vege',dest='vege', action='store_true') 
    args = parser.parse_args()

    print (args.vege)

	#Mettre en forme le df
    indice = "{}".format(str(args.indice).strip("[]"))
    df=pd.read_csv("{}".format(str(args.inlist).strip("['']")),sep=',', header=None)
    df1=df.T
    df1.columns=["band_name"]
    df1.index = np.arange(1, len(df1)+1)
    df1["band"] = df1.index
    if indice in ['des_vv','asc_vv','des_vh','asc_vh']:
		df1["indice"] = df1.band_name.apply(lambda s: s[12:18])
		df1["month"] = df1.band_name.apply(lambda s:s [-5:-3])
		df_indice = df1[df1['indice'] == indice]
		band_indice = df_indice["band"].tolist()
		print (df_indice)
    else:
		df1["indice"] = df1.band_name.apply(lambda s: s[12:28])
		df1["month"] = df1.band_name.apply(lambda s:s [-5:-3])
		df_indice = df1[df1['indice'] == "{}".format(str(args.indice).strip("[]"))]
		band_indice = df_indice["band"].tolist()
		print (df_indice)
    

    path_folder = "{}".format(str(args.path).strip("['']"))
    
    if args.vege == False:
		frequence=3 # interpolation tous les 10 jours -> cumul tous les 30 jours
		t=1
		lst_img=[]

		while frequence <= len(band_indice):
			expres = []
			for i in band_indice[:frequence]:
				i= "im1b"+str(i)
				expres.append(i) 
				
			expres = '+'.join(str(x) for x in expres)
			
			tile ="{}".format(str(args.tile).strip("['']"))
			print (tile)
			print (path_folder)
			print(indice)

			print("===================")
			print(expres)
			
			BMapp = otbApplication.Registry.CreateApplication("BandMath")
			BMapp.SetParameterStringList("il",[path_folder +"Sentinel1_%s_Features.tif"%tile])
			BMapp.SetParameterString("out",path_folder+"SUMmensuel_%s_%s_temp%s.tif"% (tile,indice,t))
			BMapp.SetParameterString("exp", expres)
			BMapp.ExecuteAndWriteOutput()
			
			lst_img.append(path_folder+"SUMmensuel_%s_%s_temp%s.tif"% (tile,indice,t))
			print("image SUMmensuel_%s_%s_temp%s.tif ajoutée à la liste"% (tile,indice,t))
			print (lst_img)
			frequence+=3
			t+=1
		tile ="{}".format(str(args.tile).strip("['']"))
		ConcatImg = otbApplication.Registry.CreateApplication("ConcatenateImages")
		ConcatImg.SetParameterStringList("il", lst_img)
		ConcatImg.SetParameterString("out", path_folder+"SUMmensuel_%s_%s.tif"% (indice,tile))
		ConcatImg.ExecuteAndWriteOutput()
		
    else:	

		freq=pd.value_counts(df_indice['month'],sort=False)
		freq=pd.DataFrame(freq)
		freq.sort_index(ascending=True,inplace=True)
		print (freq.month[3:-1])
		lst_img=[]
		t=1
		frequence =0
		for i in freq.month[3:-1]:
			frequence+=i
			print (frequence)
			print(band_indice[9:(9+frequence)])
			expres=[]
			for j in band_indice[9:(9+frequence)]:
					j= "im1b"+str(j)
					expres.append(j) 
			expres = '+'.join(str(x) for x in expres)
			print (expres)
			
				
			tile ="{}".format(str(args.tile).strip("['']"))
			print (tile)
			print (path_folder)
			print(indice)
			print("===================")
			print(expres)
	

			BMapp1 = otbApplication.Registry.CreateApplication("BandMath")
			BMapp1.SetParameterStringList("il",[path_folder +"Sentinel1_%s_Features.tif"%tile])
			BMapp1.SetParameterString("out",path_folder+"SUMmensuel_%s_%s_temp%s.tif"% (tile,indice,t))
			BMapp1.SetParameterString("exp", expres)
			BMapp1.ExecuteAndWriteOutput() 

			lst_img.append(path_folder+"SUMmensuel_%s_%s_temp%s.tif"% (tile,indice,t))
			print("image SUMmensuel_%s_%s_temp%s.tif ajoutée à la liste"% (tile,indice,t))
			print (lst_img)

			t+=1	
		tile ="{}".format(str(args.tile).strip("['']"))
		ConcatImg = otbApplication.Registry.CreateApplication("ConcatenateImages")
		ConcatImg.SetParameterStringList("il", lst_img)
		ConcatImg.SetParameterString("out", path_folder+"SUMmensuelVEGE_%s_%s.tif"% (indice,tile))
		ConcatImg.ExecuteAndWriteOutput()
