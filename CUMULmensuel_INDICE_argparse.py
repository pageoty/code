#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:45:03 2019

@author: dahingerv & pageot

Calcul des indices cumuls mensuels, à utiliser en bash, deux mode de calcul sur la saison (Avril à novembre) ou ensemble de l'année
"""


import gdal
import otbApplication
import numpy as np
import pandas as pd
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create cumul Indice')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inl', dest='inlist')
    parser.add_argument('-indice', dest='indice')
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
    df1["indice"] = df1.band_name.apply(lambda s: s[12:16])
    df1["month"] = df1.band_name.apply(lambda s:s [-5:-3])
   
    #Tableaux et liste avec les bandes NDVI
    indice = "{}".format(str(args.indice).strip("[]"))
    df_indice = df1[df1['indice'] == "{}".format(str(args.indice).strip("[]"))]
    band_indice = df_indice["band"].tolist()
    print (df_indice)
    print (band_indice)
    path_folder = "{}".format(str(args.path).strip("['']"))
    expres = []
    if args.vege == False :
    
        frequence=3 # interpolation tous les 10 jours -> cumul tous les 30 jours
        t=1
        lst_img=[]
        while frequence <= len(band_indice):
            expres = []
            
            for i in band_indice[:frequence]:
            
                i= "im1b"+str(i)
                expres.append(i) 
        
            expres = '+'.join(str(x) for x in expres)     
            tile="{}".format(str(args.tile).strip("['']"))
            print (tile)
            print (path_folder)
            print (df_indice)
            print(indice)
            print("===================")
            print(expres)
            
            BMapp = otbApplication.Registry.CreateApplication("BandMath")
            BMapp.SetParameterStringList("il",[path_folder +"Sentinel2_%s_Features.tif"%tile])
            BMapp.SetParameterString("out",path_folder+"SUMmensuel_%s_%s_temp%s.tif"% (tile,indice,t))
            BMapp.SetParameterString("exp", expres)
            BMapp.ExecuteAndWriteOutput()
            
            lst_img.append(path_folder+"SUMmensuel_%s_%s_temp%s.tif"% (tile,indice,t))
            print("image SUMmensuel_%s_%s_temp%s.tif ajoutée à la liste"% (tile,indice,t))
            
            frequence+=3
            t+=1
        
        ConcatImg = otbApplication.Registry.CreateApplication("ConcatenateImages")
        ConcatImg.SetParameterStringList("il", lst_img)
        ConcatImg.SetParameterString("out", path_folder+"SUMmensuel_%s_%s.tif"% (tile,indice))
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
            print(band_indice[8:(8+frequence)])
            expres=[]
            for i in band_indice[8:(8+frequence)]: # A modifier pour l'année 2017 avec 8 a la passe de 9
                i= "im1b"+str(i)
                expres.append(i) 
            expres = '+'.join(str(x) for x in expres)     
            tile="{}".format(str(args.tile).strip("['']"))
            print (tile)
            print (path_folder)
            print(indice)
            print("===================")
            print(expres)
            
            BMapp = otbApplication.Registry.CreateApplication("BandMath")
            BMapp.SetParameterStringList("il",[path_folder +"Sentinel2_%s_Features.tif"%tile])
            BMapp.SetParameterString("out",path_folder+"SUMmensuel_%s_%s_vege_temp%s.tif"% (tile,indice,t))
            BMapp.SetParameterString("exp", expres)
            BMapp.ExecuteAndWriteOutput()
            
            lst_img.append(path_folder+"SUMmensuel_%s_%s_vege_temp%s.tif"% (tile,indice,t))
            print("image SUMmensuel_%s_%s_vege_temp%s.tif ajoutée à la liste"% (tile,indice,t))
            
            t+=1
        
        ConcatImg = otbApplication.Registry.CreateApplication("ConcatenateImages")
        ConcatImg.SetParameterStringList("il", lst_img)
        ConcatImg.SetParameterString("out", path_folder+"SUMmensuelVEGE_%s_%s.tif"% (tile,indice))
        ConcatImg.ExecuteAndWriteOutput()
