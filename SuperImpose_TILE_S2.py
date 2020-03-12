#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 10:08:55 2019

@author: pageot
"""


import os
import otbApplication
import argparse

#def superimposeotb(img_input,imgref,output):


#list_IMG= os.listdir(d["SM"])
#list_tile=os.listdir(d["TILE"])
#for i in list_IMG:
#    print (i)
#    for j in list_tile:
#        tile=os.path.basename(j)[-16:-11]
#        print(tile)
#        Superimpose = otbApplication.Registry.CreateApplication("Superimpose")
#        Superimpose.SetParameterString('inm',d["SM"]+i)
#        Superimpose.SetParameterString('inr',d["TILE"]+ j)
#        Superimpose.SetParameterInt('interpolator.bco.radius',2)
#        Superimpose.SetParameterString("out",d["SM"]+"SM_TILES/%s_%s"%(tile,i))
#        Superimpose.ExecuteAndWriteOutput()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Superimposeotb_programme. this programme allows to reproject and cut image')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inref', dest='pathref',nargs='+',required = True)
    parser.add_argument('-inp', dest='path',nargs='+',required = True)
    parser.add_argument('-ram', dest='ram',nargs='+',required = True)
    args = parser.parse_args()
    print (args.path)
    print (args.out)
        
#    d={}
#    d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/"
#    d["GSM"]=d["data_file"]+"DONNES_SIG/CARTES_DES_SOLS/GSM/Raster_all/"
#    d["TILE"]=d["data_file"]+"TRAITEMENT/DATA_GROUND_2017/DPI_2017/IMG_NDVI_2017_TILES/"
#    d["SAFRAN"]=d["data_file"]+"TRAITEMENT/DATA_METEO/PRELIQ/SAFRAN_ALL/RASTER_10J/"
#    d["SM"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SOIL_MOISTURE/"
#    

    
    
    list_IMG= os.listdir("{}".format(str(args.path).strip("['']")))
    list_tile=os.listdir("{}".format(str(args.pathref).strip("['']")))
    for i in list_IMG:
        print (i)
        for j in list_tile:
#            tile=os.path.basename(j)[-16:-11]
            tile=os.path.basename(j)[0:3]
            print(tile)
            Superimpose = otbApplication.Registry.CreateApplication("Superimpose")
            Superimpose.SetParameterString('inm',"{}".format(str(args.path).strip("['']"))+i)
            Superimpose.SetParameterString('inr',"{}".format(str(args.pathref).strip("['']"))+j)
            Superimpose.SetParameterString('ram',"{}".format(str(args.ram).strip("['']")))
            Superimpose.SetParameterString('interpolator','nn')
            Superimpose.SetParameterString("out","{}".format(str(args.out).strip("['']"))+"/%s_%s"%(tile,i))
            Superimpose.ExecuteAndWriteOutput()
