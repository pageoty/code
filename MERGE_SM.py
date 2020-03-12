#!/usr/bin/python
# coding: utf-8

# 

import os
import csv
import gdal
import argparse


#def fusion_sm(empriseA,empriseB):
#	for i in empriseA:
#		date=i[0:8]
#		for j in empriseB :
#			date2=j[0:8]
#			if date==date2:
#				print ("processing en cours")
#				os.system("gdal_merge.py -o SM_%s.tif %s/%s %s/%s -n 0"%(date,PATHA,i,PATHB,j))
#			else : 
#				print ("pas de Fusion")

#	os.system("mv SM* %s"%(outpath))

def fusion_sm1(PATHA,PATHB,outpath):
    for i in os.listdir(PATHA):
        for j in os.listdir(PATHB):
            print (i,j)
            date=i[-19:-11]
            date2=j[-19:-11]
            print (date)
            print (date2)
            if date==date2:
                print( 'fus')
                os.system("gdal_merge.py -o SM_%s.tif %s/%s %s/%s -n 0"%(date,PATHA,i,PATHB,j))
            else : 
                print ("pas de Fusion")
    os.system("mv SM* %s"%(outpath))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FUSION_RASTER.')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inp1', dest='path1',nargs='+',required = True)
    parser.add_argument('-inp2', dest='path2',nargs='+',required = True)
    args = parser.parse_args()
    print (args.out)
    

#	outpath = "/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/SOIL_MOISTURE/"	

    fusion_sm1("{}".format(str(args.path1).strip("['']")),"{}".format(str(args.path2).strip("['']")),"{}".format(str(args.out).strip("['']")))

 
 