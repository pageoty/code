#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:36:28 2019

@author: pageot
"""

import os
import otbApplication
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='SAR_fusion_impose_programme. this programme allows to fusion and cut image')
    parser.add_argument('-out', dest='out')
    parser.add_argument('-inp', dest='path',nargs='+',required = True)
    args = parser.parse_args()
    print (args.path)
    print (args.out)

    list_vv=[]
    list_vh=[]
    for i in  os.listdir("{}".format(str(args.path).strip("['']"))):
        #print (i)
        polar=i[10:12]
        orbite=i[13:16]
        if (polar == "vv" and "s1b_31TCJ_vv" and ".tif" in i and "BorderMask" not in i):
            list_vv.append("{}".format(str(args.path).strip("['']"))+i)
        if (polar == "vh" and "s1b_31TCJ_vv" and ".tif" in i and "BorderMask" not in i):
            list_vh.append("{}".format(str(args.path).strip("['']"))+i)
    print(list_vv)     
    list_vv_sorted=[]
    for x in sorted(list_vv):
        list_vv_sorted.append(x)
    list_vh_sorted=[]   
    for h in sorted(list_vh):
        list_vh_sorted.append(h)
    
            
            
            
    for p in ["vv","vh"]:
        if (p == "vv"):
            App2 = otbApplication.Registry.CreateApplication("ConcatenateImages")
            App2.SetParameterStringList("il",list_vv_sorted)
            App2.SetParameterString("out","%sSAR_%s%s.tif"%("{}".format(str(args.out).strip("['']")),'vv',orbite))
            App2.ExecuteAndWriteOutput()                                                                                                                            
        else:
            App2 = otbApplication.Registry.CreateApplication("ConcatenateImages")
            App2.SetParameterStringList("il",list_vh_sorted)
            App2.SetParameterString("out","%sSAR_%s%s.tif"%("{}".format(str(args.out).strip("['']")),'vh',orbite))
            App2.ExecuteAndWriteOutput()  