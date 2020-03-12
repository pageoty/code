#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os 

df=pd.read_csv("/work/CESBIO/projects/Irrigation/dataExogene/2018/list_features_optic2018.txt",sep=',', header=None)
df1=df.T
df1.columns=["band_name"]
df1.index = np.arange(1, len(df1)+1)
df1["band"] = df1.index
df1["indice"] = df1.band_name.apply(lambda s: s[12:15])



list_channel=[]
f=df1.iloc[90:340]["band"].tolist()
for i in f:
    a = "Channel"+ str(i)
    list_channel.append(a)
print (df1.iloc[90:340])

f=df1.band[379:404].tolist()
for i in f:
	a = "Channel"+ str(i)
	list_channel.append(a)

f=df1.band[416:441].tolist()
for i in f:
	a = "Channel"+ str(i)
	list_channel.append(a)

f=df1.band[454:478].tolist()
for i in f:
	a = "Channel"+ str(i)
	list_channel.append(a)

list_cha = ' '.join(str(x) for x in list_channel)
print(list_cha)

#~ os.system("otbcli_ExtractROI -in /work/CESBIO/projects/Irrigation/dataExogene/T31TDJ/Sentinel2_T31TDJ_Features.tif -cl "+list_cha+" -out /work/CESBIO/projects/Irrigation/dataExogene/T31TDJ/SAISON_S2_NDVI_NDWI_BRI_2018.tif")
