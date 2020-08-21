#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:13:13 2019

@author: pageot

script pour comparer les trajectoires des signture temporelles sur une petit zone a la parcelle
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import csv
from STAT_ZONAL_SPECRTE import *
from scipy import stats


def pltmutli(x,x2,y1,y1bis,y2,y2bis,y3,y3bis,y4,y4bis,y5,y5bis,x3,y6):
    plt.figure(figsize=(20,20))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(611)
    p1=plt.plot(x,y1,color='blue')
    p2=plt.plot(x,y1bis,color='red',linestyle='--')
    plt.ylabel("NDVI")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(612)
    plt.plot(x,y2,color="blue")
    plt.plot(x,y2bis,color='red',linestyle='--')
    plt.ylabel("NDWI")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = plt.subplot(613)
    plt.plot(x2,y3,color='blue')
    plt.plot(x2, y3bis,color="red",linestyle='--')
    plt.ylabel("VV")
    plt.setp(ax3.get_xticklabels(), visible=False)    
    ax4 = plt.subplot(614)
    plt.plot(x2,y4,color='blue')
    plt.plot(x2, y4bis,color='red',linestyle='--')
    plt.ylabel("VH")
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax5 = plt.subplot(615)
    plt.plot(x2,y5,color='blue')
    plt.plot(x2, y5bis,color='red',linestyle='--')
    plt.ylabel("VH/VV")
    plt.setp(ax5.get_xticklabels(),visible=False)
    plt.legend((p1[0],p2[0]),("Irrigué","Non Irrigué"))
    ax6 = plt.subplot(616)
    plt.bar(x3,y6,color='blue',width=1)
    plt.ylabel("pluviométrie en mm")
    plt.setp(ax6.get_xticklabels(),rotation=90)
    
def pltmutlimode(x,x2,x3,x4,y1,y1bis,y2,y2bis,y3,y3bis,y3tris,y3quad,y4,y4bis,y4tris,y4quad,y5,y5bis,x6,y6):
    plt.figure(figsize=(20,20))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(611)
    p1=plt.plot(x,y1,color='blue')
    p2=plt.plot(x,y1bis,color='red',linestyle='--')
    plt.ylabel("NDVI")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(612)
    plt.plot(x,y2,color="blue")
    plt.plot(x,y2bis,color='red',linestyle='--')
    plt.ylabel("NDWI")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = plt.subplot(613)
    p3=plt.plot(x2,y3,color='blue')
    plt.plot(x2, y3bis,color="red",linestyle='-')
    p4=plt.plot(x3,y3tris,color='blue',linestyle=':')
    plt.plot(x3,y3quad,color='red',linestyle=':')
    plt.legend((p3[0],p4[0]),("des","des"))
    plt.ylabel("VV")
    plt.setp(ax3.get_xticklabels(), visible=False)    
    ax4 = plt.subplot(614)
    plt.plot(x2,y4,color='blue')
    plt.plot(x2,y4bis,color='red',linestyle='-')
    plt.plot(x3,y4tris,color='blue',linestyle=':')
    plt.plot(x3,y4quad,color='red',linestyle=':')
    plt.legend((p3[0],p4[0]),("des","des"))
    plt.ylabel("VH")
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax5 = plt.subplot(615)
    plt.plot(x4,y5,color='blue')
    plt.plot(x4, y5bis,color='red',linestyle='--')
    plt.ylabel("VH/VV")
    plt.setp(ax5.get_xticklabels(),visible=False)
    plt.legend((p1[0],p2[0]),("Irrigué","Non Irrigué"))
    ax6 = plt.subplot(616)
    plt.bar(x6,y6,color='blue',width=1)
    plt.ylabel("pluviométrie en mm")
    plt.setp(ax6.get_xticklabels(),rotation=90)
    
def plotconfi(x,x2,x3,y1,y1bis,y3,y3bis,y3tris,y3quad,y4,y4bis,y4tris,y4quad,yr,yrbis,yr1,yrbis1,x6,y6,confsup,confinf):
    plt.figure(figsize=(20,20))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(811)
    p1=plt.plot(x,y1.T.mean(),color='blue')
    p2=plt.plot(x,y1bis.T.mean(),color='red',linestyle='--')
    plt.fill_between(x, confsup[0], confinf[0], facecolor='blue', alpha=0.2)
    plt.fill_between(x, confsup[1], confinf[1], facecolor='red', alpha=0.2)
    plt.ylabel("NDVI")
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(812)
    plt.plot(x2, y3.T.mean(), color='blue',linestyle=":")
    plt.fill_between(x2, confsup[2], confinf[2], facecolor='blue', alpha=0.2)
    plt.plot(x2, y3bis.T.mean(), color='red',linestyle=":")
    plt.fill_between(x2, confsup[3], confinf[3], facecolor='red', alpha=0.2)
    plt.ylabel("VV des")
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = plt.subplot(813)
    plt.plot(x3, y3tris.T.mean(), color='blue')
    plt.fill_between(x3, confsup[4], confinf[4], facecolor='blue', alpha=0.2)
    plt.plot(x3, y3quad.T.mean(), color='red')
    plt.fill_between(x3, confsup[5], confinf[5], facecolor='red', alpha=0.2)
    plt.ylabel("VV des")
    plt.legend((p1[0],p2[0]),("Irrigué","Non Irrigué"))
    plt.setp(ax3.get_xticklabels(), visible=False)    
    ax4 = plt.subplot(814)
    plt.plot(x2, y4.T.mean(), color='blue')
    plt.fill_between(x2, confsup[6], confinf[6], facecolor='blue', alpha=0.2)
    plt.plot(x2, y4bis.T.mean(), color='red')
    plt.fill_between(x2, confsup[7], confinf[7], facecolor='red', alpha=0.2)
    plt.ylabel("VH des")
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax5 = plt.subplot(815)
    plt.plot(x3, y4tris.T.mean(), color='blue',linestyle=":")
    plt.fill_between(x3, confsup[8], confinf[8], facecolor='blue', alpha=0.2)
    plt.plot(x3, y4quad.T.mean(), color='red',linestyle=":")
    plt.fill_between(x3, confsup[9], confinf[9], facecolor='red', alpha=0.2)
    plt.ylabel("VH des")
    plt.setp(ax5.get_xticklabels(),visible=False)
    plt.legend((p1[0],p2[0]),("Irrigué","Non Irrigué"))
    ax6= plt.subplot(816)
    plt.plot(yr.index, yr.T.mean(), color='blue',linestyle=":")
    plt.fill_between(yr.index, confsup[10], confinf[10], facecolor='blue', alpha=0.2)
    plt.plot(yr.index, yrbis.T.mean(), color='red',linestyle=":")
    plt.fill_between(yr.index, confsup[11], confinf[11], facecolor='red', alpha=0.2)
    plt.ylabel("ratio des_VH/VV")
    plt.setp(ax6.get_xticklabels(),visible=False)
    ax7= plt.subplot(817)
    plt.plot(yr1.index, yr1.T.mean(), color='blue',linestyle=":")
    plt.fill_between(yr1.index, confsup[12], confinf[12], facecolor='blue', alpha=0.2)
    plt.plot(yr1.index, yrbis1.T.mean(), color='red',linestyle=":")
    plt.fill_between(yr1.index, confsup[13], confinf[13], facecolor='red', alpha=0.2)
    plt.ylabel("ratio des_VH/VV")
    plt.setp(ax7.get_xticklabels(),visible=False)
    ax8 = plt.subplot(818)
    plt.bar(x6,y6,color='blue',width=1)
    plt.ylabel("pluviométrie en mm")
    plt.setp(ax8.get_xticklabels(),rotation=90)

def col_sqlite(path,name,list_bd_drop,pathlist_names_feature): # modifier par labcroirr non générique
    """
    Permet d'attribuer le nom des primitives dans le sqlite a partir de la list des primitives"""
    
    
    dfnames=pd.read_csv(pathlist_names_feature,sep=',', header=None) 
    df1=dfnames.T
    df1.columns=["band_name"]
    colnames=list(df1.band_name.apply(lambda s: s[2:-1]))
    
    if ".csv" in path:
        df=pd.read_csv(path)
        globals()["%s"% name ]=df.groupby("originfid").mean()
        labcroirr=globals()["df%s"% name ].labcroirr
        globals()["df%s"% name ].drop(columns=list_bd_drop,inplace=True)
        globals()["df%s"% name ]=globals()["df%s"% name ].T

        globals()["df%s"% name ]["band_names"]=colnames
        globals()["df%s"% name ]["date"] = globals()["%s"% name ].band_names.apply(lambda s: s[-8:])
        globals()["df%s"% name ].set_index("band_names",inplace=True)
        globals()["df%s"% name ]=globals()["%s"% name ].T
        globals()["df%s"% name ]["labcroirr"]= labcroirr
    else:
       sql=sqlite3.connect(path)
       df=pd.read_sql_query("SELECT * FROM output", sql)
       globals()["df%s"%name]=df.groupby("originfid").mean()
       labcroirr=globals()["df%s"%name]["labcroirr"]
       globals()["df%s"%name].drop(columns=list_bd_drop,inplace=True)
       globals()["df%s"% name ]=globals()["df%s"%name].T
       globals()["df%s"% name ]["band_names"]=colnames
       globals()["df%s"% name ]["date"] = globals()["df%s"% name ].band_names.apply(lambda s: s[-8:])
       globals()["df%s"% name ].set_index("band_names",inplace=True)
       globals()["df%s"% name ]=globals()["df%s"% name ].T
       globals()["df%s"% name ]["labcroirr"]= labcroirr
    return  globals()["df%s"% name ]

def indexdate(df,intervalle_inf,out):
    globals()["%s"% out ] = df.rename(index=lambda x: x[intervalle_inf:])
    globals()["%s"%out].sort_index(inplace=True)
    globals()["%s"%out].index=pd.to_datetime(globals()["%s"%out].index,format="%Y%m%d")

if __name__ == '__main__':
    d={}
    d["SAVE"]="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/"
#    df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/TEST_ANALYSE_SIGNAL/ZONE_TEST_MAIZE.csv")
#    dfpar=df.groupby("originfid").mean()
    Label=[1,2,11,22,33,44]
    label=[1,11]
    polarisation=["des_vv","des_vh","des_userfeature","asc_vv","asc_vh","asc_userfeature"]
    indice=["NDVI","NDWI"]
    Indice=["ndvi","ndwi"]
    features=indice[0:2]+polarisation
#    drop_band=['X', 'Y', 'region', 'labcroirr',"alt_band_0"]
    list_bd_drop=["region","labcroirr","ogc_fid","alt_band_0"]
    list_drop=["labcroirr","ogc_fid"]
    list_drop_bv=["labcroirr","ogc_fid","alt_band_0"]
    years= "2018" #or 2018
    BV="TARN" # ADOUR
    
# =============================================================================
#     Ceate plot
# =============================================================================
    list_drop=["labcroirr","OGC_FID","alt_band_0"]
    list_drop_2018=["labcroirr","ogc_fid"]
    cropslab1=[]
    for z in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/"):
        if years in z:
            if BV in z and "NORD" not in z and "SUD" not in z :
                tuile=z[:5]
                if years == "2017":
                    print( z)
                    col_sqlite("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/%s"%z ,tuile+years,list_drop_bv,"/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt")     
                else:
                    print (z)
                    col_sqlite("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/%s" %z, tuile+years ,list_drop_2018,"/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_TYN2018.txt")

                for p in polarisation:
                    print (p)
                    for i in Label:
                        globals()['cropslab%s' % i] = pd.DataFrame( globals()["df%s"% tuile+years ][ globals()["df%s"% tuile+years ].labcroirr==i]).T
                        globals()['%s%s' % (p,i)]=[]
                        for index,row in globals()['cropslab%s' % i].iterrows():
                            if p in index:
                                globals()['%s%s' % (p,i)].append (row)
                                globals()['df%s%s' %(p,i)]=pd.DataFrame(globals()['%s%s' % (p,i)])
                                globals()["df%s%s"% (p,i)].replace(to_replace =0 , value= pd.NaT,inplace=True) # Suppression des 0 
                                globals()["dbdf%s%s" %(p ,i)]=10*np.log10(globals()['df%s%s' % (p,i)]) # passage du coefficant retodiffusion en DB
    
    
                for ind in features:
                    print (ind)
                    for i in Label:
                        globals()['cropslab%s' % i] = pd.DataFrame(globals()["df%s"% tuile+years ][globals()["df%s"% tuile+years ].labcroirr==i]).T
                        globals()['%s%s' %(ind,i)]=[]
                        for index,row in globals()['cropslab%s' % i].iterrows():
                            if ind in index:
                                globals()['%s%s' % (ind,i)].append (row)
                                globals()['df%s%s' %(ind,i)]=pd.DataFrame(globals()['%s%s' %(ind,i)])
                                
                for f in features: # passage de date au format datetime pandas 
                    for i in Label:
                        if f in polarisation:
                            globals()["%s%s"% (f,i) ] = globals()["dbdf%s%s"% (f,i)].rename(index=lambda x: x[-8:])
                            globals()["%s%s"%(f,i)].sort_index(inplace=True)
                            globals()["%s%s"%(f,i)].index=pd.to_datetime(globals()["%s%s"%(f,i)].index,format="%Y%m%d")
                        else:
                            globals()["%s%s"% (f,i) ] = globals()["df%s%s"% (f,i)].rename(index=lambda x: x[-8:])
                            globals()["%s%s"%(f,i)].sort_index(inplace=True)
                            globals()["%s%s"%(f,i)].index=pd.to_datetime(globals()["%s%s"%(f,i)].index,format="%Y%m%d")
                            
    
                confianceinf=[] # list pour introduire les intervalle de confiance 
                confiancesup=[]
        
                for i in features:
                    for l in Label:
                        print ("%s%s"%(i,l))
    #                    a="%s%s"%(i,l)
                        globals()["_%s%s"% (i,l)],globals()["b_sup%s%s"% (i,l)]=stats.t.interval(0.95,globals()["%s%s"%(i,l)].shape[1]-1,loc=globals()["%s%s"%(i,l)].T.mean(),scale=stats.sem(globals()["%s%s"%(i,l)].T)) # Calcule de l'intervalle de confiance 
                        confianceinf.append(globals()["_%s%s"% (i,l)])
                        confiancesup.append(globals()["b_sup%s%s"% (i,l)])

#        
                for m in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/SAFRAN/"): # regle meteo pour 2018 sur les auters zones 
#                    print (m)
                    if years=="2017" :
                        if m[4:9] == tuile and m[-8:-4]== "2017" :
                            df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/SAFRAN/%s" % m)
                            df=df[["DATE","PRELIQ_Q"]]
                            globals()["dfSAFR%s"% (tuile)]=df.set_index("DATE")
                            globals()["dfSAFR%s"% (tuile)].index=pd.to_datetime(globals()["dfSAFR%s"% (tuile)].index,format="%Y%m%d")
                    else:
                        if m[4:9] == tuile and m[-8:-4]=="2018":
                            df=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/SAFRAN/%s" % m)
                            df=df[["DATE","PRELIQ_Q"]][0:365]
                            globals()["dfSAFR%s"% (tuile)]=df.set_index("DATE")
                            globals()["dfSAFR%s"% (tuile)].index=pd.to_datetime(globals()["dfSAFR%s"% (tuile)].index,format="%Y%m%d")
                            

#        pltmutli(NDVI1.index,VV1.index,NDVI1,NDVI11,NDWI1,NDWI11,VV1,VV11,VH1,VH11,VH_VV1,VH_VV11,globals()["dfSAFR%s"% (tile)].index,globals()["dfSAFR%s"% (tile)].PRELIQ_Q)    
#        plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/comparaison_IRR_NIRR_2017%s.png"% tile)
#        pltmutlimode(NDVI1.index,VVa1.index,VVd1.index,VH_VV1.index,NDVI1,NDVI11,NDWI1,NDWI11,VVa1,VVa11,VVd1,VVd11,VHa1,VHa11,VHd1,VHd11,VH_VV1,VH_VV11,globals()["dfSAFR%s"% (tile)].index,globals()["dfSAFR%s"% (tile)].PRELIQ_Q)
#        plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/comparaison_IRR_NIRR_SARMODE_2017%s.png"% tile)

#        plotconfi(NDVI1.index,des_vv11.index,des_vv1.index,NDVI1,NDVI11,des_vv1,des_vv11,des_vv1,des_vv11,des_vh1,des_vh11,des_vh1,des_vh11,des_userfeature1,des_userfeature11,des_userfeature1,des_userfeature11,globals()["dfSAFR%s"% (tile)].index,globals()["dfSAFR%s"% (tile)].PRELIQ_Q,confiancesup,confianceinf)
#        plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/comparaison_IRR_NIRR_SARMODE_confiance_2017%s.png"% tile)
#        plotconfi(NDVI1.index,des_vv11.index,des_vv1.index,NDVI1,NDVI11,des_vv1,des_vv11,des_vv1,des_vv11,des_vh1,des_vh11,des_vh1,des_vh11,des_userfeature1,des_userfeature11,des_userfeature1,des_userfeature11,dfSAFRTCJ.index,dfSAFRTCJ.PRELIQ_Q,confiancesup,confianceinf)

#        
        
        
        
        
        
# =============================================================================
# Plot illustation overlap des partiques agricoles entre zones sèche et zone humide
# ========================================================================
#     plt.figure(figsize=(10,8)) 
#     sns.set(style="whitegrid")
#     sns.set_context('paper')
#     ax1 = plt.subplot(221)
#     p1=plt.plot(NDVI1.index,NDVI1.T.mean()/1000,color='blue',linestyle="-")
#     plt.fill_between(NDVI1.index, confiancesup[0]/1000, confianceinf[0]/1000, facecolor='blue', alpha=0.1)
# #    p1=plt.plot(NDVI1.index,NDVI1.T.loc[175],color='blue',linestyle='--')
# ###        plt.fill_between(NDWI1.index, confiancesup[1], confianceinf[1], facecolor='red', alpha=0.1)
#     p3=plt.plot(NDVI11.index,NDVI11.T.mean()/1000,color='red',linestyle='-')
#     plt.fill_between(NDVI11.index, confiancesup[2]/1000, confianceinf[2]/1000, facecolor='red', alpha=0.1)
# #    p4=plt.plot(NDVI11.index,NDVI11.T.mean(),color='red',linestyle='--')
# ###        plt.fill_between(NDWI1.index, confiancesup[3], confianceinf[3], facecolor='red', alpha=0.1)
# ##        p5=plt.plot(NDWI33.index,NDWI33.T.mean(),color='green')
# ###        plt.fill_between(NDWI1.index, confiancesup[4], confianceinf[4], facecolor='green', alpha=0.1)
# ##        p6=plt.plot(NDWI44.index,NDWI44.T.mean(),color='pink')
# ###        plt.fill_between(NDWI1.index, confiancesup[5], confianceinf[5], facecolor='pink', alpha=0.1)
#     plt.ylabel("NDVI")
#     plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.xticks(size='large',rotation=45,visible=False)
#     plt.yticks(size='large')
#     # plt.legend((p1,p3),("Maize_Irr","Maize_Nirr"))
# #    plt.legend((p1[0],p1[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
# ###        plt.legend((p1[0],p3[0]),("Maize_Irr","Maize_Nirr"))
# ###        plt.setp(ax1.get_xticklabels(),visible=True)
# ###        plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/NDWI_ADOUR_NORD_SUD/NDWI%s%s.png"% (tuile,date))
#     ax1 = plt.subplot(223)
#     p1=plt.plot(NDWI1.index,NDWI1.T.mean(),color='blue',linestyle="-")
#     plt.fill_between(NDWI1.index, confiancesup[6], confianceinf[6], facecolor='blue', alpha=0.1)
# #    p1=plt.plot(NDWI1.index,NDWI1.T.loc[101],color='blue',linestyle='--')
#     p3=plt.plot(NDWI11.index,NDWI11.T.mean(),color='red',linestyle='-')
#     plt.fill_between(NDWI11.index, confiancesup[8], confianceinf[8], facecolor='red', alpha=0.1)
# #    p4=plt.plot(NDWI11.index,NDWI11.T.loc[193],color='red',linestyle='--')
#     plt.ylabel("NDWI")
#     plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.xticks(size='large',rotation=45)
#     plt.yticks(size='large')
# #    plt.legend((p1[0],p1[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#     ax3=plt.subplot(222)
#     # p1=plt.plot(des_vv1.index,des_vv1.T.mean(),color='blue',linestyle="-")
#     # plt.fill_between(des_vv1.index, confiancesup[12], confianceinf[12], facecolor='blue', alpha=0.1)
#     p1=plt.plot(asc_vv1.index,asc_vv1.T.mean(),color='blue',linestyle="-")
#     plt.fill_between(asc_vv1.index, confiancesup[30], confianceinf[30], facecolor='blue', alpha=0.1)
# #    p1=plt.plot(des_vv1.index,des_vv1.T.loc[101],color='blue',linestyle='--')
#     # p3=plt.plot(des_vv11.index,des_vv11.T.mean(),color='red',linestyle='-')
#     # plt.fill_between(des_vv11.index, confiancesup[14], confianceinf[14], facecolor='red', alpha=0.1)
#     p3=plt.plot(asc_vv11.index,asc_vv11.T.mean(),color='red',linestyle='-')
#     plt.fill_between(asc_vv11.index, confiancesup[32], confianceinf[32], facecolor='red', alpha=0.1)
# #    p4=plt.plot(des_vv11.index,des_vv11.T.loc[193],color='red',linestyle='--')
#     plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.ylabel("VV")
#     plt.xticks(size='large',rotation=45)
#     plt.yticks(size='large')
#     ax32=plt.twinx()
#     ax32.yaxis.grid(False) 
#     ax32.bar(dfSAFRADOUR.index,dfSAFRADOUR.PRELIQ_Q,linewidth=0.1,color='blue')
#     # ax32.ylabel("pluviomètrie en mm")
# #    plt.legend((p1[0],p1[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#     ax4=plt.subplot(223)
#     # p1=plt.plot(des_vh1.index,des_vh1.T.mean(),color='blue',linestyle="-")
#     # plt.fill_between(des_vh1.index, confiancesup[18], confianceinf[18], facecolor='blue', alpha=0.1)
# #    p1=plt.plot(des_vh1.index,des_vh1.T.loc[101],color='blue',linestyle='--')
#     # p3=plt.plot(des_vh11.index,des_vh11.T.mean(),color='red',linestyle='-')
#     # plt.fill_between(des_vh11.index, confiancesup[20], confianceinf[20], facecolor='red', alpha=0.1)
# #    p4=plt.plot(des_vh11.index,des_vh11.T.loc[193],color='red',linestyle='--')
#     p1=plt.plot(asc_vh1.index,asc_vh1.T.mean(),color='blue',linestyle="-")
#     plt.fill_between(asc_vh1.index, confiancesup[36], confianceinf[36], facecolor='blue', alpha=0.1)
#     p3=plt.plot(asc_vh11.index,asc_vh11.T.mean(),color='red',linestyle='-')
#     plt.fill_between(asc_vh11.index, confiancesup[38], confianceinf[38], facecolor='red', alpha=0.1)
#     plt.ylabel("VH")
#     plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
#     plt.xticks(size='large',rotation=45)
#     plt.yticks(size='large')
#     plt.legend((p1[0],p3[0]),("Maize_Irr","Maize_Nirr"))
#    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/"+years+"_MAIZE_"+BV+"_OPTIC_SAR.jpg")
    
    
#     plt.figure(figsize=(10,7)) 
#     p1=plt.plot(des_userfeature1.index,des_userfeature1.T.mean(),color='blue')
#     plt.fill_between(des_userfeature1.index, confiancesup[24], confianceinf[24], facecolor='blue', alpha=0.1)
# #        plt.plot(des_userfeature2.index,des_userfeature2.T.mean(),color='red',linestyle='--')
# #        plt.fill_between(des_userfeature1.index, confiancesup[7], confianceinf[7], facecolor='red', alpha=0.2)
#     p3=plt.plot(des_userfeature11.index,des_userfeature11.T.mean(),color='red',linestyle='-')
#     plt.fill_between(des_userfeature1.index, confiancesup[26], confianceinf[26], facecolor='red', alpha=0.1)
#     plt.xticks(size='large')
#     plt.yticks(size='large')
#     plt.ylabel("VH/VV en db")
#     plt.legend((p1[0],p3[0]),("Maize_Irr","Maize_Nirr"))
#    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/"+years+"_MAIZE_"+BV+"_des_VV_VH.jpg")
    
#    plt.figure(figsize=(12,5)) 
#    plt.bar(dfSAFRTARN_.index,dfSAFRTARN_.PRELIQ_Q,linewidth=0.1,color='blue')
#    plt.ylabel("pluviomètrie en mm")
#    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/pluvio"+BV+"_"+years+".jpg")
        # p1=plt.plot(NDVI1.index,NDVI1.T.mean(),color='blue',linestyle="-")
        # plt.plot(des_userfeature22.index,des_userfeature22.T.mean(),color='red')
        # # plt.fill_between(des_userfeature1.index, confiancesup[9], confianceinf[9], facecolor='red', alpha=0.2)
        # plt.plot(des_userfeature33.index,des_userfeature33.T.mean(),color='green')
        # # plt.fill_between(des_userfeature1.index, confiancesup[10], confianceinf[10], facecolor='green', alpha=0.2)
        # plt.plot(des_userfeature44.index,des_userfeature44.T.mean(),color='pink')
        # # plt.fill_between(des_userfeature1.index, confiancesup[11], confianceinf[11], facecolor='pink', alpha=0.2)
        # plt.ylabel("VH/VV")
        # plt.setp(ax2.get_xticklabels(),visible=False)
        # ax3 = plt.subplot(313)
        # # plt.bar(globals()["dfSAFR%s"% (tile)].index,globals()["dfSAFR%s"% (tile)].PRELIQ_Q,color='blue',width=1)
        # plt.ylabel("rainfall (mm)")
        # plt.setp(ax3.get_xticklabels(),rotation=90)
#        plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/PLOT_NDVI_VV_VH_BV/NDVI%s%s.png"% (tile,date))
        
    
# =============================================================================
#     SAR only with rainfall
# =============================================================================
    plt.figure(figsize=(15,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(411)
    # p1=plt.plot(des_vv1.index,des_vv1.T.mean(),color='blue',linestyle="-")
    # plt.fill_between(des_vv1.index, confiancesup[12], confianceinf[12], facecolor='blue', alpha=0.1)
    p1=plt.plot(asc_vv1.index,asc_vv1.T.mean(),color='blue',linestyle="--")
    plt.fill_between(asc_vv1.index, confiancesup[30], confianceinf[30], facecolor='blue', alpha=0.1)
    # p3=plt.plot(des_vv11.index,des_vv11.T.mean(),color='red',linestyle='-')
    # plt.fill_between(des_vv11.index, confiancesup[14], confianceinf[14], facecolor='red', alpha=0.1)
    p3=plt.plot(asc_vv11.index,asc_vv11.T.mean(),color='red',linestyle='--')
    plt.fill_between(asc_vv11.index, confiancesup[32], confianceinf[32], facecolor='red', alpha=0.1)
    plt.ylabel("VV")
    plt.xticks(size='large')
    plt.yticks(size='large')
    ax2=plt.subplot(412)
    # p1=plt.plot(des_vh1.index,des_vh1.T.mean(),color='blue',linestyle="-")
    # plt.fill_between(des_vh1.index, confiancesup[18], confianceinf[18], facecolor='blue', alpha=0.1)
    # p3=plt.plot(des_vh11.index,des_vh11.T.mean(),color='red',linestyle='-')
    # plt.fill_between(des_vh11.index, confiancesup[20], confianceinf[20], facecolor='red', alpha=0.1)
    p1=plt.plot(asc_vh1.index,asc_vh1.T.mean(),color='blue',linestyle="--")
    plt.fill_between(asc_vh1.index, confiancesup[36], confianceinf[36], facecolor='blue', alpha=0.1)
    p3=plt.plot(asc_vh11.index,asc_vh11.T.mean(),color='red',linestyle='--')
    plt.fill_between(asc_vh11.index, confiancesup[38], confianceinf[38], facecolor='red', alpha=0.1)
    plt.ylabel("VH")
    plt.xticks(size='large')
    plt.yticks(size='large')
    plt.legend((p1[0],p3[0]),("Maize_Irr","Maize_Nirr"))
    ax3=plt.subplot(413)
    # p1=plt.plot(des_userfeature1.index,des_userfeature1.T.mean(),color='blue',linestyle="-")
    # plt.fill_between(des_userfeature1.index, confiancesup[24], confianceinf[24], facecolor='blue', alpha=0.1)
    p3=plt.plot(asc_userfeature1.index,asc_userfeature1.T.mean(),color='blue',linestyle='--')
    plt.fill_between(asc_userfeature1.index, confiancesup[42], confianceinf[42], facecolor='red', alpha=0.1)
    # p3=plt.plot(des_userfeature11.index,des_userfeature11.T.mean(),color='red',linestyle='-')
    # plt.fill_between(des_userfeature1.index, confiancesup[26], confianceinf[26], facecolor='red', alpha=0.1)
    p3=plt.plot(asc_userfeature11.index,asc_userfeature11.T.mean(),color='red',linestyle='--')
    plt.fill_between(asc_userfeature1.index, confiancesup[44], confianceinf[44], facecolor='red', alpha=0.1)
    plt.ylabel("VH/VV")
    plt.xticks(size='large')
    plt.yticks(size='large')
    ax4=plt.subplot(414)
    plt.bar(dfSAFRADOUR.index,dfSAFRADOUR.PRELIQ_Q,linewidth=0.1,color='blue')
    plt.ylabel("pluviomètrie en mm")
#    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/SAR_mode_acq"+BV+"_"+years+".png")
    
# =============================================================================
# # paper plot sar opic safran 
# =============================================================================
    plt.figure(figsize=(15,10))
    sns.set(style="darkgrid")
    sns.set_context('paper')
    ax1 = plt.subplot(411)
    p1=plt.plot(NDVI1.iloc[8:-3].index,NDVI1.iloc[8:-3].T.mean(),color='blue',linestyle="-")
    plt.fill_between(NDVI1.iloc[8:-3].index, confiancesup[0][8:-3], confianceinf[0][8:-3], facecolor='blue', alpha=0.2)
    p3=plt.plot(NDVI11.iloc[8:-3].index,NDVI11.iloc[8:-3].T.mean(),color='red',linestyle='-')
    plt.fill_between(NDVI11.iloc[8:-3].index, confiancesup[2][8:-3], confianceinf[2][8:-3], facecolor='red', alpha=0.2)
#    plt.plot([NDVI11.index[12],NDWI11.index[12]], [min(NDVI11.T.mean())-50,max(NDVI11.T.mean())+50], lw=2,color="black")
#    plt.plot([NDVI11.index[-7],NDWI11.index[-7]], [min(NDVI11.T.mean())-50,max(NDVI11.T.mean())+50], lw=2,color="black")
    plt.ylabel("NDVI")
    plt.xticks(size='large')
    plt.yticks(size='large')
    ax2=plt.subplot(412)
    p1=plt.plot(NDWI1.iloc[8:-3].index,NDWI1.iloc[8:-3].T.mean(),color='blue',linestyle="-")
    plt.fill_between(NDWI1.iloc[8:-3].index, confiancesup[6][8:-3], confianceinf[6][8:-3], facecolor='blue', alpha=0.2)
    p3=plt.plot(NDWI11.iloc[8:-3].index,NDWI11.iloc[8:-3].T.mean(),color='red',linestyle='-')
    plt.fill_between(NDWI11.iloc[8:-3].index, confiancesup[8][8:-3], confianceinf[8][8:-3], facecolor='red', alpha=0.2)
#    plt.plot([NDWI11.index[12],NDWI11.index[12]], [min(NDWI1.T.mean())-50,max(NDWI1.T.mean())+50], lw=2,color="black")
#    plt.plot([NDWI11.index[-7],NDWI11.index[-7]], [min(NDWI1.T.mean())-50,max(NDWI1.T.mean())+50], lw=2,color="black")
    plt.ylabel("NDWI")
    plt.xticks(size='large')
    plt.yticks(size='large')
    plt.legend((p1[0],p3[0]),("Maize_Irr","Maize_Nirr"))
    ax3=plt.subplot(413)
    p3=plt.plot(asc_userfeature1.iloc[9:-3].index,asc_userfeature1.iloc[9:-3].T.mean(),color='blue',linestyle='-')
    plt.fill_between(asc_userfeature1.iloc[9:-3].index, confiancesup[42][9:-3], confianceinf[42][9:-3], facecolor='blue', alpha=0.2)
    p3=plt.plot(asc_userfeature11.iloc[9:-3].index,asc_userfeature11.iloc[9:-3].T.mean(),color='red',linestyle='-')
    plt.fill_between(asc_userfeature1.iloc[9:-3].index, confiancesup[44][9:-3], confianceinf[44][9:-3], facecolor='red', alpha=0.2)
    plt.ylabel("VH/VV")
#    plt.plot([asc_userfeature1.index[12],asc_userfeature1.index[12]], [min(asc_userfeature1.T.mean())-1,max(asc_userfeature1.T.mean())+1], lw=2,color="black")
#    plt.plot([asc_userfeature1.index[-7],asc_userfeature1.index[-7]], [min(asc_userfeature1.T.mean())-1,max(asc_userfeature1.T.mean())+1], lw=2,color="black")
    plt.xticks(size='large')
    plt.yticks(size='large')
    ax4=plt.subplot(414)
    plt.bar(dfSAFRADOUR.iloc[90:-31].index,dfSAFRADOUR.iloc[90:-31].PRELIQ_Q,linewidth=0.1,color='blue')
    plt.ylabel("pluviomètrie en mm")
    plt.xticks(size='large')
    plt.yticks(size='large')
    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/optic_sar_pluvio"+BV+"_"+years+".png")
    
    
# =============================================================================
#     pour 2017
# =============================================================================
#    plt.figure(figsize=(18,10))
#    sns.set(style="darkgrid")
#    sns.set_context('paper')
#    ax1 = plt.subplot(221)
#    p1=plt.plot(NDVI1.index,NDVI1.T.loc[12],color='blue',linestyle="-")
##        plt.fill_between(NDVI1.index, confiancesup[0], confianceinf[0], facecolor='blue', alpha=0.2)
#    p2=plt.plot(NDVI1.index,NDVI1.T.loc[309],color='blue',linestyle='--')
###        plt.fill_between(NDWI1.index, confiancesup[1], confianceinf[1], facecolor='red', alpha=0.2)
#    p3=plt.plot(NDVI11.index,NDVI11.T.loc[113],color='red',linestyle='-')
##        plt.fill_between(NDWI1.index, confiancesup[2], confianceinf[2], facecolor='red', alpha=0.2)
#    p4=plt.plot(NDVI11.index,NDVI11.T.loc[277],color='red',linestyle='--')
#    
###        plt.fill_between(NDWI1.index, confiancesup[3], confianceinf[3], facecolor='red', alpha=0.2)
##        p5=plt.plot(NDWI33.index,NDWI33.T.mean(),color='green')
###        plt.fill_between(NDWI1.index, confiancesup[4], confianceinf[4], facecolor='green', alpha=0.2)
##        p6=plt.plot(NDWI44.index,NDWI44.T.mean(),color='pink')
###        plt.fill_between(NDWI1.index, confiancesup[5], confianceinf[5], facecolor='pink', alpha=0.2)
#    plt.ylabel("NDVI")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
##        plt.legend((p1[0],p3[0]),("Maize_Irr","Maize_Nirr"))
##        plt.setp(ax1.get_xticklabels(),visible=True)
##        plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/NDWI_ADOUR_NORD_SUD/NDWI%s%s.png"% (tuile,date))
#    ax2 = plt.subplot(222)
#    p1=plt.plot(NDWI1.index,NDWI1.T.loc[12],color='blue',linestyle="-")
#    p2=plt.plot(NDWI1.index,NDWI1.T.loc[309],color='blue',linestyle='--')
#    p3=plt.plot(NDWI11.index,NDWI11.T.loc[113],color='red',linestyle='-')
#    p4=plt.plot(NDWI11.index,NDWI11.T.loc[277],color='red',linestyle='--')
#    plt.ylabel("NDWI")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax3=plt.subplot(223)
#    p1=plt.plot(des_vv1.index,des_vv1.T.loc[12],color='blue',linestyle="-")
#    p2=plt.plot(des_vv1.index,des_vv1.T.loc[309],color='blue',linestyle='--')
#    p3=plt.plot(des_vv11.index,des_vv11.T.loc[113],color='red',linestyle='-')
#    p4=plt.plot(des_vv11.index,des_vv11.T.loc[277],color='red',linestyle='--')
#    plt.ylabel("VV ")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax4=plt.subplot(224)
#    p1=plt.plot(des_vh1.index,des_vh1.T.loc[12],color='blue',linestyle="-")
#    p2=plt.plot(des_vh1.index,des_vh1.T.loc[309],color='blue',linestyle='--')
#    p3=plt.plot(des_vh11.index,des_vh11.T.loc[113],color='red',linestyle='-')
#    p4=plt.plot(des_vh11.index,des_vh11.T.loc[277],color='red',linestyle='--')
#    plt.ylabel("VH")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/2017_MAIZE_ZONE_OVERLAP_CLASSE.jpg")
#        

# =============================================================================
#     Zone sud et NORD dyna tempo
# =============================================================================
#    for z in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/"):
#        if '2017' in z and "ADOUR" in z and "NORD_v2"  in z and "SUD" not in z:
#            tuile=z[:5]
#            print (tuile)
#            date="2017_NORD"
#            col_sqlite("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/%s"%z ,tuile+date,list_drop,"/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt")     
#            for p in polarisation:
#                print (p)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame( globals()["df%s"% tuile+date ][ globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' % (p,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if p in index:
#                            globals()['%s%s' % (p,i)].append (row)
#                            globals()['df%s%s' %(p,i)]=pd.DataFrame(globals()['%s%s' % (p,i)])
#                            globals()["df%s%s"% (p,i)].replace(to_replace =0 , value= pd.NaT,inplace=True)
#                            globals()["dbdf%s%s" %(p ,i)]=10*np.log10(globals()['df%s%s' % (p,i)])
#            for ind in features:
#                print (ind)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame(globals()["df%s"% tuile+date ][globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' %(ind,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if ind in index:
#                            globals()['%s%s' % (ind,i)].append (row)
#                            globals()['df%s%s' %(ind,i)]=pd.DataFrame(globals()['%s%s' %(ind,i)])
#            for f in features:
#                for i in Label:
#                    if f in polarisation:
#                        indexdate(globals()["dbdf%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#                    else:
#                        indexdate(globals()["df%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#            confianceinf2017N=[]
#            confiancesup2017N=[]
#            for i in features:
#                for l in Label:
##                    print ("%s%s%s"%(tuile+date,i,l))
##                    a="%s%s"%(i,l)
#                    globals()["_%s%s%s"% (tuile+date,i,l)],globals()["b_sup%s%s%s"% (tuile+date,i,l)]=stats.t.interval(0.95,globals()["%s%s%s"%(tuile+date,i,l)].shape[1]-1,loc=globals()["%s%s%s"%(tuile+date,i,l)].T.mean(),scale=stats.sem(globals()["%s%s%s"%(tuile+date,i,l)].T))
#                    confianceinf2017N.append(globals()["_%s%s%s"% (tuile+date,i,l)])
#                    confiancesup2017N.append(globals()["b_sup%s%s%s"% (tuile+date,i,l)])
##                        
#        elif "2017" in z and "ADOUR" in z and "NORD" not in z and "SUD_v2" in z: 
#            tuile=z[:5]
#            print (tuile)
#            date="2017_SUD"
#            col_sqlite("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/%s"%z ,tuile+date,list_drop,"/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt")     
#            for p in polarisation:
#                print (p)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame( globals()["df%s"% tuile+date ][ globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' % (p,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if p in index:
#                            globals()['%s%s' % (p,i)].append (row)
#                            globals()['df%s%s' %(p,i)]=pd.DataFrame(globals()['%s%s' % (p,i)])
#                            globals()["df%s%s"% (p,i)].replace(to_replace =0 , value= pd.NaT,inplace=True)
#                            globals()["dbdf%s%s" %(p ,i)]=10*np.log10(globals()['df%s%s' % (p,i)])
#            for ind in features:
#                print (ind)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame(globals()["df%s"% tuile+date ][globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' %(ind,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if ind in index:
#                            globals()['%s%s' % (ind,i)].append (row)
#                            globals()['df%s%s' %(ind,i)]=pd.DataFrame(globals()['%s%s' %(ind,i)])
#                            
#            for f in features:
#                for i in Label:
#                    if f in polarisation:
#                        indexdate(globals()["dbdf%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#                    else:
#                        indexdate(globals()["df%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#            confianceinf2017S=[]
#            confiancesup2017S=[]
#            for i in features:
#                for l in Label:
##                    print ("%s%s%s"%(tuile+date,i,l))
##                    a="%s%s"%(i,l)
#                    globals()["_%s%s%s"% (tuile+date,i,l)],globals()["b_sup%s%s%s"% (tuile+date,i,l)]=stats.t.interval(0.95,globals()["%s%s%s"%(tuile+date,i,l)].shape[1]-1,loc=globals()["%s%s%s"%(tuile+date,i,l)].T.mean(),scale=stats.sem(globals()["%s%s%s"%(tuile+date,i,l)].T))
#                    confianceinf2017S.append(globals()["_%s%s%s"% (tuile+date,i,l)])
#                    confiancesup2017S.append(globals()["b_sup%s%s%s"% (tuile+date,i,l)])
#        elif "2018" in z and "ADOUR" in z and "NORD_v2" in z and "SUD" not in z: 
#            tuile=z[:5]
#            print (tuile)
#            date="2018_NORD"
#            col_sqlite("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/%s"%z ,tuile+date,list_drop[0:2],"/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_TYN2018.txt")     
#            for p in polarisation:
#                print (p)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame( globals()["df%s"% tuile+date ][ globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' % (p,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if p in index:
#                            globals()['%s%s' % (p,i)].append (row)
#                            globals()['df%s%s' %(p,i)]=pd.DataFrame(globals()['%s%s' % (p,i)])
#                            globals()["df%s%s"% (p,i)].replace(to_replace =0 , value= pd.NaT,inplace=True)
#                            globals()["dbdf%s%s" %(p ,i)]=10*np.log10(globals()['df%s%s' % (p,i)])
#            for ind in features:
#                print (ind)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame(globals()["df%s"% tuile+date ][globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' %(ind,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if ind in index:
#                            globals()['%s%s' % (ind,i)].append (row)
#                            globals()['df%s%s' %(ind,i)]=pd.DataFrame(globals()['%s%s' %(ind,i)])
#            for f in features:
#                for i in Label:
#                    if f in polarisation:
#                        indexdate(globals()["dbdf%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#                    else:
#                        indexdate(globals()["df%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#            confianceinf2018N=[]
#            confiancesup2018N=[]
#            for i in features:
#                for l in Label:
##                    print ("%s%s%s"%(tuile+date,i,l))
##                    a="%s%s"%(i,l)
#                    globals()["_%s%s%s"% (tuile+date,i,l)],globals()["b_sup%s%s%s"% (tuile+date,i,l)]=stats.t.interval(0.95,globals()["%s%s%s"%(tuile+date,i,l)].shape[1]-1,loc=globals()["%s%s%s"%(tuile+date,i,l)].T.mean(),scale=stats.sem(globals()["%s%s%s"%(tuile+date,i,l)].T))
#                    confianceinf2018N.append(globals()["_%s%s%s"% (tuile+date,i,l)])
#                    confiancesup2018N.append(globals()["b_sup%s%s%s"% (tuile+date,i,l)])
#        elif "2018" in z and "ADOUR" in z and "NORD" not in z and "SUD_v2" in z: 
#            tuile=z[:5]
#            date="2018_SUD"
#            col_sqlite("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/%s" %z, tuile+date ,list_drop[0:2],"/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_TYN2018.txt")
#            for p in polarisation:
#                print (p)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame( globals()["df%s"% tuile+date ][ globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' % (p,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if p in index:
#                            globals()['%s%s' % (p,i)].append (row)
#                            globals()['df%s%s' %(p,i)]=pd.DataFrame(globals()['%s%s' % (p,i)])
#                            globals()["df%s%s"% (p,i)].replace(to_replace =0 , value= pd.NaT,inplace=True)
#                            globals()["dbdf%s%s" %(p ,i)]=10*np.log10(globals()['df%s%s' % (p,i)])
#            for ind in features:
#                print (ind)
#                for i in Label:
#                    globals()['cropslab%s' % i] = pd.DataFrame(globals()["df%s"% tuile+date ][globals()["df%s"% tuile+date ].labcroirr==i]).T
#                    globals()['%s%s' %(ind,i)]=[]
#                    for index,row in globals()['cropslab%s' % i].iterrows():
#                        if ind in index:
#                            globals()['%s%s' % (ind,i)].append (row)
#                            globals()['df%s%s' %(ind,i)]=pd.DataFrame(globals()['%s%s' %(ind,i)])
#                            
#            for f in features:
#                for i in Label:
#                    if f in polarisation:
#                        indexdate(globals()["dbdf%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#                    else:
#                        indexdate(globals()["df%s%s"% (f,i)],-8,'%s%s%s' % (tuile+date,f,i))
#            confianceinf2018S=[]
#            confiancesup2018S=[]
#            for i in features:
#                for l in Label:
##                    print ("%s%s%s"%(tuile+date,i,l))
##                    a="%s%s"%(i,l)
#                    globals()["_%s%s%s"% (tuile+date,i,l)],globals()["b_sup%s%s%s"% (tuile+date,i,l)]=stats.t.interval(0.95,globals()["%s%s%s"%(tuile+date,i,l)].shape[1]-1,loc=globals()["%s%s%s"%(tuile+date,i,l)].T.mean(),scale=stats.sem(globals()["%s%s%s"%(tuile+date,i,l)].T))
#                    confianceinf2018S.append(globals()["_%s%s%s"% (tuile+date,i,l)])
#                    confiancesup2018S.append(globals()["b_sup%s%s%s"% (tuile+date,i,l)])
# =============================================================================
# 2017
# =============================================================================
#    plt.figure(figsize=(18,10))
#    sns.set(style="darkgrid")
#    sns.set_context('paper')
#    ax1 = plt.subplot(221)
#    p1=plt.plot(ADOUR2017_SUDNDVI1.index,ADOUR2017_SUDNDVI1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2017_SUDNDVI1.index, confiancesup2017S[0], confianceinf2017S[0], facecolor='blue', alpha=0.2)
#    p2=plt.plot(ADOUR2017_NORDNDVI1.index,ADOUR2017_NORDNDVI1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2017_NORDNDVI1.index, confiancesup2017N[0], confianceinf2017N[0], facecolor='blue', alpha=0.2)
#    p3=plt.plot(ADOUR2017_SUDNDVI11.index,ADOUR2017_SUDNDVI11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2017_SUDNDVI11.index, confiancesup2017S[2], confianceinf2017S[2], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2017_NORDNDVI11.index,ADOUR2017_NORDNDVI11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2017_NORDNDVI11.index, confiancesup2017N[2], confianceinf2017N[2], facecolor='red', alpha=0.2)
#    plt.ylabel("NDVI")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax2 = plt.subplot(222)
#    p1=plt.plot(ADOUR2017_SUDNDWI1.index,ADOUR2017_SUDNDWI1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2017_SUDNDWI1.index, confiancesup2017S[6], confianceinf2017S[6], facecolor='blue', alpha=0.2)
#    p2=plt.plot(ADOUR2017_NORDNDWI1.index,ADOUR2017_NORDNDWI1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2017_SUDNDWI1.index, confiancesup2017N[6], confianceinf2017N[6], facecolor='red', alpha=0.2)
#    p3=plt.plot(ADOUR2017_SUDNDWI11.index,ADOUR2017_SUDNDWI11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2017_SUDNDWI1.index, confiancesup2017S[8], confianceinf2017S[8], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2017_NORDNDWI11.index,ADOUR2017_NORDNDWI11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2017_SUDNDWI1.index, confiancesup2017N[8], confianceinf2017N[8], facecolor='red', alpha=0.2)
#    plt.ylabel("NDWI")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax3 = plt.subplot(223)
#    p1=plt.plot(ADOUR2017_SUDdes_vv1.index,ADOUR2017_SUDdes_vv1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2017_SUDdes_vv1.index, confiancesup2017S[12], confianceinf2017S[12], facecolor='blue', alpha=0.2)
#    p2=plt.plot(ADOUR2017_NORDdes_vv1.index,ADOUR2017_NORDdes_vv1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2017_SUDdes_vv1.index, confiancesup2017N[12], confianceinf2017N[12], facecolor='red', alpha=0.2)
#    p3=plt.plot(ADOUR2017_SUDdes_vv11.index,ADOUR2017_SUDdes_vv11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2017_SUDdes_vv1.index, confiancesup2017S[14], confianceinf2017S[14], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2017_NORDdes_vv11.index,ADOUR2017_NORDdes_vv11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2017_SUDdes_vv1.index, confiancesup2017N[14], confianceinf2017N[14], facecolor='red', alpha=0.2)
#    plt.ylabel("VV")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax4 = plt.subplot(224)
#    p1=plt.plot(ADOUR2017_SUDdes_vh1.index,ADOUR2017_SUDdes_vh1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2017_SUDdes_vh1.index, confiancesup2017S[18], confianceinf2017S[18], facecolor='blue', alpha=0.2)
#    p2=plt.plot(ADOUR2017_NORDdes_vh1.index,ADOUR2017_NORDdes_vh1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2017_SUDdes_vh1.index, confiancesup2017N[18], confianceinf2017N[18], facecolor='red', alpha=0.2)
#    p3=plt.plot(ADOUR2017_SUDdes_vh11.index,ADOUR2017_SUDdes_vh11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2017_SUDdes_vh1.index, confiancesup2017S[20], confianceinf2017S[20], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2017_NORDdes_vh11.index,ADOUR2017_NORDdes_vh11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2017_SUDdes_vh1.index, confiancesup2017N[20], confianceinf2017N[20], facecolor='red', alpha=0.2)
#    plt.ylabel("VH")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/2017_MAIZE_OVERLAP_zone_intervalle.jpg")
    
# =============================================================================
#2018
## =============================================================================
#    plt.figure(figsize=(18,10))
#    sns.set(style="darkgrid")
#    sns.set_context('paper')
#    ax1 = plt.subplot(221)
#    p1=plt.plot(ADOUR2018_SUDNDVI1.index,ADOUR2018_SUDNDVI1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2018_SUDNDVI1.index, confiancesup2018S[0], confianceinf2018S[0], facecolor='blue', alpha=0.2)
#    p2=plt.plot(ADOUR2018_NORDNDVI1.index,ADOUR2018_NORDNDVI1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2018_NORDNDVI1.index, confiancesup2018N[0], confianceinf2018N[0], facecolor='blue', alpha=0.2)
#    p3=plt.plot(ADOUR2018_SUDNDVI11.index,ADOUR2018_SUDNDVI11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2018_SUDNDVI11.index, confiancesup2018S[2], confianceinf2018S[2], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2018_NORDNDVI11.index,ADOUR2018_NORDNDVI11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2018_NORDNDVI11.index, confiancesup2018N[2], confianceinf2018S[2], facecolor='red', alpha=0.2)
#    plt.ylabel("NDVI")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax2 = plt.subplot(222)
#    p1=plt.plot(ADOUR2018_SUDNDWI1.index,ADOUR2018_SUDNDWI1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2018_SUDNDWI1.index, confiancesup2018S[6], confianceinf2018S[6], facecolor='red', alpha=0.2)
#    p2=plt.plot(ADOUR2018_NORDNDWI1.index,ADOUR2018_NORDNDWI1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2018_SUDNDWI1.index, confiancesup2018N[6], confianceinf2018N[6], facecolor='red', alpha=0.2)
#    p3=plt.plot(ADOUR2018_SUDNDWI11.index,ADOUR2018_SUDNDWI11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2018_SUDNDWI1.index, confiancesup2018S[8], confianceinf2018S[8], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2018_NORDNDWI11.index,ADOUR2018_NORDNDWI11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2018_SUDNDWI1.index, confiancesup2018N[8], confianceinf2018N[8], facecolor='red', alpha=0.2)
#    plt.ylabel("NDWI")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax3 = plt.subplot(223)
#    p1=plt.plot(ADOUR2018_SUDdes_vv1.index,ADOUR2018_SUDdes_vv1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018S[12], confianceinf2018S[12], facecolor='red', alpha=0.2)
#    p2=plt.plot(ADOUR2018_NORDdes_vv1.index,ADOUR2018_NORDdes_vv1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018N[12], confianceinf2018N[12], facecolor='red', alpha=0.2)
#    p3=plt.plot(ADOUR2018_SUDdes_vv11.index,ADOUR2018_SUDdes_vv11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018S[14], confianceinf2018S[14], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2018_NORDdes_vv11.index,ADOUR2018_NORDdes_vv11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018N[14], confianceinf2018N[14], facecolor='red', alpha=0.2)
#    plt.ylabel("VV")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    ax4 = plt.subplot(224)
#    p1=plt.plot(ADOUR2018_SUDdes_vh1.index,ADOUR2018_SUDdes_vh1.T.mean(),color='blue',linestyle="-")
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018S[18], confianceinf2018S[18], facecolor='red', alpha=0.2)
#    p2=plt.plot(ADOUR2018_NORDdes_vh1.index,ADOUR2018_NORDdes_vh1.T.mean(),color='blue',linestyle='--')
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018N[18], confianceinf2018N[18], facecolor='red', alpha=0.2)
#    p3=plt.plot(ADOUR2018_SUDdes_vh11.index,ADOUR2018_SUDdes_vh11.T.mean(),color='red',linestyle='-')
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018S[20], confianceinf2018S[20], facecolor='red', alpha=0.2)
#    p4=plt.plot(ADOUR2018_NORDdes_vh11.index,ADOUR2018_NORDdes_vh11.T.mean(),color='red',linestyle='--')
#    plt.fill_between(ADOUR2018_SUDdes_vv1.index, confiancesup2018N[20], confianceinf2018N[20], facecolor='red', alpha=0.2)
#    plt.ylabel("VH")
#    plt.xticks(size='large')
#    plt.yticks(size='large')
#    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Maize_Irr_rain","Maize_Irr_dry","Maize_Nirr_rain","Maize_Nirr_dry"))
#    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/2018_MAIZE_OVERLAP_zone_intervalle.jpg")
#### =============================================================================
####        Calcule de la différence entre IRR et NIRR sur VV
#### =============================================================================

##    
#    dfnames=pd.read_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/list_features_SAR.txt",sep=',', header=None) 
#    df1=dfnames.T
#    df1.columns=["band_name"]
#    colnames=list(df1.band_name.apply(lambda s: s[2:-1]))
#    BV=['TARN',"ADOUR"]
#    for bv in BV:
#        sql=sqlite3.connect("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/ANALYSE_SIGNAL_SAR/DATA_SQLITE/%s_Samples_2017.sqlite"%bv)
#        df=pd.read_sql_query("SELECT * FROM output", sql)
#        globals()["df%s"%bv]=df.groupby("originfid").mean()
#        labcrobvrr=globals()["df%s"%bv]["labcroirr"]
#        globals()["df%s"%bv].drop(columns=list_drop_bv,inplace=True)
#        globals()["df%s"% bv ]=globals()["df%s"%bv].T
#        globals()["df%s"% bv ]["band_names"]=colnames
#        globals()["df%s"% bv ]["date"] = globals()["df%s"% bv ].band_names.apply(lambda s: s[-8:])
#        globals()["df%s"% bv ].set_index("band_names",inplace=True)
#        globals()["df%s"% bv ]=globals()["df%s"% bv ].T
#        globals()["df%s"% bv ]["labcroirr"]= labcroirr
#        
#        for p in polarisation:
#            SAR_process_db(label,globals()["df%s"% bv],p)
#            for i in label :
#                indexdate(globals()["dbdf%s%s"% (p,i)],-8,'%s%s' % (p,i))
#    
#        for ind in indice:
#                print (ind)
#                Optique_Process(label,globals()["df%s"% bv],ind)
#                for i in label:
#                    indexdate(globals()["df%s%s"% (ind,i)],-8,ind.upper()+str(i))
#                    
#        confianceinf=[]
#        confiancesup=[]
#        for i in features:
#            for l in label:
#                print ("%s%s"%(i,l))
#                a="%s%s"%(i,l)
#                globals()["_%s%s"% (i,l)],globals()["b_sup%s%s"% (i,l)]=stats.t.interval(0.95,globals()["%s%s"%(i,l)].shape[1]-1,loc=globals()["%s%s"%(i,l)].T.mean(),scale=stats.sem(globals()["%s%s"%(i,l)].T))
#                confianceinf.append(globals()["_%s%s"% (i,l)])
#                confiancesup.append(globals()["b_sup%s%s"% (i,l)])
#
#        print (bv)    
#        plotconfi(NDVI1.index,des_vv11.index,des_vv1.index,NDVI1,NDVI11,des_vv1,des_vv11,des_vv1,des_vv11,des_vh1,des_vh11,des_vh1,des_vh11,des_userfeature1,des_userfeature11,des_userfeature1,des_userfeature11,globals()["dfSAFR%s"% (tile)].index,globals()["dfSAFR%s"% (tile)].PRELIQ_Q,confiancesup,confianceinf)
#
    # plt.figure(figsize=(13,10))
    # sns.set(style="darkgrid")
    # sns.set_context('paper')
    # ax1 = plt.subplot(411)
    # p1=plt.plot(des_userfeature1.index,des_userfeature1.T.mean(),color='blue')
    # # plt.fill_between(des_userfeature1.index, sup[12], inf[12], facecolor='blue', alpha=0.2)
    # plt.plot(NDVI11.index,NDVI11.T.mean(),color='blue',linestyle='--')
    # plt.plot(NDVI2.index,NDVI2.T.mean(),color='red',linestyle='--')
    # plt.plot(NDVI22.index,NDVI22.T.mean(),color='red')
    # plt.plot(NDVI33.index,NDVI33.T.mean(),color='green')
    # plt.plot(NDVI44.index,NDVI44.T.mean(),color='pink')
    

# =============================================================================
# Plot paper review illustre la phénologie 
# =============================================================================
    plt.figure(figsize=(10,8)) 
    sns.set(style="whitegrid")
    sns.set_context('paper')
    ax1 = plt.subplot(221)
    p1=plt.plot(NDVI1.index,NDVI1.T.mean()/1000,color='blue',linestyle="-")
    plt.fill_between(NDVI1.index, confiancesup[0]/1000, confianceinf[0]/1000, facecolor='blue', alpha=0.1)
    p3=plt.plot(NDVI11.index,NDVI11.T.mean()/1000,color='red',linestyle='-')
    plt.fill_between(NDVI11.index, confiancesup[2]/1000, confianceinf[2]/1000, facecolor='red', alpha=0.1)
    p2=plt.plot(NDVI2.index,NDVI2.T.mean()/1000,color='green',linestyle="--")
    plt.fill_between(NDVI1.index, confiancesup[1]/1000, confianceinf[1]/1000, facecolor='green', alpha=0.1)
    p4=plt.plot(NDVI22.index,NDVI22.T.mean()/1000,color='black',linestyle='--')
    plt.fill_between(NDVI11.index, confiancesup[3]/1000, confianceinf[3]/1000, facecolor='black', alpha=0.1)
    plt.ylabel("NDVI")
    plt.legend((p1[0],p2[0],p3[0],p4[0]),("Irrigated Maize","Irrigated Soybean","Rainfed Maize","Rainfed Soybean"))
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.xticks(size='large',rotation=45,visible=False)
    plt.yticks(fontsize= 12)
    ax1 = plt.subplot(223)
    p1=plt.plot(NDWI1.index,NDWI1.T.mean()/1000,color='blue',linestyle="-")
    plt.fill_between(NDWI1.index, confiancesup[6]/1000, confianceinf[6]/1000, facecolor='blue', alpha=0.1)
    p3=plt.plot(NDWI11.index,NDWI11.T.mean()/1000,color='red',linestyle='-')
    plt.fill_between(NDWI11.index, confiancesup[8]/1000, confianceinf[8]/1000, facecolor='red', alpha=0.1)
    p1=plt.plot(NDWI2.index,NDWI2.T.mean()/1000,color='green',linestyle="--")
    plt.fill_between(NDWI1.index, confiancesup[7]/1000, confianceinf[7]/1000, facecolor='green', alpha=0.1)
    p3=plt.plot(NDWI22.index,NDWI22.T.mean()/1000,color='black',linestyle='--')
    plt.fill_between(NDWI11.index, confiancesup[9]/1000, confianceinf[9]/1000, facecolor='black', alpha=0.1)
    plt.ylabel("NDWI")
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.xticks(size='large',rotation=45)
    plt.yticks(fontsize= 12)
    ax3=plt.subplot(222)
    p1=plt.plot(asc_vv1.index,asc_vv1.T.mean(),color='blue',linestyle="-")
    plt.fill_between(asc_vv1.index, confiancesup[30], confianceinf[30], facecolor='blue', alpha=0.1)
    p3=plt.plot(asc_vv11.index,asc_vv11.T.mean(),color='red',linestyle='-')
    plt.fill_between(asc_vv11.index, confiancesup[32], confianceinf[32], facecolor='red', alpha=0.1)
    p1=plt.plot(asc_vv2.index,asc_vv2.T.mean(),color='green',linestyle="--")
    plt.fill_between(asc_vv2.index, confiancesup[31], confianceinf[31], facecolor='green', alpha=0.1)
    p3=plt.plot(asc_vv22.index,asc_vv22.T.mean(),color='black',linestyle='--')
    plt.fill_between(asc_vv22.index, confiancesup[33], confianceinf[33], facecolor='black', alpha=0.1)
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    plt.ylabel("sigma0 VV")
    plt.xticks(size='large',rotation=45,visible=False)
    plt.yticks(fontsize= 12)
    ax4=plt.subplot(224)
    p1=plt.plot(asc_vh1.index,asc_vh1.T.mean(),color='blue',linestyle="--")
    plt.fill_between(asc_vh1.index, confiancesup[36], confianceinf[36], facecolor='blue', alpha=0.1)
    p3=plt.plot(asc_vh11.index,asc_vh11.T.mean(),color='red',linestyle='--')
    plt.fill_between(asc_vh11.index, confiancesup[38], confianceinf[38], facecolor='red', alpha=0.1)
    p1=plt.plot(asc_vh2.index,asc_vh2.T.mean(),color='green',linestyle="--")
    plt.fill_between(asc_vh2.index, confiancesup[37], confianceinf[37], facecolor='green', alpha=0.1)
    p3=plt.plot(asc_vh22.index,asc_vh22.T.mean(),color='black',linestyle='--')
    plt.fill_between(asc_vh22.index, confiancesup[39], confianceinf[39], facecolor='black', alpha=0.1)
    plt.ylabel("sigma0 VH")
    plt.xticks(size='large',rotation=45)
    plt.yticks(fontsize= 12)
    ax32=plt.twinx()
    ax32.yaxis.grid(False) 
    ax32.bar(dfSAFRADOUR.index,dfSAFRADOUR.PRELIQ_Q,linewidth=0.1,color='blue')
    ax32.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    ax32.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    plt.ylabel("Rainfall in mm")
    plt.ylim(0,100)
    plt.yticks(fontsize= 12)
    plt.xticks(visible=False)

    plt.savefig(d["SAVE"]+"PLOT_TEMPOREL/Paper_PLOT_all_crops_optic_sar_pluvio"+BV+"_"+years+".png",dpi=600,bbox_inches='tight', pad_inches=0.5)
    
   