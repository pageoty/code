# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:03:39 2020

@author: Yann Pageot

Code compare les indices de confiances entre plusieurs classifications
"""


import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
from scipy.optimize import minimize
from sklearn.metrics import *
from scipy.optimize import linprog
from scipy import optimize
import random
import pickle
from SAMIR_optimi import RMSE
import geopandas as geo
import shapely.geometry as geom
import seaborn as sns
import descartes
from datetime import datetime, date, time, timezone
from scipy import *
from scipy import stats
from pylab import *
from sklearn.linear_model import LinearRegression
# from ambhas.errlib import NS

if __name__ == '__main__':
    
    d={}
    d["path_disk"]="/run/media/pageot/Transcend/Yann_THESE/CLASSIFICATION/DATA_CLASSIFICATION/ANALYSE_CONFIANCE_MAP/"
    
    all_res_2017=pd.DataFrame()
    jobs_2017=[]
    all_res_2018=pd.DataFrame()
    jobs_2018=[]
    all_stock2017=pd.DataFrame()
    all_stock2018=pd.DataFrame()
    for y in ["2017","2018"]:
        if y =="2017":
            for file in os.listdir(d["path_disk"]+"2017_S5/") : 
                if '.shp' in file and 'conf' in file:
                    seed=file[-5:-4]
                    conf_2017=geo.read_file(d["path_disk"]+"2017_S5/"+file)
                    conf_2017["surf"]=conf_2017.area/10000
                    stock=conf_2017[["classmode","confmean","confstd",'surf']]
                    mean=stock.groupby('classmode').mean()
                    meanp=stock.groupby('classmode').apply(lambda x: np.average(x['confmean'], weights = x['surf']))
                    stdp=stock.groupby('classmode').apply(lambda x: np.average(x['confstd'], weights = x['surf']))
                    a=pd.concat([meanp,stdp],axis=1)
                    jobs_2017.append(seed)
                    all_res_2017=all_res_2017.append(mean)
                    all_stock2017=all_stock2017.append(stock)
        else:
            for file in os.listdir(d["path_disk"]+"2018_S5/") : 
                if '.shp' in file and 'conf' in file:
                    seed=file[-5:-4]
                    conf_2018=geo.read_file(d["path_disk"]+"2018_S5/"+file)
                    conf_2018["surf"]=conf_2018.area/10000
                    stock=conf_2018[["classmode","confmean","confstd",'surf']]
                    mean=stock.groupby('classmode').mean()
                    meanp=stock.groupby('classmode').apply(lambda x: np.average(x['confmean'], weights = x['surf']))
                    stdp=stock.groupby('classmode').apply(lambda x: np.average(x['confstd'], weights = x['surf']))
                    a=pd.concat([meanp,stdp],axis=1)
                    jobs_2018.append(seed)
                    all_res_2018=all_res_2018.append(mean)
                    all_stock2018=all_stock2018.append(stock)

    plt.figure(figsize=(10,5))         
    sns.boxplot(all_stock2017.classmode,all_stock2017.confmean)
    plt.figure(figsize=(10,5)) 
    sns.boxplot(all_stock2018.classmode,all_stock2018.confmean)
    jobs_7=list(np.repeat(jobs_2017,7))
    jobs_8=list(np.repeat(jobs_2017,6))
    all_res_2017["seed"]=jobs_7
    all_res_2018["seed"]=jobs_8
    all_res_2017.drop([0.0,33.0],inplace=True)
    all_res_2018.drop([0.0],inplace=True)
    mean_res2018=all_res_2018.groupby(all_res_2018.index).mean()
    mean_res2017=all_res_2017.groupby(all_res_2017.index).mean()
    # mean_res2017.columns=["confmean","confstd"]
    # mean_res2018.columns=["confmean","confstd"] 
    name_crop=["Irrigated Maize","Irrigated Soybean","Rainfed Maize","Rainfed Soybean",'Sunflower']
    #  plot
    # for s in jobs_2017:
    #     plt.figure(figsize=(10,5)) 
    #     sns.set(style="darkgrid")
    #     sns.set_context('paper')
    #     barWidth = 0.3
    #     bars1 = all_res_2017[all_res_2017.seed==s]["confmean"]
    #     bars2 = all_res_2018[all_res_2018.seed==s]["confmean"]
    #     yer1 = all_res_2017[all_res_2017.seed==s]['confstd']
    #     yer2 = all_res_2018[all_res_2018.seed==s]['confstd']
    #     r1 = np.arange(len(bars1))
    #     r2 = [x + barWidth for x in r1]
    #     plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='2017')
    #     plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='2018')
    #     plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],name_crop,rotation=90) # fixer problème du nom de l'axe"
    #     plt.ylabel('confidence percentage')
    #     # plt.ylim(0,1)
    #     # plt.title(str(i[0][:-5]))
    #     plt.legend()
    #     plt.xticks(size='large')
    #     plt.yticks(size='large')
#plot V2
    plt.figure(figsize=(10,5)) 
    sns.set(style="darkgrid")
    sns.set_context('paper')
    barWidth = 0.3
    bars1 = mean_res2017["confmean"]
    bars2 = mean_res2018["confmean"]
    yer1 = mean_res2017['confstd']
    yer2 = mean_res2018['confstd']
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='2017')
    plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='2018')
    plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],name_crop,rotation=90) # fixer problème du nom de l'axe"
    plt.ylabel('confidence percentage')
    # plt.ylim(0,1)
    plt.title("Scenario 5")
    plt.legend()
    plt.xticks(size='large')
    plt.yticks(size='large')
    plt.savefig(d["path_disk"]+"plot_confiance_maps_Scenario5.png",dpi=300,bbox_inches='tight', pad_inches=0.5)