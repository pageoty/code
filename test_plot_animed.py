#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:06:10 2021

@author: pageot
"""
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import pandas_alive

import pandas as pd
import pickle
import matplotlib.animation as animation
import seaborn as sns
import matplotlib

if __name__ == '__main__':

    d={}
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"

    UTS=pickle.load(open(d["PC_disk"]+"/TRAITEMENT/RUNS_SAMIR/RUN_PKGC/GERS/PKGC_Fcover_GSM_irri_auto_v2/2017/Output/maxZr/output_test_maize_irri_12.df","rb"))
    UTS1=UTS[UTS.id==10]
    UTS1.set_index("date",inplace=True)
    Zr=UTS1[["Dr","TAW"]].resample("SM").asfreq()
    Zr.dropna(inplace=True)
    # ETR=pd.read_csv(d["PC_disk"]+"/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/ETR_maize_irri2019.csv",decimal='.',sep=",",parse_dates=[0])
    # ETR.set_index("date",inplace=True)
    # plt.show()
    # Zr.plot_animated(filename='/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/example-line-chart.gif',kind='line')


    fig = plt.figure(figsize=(10,6))
    # plt.xlim(1999, 2016)
    plt.ylim(0, 100)
    plt.xlabel('Year',fontsize=20)
    plt.ylabel("Dr",fontsize=20)
    plt.title('Dr',fontsize=20)

    def animate(i):
        data = Zr.iloc[:int(i+1)] #select data range
        p = sns.lineplot(x=data.index, y=data["Dr"], data=data, color="r")
        sns.lineplot(x=data.index, y=data["TAW"]*0.55, data=data, color="blue")
        p.tick_params(labelsize=14)
        plt.setp(p.lines,linewidth=2)
    

    ani = matplotlib.animation.FuncAnimation(fig, animate, repeat=False)
    plt.show()
    # ani.save('HeroinOverdosesJumpy.mp4', writer=writer)