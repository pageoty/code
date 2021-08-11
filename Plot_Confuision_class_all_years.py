#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:07:10 2019

@author: pageot
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import csv
from scipy import stats
from ResultsUtils import *

def fig_conf_mat_rec(conf_mat_dic, nom_dict, kappa, oacc, p_dic, r_dic, f_dic,
                 out_png, dpi=600, write_conf_score=True,
                 grid_conf=False, conf_score="count",
                 point_of_view="ref", threshold=0):

    import numpy as np
    import matplotlib
    matplotlib.get_backend()
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from decimal import Decimal

    labels_ref = [nom_dict[lab] for lab in list(conf_mat_dic.keys())]
    labels_prod = [nom_dict[lab] for lab in list(conf_mat_dic[list(conf_mat_dic.keys())[0]].keys())]
    nb_labels = len(set(labels_prod + labels_ref))

    # convert conf_mat_dic to a list of lists
    conf_mat_array = np.array([[v for __, v in list(prod_dict.items())] for _, prod_dict in list(conf_mat_dic.items())])

    color_map_coeff = plt.cm.RdYlGn
    diag_cmap = plt.cm.RdYlGn
    not_diag_cmap = plt.cm.Reds

    # normalize by ref samples
    norm_conf = normalize_conf(conf_mat_array, norm=point_of_view)
    rgb_matrix = get_rgb_mat(norm_conf, diag_cmap, not_diag_cmap)

    maxtrix = norm_conf

    vals=[]
    width, height = maxtrix.shape
    if write_conf_score:
        for x_coord in range(width):
            for y_coord in range(height):
                val_percentage = norm_conf[x_coord][y_coord] * 100.0
                if conf_score.lower() == "count":
                    val_to_print = str(int(conf_mat_array[x_coord][y_coord]))
                elif conf_score.lower() == "count_sci" :
                    conf_value = conf_mat_array[x_coord][y_coord]
                    if conf_value > 999.0:
                        val_to_print = "{:.2E}".format(Decimal(conf_value))
                    else:
                        val_to_print = str(int(conf_value))
                elif conf_score.lower() == "percentage":
                    val_to_print = "{:.1f}%".format(val_percentage)
                if val_percentage <= float(threshold):
                    val_to_print = ""

                vals.append(val_percentage)
        return vals

if __name__ == "__main__":
    years='All_Years_ASC' # nom du ficher comptenant l'ensemble des résultats # SEASON_TIME
    bv="ADOUR"
    d={}
    d["disk_PC"]="/run/media/pageot/Transcend/Yann_THESE/RESULTAT_CLASSIFICATION/"
    d["SAVE_disk"]="/run/media/pageot/Transcend/Yann_THESE/RESULTAT_CLASSIFICATION/PLOT/"
    d["SAVE"]="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/PLOT_SYNTH_CLASSIF/" # path où seront save les graphiques finaux 
    for b in [bv]: 
        step = []
        stock_res=[]
        stock_res_std=[]    
        for classif in os.listdir(d["disk_PC"]+'/FILE_TXT_RESULAT/FIxe_seed/SHARK/'+years+'/'): # FIxe_seed/SHARK/'+years+''chemin où sont stocker les matrices de confusion géner avec le script Validation BV
            # if "&" in classif or 'Not' in classif or 'Climate' in classif: 
            if "Optical" in classif: 
                print ("=============")
                print (r" RUN : %s " %classif)
                print ("=============")
                nom=get_nomenclature(d["disk_PC"]+"/nomenclature_paper.txt") # Nomenclature utlisé dans Iota²
                pathNom=d["disk_PC"]+"/nomenclature_paper.txt"
                pathRes=d["disk_PC"]+"/FILE_TXT_RESULAT/FIxe_seed/SHARK/"+years+"/"+classif+"/" # FIxe_seed/SHARK/"+years+"/"+classif+"/" ¬ path où sont stocker les fichiers les matrices de confusion
                all_k = []
                all_oa = []
                all_p = []
                all_r = []
                all_f = []
                all_matrix = []
                for j in os.listdir(pathRes):
                        if ".csv" in j and "regularized" in j and b in j: # récupération uniquement des .csv et de matrices issues des classifiactiosn régularisés
                            print (j)
                            conf_mat_dic = parse_csv(pathRes+j)
                            kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
                            all_matrix.append(conf_mat_dic)
                            all_k.append(kappa)
                            all_oa.append(oacc)
                            all_p.append(p_dic)
                            all_r.append(r_dic)
                            all_f.append(f_dic)
                step.append(classif)
                conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
                conf_mat_dic_std = compute_interest_matrix(all_matrix, f_interest="std")
                size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom)
                p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
                r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
                f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean")
                p_std = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="std")
                r_std = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="std")
                f_std = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="std")
                
                
                globals()["val_conf_%s"%classif]=fig_conf_mat_rec(conf_mat_dic,nom,np.mean(all_k),np.mean(all_oa),p_mean,r_mean,f_mean,"/datalocal/",conf_score="percentage")
                globals()["val_conf_std%s"%classif]=fig_conf_mat_rec(conf_mat_dic_std,nom,np.std(all_k),np.std(all_oa),p_std,r_std,f_std,"/datalocal/",conf_score="percentage")
                stock_res.append(globals()["val_conf_%s"%classif])
                stock_res_std.append( globals()["val_conf_std%s"%classif])
                print(globals()["val_conf_std%s"%classif])
    name_index=labels_prod*len(labels_prod) # génration variable pour crée l'index du tableau
    if bv !="NESTE":
        Multi=list(np.repeat(labels_prod,6)) # création du multi index car eépartion des variables
    else:
        Multi=list(np.repeat(labels_prod,4))
    test=[Multi,name_index]
    tuples=list(zip(*test))
    multi_index=pd.MultiIndex.from_tuples(tuples,names=["user","prod"])
    stock_res=pd.DataFrame(stock_res).T
    stock_res_std=pd.DataFrame(stock_res_std).T
    df_multi=pd.DataFrame(stock_res.values,index=multi_index,columns=step) # tableau mutli_index crée
    df_multi_std=pd.DataFrame(stock_res_std.values,index=multi_index,columns=step)
    if bv !="NESTE":
        fig, ax = plt.subplots(figsize=(12, 10))
        ax1=plt.subplot(221)
        ax1.grid(axis='x')
        # sns.set(style="whitegrid")
        # sns.set_context('paper')
        plt.title("Irrigated Maize",fontsize=14)
        # df_multi.xs("Maize irrigated").iloc[1:-2].plot(kind='bar',ax=ax1)
        a=df_multi.xs("Irrigated Maize").iloc[1:-2].T.sort_values(by="Rainfed Maize")
        if bv =="ADOUR":
            # a.T.plot(kind="bar",color=["salmon","darkorange",'red','deepskyblue',"royalblue",'blue'],ax=ax1,legend=True)
            a.T.plot(kind="bar",color=["dimgrey","darkgrey",'lightgrey','lightgrey',"dimgrey","darkgrey"],ax=ax1,legend=True)
            bars1 = ax1.patches
            hatches = ("//","//","//","//","//","//","//","//","//","..","..","..","..","..","..","..","..","..")
            for bar, hatch in zip(bars1, hatches):
                print(bar)
                bar.set_hatch(hatch)
        else:
            a.T.plot(kind="bar",color=["royalblue","red",'blue','deepskyblue',"darkorange",'salmon'],ax=ax1,legend=True)
        plt.xticks(rotation=0,fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12,ncol=1)
        plt.text(-0.25,60,"a",size="20")
        ax1.xaxis.set_label_text("")
        ax1.yaxis.set_label_text("percentage confusion",fontsize=12)
        plt.ylim(0,65)
        ax2=plt.subplot(222)
        plt.title("Irrigated Soybean",fontsize=14)
        # df_multi.xs("Soybean irrigated").iloc[[0,2,3]].plot(kind='bar',ax=ax2,legend=False,yerr=df_multi_std.xs("Soybean irrigated").iloc[[0,2,3]])
        # df_multi.xs("Soybean irrigated").iloc[[0,2,3]].plot(kind='bar',ax=ax2,legend=False)
        b=df_multi.xs("Irrigated Soybean").iloc[[0,2,3]].T.sort_values(by="Rainfed Soybean")
        if bv == "ADOUR":
            # b.T.plot(kind="bar",color=["salmon","darkorange","royalblue",'red','deepskyblue','blue'],ax=ax2,legend=False) # Problème color other BV
            b.T.plot(kind="bar",color=['dimgrey','darkgrey',"dimgrey",'lightgrey',"lightgrey","darkgrey"],ax=ax2,legend=False)
            bars2 = ax2.patches
            hatches = ("..","..","..","..","..","..","//","//","//","..","..","..","//","//","//","//","//","//")
            for bar, hatch in zip(bars2, hatches):
                bar.set_hatch(hatch)
        else:
            b.T.plot(kind="bar",color=["blue","royalblue","deepskyblue",'darkorange','red','salmon'],ax=ax2,legend=False) # Problème color other BV
        plt.text(-0.25,60,"b",size="20")
        plt.xticks(rotation=0,fontsize=12)
        plt.yticks(fontsize=12)
        ax2.xaxis.set_label_text("")
        plt.ylim(0,65)
        # plt.legend()
        ax3=plt.subplot(223)
        plt.title("Rainfed Maize",fontsize=14)
        # df_multi.xs("Maize no irrigated").iloc[[0,1,3]].plot(kind='bar',ax=ax3,legend=False)
        c=df_multi.xs("Rainfed Maize").iloc[[0,1,3]].T.sort_values(by="Irrigated Maize")
        if bv =="ADOUR":
            # c.T.plot(kind="bar",color=["royalblue",'deepskyblue','blue',"salmon",'red',"darkorange"],ax=ax3,legend=False)
            c.T.plot(kind="bar",color=['dimgrey','lightgrey',"darkgrey",'dimgrey',"lightgrey",'darkgrey'],ax=ax3,legend=False)
            bars3 = ax3.patches
            hatches = ("//","//","//","//","//","//","//","//","//","..","..","..","..","..","..","..","..","..")
            for bar, hatch in zip(bars3, hatches):
                bar.set_hatch(hatch)
        else:
            c.T.plot(kind="bar",color=["royalblue",'salmon','darkorange',"red",'blue',"deepskyblue"],ax=ax3,legend=False)
        plt.text(-0.25,60,"c",size="20")
        plt.xticks(rotation=0,fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend()
        ax3.xaxis.set_label_text("")
        ax3.yaxis.set_label_text("percentage confusion",fontsize=12)
        plt.ylim(0,65)
        ax4=plt.subplot(224)
        plt.title("Rainfed Soybean",fontsize=14)
        # df_multi.xs("Soybean no irrigated").iloc[[0,1,2]].plot(kind='bar',ax=ax4,legend=False)
        d=df_multi.xs("Rainfed Soybean").iloc[[0,1,2]].T.sort_values(by="Irrigated Soybean")
        if bv =="ADOUR":
            # d.T.plot(kind="bar",color=['deepskyblue',"royalblue",'blue',"salmon","darkorange",'red'],ax=ax4,legend=False)
            d.T.plot(kind="bar",color=['lightgrey','dimgrey',"darkgrey",'dimgrey',"darkgrey","lightgrey"],ax=ax4,legend=False)

            bars = ax4.patches
            hatches = ("//","//","//","//","//","//","//","//","//","..","..","..","..","..","..","..","..","..")
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            # ax.legend(loc='center right', bbox_to_anchor=(1, 1), ncol=4)
        else:
            d.T.plot(kind="bar",color=['salmon',"royalblue",'blue',"darkorange","red",'deepskyblue'],ax=ax4,legend=False)
        plt.text(-0.25,60,"d",size="20")
        plt.xticks(rotation=0,fontsize=12)
        plt.yticks(fontsize=12)
        # plt.legend()
        ax4.xaxis.set_label_text("")
        plt.ylim(0,65)
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax1=plt.subplot(221)
        sns.set(style="darkgrid")
        sns.set_context('paper')
        plt.title("Irrigated Maize",fontsize=14)
        df_multi.xs("Maize irrigated").iloc[1:].plot(kind='bar',ax=ax1)
        plt.xticks(rotation=0)
        ax1.xaxis.set_label_text("")
        ax1.yaxis.set_label_text("percentage confusion",fontsize=12)
        plt.ylim(0,65)
        ax2=plt.subplot(222)
        plt.title("Irrigated Soybean ",fontsize=14)
    #    df_multi.xs("Soybean irrigated").iloc[[0,2,3]].plot(kind='bar',ax=ax2,legend=False,yerr=df_multi_std.xs("Soybean irrigated").iloc[[0,2,3]])
        df_multi.xs("Soybean irrigated").iloc[[0,2,3]].plot(kind='bar',ax=ax2,legend=False)
        plt.xticks(rotation=0)
        ax2.xaxis.set_label_text("")
        plt.ylim(0,65)
    plt.savefig("/run/media/pageot/Transcend/Yann_THESE/RESULTAT_CLASSIFICATION/PLOT/Confusion_"+bv+"_"+years+".png")
#    ax4=plt.subplot(224)
#    df_multi.xs("Sunflower ").iloc[0:-1].plot(kind='bar',ax=ax4,legend=True)
#    plt.xticks(rotation=0)
    
    

