#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:50:09 2019

@author: pageot
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
from  PLOT_RESULT_CLASSIF import *
from ResultsUtils import *
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def get_interest_coeff(runs_coeff, nb_lab, f_interest="mean"):

    import collections
    nb_run = len(runs_coeff)

    # get all labels
    for run in runs_coeff:
        ref_labels = [label for label, value in list(run.items())]
    ref_labels = sorted(list(set(ref_labels)))
    # init
    coeff_buff = collections.OrderedDict()
    for ref_lab in ref_labels:
        coeff_buff[ref_lab] = []
    # fill-up
    for run in runs_coeff:
        for label, value in list(run.items()):
            coeff_buff[label].append(value)
    # Compute interest coeff
    coeff_out = collections.OrderedDict()
    for label, values in list(coeff_buff.items()):
        if f_interest.lower() == "mean":
            mean = np.mean(values)
            _, b_sup = stats.t.interval(0.95, nb_lab - 1,
                                        loc=np.mean(values),
                                        scale=stats.sem(values))
            if nb_run > 1:
                coeff_out[label] = "{:.3f} +- {:.3f}".format(mean, b_sup - mean)
    return coeff_out

def plt_classif_kappa(df,var1,var2):
    plt.figure(figsize=(10,5))
    y_pos=np.arange(df.shape[0])
    # sns.set(style="darkgrid")
    sns.set(style="whitegrid")
    sns.set_context('paper')
    plt.grid(axis='x')
    plt.bar(y_pos,df["mean_"+var1],yerr=df["std_"+var1],capsize=3,width = 1,label=var1)
    plt.bar(y_pos+df.shape[0]+0.5,df["mean_"+var2],yerr=df["std_"+var2],capsize=3,width = 1,label=var2)
    # plt.ylabel("score",fontsize=14)
    # plt.xlabel("step")
    plt.xticks(y_pos, tuple(df.index),rotation=90,size=9)
    y_pos3=y_pos+df.shape[0]+0.5
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos3[j]-0.25 , y = -0.02, s = list(df.index)[j],size=14,rotation=90,va="top")
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos[j]-0.3, y=list(df["mean_"+var1]+df["std_"+var1])[j]+0.01,s = list(round(df["mean_"+var1],2))[j],size=12)
    for j in np.arange(len(df.index)):
        plt.text(x = y_pos3[j]-0.3, y=list(df["mean_"+var2]+df["std_"+var2])[j]+0.01,s = list(round(df["mean_"+var2],2))[j],size=12)
    plt.legend(fontsize=14)

if __name__ == "__main__":
    years='All_Years_ASC' # nom du ficher comptenant l'ensemble des résultats # SEASON_TIME
    bv="ADOUR"
    d={}
    d["SAVE"]="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/PLOT/PLOT_SYNTH_CLASSIF/" # path où seront save les graphiques finaux 
    d["disk_PC"]="G:/Yann_THESE/RESULTAT_CLASSIFICATION/"
    d["SAVE_disk"]="G:/Yann_THESE/RESULTAT_CLASSIFICATION/PLOT/"
    for b in [bv]: 
        step = []
        jobs=pd.DataFrame()
        KAPPA = []
        OA = []
        KAPPA_s = []
        OA_s = []
        Recall=pd.DataFrame()
        Prec=pd.DataFrame()
        Fscore=pd.DataFrame()
        for classif in os.listdir(d["disk_PC"]+'FILE_TXT_RESULAT/FIxe_seed/SHARK/'+years+'/'): # FIxe_seed/SHARK/'+years+''chemin où sont stocker les matrices de confusion géner avec le script Validation BV
            print (classif)
#            classif="DES_F_3ind_SAFRAN_2017"
            nom=get_nomenclature(d["disk_PC"]+"nomenclature_paper.txt") # Nomenclature utlisé dans Iota²
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
                        if bv == "NESTE":
                            if "2018" in classif:
                                dico_sans_22 = OrderedDict()
                                for k, v in conf_mat_dic.items():
                                    dico_sans_22_tmp = OrderedDict()
                                    if k == 11 or k ==22 :
                                        continue
                                    for class_name, class_count in v.items():
                                        if class_name == 11 or class_name == 22 :
                                            print( class_name)
                                            continue
                                        dico_sans_22_tmp[class_name] = class_count # supprimer les classes Soja non irriguée dans le matrix car pose probleme
                                    dico_sans_22[k]=dico_sans_22_tmp
    #                            print (dico_sans_22)
                                conf_mat_dic = dico_sans_22
                                kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
                                all_matrix.append(conf_mat_dic)
                                all_k.append(kappa) # Apprend les résultats de chaque métrique 
                                all_oa.append(oacc)
                                all_p.append(p_dic)
                                all_r.append(r_dic)
                                all_f.append(f_dic)
                            elif "2017" in classif:
                                dico_sans_22 = OrderedDict()
                                for k, v in conf_mat_dic.items():
                                    print(k)
                                    print( v)
                                    dico_sans_22_tmp = OrderedDict()
                                    if k == 11 or k ==22 :
                                        continue
                                    for class_name, class_count in v.items():
                                        if class_name == 11 or class_name == 22:
                                            print( class_name)
                                            continue
                                        dico_sans_22_tmp[class_name] = class_count
                                    dico_sans_22[k]=dico_sans_22_tmp
    #                            print (dico_sans_22)
                                conf_mat_dic = dico_sans_22
                                kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
                                all_matrix.append(conf_mat_dic)
                                all_k.append(kappa)
                                all_oa.append(oacc)
                                all_p.append(p_dic)
                                all_r.append(r_dic)
                                all_f.append(f_dic)
                        else:
                            kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
                            all_matrix.append(conf_mat_dic)
                            all_k.append(kappa)
                            all_oa.append(oacc)
                            all_p.append(p_dic)
                            all_r.append(r_dic)
                            all_f.append(f_dic)
            step.append(classif)
            conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
            nom_dict = get_nomenclature(d["disk_PC"]+"/nomenclature_paper.txt")
            size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
            origin=pd.DataFrame({'step':classif},index=[0],dtype="category") # recuper les noms des différents configurations testées
            origindup=pd.DataFrame(np.repeat(origin.values,len(labels_ref)),dtype="category") 
            p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
            r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
            f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean") # Calcule d'une moyenne pondérer au nombre de classe et de run
            p_mean_df=pd.DataFrame(p_mean.items()) # Transformation en dataframe Pandas
            p_mean_df=p_mean_df[1].str.split(expand=True)
            p_mean_df.drop([1],axis=1,inplace=True) # suppression d'une colonne inutile 
            r_mean_df=pd.DataFrame(r_mean.items())
            r_mean_df=r_mean_df[1].str.split(expand=True)
            r_mean_df.drop([1],axis=1,inplace=True)
            f_mean_df=pd.DataFrame(f_mean.items())
            f_mean_df=f_mean_df[1].str.split(expand=True)
            f_mean_df.drop([1],axis=1,inplace=True)
            KAPPA.append(round(np.mean(all_k),3))
            OA.append(round(np.mean(all_oa),3))
            KAPPA_s.append(round(np.std(all_k),3)) # Arrondi les résulats au centièmes
            OA_s.append(round(np.std(all_oa),3))
            Recall=Recall.append(r_mean_df).astype(float)
            Recall.loc[Recall[0]<=0]=0
            Prec=Prec.append(p_mean_df).astype(float)
            Prec.loc[Prec[0]<=0]=0
            Fscore=Fscore.append(f_mean_df).astype(float)
            Fscore.loc[Fscore[0]<=0]=0 # Supprime les Fscore négatif en les remplacant par 0 
            jobs=jobs.append(origindup)
           
        dfindice_bv=pd.DataFrame([step,KAPPA,KAPPA_s,OA,OA_s],index=["step","mean_Kappa","std_Kappa","mean_OA","std_OA"])
        dfindice_bv=dfindice_bv.T
        dfindice_bv[["mean_Kappa","std_Kappa","mean_OA","std_OA"]]=dfindice_bv[["mean_Kappa","std_Kappa","mean_OA","std_OA"]].apply(pd.to_numeric)
        dfindice_bv.set_index("step",inplace=True)
#        dfindice_bv.sort_index(inplace=True)
        dfindice_bvstep=dfindice_bv.sort_values(by=["mean_Kappa"])
        plt_classif_kappa(dfindice_bvstep,"Kappa","OA") # Production du graphique
        plt.title(years[:-4],size="large")
        # plt.xticks(size='large')
        plt.xticks(fontsize=14)
        plt.yticks(size='large')
        plt.savefig(d["SAVE_disk"]+"KAPPA_RUN_"+b+"_"+years+".png",dpi=600,bbox_inches='tight', pad_inches=0.5)
                
        df_names=["step","mean_fscore","std_fscore","mean_Recall","std_Recall","mean_Precision","std_Precision"]
        dfmetric=pd.concat([jobs,Fscore,Recall,Prec],axis=1)
        dfmetric.columns=df_names
        Classe=labels_ref*len(step)
        dfmetric["Classe"]=Classe  
        dfmetric.sort_values("step",inplace=True)
        dfMetric=pd.concat([dfmetric[dfmetric['Classe'] != "Sorghum"]]) # Génération du tabaleua sans la classes Sorghum
        
        
# =============================================================================
#     Génération des barplot par métriques d'évaluation 
# =============================================================================
#        dfMetric.sort_values(by=["mean_fscore"],inplace=True)
#        for i in dfMetric[["mean_fscore","mean_Recall","mean_Precision"]]:
#            print(i)
#            var=i[5:]
#            plt.figure(figsize=(20,20))
#            sns.set(style="darkgrid")
#            sns.set_context('paper')
#            g = sns.FacetGrid(dfMetric, col="Classe", col_wrap=6, palette="Set1",height=5,margin_titles=False,legend_out=False)# Gerer la color par run et +3 a modifier en focntion du nb de run 
#            g.map_dataframe(errplot, "step", "mean_"+var, "std_"+var,)
#            g.savefig(d["SAVE"]+var+"_plot_classe_run_"+b+"_"+years+".png",dpi=600,bbox_inches='tight', pad_inches=0.5)
        
# =============================================================================
# GRAPHIQUE_scatter
# =============================================================================
    # if bv != "NESTE" :
    #     plt.figure(figsize=(15,8)) 
    #     axes = plt.gca()
    #     ax1=plt.subplot(122)
    #     axes = plt.gca()
    #     for i,step in enumerate(dfindice_bv.index):
    #         x=list(dfMetric[dfMetric.index==0].mean_fscore)[i]
    #         y=list(dfMetric[dfMetric.index==2].mean_fscore)[i]
    #         stdx=list(dfMetric[dfMetric.index==0].std_fscore)[i]
    #         stdy=list(dfMetric[dfMetric.index==2].std_fscore)[i]
    #         if "2017" in step:
    #             p=plt.scatter(x,y,color="blue",marker="o")
    #             a=Ellipse((x,y),width=stdx,height=stdy,alpha=0.2,color="blue",zorder=5)
    #             axes.add_artist(a)
    #         else:
    #             b=plt.scatter(x,y,color="red")
    #             g=Ellipse((x,y),width=stdx,height=stdy,alpha=0.2,color="red",zorder=5)
    #             axes.add_artist(g)
    #         plt.plot([0.1,0.9],[0.1,0.9], 'r-', lw=1)
    #         plt.xlim(0.2,0.9)
    #         plt.ylim(0.2,0.9)
    #         plt.text(x+0.01,y+0.01,step,fontsize=9)
    #         plt.title("Comparaison of fscore between 2 classe")
    #         plt.xlabel("F_score Maize Irrigated")
    #         plt.ylabel("F_score Maize non irrigated")
    #         plt.xticks(size='large')
    #         plt.yticks(size='large')
    #         plt.legend((p,b),("2017","2018"))
    # #        plt.legend((a,g),('Intervalle of 95%'))
    #     ax2=plt.subplot(121)
    #     axes = plt.gca()
    #     for i,step in enumerate(dfindice_bv.index):
    #         x=list(dfMetric[dfMetric.index==1].mean_fscore)[i]
    #         y=list(dfMetric[dfMetric.index==3].mean_fscore)[i]
    #         stdx=list(dfMetric[dfMetric.index==1].std_fscore)[i]
    #         stdy=list(dfMetric[dfMetric.index==3].std_fscore)[i]
    #         if "2017" in step:
    #             p=plt.scatter(x,y,color="blue")
    #             a=Ellipse((x,y),width=stdx,height=stdy,alpha=0.2,color="blue",zorder=5)
    #             axes.add_artist(a)
    #         else:
    #             b=plt.scatter(x,y,color="red")
    #             g=Ellipse((x,y),width=stdx,height=stdy,alpha=0.2,color="red",zorder=4)
    #             axes.add_artist(g)
    #         plt.plot([0.1,0.9],[0.1,0.9], 'r-', lw=1)
    #         plt.xlim(0.2,0.9)
    #         plt.ylim(0.2,0.9)
    #         plt.text(x+0.01,y+0.01,step,fontsize=9)
    #         plt.title("Comparaison of F-score between 2 classe")
    #         plt.xlabel("F-score Soybean Irrigated")
    #         plt.ylabel("F-score Soybean non irrigated")
    #         plt.xticks(size='large')
    #         plt.yticks(size='large')
    #         plt.legend((p,b),("2017","2018"))
    #     plt.savefig(d["SAVE"]+"scatter_Classe_"+"_"+years+"_"+bv+".png",format="png",dpi=600,bbox_inches='tight', pad_inches=0.5)

# =============================================================================
# Barplot recall and Acccuracy
## =============================================================================
#    df2017SAFRAN=dfMetric[dfMetric.step.isin(['All_Data_Not_Cumul_2017','SAR_&_Optic_2017','SAR_Optic_Climate_2017'])]
#    maize2017=df2017SAFRAN[df2017SAFRAN.index.isin([1,3, 0, 2])]
#    maize2017.set_index(['Classe'],inplace=True)
##    maize2017.sort_index(by=["Classe"],inplace=True)
#    # si soybean modifier les index avec 1&3
##    df2017SAFRAN.plot(kind="bar",x="Classe",y=["mean_Recall","mean_Precision"])
#    df2018SAFRAN=dfMetric[dfMetric.step.isin(['All_Data_Not_Cumul_2018','SAR_&_Optic_2018','SAR_Optic_Climate_2018'])]
#    maize2018=df2018SAFRAN[df2018SAFRAN.index.isin([1,3,0,2])]
##    maize2018.sort_index(by=["Classe"],inplace=True)
#    maize2018.set_index(['Classe'],inplace=True)
##    name=set(tuple(maize2017["Classe"]))
#    for i in set(zip(maize2017.step,maize2018.step)):
#        print(i)
#        plt.figure(figsize=(12,5)) 
#        x1=plt.subplot(121)
#        barWidth = 0.3
#        bars1 = maize2017[maize2017.step==i[0]]["mean_Recall"]
#        print (bars1)
#        bars2 = maize2017[maize2017.step==i[0]]["mean_Precision"]
#        yer1 = maize2017[maize2017.step==i[0]]['std_Recall']
#        yer2 = maize2017[maize2017.step==i[0]]['std_Precision']
#        r1 = np.arange(len(bars1))
#        r2 = [x + barWidth for x in r1]
#        plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='Recall')
#        plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='Precision')
#        plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))], maize2017.index,rotation=90) # fixer problème du nom de l'axe"
#        plt.ylabel('value')
#        plt.ylim(0,1)
#        plt.title(str(i[0]))
#        plt.legend()
#        x2=plt.subplot(122)
#        bars1 = maize2018[maize2018.step==i[1]]["mean_Recall"]
#        bars2 = maize2018[maize2018.step==i[1]]["mean_Precision"]
#        yer1 = maize2018[maize2018.step==i[1]]['std_Recall']
#        yer2 = maize2018[maize2018.step==i[1]]['std_Precision']
#        r1 = np.arange(len(bars1))
#        r2 = [x + barWidth for x in r1]
#        plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='Recall')
#        plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='Precision')
#        plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))], maize2018.index,rotation=90)
#        plt.ylabel('value')
#        plt.ylim(0,1)
#        plt.title(str(i[1]))
#        plt.legend()
#        plt.savefig(d["SAVE"]+"barplot_Recall_Accura_crops irrigated"+"_"+years+i[0]+"_"+bv+".png",format="png",dpi=600,bbox_inches='tight', pad_inches=0.5)
#    
        
# =============================================================================
    # graphi Fscre barplot
# =============================================================================
    
    if "2017" not in years or "2018" not in years:
        # df2017SAFRAN=dfMetric[dfMetric.step.isin(['All_Data_Not_Cumul_2017','SAR_&_Optic_2017','SAR_Optic_Climate_2017'])]
        df2017SAFRAN=dfMetric[dfMetric.step.str.endswith('2017')]
        maize2017=df2017SAFRAN.set_index(['Classe'])
        maize2017.sort_index(ascending=True,inplace=True)
        # df2018SAFRAN=dfMetric[dfMetric.step.isin(['All_Data_Not_Cumul_2018','SAR_&_Optic_2018','SAR_Optic_Climate_2018'])]
        df2018SAFRAN=dfMetric[dfMetric.step.str.endswith('2018')]
        maize2018=df2018SAFRAN.set_index(['Classe'])
        maize2018.sort_index(ascending=True,inplace=True)
        M7=maize2017.sort_values(by='step',ascending=True)
        M8=maize2018.sort_values(by='step',ascending=True)
        if "ALL_Years" in years:
            if bv == "NESTE":
                for i in set(zip(maize2017.step,maize2018.step)):
                    print(i)
                    plt.figure(figsize=(12,5)) 
                    plt.grid(axis='x')
                    x1=plt.subplot(121)
                    barWidth = 0.3
                    bars1 = maize2017[maize2017.step==i[0]]["mean_fscore"]
                    bars2 = maize2018[maize2018.step==i[1]]["mean_fscore"]
                    yer1 = maize2017[maize2017.step==i[0]]['std_fscore']
                    yer2 = maize2018[maize2018.step==i[1]]['std_fscore']
                    r1 = np.arange(len(bars1))
                    r2 = [x + barWidth for x in r1]
                    plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='2017')
                    plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='2018')
                    plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],bars1.index,rotation=90) # fixer problème du nom de l'axe"
                    # plt.ylabel('value')
                    plt.ylim(0,1)
                    plt.title(str(i[0][:-5]))
                    plt.legend()
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.savefig(d["SAVE_disk"]+"Fscore_Barplot"+"_"+i[0][:-5]+"_"+bv+".png",format="png",dpi=900,bbox_inches='tight', pad_inches=0.5)
        
            else: # A revoir probleme dans la conception du plot
                for i in set(zip(M7.step,M8.step)):
                    print(i)
                    plt.figure(figsize=(12,5)) 
                    plt.grid(axis='x')
                    x1=plt.subplot(121)
                    barWidth = 0.3
                    bars1 = maize2017[maize2017.step==i[0]]["mean_fscore"]
                    bars2 = maize2018[maize2018.step==i[1]]["mean_fscore"]
                    yer1 = maize2017[maize2017.step==i[0]]['std_fscore']
                    yer2 = maize2018[maize2018.step==i[1]]['std_fscore']
                    r1 = np.arange(len(bars1))
                    r2 = [x + barWidth for x in r1]
                    plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='2017')
                    plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='2018')
                    plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],bars1.index,rotation=90) # fixer problème du nom de l'axe"
                    # plt.ylabel('value')
                    plt.ylim(0,1)
                    plt.title(str(i[0][:-5]))
                    plt.legend()
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.savefig(d["SAVE_disk"]+"Fscore_Barplot"+"_"+i[0][:-5]+"_"+bv+".png",format="png",dpi=900,bbox_inches='tight', pad_inches=0.5)
                    
    # Comparer les runs cumil et non cumuls 
        if years =="CUMUL_VS_NOT":
            stepvs={"Cu_Optic_2017":"OPTIC_ALL_2017","Cu_SAR_2017":"SAR_ALL_2017","Cu_SAR_&_Optic_2017":"ALL_index_2017","Cu_Optic_2018":"OPTIC_ALL_2018","Cu_SAR_2018":"SAR_ALL_2018","Cu_SAR_&_Optic_2018":"ALL_index_2018"}
            for i,j in zip(stepvs.keys(),stepvs.values()):
                if "2017" in i:
                    plt.figure(figsize=(10,10)) 
                    plt.grid(axis='x')
                    # x1=plt.subplot(121)
                    barWidth = 0.3
                    bars1 = maize2017[maize2017.step==i]["mean_fscore"]
                    bars2 = maize2017[maize2017.step==j]["mean_fscore"]
                    yer1 = maize2017[maize2017.step==i]['std_fscore']
                    yer2 = maize2017[maize2017.step==j]['std_fscore']
                    r1 = np.arange(len(bars1))
                    r2 = [x + barWidth for x in r1]
                    plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='Cumul')
                    plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='Not_cumul')
                    plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],bars1.index,rotation=45) # fixer problème du nom de l'axe"
                    # plt.ylabel('value')
                    plt.ylim(0,1)
                    plt.title(r'{} vs {}'.format(str(i),str(j)))
                    plt.legend()
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.savefig(d["SAVE_disk"]+"VERSUS_cumul_not_cumul/Fscore_Barplot"+"_"+i[3:]+"_"+bv+".png",format="png",dpi=900,bbox_inches='tight', pad_inches=0.5)
                else:
                
                    plt.figure(figsize=(10,10)) 
                    plt.grid(axis='x')
                    bars1 = maize2018[maize2018.step==i]["mean_fscore"]
                    bars2 = maize2018[maize2018.step==j]["mean_fscore"]
                    yer1 = maize2018[maize2018.step==i]['std_fscore']
                    yer2 = maize2018[maize2018.step==j]['std_fscore']
                    r1 = np.arange(len(bars1))
                    r2 = [x + barWidth for x in r1]
                    plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='Cumul')
                    plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='Not_cumul')
                    plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],bars1.index,rotation=45) # fixer problème du nom de l'axe"
                    # plt.ylabel('value')
                    plt.ylim(0,1)
                    plt.title(r'{} vs {}'.format(str(i),str(j)))
                    plt.legend()
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.savefig(d["SAVE_disk"]+"VERSUS_cumul_not_cumul/Fscore_Barplot"+"_"+i[3:]+"_"+bv+".png",format="png",dpi=900,bbox_inches='tight', pad_inches=0.5)
    # =============================================================================
#     SEASON_TIME
# =============================================================================
#    plt.figure(figsize=(10,7)) 
#    a=dfMetric[dfMetric.index==0]
#    b=dfMetric[dfMetric.index==2]
#    plt.plot(a.step,a["mean_fscore"],color='blue',label='Irr')
#    plt.fill_between(a.step,a["mean_fscore"]+a["std_fscore"],a["mean_fscore"]-a["std_fscore"],alpha=0.2,color='blue')
#    plt.plot(b.step,b["mean_fscore"],color='red',label='non Irr')
#    plt.fill_between(a.step,b["mean_fscore"]+b["std_fscore"],b["mean_fscore"]-b["std_fscore"],alpha=0.2,color='red')
#    plt.xticks(rotation=45)
#    plt.ylim(0.1,0.9)
#    plt.ylabel("F_score")
#    plt.title(str(bv)+"_"+str(years))
#    plt.legend()
#    plt.savefig(d["SAVE"]+"fscore_maize"+"_"+years+"_"+bv+".png",format="png",dpi=600,bbox_inches='tight', pad_inches=0.5)


# =============================================================================
#  Test sur Fscore en focntion des classes et non des runs
# =============================================================================
    df2017SAFRAN=dfMetric[dfMetric.step.str.endswith('2017')]
    maize2017=df2017SAFRAN.set_index(['Classe'])
    maize2017.sort_index(ascending=True,inplace=True)
    # df2018SAFRAN=dfMetric[dfMetric.step.isin(['All_Data_Not_Cumul_2018','SAR_&_Optic_2018','SAR_Optic_Climate_2018'])]
    df2018SAFRAN=dfMetric[dfMetric.step.str.endswith('2018')]
    maize2018=df2018SAFRAN.set_index(['Classe'])
    maize2018.sort_index(ascending=True,inplace=True)
    M7=maize2017.sort_values(by='step',ascending=True)
    M8=maize2018.sort_values(by='step',ascending=True)
    if "All_Years" in years:
        # if bv == "NESTE":
        #     for i in set(zip(maize2017.step,maize2018.step)):
        #         print(i)
        #         plt.figure(figsize=(12,5)) 
        #         x1=plt.subplot(121)
        #         barWidth = 0.3
        #         bars1 = maize2017[maize2017.step==i[0]]["mean_fscore"]
        #         bars2 = maize2018[maize2018.step==i[1]]["mean_fscore"]
        #         yer1 = maize2017[maize2017.step==i[0]]['std_fscore']
        #         yer2 = maize2018[maize2018.step==i[1]]['std_fscore']
        #         r1 = np.arange(len(bars1))
        #         r2 = [x + barWidth for x in r1]
        #         plt.bar(r1, bars1, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='2017')
        #         plt.bar(r2, bars2, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='2018')
        #         plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],bars1.index,rotation=90) # fixer problème du nom de l'axe"
        #         plt.ylabel('value')
        #         plt.ylim(0,1)
        #         plt.title(str(i[0][:-5]))
        #         plt.legend()
        #         plt.xticks(size='large')
        #         plt.yticks(size='large')
        #         plt.savefig(d["SAVE"]+"Fscore_Barplot"+"_"+i[0][:-5]+"_"+bv+".png",format="png",dpi=900,bbox_inches='tight', pad_inches=0.5)
    
        for i in set(zip(M7.index,M8.index)):
            print(i)
            plt.figure(figsize=(5,5)) 
            plt.grid(axis='x')
            x1=plt.subplot(111)
            barWidth = 0.3
            bars1 = maize2017[maize2017.index==i[0]][["mean_fscore","step"]]
            bars2 = maize2018[maize2018.index==i[1]][["mean_fscore","step"]]
            yer1 = maize2017[maize2017.index==i[0]]['std_fscore']
            yer2 = maize2018[maize2018.index==i[1]]['std_fscore']
            name=bars1.step.to_list()
            list_name=[]
            for n in name : 
                list_name.append(n[:-5])
            r1 = np.arange(len(bars1))
            r2 = [x + barWidth for x in r1]
            plt.bar(r1, bars1.mean_fscore, width = barWidth, color = 'orange', edgecolor = 'black', yerr=yer1, capsize=5, label='2017')
            plt.bar(r2, bars2.mean_fscore, width = barWidth, color = 'royalblue', edgecolor = 'black', yerr=yer2, capsize=5, label='2018')
            plt.xticks([r + barWidth - 0.1 for r in range(len(bars2))],list_name,rotation=90,size =9) # fixer problème du nom de l'axe"
            # plt.ylabel('value')
            plt.ylim(0,1)
            # plt.title(str(i[0]))
            if 'Rainfed Maize'in i:
               plt.legend(fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            for j in np.arange(len(set(M7.step))):
                plt.text(x =np.arange(len(set(M7.step)))[j] -0.1 , y= list(yer1+bars1.mean_fscore)[j] +0.01,s = list(bars1.mean_fscore)[j],size=12)
                plt.text(x =np.arange(len(set(M8.step)))[j] +0.2, y= list(yer2+bars2.mean_fscore)[j]+0.01,s = list(bars2.mean_fscore)[j],size=12)
            plt.savefig(d["SAVE_disk"]+"bartest"+"_"+years+"_"+i[0]+"paper.png",format="png",dpi=600,bbox_inches='tight', pad_inches=0.5)