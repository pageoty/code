#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:22:42 2019

@author: pageot

Script Confusion matrix between  Irrigated Crops and No irrigated Crops 
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import collections
#from ResultsUtils import get_coeff, get_conf_max 
from ResultsUtils import *
from Common import FileUtils as fu
from Common.Utils import run
import ResultsUtils as resU


#def parse_csv(path_csv):
#    with open(path_csv, 'r') as csvfile:
#        csv_reader = csv.reader(csvfile)
#        ref_lab = [elem.replace("#Reference labels (rows):", "") for elem in next(csv_reader)]
#        prod_lab = [elem.replace("#Produced labels (columns):", "") for elem in next(csv_reader)]
#        all_labels = sorted([int(label) for label in list(set(ref_lab + prod_lab))])
#
#        # construct confusion matrix structure and init it at 0
#        matrix = collections.OrderedDict()
#        for lab_ref in all_labels:
#            matrix[lab_ref] = collections.OrderedDict()
#            for lab_prod in all_labels:
#                matrix[lab_ref][lab_prod] = 0
#
#        # fill-up confusion matrix
#        csv_dict = csv.DictReader(csvfile, fieldnames=prod_lab)
#        for row_num, row_ref in enumerate(csv_dict):
#            for klass, value in list(row_ref.items()):
#                ref = int(ref_lab[row_num])
#                prod = int(klass)
#                matrix[ref][prod] += float(value)
#    return matrix

def get_coeff_modif(matrix,list_lab):
    """
    use to extract coefficients (Precision, Recall, F-Score, OA, K)
    from a confusion matrix.

    Parameters
    ----------

    matrix : collections.OrderedDict
        a confusion matrix stored in collections.OrderedDict dictionnaries

    Example
    -------
        >>> conf_mat_dic = OrderedDict([(1, OrderedDict([(1, 50), (2, 78), (3, 41)])),
        >>>                             (2, OrderedDict([(1, 20), (2, 52), (3, 31)])),
        >>>                             (3, OrderedDict([(1, 27), (2, 72), (3, 98)]))])
        >>> kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
        >>> print p_dic[1]
        >>> 0.5154639175257731

    Return
    ------
    list
        Kappa, OA, Precision, Recall, F-Score. Precision, Recall, F-Score
        are collections.OrderedDict
    """
    import collections
    
    nan = -1000
    classes_labels = list(matrix.keys())
    if list_lab[0] and list_lab[1] in classes_labels: 
        classes_labels=list_lab
        oacc_nom = sum([matrix[class_name][class_name] for class_name in classes_labels])
        nb_samples = sum([matrix[ref_class_name][prod_class_name] for ref_class_name in classes_labels for prod_class_name in classes_labels])

        # compute overall accuracy
        if nb_samples != 0.0:
            oacc = float(oacc_nom) / float(nb_samples)
        else:
            oacc = nan
    
        p_dic = collections.OrderedDict()
        r_dic = collections.OrderedDict()
        f_dic = collections.OrderedDict()
        lucky_rate = 0.
        for classe_name in classes_labels:
            oacc_class = matrix[classe_name][classe_name]
            p_denom = sum([matrix[ref][classe_name] for ref in classes_labels])
            r_denom = sum([matrix[classe_name][ref] for ref in classes_labels])
            if float(p_denom) != 0.0:
                p_class = float(oacc_class) / float(p_denom)
            else:
                p_class = nan
            if float(r_denom) != 0.0:
                r_class = float(oacc_class) / float(r_denom)
            else:
                r_class = nan
            if float(p_class + r_class) != 0.0:
                f_class = float(2.0 * p_class * r_class) / float(p_class + r_class)
            else:
                f_class = nan
            p_dic[classe_name] = p_class
            r_dic[classe_name] = r_class
            f_dic[classe_name] = f_class
    
            lucky_rate += p_denom * r_denom
    
        k_denom = float((nb_samples * nb_samples) - lucky_rate)
        k_nom = float((oacc * nb_samples * nb_samples) - lucky_rate)
        if k_denom != 0.0:
            kappa = k_nom / k_denom
        else:
            kappa = nan
    
        return kappa, oacc, p_dic, r_dic, f_dic
    

if __name__ == "__main__":
#    d={}
#    d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/RUN/"
#    lab=[1,11]
#    all=[]
#    for j in os.listdir(d["data_file"]):
#        for i in os.listdir(d["data_file"] +j):
#            for k in os.listdir(d["data_file"] +j+"/"+i+"/TMP"):
#                if "Classif_Seed" in k:
#                    globals()["matrix_%s_%s"% (j,k[:-4])] = parse_csv(d["data_file"] +j+"/"+i+"/TMP/"+k)                 
#                    kappa, oacc, p_dic, r_dic, f_dic=get_coeff_modif(globals()["matrix_%s_%s"% (j,k[:-4])],lab)
#                    
#                    all.append(kappa)  
#    lab=[1,11,2,22,33,44]
#    Fscore=[]
#    nom=get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/MATRIX_CONFU_TYP_RPG/Nomenclature_RPG.txt")
#    for i in os.listdir("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/MATRIX_CONFU_TYP_RPG/MATRIX/"):
#        print (i)                
#        globals()["matrix_%s"% (i[:-4])]=parse_csv("/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/MATRIX_CONFU_TYP_RPG/MATRIX/%s" %i)
#
#        kappa, oacc, p_dic, r_dic, f_dic=get_coeff_modif(globals()["matrix_%s"% (i[:-4])],lab)
#        Fscore.append(f_dic)
#        
#        fig_conf_mat(globals()["matrix_%s"% (i[:-4])],nom,kappa, oacc, p_dic, r_dic, f_dic,"/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/tmp/MATRIX_CONFU_TYP_RPG/matrix_%s.png"%(i[:-4]) )
   

# =============================================================================
#  Fusion matrix run 5     
# =========================================================== ==================
    nom=get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
    pathNom="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt"
    pathRes="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/2017/MT/RUN_POLA_DES_FULL_3ind/final/"
    


    all_k = []
    all_oa = []
    all_p = []
    all_r = []
    all_f = []
    all_matrix = []
    for csv in os.listdir(pathRes):
        if "csv" in csv and "ColorIndexed" in csv:
#            if "CM" in csv:
            print (csv)
            conf_mat_dic = parse_csv(pathRes+csv)
            kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
            all_matrix.append(conf_mat_dic)
            all_k.append(kappa)
            all_oa.append(oacc)
            all_p.append(p_dic)
            all_r.append(r_dic)
            all_f.append(f_dic)
    
    conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
    nom_dict = get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
    size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
    p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
    r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
    f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean")

   
    fig_conf_mat(conf_mat_dic,nom,np.mean(all_k),np.mean(all_oa),p_mean,r_mean,f_mean,pathRes[:-4]+"Matrix_fusion.png",conf_score="percentage")



    nom=get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
    pathNom="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt"
    pathRes="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/2018/RUN_MT/RUN_POLA_DES_FULL_3ind_RPG_SOG/final/TMP/"
    


    all_k = []
    all_oa = []
    all_p = []
    all_r = []
    all_f = []
    all_matrix = []
    for csv in os.listdir(pathRes):
        if "csv" in csv and "Classif" in csv:
            print (csv)
            conf_mat_dic = parse_csv(pathRes+csv)
            kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
            all_matrix.append(conf_mat_dic)
            all_k.append(kappa)
            all_oa.append(oacc)
            all_p.append(p_dic)
            all_r.append(r_dic)
            all_f.append(f_dic)
        
    conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
    nom_dict = get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
    size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
    p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
    r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
    f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean")

   
    fig_conf_mat(conf_mat_dic,nom,np.mean(all_k),np.mean(all_oa),p_mean,r_mean,f_mean,pathRes[:-4]+"Matrix_fusion.png",conf_score="percentage")
