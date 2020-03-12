#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:30:30 2019

@author: pageot
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:09:22 2019

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
from scipy import stats
import otbApplication
from ResultsUtils import *



def mergeVectors(outname, opath, files, ext="shp", out_Tbl_name=None):
    """
    Merge a list of vector files in one
    """
    done = []

    outType = ''
    if ext == 'sqlite':
        outType = ' -f SQLite '
    file1 = files[0]
    nbfiles = len(files)
    filefusion = opath + "/" + outname + "." + ext
    if os.path.exists(filefusion):
        os.remove(filefusion)

    table_name = outname
    if out_Tbl_name:
        table_name = out_Tbl_name
    fusion = 'ogr2ogr '+filefusion+' '+file1+' '+outType+' -nln '+table_name
    os.system(fusion)

    done.append(file1)
    for f in range(1, nbfiles):
        fusion = 'ogr2ogr -update -append '+filefusion+' '+files[f]+' -nln '+table_name+' '+outType
        os.system(fusion)
        done.append(files[f])

    return filefusion


if __name__ == "__main__":
    os.environ['OTB_MAX_RAM_HINT'] = str(8000) # 8Gb 
    years ="2017"
    method="SHARK"# or "OPEN_CV" or "SEASON_TIME" or "SHARK"
    bv="ADOUR" # or TARN
    d={}
    file=d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/"
    d["vector_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_LEARN_VAL_CLASSIF_MT/RUN_FIXE_SEED/"+years+"/"
    ram=8096
    tuiles=["T31TCJ","T31TDJ","T30TYP","T30TYN"]
    grain=range(0,5)
#    Raster_classif=[]
        
    for classif in os.listdir('/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/'+years+'/RUN_fixe_seed/'+method):
        if "ASC" in classif :
            print ("=============")
            print (r" RUN : %s " %classif)
            print ("=============")
            Raster_classif=[]
            for Rast in os.listdir('/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/'+years+'/RUN_fixe_seed/'+method+"/"+classif+"/final/"):
                if "Classif_"+bv+"" and "regularized.tif" in Rast and "aux.xml" not in Rast :
                    print (Rast)
                    Raster_classif.append('/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/'+years+'/RUN_fixe_seed/'+method+"/"+classif+"/final/"+Rast)
                    
            Fusion_class = otbApplication.Registry.CreateApplication("FusionOfClassifications")
            Fusion_class.SetParameterStringList("il", Raster_classif)
            Fusion_class.SetParameterString("method","majorityvoting")
#            Fusion_class.SetParameterString("method.dempstershafer.mob","precision")
            Fusion_class.SetParameterInt("nodatalabel", 0)
#            Fusion_class.SetParameterInt("undecidedlabel", 10)
            Fusion_class.SetParameterString("out", "/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/"+years+"/RUN_fixe_seed/"+method+"/"+classif+"/final/{}_{}_{}_MAJORITY.tif".format(classif,bv,years))
            Fusion_class.ExecuteAndWriteOutput()

# Matrix de confusion 
            print ("Calcul confusion matrix")
            ComputeConfusionMatrix = otbApplication.Registry.CreateApplication("ComputeConfusionMatrix")
            ComputeConfusionMatrix.SetParameterString("in", '/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/{}_{}_{}_MAJORITY.tif'.format(classif,bv,years))          
            ComputeConfusionMatrix.SetParameterString("out", '/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/ConfusionMatrix_fusion_{}_{}_{}.csv'.format(classif,bv,years))       
            ComputeConfusionMatrix.SetParameterString("ref","vector")      
            ComputeConfusionMatrix.SetParameterString("ref.vector.in", d["vector_file"]+classif+'/Fusion_all/%s_seed_0.shp'% (bv.upper())) # Probleme liée au seed (le quel choisir) ou utlisr des données exterieurs à la classifcation pour valider 
            ComputeConfusionMatrix.UpdateParameters()
            ComputeConfusionMatrix.SetParameterString("ref.vector.field", "labcroirr")
            ComputeConfusionMatrix.SetParameterString('nodatalabel', str(0))
            ComputeConfusionMatrix.SetParameterString('ram',str(ram))
            ComputeConfusionMatrix.ExecuteAndWriteOutput()
                


## =============================================================================
##     Matric confusion_build
## =============================================================================
            for b in [bv] :
                print(b)
                nom=get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
                pathNom="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt"
                pathRes="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/"+years+"/RUN_fixe_seed/"+method+"/"+classif+"/final/"
                if b == "NESTE":
                    from collections import OrderedDict
                    dico_sans_22 = OrderedDict()
                    for j in os.listdir(pathRes):
                       if "ConfusionMatrix_fusion_"+classif+"_"+b+"" in j:
                            conf_mat_dic = parse_csv(pathRes+j)
                            for k, v in conf_mat_dic.items():
                                dico_sans_22_tmp = OrderedDict()
                                if k == 11 or k ==22 or  k== 44:
                                    print (k)
                                    continue
                                for class_name, class_count in v.items():
                                    if class_name == 11 or class_name == 22 or class_name ==44:
                                        print( class_name)
                                        continue
                                    dico_sans_22_tmp[class_name] = class_count
                                dico_sans_22[k]=dico_sans_22_tmp
#                            print (dico_sans_22)
                            conf_mat_dic = dico_sans_22
                            kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
                            conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
                            nom_dict = get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
                            size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)

            
               
                            fig_conf_mat(conf_mat_dic,nom,kappa,oacc,p_dic,r_dic,f_dic,pathRes+"Matrix_fusion"+b+"_"+classif+"_Fusion_regularized.png",conf_score="percentage", grid_conf=True)
                    
                else:
                     for j in os.listdir(pathRes):
                         if "ConfusionMatrix_fusion_"+classif+"_"+b+"" in j:
                             conf_mat_dic = parse_csv(pathRes+j)
                             kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
                             nom_dict = get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
                             size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
                             fig_conf_mat(conf_mat_dic,nom,kappa,oacc,p_dic,r_dic,f_dic,pathRes+"Matrix_fusion"+b+"_"+classif+"_Fusion_regularized.png",conf_score="percentage")
                         
    print ("end processing")