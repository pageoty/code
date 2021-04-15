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
# import seaborn as sns
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
    
    years ="2017"
    method="SEASON_TIME"# or "OPEN_CV" or "SEASON_TIME" or "SHARK"
    bv="ADOUR" # or TARN
    d={}
    d["data_file"]="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/"
    d["data_tra"]="/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/"
    d["output_file"]='/datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_LEARN_VAL_CLASSIF_MT/RUN_FIXE_SEED/'+years+'/'
    d["output_file_unix"]='/mnt/h/Yann_THESE/DATA_CLASSIFICATION/DATA_LEARN_VAL_CLASSIF_MT/RUN_FIXE_SEED/'+years+'/'
    d["data_file_unix"]="/mnt/h/Yann_THESE/RESULTAT_CLASSIFICATION/"
    d["data_tra_unix"]="/mnt/h/Yann_THESE/RESULTAT_CLASSIFICATION/"
    d["data_tra_usb"]="/mnt/g/THESE/CLASSIFICATION/TRAITEMENT/"
    d["data_tra_data"]="/mnt/g/THESE/CLASSIFICATION/"
    ram=5000
    tuiles=["T31TCJ","T31TDJ","T30TYP","T30TYN"]
    grain=range(0,5)
    # d["unix_data"]="/mnt/d/THESE_TMP/classif/"
    
#    for jobs in os.listdir(d["data_file"]+"DATA_LEARN_VAL_CLASSIF_MT/NESTE_2017/"): # fonction utlisé pour générer les lotts validation à l'échelle du BV 
#        if "NESTE" in jobs :
#            print (jobs)
#            for s in grain:
#                print ("=========")
#                print (s)
#                print ("=========")
#                SHP_seed=[]
#                for t in tuiles:
#                    print (t)
#                    sortie=d["output_file"]+str(jobs)+'/Fusion_all/%s_seed_%s_val.shp'%(t,s)
#                    entre=d["data_file"]+'DATA_LEARN_VAL_CLASSIF_MT/NESTE_2017/'+str(jobs)+'/dataAppVal/%s_seed_%s_val.sqlite'%(t,str(s))
#                    cmd='ogr2ogr -f "ESRI Shapefile" '+sortie+'  '+entre+' -skipfailures'
#                    os.system(cmd)
#                    SHP_seed.append(sortie)    
#                print("Merge of shapefile")
#                name="merge%s"%s
#                mergeVectors(name,sortie[:-21],SHP_seed,ext="shp")
#                os.system("rm /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_LEARN_VAL_CLASSIF_MT/NESTE_2017/%s/Fusion_all/*seed*val*"%(jobs))
#                for bv,BV in zip([1,2,3,4],["OTHERS","NESTE","TARN","ADOUR"]):
#                    print ("Extraction of region")
#                    sortie=d["output_file"]+str(jobs)+'/Fusion_all/%s_seed_%s.shp'% (BV,s)
#                    entre=d["output_file"]+str(jobs)+"/Fusion_all/merge%s.shp"% s
#                    arg="region='%s'" % bv
#                    print( arg)
#                    commande='ogr2ogr -f "ESRI Shapefile"  -where "'+arg+'" '+sortie+'  '+entre+''
#    
#                    os.system(commande)
#        os.system("rm /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/DATA_LEARN_VAL_CLASSIF_MT/2017/%s/Fusion_all/others_seed_*"% (jobs))

        
    for classif in os.listdir(d["data_file_unix"]+'/'+years+'/RUN_fixe_seed/'+method):
        if "SAISON" in classif and "NESTE" not in classif and "DES" not in classif and "3ind" not in classif : 
            print ("=============")
            print (r" RUN : %s " %classif)
            print ("=============")
            for seed in os.listdir(d["data_file_unix"]+'/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/'):
                if "ColorIndexed.tif" in seed:
                    print (r'seed : %s' %seed[13:14])
                    for BV in os.listdir(d["data_tra_data"]+'/DONNES_SIG/EMPRISE/EMPRISE_RASTER_BV/'):
                        print (r" watershed : %s" %(BV[:-4]))
                        if BV[:-4] == bv :
                            print (True)
                            ConcatenateImages_CROPS = otbApplication.Registry.CreateApplication("ConcatenateImages") # Create Otb Application 
                            ConcatenateImages_CROPS.SetParameterStringList("il",[d["data_tra_usb"]+'/RPG/MASk_'+years+'_RECAL_ADOUR.tif']) # or MASK_RPG_2018_Er10.tif & MASK_MT_RPG2017.tif
                            ConcatenateImages_CROPS.SetParameterString("out", "mask_crops.tif")
                            ConcatenateImages_CROPS.Execute()
                            ConcatenateImages_MASK = otbApplication.Registry.CreateApplication("ConcatenateImages") # Create Otb Application 
                            ConcatenateImages_MASK.SetParameterStringList("il",[d["data_tra_data"]+'/DONNES_SIG/EMPRISE/EMPRISE_RASTER_BV/'+BV])
                            ConcatenateImages_MASK.SetParameterString("out", "Mask_BV.tif")
                            ConcatenateImages_MASK.Execute()
        
                            ConcatenateImages_Class = otbApplication.Registry.CreateApplication("ConcatenateImages") # Create Otb Application 
                            ConcatenateImages_Class.SetParameterStringList("il",[d["data_file_unix"]+'/'+years+'/RUN_fixe_seed/'+method+'/'+classif+"/final/"+seed])
                            ConcatenateImages_Class.SetParameterString("out", "Classif.tif")
                            ConcatenateImages_Class.Execute()

                            print ('Band_math processing')
                            BandMath = otbApplication.Registry.CreateApplication("BandMath") # Application du mask sur classification 
                            BandMath.AddImageToParameterInputImageList("il",ConcatenateImages_MASK.GetParameterOutputImage("out"))
                            BandMath.AddImageToParameterInputImageList("il",ConcatenateImages_CROPS.GetParameterOutputImage("out"))
                            BandMath.AddImageToParameterInputImageList("il",ConcatenateImages_Class.GetParameterOutputImage("out"))
                            BandMath.SetParameterString("out",d["data_file_unix"]+'/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/Classif_paper_'+BV[:-4]+"_"+seed[13:14]+".tif")
                            BandMath.SetParameterString("exp", "im1b1*im2b1*im3b1")
                            BandMath.SetParameterString("ram",str(ram))
                            BandMath.SetParameterOutputImagePixelType("out",otbApplication.ImagePixelType_uint8)
                            BandMath.ExecuteAndWriteOutput()
        
                            print ('Map Regularization processing') # Filtrage par régualtion en supprimant les pixles isolées selon une fenetre
                            ClassificationMapRegularization = otbApplication.Registry.CreateApplication("ClassificationMapRegularization")
                            ClassificationMapRegularization.SetParameterString("io.in", d["data_file_unix"]+'/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/Classif_paper_'+BV[0:-4]+"_"+seed[13:14]+".tif")
                            ClassificationMapRegularization.SetParameterString("io.out",d["data_file_unix"]+ '/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/Classif_paper_'+BV[0:-4]+"_"+seed[13:14]+"_regularized.tif")
                            ClassificationMapRegularization.SetParameterInt("ip.radius", 1)
                            ClassificationMapRegularization.SetParameterInt("ip.nodatalabel", 0)
                            ClassificationMapRegularization.SetParameterString("ip.onlyisolatedpixels",'True')
                            ClassificationMapRegularization.SetParameterInt('ip.isolatedthreshold', 3)
                            ClassificationMapRegularization.SetParameterString('ram',str(ram))
                            ClassificationMapRegularization.ExecuteAndWriteOutput()
                            
                            print ("Confusion_matrix processing") # génratino de la matrix de confusion
                            ComputeConfusionMatrix = otbApplication.Registry.CreateApplication("ComputeConfusionMatrix")
                            ComputeConfusionMatrix.SetParameterString("in", d["data_file_unix"]+'/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/Classif_paper_'+BV[0:-4]+"_"+seed[13:14]+"_regularized.tif")          
                            ComputeConfusionMatrix.SetParameterString("out", d["data_file_unix"]+'/'+years+'/RUN_fixe_seed/'+method+'/'+classif+'/final/ConfusionMatrix_regularized_%s_%s.csv'% (BV[0:-4],seed[13:14]))       
                            ComputeConfusionMatrix.SetParameterString("ref","vector")      
                            ComputeConfusionMatrix.SetParameterString("ref.vector.in", d["output_file_unix"]+classif+'/Fusion_all/%s_seed_%s.shp'% (BV[0:-4],seed[13:14])) #d["output_file"]+classif+'/Fusion_all/merge%s.shp'%(seed[13:14])
                            ComputeConfusionMatrix.UpdateParameters()
                            ComputeConfusionMatrix.SetParameterString("ref.vector.field", "labcroirr")
                            ComputeConfusionMatrix.SetParameterString('nodatalabel', str(0))
                            ComputeConfusionMatrix.SetParameterString('ram',str(ram))
                            ComputeConfusionMatrix.ExecuteAndWriteOutput()
                        
                        # print ("Confusion_matrix processing")
                        # ComputeConfusionMatrix = otbApplication.Registry.CreateApplication("ComputeConfusionMatrix")
                        # ComputeConfusionMatrix.SetParameterString("in", '/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/'+years+'/RUN_fixe_seed/'+classif+'/final/Classif_'+BV[10:-4]+"_"+seed[13:14]+".tif")          
                        # ComputeConfusionMatrix.SetParameterString("out", '/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/'+years+'/RUN_fixe_seed/'+classif+'/final/ConfusionMatrix_%s_%s.csv'% (BV[10:-4],seed[13:14]))       
                        # ComputeConfusionMatrix.SetParameterString("ref","vector")      
                        # ComputeConfusionMatrix.SetParameterString("ref.vector.in", d["output_file"]+classif+'/Fusion_all/%s_seed_%s.shp'% (BV[10:-4],seed[13:14])) #d["output_file"]+classif+'/Fusion_all/merge%s.shp'%(seed[13:14])
                        # ComputeConfusionMatrix.UpdateParameters()
                        # ComputeConfusionMatrix.SetParameterString("ref.vector.field", "labcroirr")
                        # ComputeConfusionMatrix.SetParameterString('nodatalabel', str(0))
                        # ComputeConfusionMatrix.SetParameterString('ram',str(ram))
                        # ComputeConfusionMatrix.ExecuteAndWriteOutput()
                        
    
    #         for b in ["ADOUR","TARN",'NESTE'] :
    #             print(b)
    #             nom=get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
    #             pathNom="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt"
    #             pathRes="/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/"+years+"/RUN_fixe_seed/"+classif+"/final/"
    #             all_k = []
    #             all_oa = []
    #             all_p = []
    #             all_r = []
    #             all_f = []
    #             all_matrix = []
    #             a=[]
    #             if b == "NESTE":
    #                 from collections import OrderedDict
    #                 dico_sans_22 = OrderedDict()
    #                 for j in os.listdir(pathRes):
    #                     if ".csv" in j and "regularized" not in j:
    #                         if b in j:
    #                             print(j)
    #                             conf_mat_dic = parse_csv(pathRes+j)
    #                             for k, v in conf_mat_dic.items():
    #                                 dico_sans_22_tmp = OrderedDict()
    #                                 if k == 11 or k ==22 or k ==44:
    #                                     print (k)
    #                                     continue
    #                                 for class_name, class_count in v.items():
    #                                     if class_name == 11 or class_name == 22 or class_name==44:
    #                                         print( class_name)
    #                                         continue
    #                                     dico_sans_22_tmp[class_name] = class_count
    #                                 dico_sans_22[k]=dico_sans_22_tmp
    # #                            print (dico_sans_22)
    #                             conf_mat_dic = dico_sans_22
    #                             kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
    #                             all_matrix.append(conf_mat_dic)
    #                             all_k.append(kappa)
    #                             all_oa.append(oacc)
    #                             all_p.append(p_dic)
    #                             all_r.append(r_dic)
    #                             all_f.append(f_dic)
    #                 conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
    #                 nom_dict = get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
    #                 size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
    #                 p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
    #                 r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
    #                 f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean")
        
          
    #                 fig_conf_mat(conf_mat_dic,nom,np.mean(all_k),np.mean(all_oa),p_mean,r_mean,f_mean,pathRes+"Matrix_fusion"+b+"_"+classif+".png",conf_score="percentage", grid_conf=True)
                
    #             else:
    #                 for j in os.listdir(pathRes):
    #                     if ".csv" in j and "regularized" not in j:
    #                         if b in j:
    #                             print(j)
    #                             conf_mat_dic = parse_csv(pathRes+j)
    #                             kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
    #                             all_matrix.append(conf_mat_dic)
    #                             all_k.append(kappa)
    #                             all_oa.append(oacc)
    #                             all_p.append(p_dic)
    #                             all_r.append(r_dic)
    #                             all_f.append(f_dic)
    #                 conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
    #                 nom_dict = get_nomenclature("/datalocal/vboxshare/THESE/CLASSIFICATION/RESULT/nomenclature_T31TDJ.txt")
    #                 size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
    #                 p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
    #                 r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
    #                 f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean")
            
              
    #                 fig_conf_mat(conf_mat_dic,nom,np.mean(all_k),np.mean(all_oa),p_mean,r_mean,f_mean,pathRes+"Matrix_fusion"+b+"_"+classif+".png",conf_score="percentage")

                    
            for b in [bv] : # génration de figurz matrix 
                print(b)
                nom=get_nomenclature( d["data_file_unix"]+"/nomenclature_T31TDJ.txt")
                pathNom= d["data_file_unix"]+"/nomenclature_T31TDJ.txt"
                pathRes= d["data_file_unix"]+"/"+years+"/RUN_fixe_seed/"+method+"/"+classif+"/final/"
                all_k = []
                all_oa = []
                all_p = []
                all_r = []
                all_f = []
                all_matrix = []
                a=[]
                if b == "NESTE":
                    from collections import OrderedDict
                    dico_sans_22 = OrderedDict()
                    for j in os.listdir(pathRes):
                        if ".csv" in j and "regularized" in j:
                            if b in j:
                                print(j)
                                conf_mat_dic = parse_csv(pathRes+j)
                                if "2017" in classif :
                                    for k, v in conf_mat_dic.items():
                                        dico_sans_22_tmp = OrderedDict()
                                        if k == 11 or k ==22 :
                                            print (k)
                                            continue
                                        for class_name, class_count in v.items():
                                            if class_name == 11 or class_name == 22 :
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
                    conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
                    nom_dict = get_nomenclature( d["data_file_unix"]+"/nomenclature_T31TDJ.txt")
                    size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
                    p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
                    r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
                    f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean")
            
               
                    fig_conf_mat(conf_mat_dic,nom,np.mean(all_k),np.mean(all_oa),p_mean,r_mean,f_mean,pathRes+"Matrix_fusion"+b+"_"+classif+"_regularized.png",conf_score="percentage", grid_conf=True)
                    
                else:
                      for j in os.listdir(pathRes):
                        if ".csv" in j and "regularized" in j:
                            if b in j:
                                print(j)
                                conf_mat_dic = parse_csv(pathRes+j)
                                kappa, oacc, p_dic, r_dic, f_dic = get_coeff(conf_mat_dic)
                                all_matrix.append(conf_mat_dic)
                                all_k.append(kappa)
                                all_oa.append(oacc)
                                all_p.append(p_dic)
                                all_r.append(r_dic)
                                all_f.append(f_dic)
                      conf_mat_dic = compute_interest_matrix(all_matrix, f_interest="mean")
                      nom_dict = get_nomenclature(d["data_file_unix"]+"/nomenclature_T31TDJ.txt")
                      size_max, labels_prod, labels_ref = get_max_labels(conf_mat_dic, nom_dict)
                      p_mean = get_interest_coeff(all_p, nb_lab=len(labels_ref), f_interest="mean")
                      r_mean = get_interest_coeff(all_r, nb_lab=len(labels_ref), f_interest="mean")
                      f_mean = get_interest_coeff(all_f, nb_lab=len(labels_ref), f_interest="mean")
            
                      fig_conf_mat(conf_mat_dic,nom,np.mean(all_k),np.mean(all_oa),p_mean,r_mean,f_mean,pathRes+"Matrix_fusion"+b+"_"+classif+"_regularized.png",conf_score="percentage")
