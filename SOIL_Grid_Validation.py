#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:58:45 2020

@author: pageot
"""


import os
import sqlite3
import geopandas as geo
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import TEST_ANALYSE_SIGNATURE
import shapely.geometry as geom
import descartes
from mpl_toolkits import mplot3d
from IPython.display import HTML
import matplotlib.animation as animation
import statistics as stat

def init():
    ax.scatter3D(xdata,ydata, zdata, c=zdata)
    return fig

def animate(i):
    ax.view_init(elev=30., azim=3.6*i)
    return fig

def rotate(angle):
    ax.view_init(azim=angle)

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

if __name__ == '__main__':
    d={}
    d['path_features']="/datalocal/vboxshare/THESE/BESOIN_EAU/DONNES_SOIL/SOIL_GRID/"
    d["SAVE"]="/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/PLOT/GEOSTAT_SOIL_GRID/"
    all_features={d['path_features']+'CLAY/', d['path_features']+'limon/',d['path_features']+'sand/',d["path_features"]+'Depth/'}
    d["PC_disk"]="G:/Yann_THESE/BESOIN_EAU/"
    d["SAVE_disk"]="G:/Yann_THESE/BESOIN_EAU/TRAITEMENT/PLOT/GEOSTAT_SOIL_GRID/"
    # all_features={'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_data_L93/'}
# =============================================================================
#     Zonal stats OTB
# =============================================================================
    # for bv in ['ADOUR','TARN','NESTE']:
    #     for file in all_features:
    #         for features in os.listdir(file):
    #             if "L93" in features : 
    #                 print (features)
    #                 os.environ['PYTHONPATH']='/home/pageot/anaconda3/envs/iota2/lib/python3.6/site-packages/iota2/:$PYTHONPATH'
    # #                 os.system('otbcli_ZonalStatistics -in '+str(file+features)+'  -inzone.vector.in /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/RPG/RPG_BV/RPG_SUMMER_2017_'+str(bv)+'_MAIZE_ONLY.shp -inzone.labelimage.in 1 -out.vector.filename /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/'+str(features[:-4])+'.shp' )
    #                 os.system('python /home/pageot/anaconda3/envs/iota2/lib/python3.6/site-packages/iota2/simplification/ZonalStats.py -wd /datalocal/vboxshare/THESE/BESOIN_EAU/DONNES_SOIL/SOIL_GRID/ -inr '+str(file+features)+' -shape /datalocal/vboxshare/THESE/CLASSIFICATION/TRAITEMENT/RPG/RPG_BV/RPG_SUMMER_2017_'+str(bv)+'_MAIZE_ONLY.shp -output /datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/'+str(features[:-4])+'vZSVT.shp -params 1:stats -syscall')
# =============================================================================
# Plot
# =============================================================================
    params=["RU"]
    for bv in ["ADOUR","TARN",'NESTE']:
        print(bv)
        for param in params:
            if param == 'texture' :
            #  Plot 3D textur
                depth={1:0,2:50,3:150,4:300,5:600,6:1000,7:2000}
                for i,z in zip(depth.keys(),depth.values()):
                    print(i)
                    # limon=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/SLTPPT_M_sl'+str(i)+'_250m_L93.shp')
                    # sand=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/SNDPPT_M_sl'+str(i)+'_250m_L93.shp')
                    # clay=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/CLYPPT_M_sl'+str(i)+'_250m_L93.shp')
                   
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # # ax = plt.axes(projection='3d')
                    # ydata = limon.mean_0
                    # xdata = sand.mean_0
                    # zdata = clay.mean_0
                    # c = np.abs(zdata)
                    # cmhot = plt.get_cmap("hot")
                    # fr=ax.scatter3D(xdata, ydata, zdata,c=zdata,cmap=cmhot) # plus couleur claire moins il y a de sable
                    # ax.set_title(r' depth : {} mm' .format(z))
                    # ax.set_xlabel(' % Sand')
                    # ax.set_zlabel('% Clay')
                    # ax.set_ylabel('% Limon')
                    # ax.view_init(60,0)
                    # cbar=plt.colorbar(fr)
                    # cbar.set_label("Values clay (%)")
                    # plt.show()

        # Create plt rotation scatter 3D and save in gif
        #             rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
        #             rot_animation.save('/datalocal/vboxshare/THESE/'+str(bv)+str(z)+'.gif', dpi=100, writer='imagemagick')
            elif param == "depth" :
                if bv =="NESTE":
                    Prof_NESTE_2017_Rigou=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_RIGOU/Profo_rac_NESTE_MAIZE_2017.shp")
                    plt.figure(figsize=(10,7))                    
                    # plt.hist(Prof_NESTE_2017_Rigou.ProfRacPot)
                    Prof_NESTE_2017_Rigou["Area"]=Prof_NESTE_2017_Rigou.area/10000
                    g=Prof_NESTE_2017_Rigou.groupby("ProfRacPot").sum()
                    plt.bar(g.index,g.Area,width=5)
                    plt.xlabel("Profondeur racinaire mm ")
                    plt.ylabel("Surface ha")
                    plt.savefig(d["SAVE"]+str(bv)+'profondeur_racinaire_Rigou.png')
                    depth_soilgrid=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/NESTE/BDTICM_M_250m_L93.shp')
                    plt.figure(figsize=(10,7))
                    sns.set(style="darkgrid")
                    sns.set_context('paper')
                    plt.hist(depth_soilgrid.mean_0,bins=100)
                    plt.xlabel("Profondeur soil mm ")
                    plt.ylabel("freq")
                    plt.savefig(d["SAVE"]+str(bv)+'profondeur_soil_Bedrock.png')
                else:
                    depth_soilgrid=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/BDTICM_M_250m_L93.shp')
                    plt.figure(figsize=(10,7))
                    sns.set(style="darkgrid")
                    sns.set_context('paper')
                    plt.hist(depth_soilgrid.mean_0,bins=100)
                    plt.xlabel("Profondeur soil mm ")
                    plt.ylabel("freq")
                    plt.savefig(d["SAVE"]+str(bv)+'profondeur_soil_Bedrock.png')
                
            elif param == 'class':
                classe=[]
                classenames=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNES_SOIL/SOIL_GRID/Classif/TAXNWRB_250m_ll.tif.csv")
                classenames=classenames[["Number","Group"]]
                classi=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/TAXNWRB_250_L93.shp')
                classe=list(set(classi.majority.dropna()))
                intcla=list(map(int,classe[:-1]))
                classesoil=classenames.loc[classenames.Number.isin(intcla)]
                classi["Area"]=classi.area/10000
                # freq_c=classi[classe[:-1]].count()
                freq_class=classi.groupby("majority").count().summer
                sum_surf_class=classi.groupby("majority").sum()
                plt.figure(figsize=(10,10))
                sns.set(style="darkgrid")
                sns.set_context('paper')
                # plt.bar(classesoil.Group,freq_class.iloc[:-1])
                plt.bar(classesoil.Group,sum_surf_class.Area.iloc[:-1])
                plt.xticks(rotation=45)
                plt.xlabel("classe FAO ")
                plt.ylabel("Surface en ha")
                plt.savefig(d["SAVE"]+str(bv)+'Repartion_class.png')
                depth_soilgrid=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/BDTICM_M_250m_L93.shp')
                a=geo.overlay(classi,depth_soilgrid,how='intersection')
                plt.figure(figsize=(10,10))
                sns.set(style="darkgrid")
                sns.set_context('paper')
                sns.boxplot(a.majority,a.mean_0)
                plt.xlabel("classe FAO ")
                plt.xticks(range(len(classesoil.Group)), classesoil.Group,rotation=45)
                plt.ylabel('Prof Soil')
                plt.savefig(d["SAVE"]+str(bv)+'boxplot_classe_profosoil.png')

            elif param == 'RU':
                classi=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/'+str(bv)+'/TAXNWRB_250_L93.shp')
                classe=list(set(classi.majority.dropna()))
                intcla=list(map(int,classe[:-1]))
                classesoil=classenames.loc[classenames.Number.isin(intcla)]
                for i in os.listdir('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/without_coarse'):
                    if".shp" in i :
                        print(i)
                        Hori=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/without_coarse/{}.shp'.format(i[:-4]))
                        RU_class=geo.overlay(classi,Hori,how='intersection')
                        plt.figure(figsize=(10,10))
                        sns.boxplot(RU_class.majority,RU_class.mean_0)
                        plt.xlabel("classe FAO ")
                        plt.xticks(range(len(classesoil.Group)), classesoil.Group,rotation=45)
                        plt.ylabel("RU_{} cm".format(i[13:15]))
                        plt.savefig(d["SAVE"]+str(bv)+'RU_{}cm_classe_FAO.png'.format(i[11:15]))
                        

    RUM_RRP_Gers=geo.read_file(d["PC_disk"]+'/TRAITEMENT/tmp/INTER_RPG2018_SOL_GERS.shp')
    for element in ["without_coarse","with_coarse"]:
        all=pd.DataFrame()
        for i in os.listdir(d["PC_disk"]+'/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/{}/'.format(element)):
            if".shp" in i :
                print (i)
                Hori=geo.read_file(d["PC_disk"]+'/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/{}/{}.shp'.format(element,i[:-4]))
                inter=geo.overlay(Hori,RUM_RRP_Gers,how='intersection')
                x2=inter[["RUM","mean_0",'ProfRacPot']]
                RU=x2.loc[x2.ProfRacPot==float(i[13:15])]
                all=all.append(RU)
        plt.figure(figsize=(12,10))
        sns.set(style="darkgrid")
        sns.set_context('paper')
        sns.boxplot(all.RUM,all.mean_0,fliersize=0.5,linewidth=1,hue_order=all.ProfRacPot,palette='RdBu')
        plt.xlabel("RUM RRP Gers")
        plt.ylim(0,146)
        plt.ylabel("RUM SG")
        plt.xticks(rotation=90)
        plt.title("{}".format(element))
        y_pos=range(len(sorted(list(set(all.RUM)))))
        for pro,j in zip(sorted(list(set(all.RUM))),range(len(sorted(list(set(all.RUM)))))):
            plt.text(x=y_pos[j]-0.5,y=all.mean_0.loc[all.RUM==pro].mean()+5,s=list(set(all.ProfRacPot.loc[all.RUM==pro])))
        plt.savefig(d["SAVE_disk"]+'boxplot_RU_{}_REF_SG_Gers_MAIZE2018.png'.format(element))
                # if len(RU.index) < 1:
                #     print ("Pas de comparaison")
                # else:
                #     print ('{} and {}'.format(element,i[13:15]))
                #     plt.figure(figsize=(10,10))
                #     sns.set(style="darkgrid")
                #     sns.set_context('paper')
                #     sns.jointplot(x=RU.RUM, y=RU.mean_0, color='b',kind="hex").set_axis_labels("RUM ref 0-{}cm".format(i[13:15]), "RU Soil_grid 0-{}cm {}".format(i[13:15],element))
                #     plt.savefig(d["SAVE"]+str(i[:-34])+'.png')

    


# =============================================================================
# Secteur analyse Rigou
# =============================================================================
    # colnames=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNES_SOIL/MAISEO_Rigou/Secteur_de_ref/caracterisation_sols_garganvillard_mean_value.csv").columns
    # colnames=colnames.drop(['ID_SOL', 'nom_sol', 'série','profondeur (cm) de la prospection racinaire selon analyse du profil type','profondeur (cm) du niveau imperméable selon notice SR','perméabilité mesurée K m/J'])
    # Horizon30_without_coarse=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_RIGOU/COMPAR_ref_secteur_SoilGrid/Graganvillard_RU__without_coarse_0_30_SG.shp')
    # Horizon30_without_coarse.columns= ['ID_SOL', 'SECT_REF', 'SURF_HA']+list(colnames)+['count', 'mean_0', 'stdev_0', 'min_0', 'max_0','geometry']

    # Horizon30_with_coarse=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_RIGOU/COMPAR_ref_secteur_SoilGrid/Graganvillard_RU_with_coarse_0_30_SG.shp')
    # Horizon30_with_coarse.columns= ['ID_SOL', 'SECT_REF', 'SURF_HA']+list(colnames)+['count', 'mean_0', 'stdev_0', 'min_0', 'max_0','geometry']

    # Horizon30_60_with_coarse=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_RIGOU/COMPAR_ref_secteur_SoilGrid/Graganvillard_RU_with_coarse_30_60_SG.shp')
    # Horizon30_60_with_coarse.columns= ['ID_SOL', 'SECT_REF', 'SURF_HA']+list(colnames)+['count', 'mean_0', 'stdev_0', 'min_0', 'max_0','geometry']

    # Horizon0_100_with_coarse=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/TEST_SEC_GRANV_RU__100.shp')
    # Horizon0_100_with_coarse.columns= ['ID_SOL', 'SECT_REF', 'SURF_HA']+list(colnames)+['count', 'mean_0', 'stdev_0', 'min_0', 'max_0','geometry']

# Plot
    # plt.scatter(Horizon30_with_coarse['RU 0-30cm (mm), par sol'],Horizon30_with_coarse.mean_0)
    # plt.xlabel("value_ref")
    # plt.ylabel("Soil_Grid")
    # plt.title("Horizon 0-30 cm")
    # # plt.xlim(30,100)
    # # plt.ylim(30,100)
    
    # plt.scatter(Horizon30_60_with_coarse['RU 30-55cm (mm), par sol'],Horizon30_60_with_coarse.mean_0)
    # plt.xlabel("value_ref")
    # plt.ylabel("Soil_Grid")
    # plt.title("Horizon 30-60 cm")
    # # plt.xlim(30,40)
    # # plt.ylim(30,40)
    # sns.jointplot(x=Horizon30_60_with_coarse['RU 30-55cm (mm), par sol'], y=Horizon30_60_with_coarse.mean_0, kind='scatter', s=50, color='b', edgecolor="skyblue", linewidth=1)
    
    # plt.scatter(Horizon0_100_with_coarse['RU 0-100cm (mm), par sol'],Horizon0_100_with_coarse.mean_0)
    # plt.figure(figsize=(10,10))
    # plt.xlabel("value_ref")
    # plt.ylabel("Soil_Grid")
    # plt.title("Horizon 0-100 cm")   
    # plt.xlim(20,150)
    # plt.ylim(20,150)
    
# =============================================================================
#     test RU en fonction des classe CArtes de sol du GERS 
# =============================================================================
    # Hori0_100=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/without_coarse/RU_Horizon_0_40cm_without_coarse_b_sup_SOIL_GRID_ss_nn_L93ZTotb_2018_ss_0.shp')
    # # Hori0_150=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/RU_RU_Horizon_0_150cm_with_coarse_b_sup_SOIL_GRID_L93.shp')
    # classi=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/NESTE/TAXNWRB_250_L93.shp')
    # NESTE_class=classi[["ID",'majority']]
    # NESTE_RU=Hori0_100[["mean_0","ID"]]
    # # NESTE_RU_150=Hori0_150[["mean_0","ID"]]
    # a=NESTE_RU.mean_0.astype(float)
    # b=NESTE_RU_150.mean_0.astype(float)
    # a[a==255.0]=pd.NaT
    # b[b==255.0]=pd.NaT
    # NESTE_class["RU100cm"]=a
    # NESTE_class["RU150cm"]=b
    # test=NESTE_class.groupby("majority").agg(lambda x:x.value_counts().index[0])
    # test_mean=NESTE_class.groupby("majority").mean()
    # # test["name"]=classesoil.Group[:-1]
    # plt.figure(figsize=(10,10))
    # sns.boxplot(NESTE_class.majority,NESTE_class.RU100cm)
    # plt.ylim(0,300)
    # plt.figure(figsize=(10,10))
    # sns.boxplot(NESTE_class.majority,NESTE_class.RU150cm)
    
    RUM_RRP_Gers=geo.read_file('/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/tmp/INTER_RPG2018_SOL_GERS.shp')
    RUM_0_70_SG=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/without_coarse/RU_Horizon_0_70cm_without_coarse_b_sup_SOIL_GRID_ss_nn_L93ZTotb_2018_ss_0.shp")
    RUM_0_30_SG=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/without_coarse/RU_Horizon_0_40cm_without_coarse_b_sup_SOIL_GRID_ss_nn_L93ZTotb_2018_ss_0.shp")
    RUM_0_50_SG=geo.read_file("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/GEOSTAT_SOIL_GRID/ALL_MAIZE_2018/without_coarse/RU_Horizon_0_50cm_without_coarse_b_sup_SOIL_GRID_ss_nn_L93ZTotb_2018_ss_0.shp")

    
    RUM_RRP=RUM_RRP_Gers[["ID","classe","ProfRacPot","RUM"]]
    # mode_RUM_id=RUM_RRP.groupby("ID").agg(lambda x:x.value_counts().index[0])

    plt.figure(figsize=(10,10))
    # sns.boxplot(x2.RUM.loc[x2.ProfRacPot==70.0],x2.mean_0,color='b')
    sns.boxplot(x2.RUM,x2.mean_030,color='y')
    # plt.xlim(0,140)
    plt.xticks(rotation=45)
    # inter70=geo.overlay(RUM_0_70_SG,RUM_RRP_Gers,how='intersection')
    x2=inter70[["RUM","mean_0",'ProfRacPot']]
    RU70=x2.loc[x2.ProfRacPot==70.0]
    sns.boxplot(x=RU70.RUM, y=RU70.mean_0)

    inter30=geo.overlay(RUM_0_30_SG,RUM_RRP_Gers,how='intersection')
    RU=inter30[["RUM","mean_0",'ProfRacPot']]
    RU30=RU.loc[RU.ProfRacPot==30.0]
    sns.boxplot(x=RU30.RUM, y=RU30.mean_0)

    inter50=geo.overlay(RUM_0_50_SG,RUM_RRP_Gers,how='intersection')
    RU=inter50[["RUM","mean_0",'ProfRacPot']]
    RU50=RU.loc[RU.ProfRacPot==50.0]
    sns.jointplot(x=RU50.RUM, y=RU50.mean_0, kind='scatter', s=50, color='b', edgecolor="skyblue")

    # inter60=geo.overlay(RUM_0_60_SG,RUM_RRP_Gers,how='intersection')
    # RU=inter60[["RUM","mean_0",'ProfRacPot']]
    # RU60=RU.loc[RU.ProfRacPot==60.0]
    # sns.jointplot(x=RU60.RUM, y=RU60.mean_0, kind='scatter', s=50, color='b', edgecolor="skyblue")
    
    a=x2.where(x2.ProfRacPot==70.0)
    b=RU.where(RU.ProfRacPot==30.0)
    a=a.dropna()
    b=b.dropna()
    test=pd.concat([a,b])    
    plt.figure(figsize=(10,10))

    sns.boxplot(test.RUM,test.mean_0,linewidth=0.5,color="white")
