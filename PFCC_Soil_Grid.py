#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:43:09 2019

@author: pageoty
"""
# import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import otbApplication

if __name__ == "__main__":



    #reproject GSM tifs
    SOIL_GRID_raw_path='/datalocal/vboxshare/THESE/CLASSIFICATION/DONNES_SIG/CARTES_DES_SOLS/SOIL_GRID/PFCC_calculate/'
    SOIL_GRID_L93_path='/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/'




# dst_crs = 'EPSG:2154'
# for f in gsmfiles:
#     if "aux" not in f:
#         with rio.open(GSM_raw_path+f) as src:
#             transform, width, height = calculate_default_transform(
#                 src.crs, dst_crs, src.width, src.height, *src.bounds)
#             kwargs = src.meta.copy()
#             kwargs.update({
#                 'crs': dst_crs,
#                 'transform': transform,
#                 'width': width,
#                 'height': height
#             })
#             with rio.open(GSM_L93_path+f, 'w', **kwargs) as dst:
#                 for i in range(1, src.count + 1):
#                     reproject(
#                         source=rio.band(src, i),
#                         destination=rio.band(dst, i),
#                         src_transform=src.transform,
#                         src_crs=src.crs,
#                         dst_transform=transform,
#                         dst_crs=dst_crs,
#                         resampling=Resampling.nearest)
                    
    # compute FC and WP without Coarse       
    strat_soil_depth_b_inf=[0,50,100,150,300,400,1000]
    strat_soil_depth_b_sup=[50,100,150,300,400,1000,0]# exprimer en mm
    couchedict={'1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7'}
    for f,depth in zip(couchedict.keys(),strat_soil_depth_b_sup):
        print (f)
    #     App1= otbApplication.Registry.CreateApplication("BandMath")
    #     App1.SetParameterStringList("il",[SOIL_GRID_raw_path+'CLYPPT_M_sl'+str(f)+'_250m.tif',SOIL_GRID_raw_path+'SNDPPT_M_sl'+str(f)+'_250m.tif'])
    #     App1.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC/FC_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif')
    #     App1.SetParameterString("exp", "im1b1 == 255 || im2b1 ==255 ? -9999 : 0.278 + 0.00245*im1b1 - 0.00135*im2b1 " )
    #     App1.ExecuteAndWriteOutput()
    
    #     App2= otbApplication.Registry.CreateApplication("BandMath")
    #     App2.SetParameterStringList("il",[SOIL_GRID_raw_path+'CLYPPT_M_sl'+str(f)+'_250m.tif',SOIL_GRID_raw_path+'SNDPPT_M_sl'+str(f)+'_250m.tif'])
    #     App2.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP/WP_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif')
    #     App2.SetParameterString("exp"," im1b1 == 255 || im2b1 == 255 ? -9999 : 0.08 + 0.00401*im1b1 - 0.000293*im2b1")
    #     App2.ExecuteAndWriteOutput()
    
    # Calcul de la RU
      # equation du calcul de la RU AWC = (FC -WP) *(1- coarse) or AWC = (FC -WP) *(1- coarse)*bulk density
        print(SOIL_GRID_L93_path+'FC_'+str(f)+'_Ptran_SOIL_GRID.tif',SOIL_GRID_L93_path+'WP_'+str(f)+'_Ptran_SOIL_GRID.tif','/datalocal/vboxshare/THESE/BESOIN_EAU/DONNES_SOIL/SOIL_GRID/COARSE/CRFVOL_M_sl'+str(f)+'_250m.tif' )
        
        coarse= otbApplication.Registry.CreateApplication("BandMath")
        coarse.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/DONNES_SOIL/SOIL_GRID/COARSE/CRFVOL_M_sl'+str(f)+'_250m.tif'])
        coarse.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/coarse'+str(f)+'_SOIL_GRID_ss_nn.tif')
        coarse.SetParameterString("exp"," im1b1 == 255  ? -9999 :im1b1")
        coarse.ExecuteAndWriteOutput()
        
        
        term1= otbApplication.Registry.CreateApplication("BandMath")
        term1.SetParameterStringList("il",[SOIL_GRID_L93_path+'FC/FC_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif',SOIL_GRID_L93_path+'WP/WP_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif'])
        term1.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup'+str(f)+'_SOIL_GRID_ss_nn.tif')
        term1.SetParameterString("exp","(im1b1-im2b1)*{}".format(depth))
        term1.ExecuteAndWriteOutput()
        
        
        term2=otbApplication.Registry.CreateApplication("BandMath")
        term2.SetParameterStringList("il",[SOIL_GRID_L93_path+'FC/FC_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif',SOIL_GRID_L93_path+'WP/WP_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif','/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/coarse'+str(f)+'_SOIL_GRID_ss_nn.tif'])
        term2.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_depth'+str(f)+'_SOIL_GRID_ss_nn.tif')
        term2.SetParameterString("exp","(im1b1-im2b1)*(1-im3b1/100)")
        term2.ExecuteAndWriteOutput() 
        
        #  Add élément grossier dans l'équation (term coarse/100)  obtenir une faction d'élement grossier
        term2=otbApplication.Registry.CreateApplication("BandMath")
        term2.SetParameterStringList("il",[SOIL_GRID_L93_path+'FC/FC_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif',SOIL_GRID_L93_path+'WP/WP_'+str(f)+'_Ptran_SOIL_GRID_ss_nn.tif','/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/coarse'+str(f)+'_SOIL_GRID_ss_nn.tif'])
        term2.SetParameterString("exp","((im1b1-im2b1)*(1-im3b1/100))*{}".format(depth))
        term2.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup'+str(f)+'_SOIL_GRID_ss_nn.tif')
        term2.ExecuteAndWriteOutput()
        
        # Summ des RU sur la colonne de sol (0,2000)
    # RUtest= otbApplication.Registry.CreateApplication("BandMath")
    # RUtest.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup5_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup6_SOIL_GRID.tif'])
    # RUtest.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_soil_0_2m_with_coarse_SOIL_GRID.tif')
    # RUtest.SetParameterString("exp","(im1b1+im2b1+im3b1+im4b1+im5b1+im6b1)")
    # RUtest.ExecuteAndWriteOutput()
    

    
    RU_horiz1_with_coarse=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz1_with_coarse.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif'])
    RU_horiz1_with_coarse.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_30cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz1_with_coarse.SetParameterString("exp","(im1b1+im2b1+im3b1)")
    RU_horiz1_with_coarse.ExecuteAndWriteOutput()
    
    RU_horiz0_60=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_60.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_60.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_60cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_60.SetParameterString("exp","(im1b1+im2b1+im3b1+im4b1)")
    RU_horiz0_60.ExecuteAndWriteOutput()
    
    RU_horiz0_40=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_40.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_40.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_40cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_40.SetParameterString("exp","((im1b1+im2b1+im3b1)+im4b1/3)")
    RU_horiz0_40.ExecuteAndWriteOutput()
    
    RU_horiz0_50=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_50.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_50.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_50cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_50.SetParameterString("exp","((im1b1+im2b1+im3b1)+im4b1/1.5)")
    RU_horiz0_50.ExecuteAndWriteOutput()
    
    # depth 0 at 1000 cm
    RU_horiz0_100=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_100.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup5_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_100.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_100cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_100.SetParameterString("exp","(im1b1+im2b1+im3b1+im4b1+im5b1)")
    RU_horiz0_100.ExecuteAndWriteOutput()
    
    RU_horiz0_70=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_70.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup5_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_70.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_70cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_70.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1)+im5b1/4)")
    RU_horiz0_70.ExecuteAndWriteOutput()
    
    RU_horiz0_85=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_85.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup5_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_85.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_85cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_85.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1)+im5b1/1.6)")
    RU_horiz0_85.ExecuteAndWriteOutput()
    
    # RU_horiz0_100_sum_FC_WP=otbApplication.Registry.CreateApplication("BandMath") # Sortie identique entre RU_horiz0_100 & RU_horiz0_100_sum_FC_WP (intégre la profondeur avant ou apres auncune différence)
    # RU_horiz0_100_sum_FC_WP.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_depth1_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_depth2_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_depth3_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_depth4_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_depth5_SOIL_GRID.tif'])
    # RU_horiz0_100_sum_FC_WP.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_100cm_mean_FC_WP_b_sup_SOIL_GRID.tif')
    # RU_horiz0_100_sum_FC_WP.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1)/5)*1000")
    # RU_horiz0_100_sum_FC_WP.ExecuteAndWriteOutput()
    
    
    RU_horiz0_150=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_150.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup5_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup6_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_150.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_150cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_150.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1)+im6b1/2)")
    RU_horiz0_150.ExecuteAndWriteOutput()
    
    
    RU_horiz0_180=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_180.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup5_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_with_depth_b_sup6_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_180.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_180cm_with_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_180.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1)+im6b1/1.25)")
    RU_horiz0_180.ExecuteAndWriteOutput()
    
    
# =============================================================================
#     Without granulo
# =============================================================================
    
    RU_horiz1_with_coarse=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz1_with_coarse.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif'])
    RU_horiz1_with_coarse.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_30cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz1_with_coarse.SetParameterString("exp","(im1b1+im2b1+im3b1)")
    RU_horiz1_with_coarse.ExecuteAndWriteOutput()
    
    RU_horiz0_60=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_60.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_60.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_60cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_60.SetParameterString("exp","(im1b1+im2b1+im3b1+im4b1)")
    RU_horiz0_60.ExecuteAndWriteOutput()
    
    RU_horiz0_40=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_40.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_40.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_40cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_40.SetParameterString("exp","((im1b1+im2b1+im3b1)+im4b1/3)")
    RU_horiz0_40.ExecuteAndWriteOutput()
    
    RU_horiz0_50=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_50.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_50.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_50cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_50.SetParameterString("exp","((im1b1+im2b1+im3b1)+im4b1/1.5)")
    RU_horiz0_50.ExecuteAndWriteOutput()
    
    # depth 0 at 1000 cm
    RU_horiz0_100=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_100.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup5_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_100.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_100cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_100.SetParameterString("exp","(im1b1+im2b1+im3b1+im4b1+im5b1)")
    RU_horiz0_100.ExecuteAndWriteOutput()
    
    RU_horiz0_70=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_70.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup5_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_70.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_70cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_70.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1)+im5b1/4)")
    RU_horiz0_70.ExecuteAndWriteOutput()
    
    RU_horiz0_85=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_85.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup5_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_85.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_85cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_85.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1)+im5b1/1.6)")
    RU_horiz0_85.ExecuteAndWriteOutput()
    
    # RU_horiz0_100_sum_FC_WP=otbApplication.Registry.CreateApplication("BandMath") # Sortie identique entre RU_horiz0_100 & RU_horiz0_100_sum_FC_WP (intégre la profondeur avant ou apres auncune différence)
    # RU_horiz0_100_sum_FC_WP.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID.tif',
    #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup5_SOIL_GRID.tif'])
    # RU_horiz0_100_sum_FC_WP.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_100cm_mean_FC_WP_b_sup_SOIL_GRID.tif')
    # RU_horiz0_100_sum_FC_WP.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1)/5)*1000")
    # RU_horiz0_100_sum_FC_WP.ExecuteAndWriteOutput()
    
    
    RU_horiz0_150=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_150.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup5_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup6_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_150.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_150cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_150.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1)+im6b1/2)")
    RU_horiz0_150.ExecuteAndWriteOutput()
    
    
    RU_horiz0_180=otbApplication.Registry.CreateApplication("BandMath")
    RU_horiz0_180.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup1_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup2_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup3_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup4_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup5_SOIL_GRID_ss_nn.tif',
                                        '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_without_coarse_with_depth_b_sup6_SOIL_GRID_ss_nn.tif'])
    RU_horiz0_180.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/RU_Horizon_0_180cm_without_coarse_b_sup_SOIL_GRID_ss_nn.tif')
    RU_horiz0_180.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1)+im6b1/1.25)")
    RU_horiz0_180.ExecuteAndWriteOutput()
    
#     # # WP_horiz0_200=otbApplication.Registry.CreateApplication("BandMath")
#     # # WP_horiz0_200.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP/WP_1_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP/WP_2_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP/WP_3_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP/WP_4_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP/WP_5_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP/WP_6_Ptran_SOIL_GRID.tif'])
#     # # WP_horiz0_200.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/Mean_WP_Horizon_0_200cm_b_sup_Ptran_SOIL_GRID.tif')
#     # # WP_horiz0_200.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1+im6b1)/6)")
#     # # WP_horiz0_200.ExecuteAndWriteOutput()
    
#     # # WP_horiz0_150=otbApplication.Registry.CreateApplication("BandMath")
#     # # WP_horiz0_150.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/Mean_WP_Horizon_0_200cm_b_sup_Ptran_SOIL_GRID.tif'])
#     # # WP_horiz0_150.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/WP_mean_Horizon_0_150cm_b_sup_Ptran_SOIL_GRID.tif')
#     # # WP_horiz0_150.SetParameterString("exp","(im1b1*1500/2000)")
#     # # WP_horiz0_150.ExecuteAndWriteOutput()
    
    
#     # # FC_horiz0_200=otbApplication.Registry.CreateApplication("BandMath")
#     # # FC_horiz0_200.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC/FC_1_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC/FC_2_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC/FC_3_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC/FC_4_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC/FC_5_Ptran_SOIL_GRID.tif',
#     # #                                     '/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC/FC_6_Ptran_SOIL_GRID.tif'])
#     # # FC_horiz0_200.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/Mean_FC_Horizon_0_200cm_b_sup_Ptran_SOIL_GRID.tif')
#     # # FC_horiz0_200.SetParameterString("exp","((im1b1+im2b1+im3b1+im4b1+im5b1+im6b1)/6)")
#     # # FC_horiz0_200.ExecuteAndWriteOutput()
    
#     # # FC_horiz0_150=otbApplication.Registry.CreateApplication("BandMath")
#     # # FC_horiz0_150.SetParameterStringList("il",['/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/Mean_FC_Horizon_0_200cm_b_sup_Ptran_SOIL_GRID.tif'])
#     # # FC_horiz0_150.SetParameterString("out",'/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/SOIL_GRID/PFCC/FC_mean_Horizon_0_150cm_b_sup_Ptran_SOIL_GRID.tif')
#     # # FC_horiz0_150.SetParameterString("exp","(im1b1*1500/2000)")
#     # # FC_horiz0_150.ExecuteAndWriteOutput()
    
    
#     # Ou pour affinier la profondeur de Sol en fonctino de la couche depth de soilGrid Utliser la rasterio 