#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:53:28 2021

@author: pageot
"""



import os
import sqlite3
import geopandas as geo 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from datetime import datetime, date, time, timezone


if __name__ == '__main__':
    d={}
    d["PC_disk"]="/run/media/pageot/Transcend/Yann_THESE/"
    d["PC_home"]="/mnt/d/THESE_TMP/"
    d["PC_home_Wind"]="D:/THESE_TMP/"

    d["PC_disk_water"]="/run/media/pageot/Transcend/Yann_THESE/BESOIN_EAU/BESOIN_EAU/"
    d["PC_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"

    df_surf=pd.read_csv(d["PC_disk"]+"/Manuscrit/Soutenance/Donnees_graphique/aquastat_surfaces_agricoles_world.csv",skiprows=2)
    surf_world=df_surf.groupby("Year").sum()
    
    df_pour_irr=pd.read_csv(d["PC_disk"]+"/Manuscrit/Soutenance/Donnees_graphique/Aquastat_terres_irriguees.csv",skiprows=2)
    pour_irr_world=df_pour_irr.groupby("Year").median()

    
    
