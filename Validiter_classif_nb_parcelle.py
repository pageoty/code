#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:00:45 2021

@author: pageot
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    
    df=pd.read_csv("/run/media/pageot/Transcend/Yann_THESE/RESULTAT_CLASSIFICATION/Vote_majoritaire_cours_saison_2017_Adour_data_ref.csv")
    lab_ref=df.groupby("labcroirr")
    ref_irri_maize=lab_ref.get_group(1)

    ref_irri_maize.mai_majori
