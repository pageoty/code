#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:16:05 2020

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
import pickle
import datetime 
from scipy import *
from scipy import stats
from pylab import *
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
from math import log10, cos, sin, asin, sqrt, exp,acos
from collections import namedtuple
import scipy.io
def predict(x):
   return slope * x + intercept

def penman_monteith(doy, LAT, ELEV, TMIN, TMAX, AVRAD, VAP, WIND2):
    """Calculates reference ET0 based on the Penman-Monteith model.

     This routine calculates the potential evapotranspiration rate from
     a reference crop canopy (ET0) in mm/d. For these calculations the
     analysis by FAO is followed as laid down in the FAO publication
     `Guidelines for computing crop water requirements - FAO Irrigation
     and drainage paper 56 <http://www.fao.org/docrep/X0490E/x0490e00.htm#Contents>`_

    Input variables::

        DAY   -  Python datetime.date object                   -
        LAT   -  Latitude of the site                        degrees
        ELEV  - Elevation above sea level                      m
        TMIN  - Minimum temperature                            C
        TMAX  - Maximum temperature                            C
        AVRAD - Daily shortwave radiation                   J m-2 d-1
        VAP   - 24 hour average vapour pressure               hPa
        WIND2 - 24 hour average windspeed at 2 meter          m/s

    Output is:

        ET0   - Penman-Monteith potential transpiration
                rate from a crop canopy                     [mm/d]
    """
    global DAYL
    # psychrometric instrument constant (kPa/Celsius)
    PSYCON = 0.665
    # albedo and surface resistance [sec/m] for the reference crop canopy
    REFCFC = 0.23; CRES = 70.
    # latent heat of evaporation of water [J/kg == J/mm] and
    LHVAP = 2.45E6
    # Stefan Boltzmann constant (J/m2/d/K4, e.g multiplied by 24*60*60)
    STBC = 4.903E-3
    # Soil heat flux [J/m2/day] explicitly set to zero
    G = 0.
    
    # mean daily temperature (Celsius)
    TMPA = (TMIN+TMAX)/2.

    # Vapour pressure to kPa
    VAP = hPa2kPa(VAP)
    # atmospheric pressure (kPa)
    T = Celsius2Kelvin(TMPA)
    PATM = 101.3 * pow((T - (0.0065*ELEV))/T, 5.26)

    # psychrometric constant (kPa/Celsius)
    GAMMA = PSYCON * PATM * 1.0E-3

    # Derivative of SVAP with respect to mean temperature, i.e.
    # slope of the SVAP-temperature curve (kPa/Celsius);
    SVAP_TMPA = SatVapourPressure(TMPA)
    DELTA = (4098. * SVAP_TMPA)/pow((TMPA + 237.3), 2)

    # Daily average saturated vapour pressure [kPa] from min/max temperature
    SVAP_TMAX = SatVapourPressure(TMAX)
    SVAP_TMIN = SatVapourPressure(TMIN)
    SVAP = (SVAP_TMAX + SVAP_TMIN) / 2.
    # measured vapour pressure not to exceed saturated vapour pressure
    VAP = np.minimum(VAP, SVAP)

    # Longwave radiation according at Tmax, Tmin (J/m2/d)
    # and preliminary net outgoing long-wave radiation (J/m2/d)
    STB_TMAX = STBC * pow(Celsius2Kelvin(TMAX), 4)
    STB_TMIN = STBC * pow(Celsius2Kelvin(TMIN), 4)
    RNL_TMP = ((STB_TMAX + STB_TMIN) / 2.) * (0.34 - 0.14 * np.sqrt(VAP))

    # Clear Sky radiation [J/m2/DAY] from Angot TOA radiation
    # the latter is found through a call to astro()
    r = astro(doy, LAT, AVRAD)
    CSKYRAD = (0.75 + (2e-05 * ELEV)) * r.ANGOT
    # if CSKYRAD > 0:
        # Final net outgoing longwave radiation [J/m2/day]
    RNL = RNL_TMP * (1.35 * (AVRAD/CSKYRAD) - 0.35)

    # radiative evaporation equivalent for the reference surface
    # [mm/DAY]
    RN = ((1-REFCFC) * AVRAD - RNL)/LHVAP

    # aerodynamic evaporation equivalent [mm/day]
    EA = ((900./(TMPA + 273)) * WIND2 * (SVAP - VAP))

    # Modified psychometric constant (gamma*)[kPa/C]
    MGAMMA = GAMMA * (1. + (CRES/208.*WIND2))

    # Reference ET in mm/day
    ET0 = (DELTA * (RN-G))/(DELTA + MGAMMA) + (GAMMA * EA)/(DELTA + MGAMMA)
        # ET0 = np.maximum(0., ET0)
    # else:
    #     ET0 = np.nan
    return ET0

def astro(latitude, radiation,doy, _cache={}):
    """python version of ASTRO routine by Daniel van Kraalingen.
    
    This subroutine calculates astronomic daylength, diurnal radiation
    characteristics such as the atmospheric transmission, diffuse radiation etc.

    :param day:         date/datetime object
    :param latitude:    latitude of location
    :param radiation:   daily global incoming radiation (J/m2/day)

    output is a `namedtuple` in the following order and tags::

        DAYL      Astronomical daylength (base = 0 degrees)     h      
        DAYLP     Astronomical daylength (base =-4 degrees)     h      
        SINLD     Seasonal offset of sine of solar height       -      
        COSLD     Amplitude of sine of solar height             -      
        DIFPP     Diffuse irradiation perpendicular to
                  direction of light                         J m-2 s-1 
        ATMTR     Daily atmospheric transmission                -      
        DSINBE    Daily total of effective solar height         s
        ANGOT     Angot radiation at top of atmosphere       J m-2 d-1
 
    Authors: Daniel van Kraalingen
    Date   : April 1991
 
    Python version
    Author      : Allard de Wit
    Date        : January 2011
    """
    global DAYL
    global ANGOT
    global DAYLP
    global SINLD
    global COSLD
    global DIFPP
    global ATMTR
    global DSINBE
    # Check for range of latitude
    # if abs(latitude) > 90.:
    #     msg = "Latitude not between -90 and 90"
    #     raise RuntimeError(msg)
    LAT = latitude
        
    # Determine day-of-year (IDAY) from day
    IDAY = doy
    
    # reassign radiation
    AVRAD = radiation

    # Test if variables for given (day, latitude, radiation) were already calculated
    # in a previous run. If not (e.g. KeyError) calculate the variables, store
    # in cache and return the value.
    # try:
    #     return _cache[(IDAY, LAT, AVRAD)]
    # except KeyError:
    #     pass

    # constants
    RAD = 0.0174533
    PI = 3.1415926
    ANGLE = -4.

    # map python functions to capitals
    SIN = sin
    COS = cos
    ASIN = asin
    REAL = float
    SQRT = np.sqrt
    ABS = abs

    # Declination and solar constant for this day
    DEC = -ASIN(SIN(23.45*RAD)*COS(2.*PI*(REAL(IDAY)+10.)/365.))
    SC  = 1370.*(1.+0.033*COS(2.*PI*REAL(IDAY)/365.))

    # calculation of daylength from intermediate variables
    # SINLD, COSLD and AOB
    SINLD = SIN(RAD*LAT)*SIN(DEC)
    COSLD = COS(RAD*LAT)*COS(DEC)
    AOB = SINLD/COSLD

    # For very high latitudes and days in summer and winter a limit is
    # inserted to avoid math errors when daylength reaches 24 hours in 
    # summer or 0 hours in winter.

    # Calculate solution for base=0 degrees
    if abs(AOB) <= 1.0:
        DAYL  = 12.0*(1.+2.*ASIN(AOB)/PI)
        # integrals of sine of solar height
        DSINB  = 3600.*(DAYL*SINLD+24.*COSLD*SQRT(1.-AOB**2)/PI)
        DSINBE = 3600.*(DAYL*(SINLD+0.4*(SINLD**2+COSLD**2*0.5))+
                 12.*COSLD*(2.+3.*0.4*SINLD)*SQRT(1.-AOB**2)/PI)
        ANGOT = SC*DSINB
    else:
        if AOB >  1.0: 
            DAYL=24.0
            DSINB = 3600.0*(24.0*SINLD)
            DSINBE = 3600.0*(24.0*(SINLD+0.4*(SINLD**2+COSLD**2*0.5)))
            ANGOT = SC*3600.0*(24.0*SINLD)
        elif AOB < -1.0: 
            DAYL=0.0
            DSINB = 3600.0*(0.0*SINLD)
            DSINBE = 3600.0*(0.0*(SINLD+0.4*(SINLD**2+COSLD**2*0.5)))
            ANGOT = SC*3600.0*(0.0*SINLD)
        # integrals of sine of solar height	


    # Calculate solution for base=-4 (ANGLE) degrees
    AOB_CORR = (-SIN(ANGLE*RAD)+SINLD)/COSLD
    if abs(AOB_CORR) <= 1.0:
        DAYLP = 12.0*(1.+2.*ASIN(AOB_CORR)/PI)
    elif AOB_CORR > 1.0:
        DAYLP = 24.0
    elif AOB_CORR < -1.0:
        DAYLP = 0.0

    # extraterrestrial radiation and atmospheric transmission
    # ANGOT = SC*DSINB
    # Check for DAYL=0 as in that case the angot radiation is 0 as well
    if DAYL > 0.0:
        ATMTR = AVRAD/ANGOT
    else:
        ATMTR = 0.

    # estimate fraction diffuse irradiation
    if (ATMTR > 0.75):
        FRDIF = 0.23
    elif (ATMTR <= 0.75) and (ATMTR > 0.35):
        FRDIF = 1.33-1.46*ATMTR
    elif (ATMTR <= 0.35) and (ATMTR > 0.07):
        FRDIF = 1.-2.3*(ATMTR-0.07)**2
    else:  # ATMTR <= 0.07
        FRDIF = 1.

    DIFPP = FRDIF*ATMTR*0.5*SC
    
    # Pack return values in namedtuple, add to cache and return
    astro_nt = namedtuple("AstroResults","DAYL, DAYLP, SINLD, COSLD, DIFPP, "
                                         "ATMTR, DSINBE, ANGOT")
    retvalue = astro_nt(DAYL, DAYLP, SINLD, COSLD, DIFPP, ATMTR, DSINBE, ANGOT)
    _cache[(IDAY, LAT, AVRAD)] = retvalue

    return retvalue
        
def et0_pm_simple(day1,alt,hmes,lat,tmoy,tmin,tmax,vv,hrmoy,hrmin,hrmax,rs):
    """
    Parameters
    ----------
    day1 : date
        DESCRIPTION.
    alt : Altitude
        DESCRIPTION.
    hmes : Hauteurs de la mesure
        DESCRIPTION.
    lat : latitude (43-lam)
        DESCRIPTION.
    tmoy : Tmepérature moyenne
        DESCRIPTION.
    tmin : Température min
        DESCRIPTION.
    tmax : Température max
        DESCRIPTION.
    vv : vitesse du vent (WS)
        DESCRIPTION.
    hrmoy : humidité relative moyenne
        DESCRIPTION.
    hrmin : humidité relative min
        DESCRIPTION.
    hrmax : humidité relative max
        DESCRIPTION.
    rs : rayonnement solaire
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if ((hrmax-hrmin)<.1 or tmin==tmax): 
        return(0)
    try: 
        conv_rad=lat * pi / 180.
        u2= vv * 4.87 / log(67.8*hmes-5.42) # hmes ?? dans formule u2 = fait intervenir l'altitude
        rs_mj= rs*24*3600*0.000001
        dr= 1+0.033*cos(2*pi*day1/365)
        d= 0.409*sin((2*pi*day1/365)-1.39)
        ws= acos(-tan(conv_rad)*tan(d))
        ra= (24*60/pi)*0.082*dr*(ws*sin(conv_rad)*sin(d)+cos(conv_rad)*cos(d)*sin(ws))
        rso= ra*(0.75+0.00002*alt)
        es= (0.6108*exp(17.27*tmin/(tmin+237.3))+0.6108*exp(17.27*tmax/(tmax+237.3)))/2
        ea= (hrmin*0.6108*exp(17.27*tmax/(tmax+237.3))+hrmax*0.6108*exp(17.27*tmin/(tmin+237.3)))/(2*100)
        rnl= 4.9*pow(10,-9)*0.5*(pow((tmin+273),4)+pow((tmax+273),4))*(0.34-0.14*sqrt(ea))*(1.35*(rs_mj/rso)-0.35)
        rn=(1-0.23)*rs_mj-rnl
        delta= 4098*0.6108*exp(17.27*tmoy /(tmoy +237.3))/pow((tmoy +237.3),2)
        et0= (0.408*delta*(rn)+(900*0.063/(tmoy+273))*u2*(es-ea))/(delta+0.063*(1+0.34*u2))
        return(et0)
    except:
        return(0)

def et0_pm_simple_kelv(day1,alt,hmes,lat,tmoy,tmin,tmax,vv,hrmoy,hrmin,hrmax,rs):
    """


    """
    if ((hrmax-hrmin)<.1 or tmin==tmax): 
        return(0)
    try: 
        conv_rad=lat * pi / 180.
        u2= vv * 4.87 / log(67.8*hmes-5.42) # hmes ?? dans formule u2 = fait intervenir l'altitude
        rs_mj= rs*24*3600*0.000001
        dr= 1+0.033*cos(2*pi*day1/365)
        d= 0.409*sin((2*pi*day1/365)-1.39)
        ws= acos(-tan(conv_rad)*tan(d))
        ra= (24*60/pi)*0.082*dr*(ws*sin(conv_rad)*sin(d)+cos(conv_rad)*cos(d)*sin(ws))
        rso= ra*(0.75+0.00002*alt)
        es= (0.6108*exp(17.27*tmin-273.3/(tmin))+0.6108*exp(17.27*tmax-273.3/(tmax)))/2
        ea= (hrmax/100*0.6108*exp(17.27*tmin-273.3/(tmin))+hrmin/100*0.6108*exp(17.27*tmax-273.3/(tmax)))/2
        rnl= 4.9*pow(10,-9)*0.5*(pow((tmin),4)+pow((tmax),4))*(0.34-0.14*sqrt(ea))*(1.35*(rs_mj/rso)-0.35)
        rn=(1-0.23)*rs_mj-rnl
        delta= 4098*0.6108*exp(17.27*tmoy-273.3 /(tmoy ))/pow((tmoy ),2)
        et0= (0.408*delta*(rn)+(900*0.063/(tmoy))*u2*(es-ea))/(delta+0.063*(1+0.34*u2))
        return(et0)
    except:
        return(0)

if __name__ == '__main__':
    d={}
    d["path_labo"]="/datalocal/vboxshare/THESE/BESOIN_EAU/"
    d["path_PC"]="D:/THESE_TMP/RUNS_SAMIR/R12/Inputdata/"
    d["PC_disk"]="H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/"
    Celsius2Kelvin = lambda x: x + 273.16
    hPa2kPa = lambda x: x/10.
    # Saturated Vapour pressure [kPa] at temperature temp [C]
    SatVapourPressure = lambda temp: 0.6108 * exp((17.27 * temp) / (237.3 + temp))
# =============================================================================
# Validation modelisation ETR     
# =============================================================================
    # Prépartion des datas Eddy-co au format journalière
      # Pour 2017 et autre année
    # for y in os.listdir(d["path_labo"]+"/DATA_ETR_CESBIO/DATA_LAM_lec_python/"):
    for y in os.listdir("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/PARCELLE_LABO/FLUX_ETR/Corr_fluxes/"):
        print(y)
        # years=y[5:9]
        # if years =="2017":
            # LE_lam=pd.read_csv(d["path_labo"]+"/DATA_ETR_CESBIO/DATA_LAM_lec_python/"+str(y),encoding = 'utf-8',delimiter=",")
        # LE_lam=pd.read_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/DONNEES_RAW/PARCELLE_LABO/FLUX_ETR/Corr_fluxes/"+y,encoding = 'utf-8',delimiter=",")
        LE_lam=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/PARCELLE_LABO/FLUX_ETR/Corr_fluxes/"+y,encoding = 'utf-8',delimiter=",")
        LE_lam.drop(0,inplace=True)
        LE_lam["time"]=LE_lam["Time"].astype(str)
        LE_lam["LE_Bowen"]=LE_lam["LE Bowen"].astype(float)
        LE_lam["LE_Res"]=LE_lam["LE Res"].astype(float)
        LE_lam["date"]=LE_lam["time"].apply(lambda x:x[0:10])
        LE_lam["time_hours"]=LE_lam["time"].apply(lambda x:x[10:-3]).replace(":",'.')
        LE_lam["date"]=pd.to_datetime(LE_lam["date"],format="%d/%m/%Y")
        LE_lam.drop(columns=["Time"],inplace=True)
        # Suppression des flux nocture
        LE_lam=LE_lam.loc[(LE_lam.time_hours >=' 06:00') & (LE_lam.time_hours <=' 19:00')]
        LE_lam_day=LE_lam.groupby("date")["LE_Bowen"].mean()
        ETR_lam_day=LE_lam_day*0.0352
        ETR_lam_day[ETR_lam_day < -1]=pd.NaT
        ETR_lam_day=pd.DataFrame(ETR_lam_day)
        ETR_lam_day.plot()
        ETR_lam_day
        ETR_lam_day.to_csv("H:/Yann_THESE/BESOIN_EAU/BESOIN_EAU/TRAITEMENT/DATA_VALIDATION/DATA_ETR_CESBIO/DATA_ETR_corr_maize_irri/ETR_maize_irri"+str(y[:-4])+".csv")

    ##### mesu diff ratio Bowen et LE nn corr
    diff=LE_lam.LE_Bowen-LE_lam.LE_Res
  # Pour la station de Grignon gestion des LE en ETR
    for y in ["2019"]:
        print (y)
        # df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/DATA_PARCELLE_GRIGNON/RAW_DATA/DATA_ICOS_ECO/EFDC_L2_Flx_FRGri_"+str(y)+"_30m.txt",sep=',')
        # df["time"]=df["TIMESTAMP_START"].astype(str)
        # df["date"]=df["time"].apply(lambda x:x[0:8])
        # df["time_hours"]=df["time"].apply(lambda x:x[8:]).astype(int)
        df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/DATA_PARCELLE_GRIGNON/RAW_DATA/DATA_ICOS_ECO/FR-Gri_Flux_2019.csv")
        df["time"]=df["TIMESTAMP"].astype(str)
        df["date"]=df["time"].apply(lambda x:x[0:8])
        df["time_hours"]=df["time"].apply(lambda x:x[8:])
        df["date"]=pd.to_datetime(df["date"],format="%Y%m%d")
        df.drop(columns=["TIMESTAMP"],inplace=True)
        # Suppression des flux nocture
        df=df.loc[(df.time_hours >='0600.0') & (df.time_hours <='1900.0')]
        df=df.groupby('date').mean()
        df["LE"]=df.eval("LE_1_1_1*0.0352")
        df[df.LE < -1]=np.nan
        df.drop(columns=['CO2_1_1_1', 'H2O_1_1_1', 'ZL_1_1_1', 'FC_1_1_1', 'FC_SSITC_TEST',
        'H_1_1_1', 'H_SSITC_TEST', 'LE_1_1_1', 'LE_SSITC_TEST', 'TAU_1_1_1',
        'TAU_SSITC_TEST', 'MO_LENGTH_1_1_1', 'USTAR_1_1_1', 'U_SIGMA_1_1_1',
        'V_SIGMA_1_1_1', 'W_SIGMA_1_1_1', 'WS_1_2_1', 'WD_1_2_1', 'SC_1_1_1'],inplace=True)
        df.LE.plot()
        df.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_ETR_GRIGNON/DATA_ETR_GRIGNON_ICOS/ETR_GRIGNON"+str(y)+".csv")
    
#    Pour 2019
    ETR_lam=pd.read_csv(d["path_labo"]+"/DATA_ETR_CESBIO/DATA_LAM_lec_python/eddypro_FR-Lam_full_output_2020-01-28T012345_adv.csv")
    ETR_lam["date"]=pd.to_datetime(ETR_lam["date"],format='%Y-%m-%d')
    ETR_lam=ETR_lam.loc[(ETR_lam.time >='06:00') & (ETR_lam.time <='19:00')]
    ETR_lam_day=ETR_lam.groupby("date")["LE"].mean()
    ETR_lam_day=ETR_lam_day*0.0352
    ETR_lam_day[ETR_lam_day < -1]=pd.NaT
    ETR_lam_day.plot()
    ETR_lam_day.to_csv(d["path_labo"]+"/DATA_ETR_CESBIO/DATA_ETR_LAM/ETR_LAM2019.csv")

#     ## Récuparation date végétation sur ETR 
# # =============================================================================
# # Pluvio satation mise en forme
# # =============================================================================
#  # LAM
    meteo_lam_station=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DATA_ETR_CESBIO/DATA_LE_LAM_original/eddypro_FR-Lam_biomet_2020-01-28T012345_adv_ss_unit.csv")
    meteo_lam_station=meteo_lam_station.iloc[:-49]
    Meteo=meteo_lam_station[["date","P_1_1_1"]]
    Meteo=Meteo.iloc[1:]
    Meteo["Prec"]=Meteo.P_1_1_1.astype(float)
    Meteo[Meteo.Prec<=-9998]=np.nan
    Meteo["Prec"]=Meteo.Prec*1000
    Meteo_lam=Meteo.groupby("date").sum()
    Meteo_lam=Meteo_lam.iloc[:-2]
    Meteo_lam.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/Meteo_station_2019.csv")
    
    # PLUVIO_STAT LAM autres anénes
    for y in ["2006","2008","2010","2012","2014","2015"]:
        mat = scipy.io.loadmat('/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/PARCELLE_LABO/DATA_METEO_LAM/LAM_'+y+'_IS.mat')
        Plui=pd.DataFrame(mat["Rain"])
        Plui["date"]=pd.to_datetime(np.arange(0,Plui.shape[0]), unit='D',origin=pd.Timestamp(str(y)+'-01-01'))
        Plui.columns=["Prec",'date']
        Plui.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/Meteo_station_"+y+".csv")
    # df=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/PARCELLE_LABO/DATA_METEO_LAM/LAM_MTO_N3_2005-2016_GP.csv",encoding = 'utf-8',sep=',')
    # df["time"]=df["TEMPS"].astype(str)
    # df["date"]=df["time"].apply(lambda x:x[0:10])
    # # df["time_hours"]=df["time"].apply(lambda x:x[10:])
    # df["Date"]=pd.to_datetime(df["date"],format="%j/%m/%Y")
    # for y in ["2006","2008"]:
    #     a=df.loc[(df.Date >=str(y)+str('-01-01')) & (df.Date <= str(y)+str('-12-31'))]
    #     a.groupby("Date").sum()
    
    #### Rayonement 
    # Rn_lam=meteo_lam_station[["date","time","Rn_1_1_1"]].iloc[1:]
    # # Suppression des flux nocture
    # Rn_lam=Rn_lam.loc[(Rn_lam.time >='06:00') & (Rn_lam.time <='19:00')]
    # Rn_lam=Rn_lam.iloc[1:]
    # Rn_lam["ray"]=Rn_lam.Rn_1_1_1.astype(float)
    # Rn_lam[Rn_lam.ray<=-500]=np.nan
    # Rn_lam_jour=Rn_lam.groupby("date").mean()
    ############ ET0 station
    # meteo_lam_station=meteo_lam_station.iloc[1:]
    # meteo_lam_station["date"]=pd.to_datetime(meteo_lam_station["date"],format='%Y-%m-%d')
    # meteo_lam_station=meteo_lam_station.loc[(meteo_lam_station.time >='06:00') & (meteo_lam_station.time <='19:00')]
    # meteo_lam_station.replace("-9999",np.nan,inplace=True)
    # meteo_lam_station['Ta_1_1_1'] = meteo_lam_station['Ta_1_1_1'].astype(float)
    # meteo_lam_station['Rn_1_1_1'] = meteo_lam_station['Rn_1_1_1'].astype(float)
    # meteo_lam_station['Pa_1_1_1'] = meteo_lam_station['Pa_1_1_1'].astype(float)
    # meteo_lam_station['WS_1_1_1'] = meteo_lam_station['WS_1_1_1'].astype(float)
    # meteo_lam_station['RH_1_1_1'] = meteo_lam_station['RH_1_1_1'].astype(float)
    # meteo_lam_station['DOY'] = meteo_lam_station['DOY'].astype(float)
    # meteo_lam_station["doy"]=round(meteo_lam_station.DOY).astype(int)
    # meteo_lam_station.reset_index(inplace=True)
    # Meteo_lam=meteo_lam_station.groupby("date").mean()
    # Meteo_lam["doy"]=round(Meteo_lam.doy).astype(int)
    # ET0_stat=pd.DataFrame()
    # for i in np.arange(2,366):
    #     Tmax=meteo_lam_station.Ta_1_1_1[meteo_lam_station.doy==i].max()
    #     Tmin=meteo_lam_station.Ta_1_1_1[meteo_lam_station.doy==i].min()
    #     Rhmin=meteo_lam_station.RH_1_1_1[meteo_lam_station.doy==i].min()
    #     Rhmax=meteo_lam_station.RH_1_1_1[meteo_lam_station.doy==i].max()
    #     ET0=et0_pm_simple_kelv(Meteo_lam.doy.iloc[i],5,2,43,Meteo_lam.Ta_1_1_1.iloc[i],Tmin,Tmax,Meteo_lam.WS_1_1_1.iloc[i],Meteo_lam.RH_1_1_1.iloc[i],Rhmin,Rhmax,Meteo_lam.Rn_1_1_1.iloc[i])
    #     print("Michel :%s"%ET0)
    #     ET0_stat=ET0_stat.append([ET0])
    # ET0_stat.reset_index(inplace=True)
    #     ET0_p=penman_monteith(meteo_lam_station.doy.iloc[i],43,5,meteo_lam_station.Ta_1_1_1.iloc[i]-273,meteo_lam_station.Ta_1_1_1.iloc[i]-273.15,meteo_lam_station.Rn_1_1_1.iloc[i],meteo_lam_station.Pa_1_1_1.iloc[i],meteo_lam_station.WS_1_1_1.iloc[i])
    #     # ET0_p=penman_monteith(Meteo_lam.doy.iloc[i],43,5,Meteo_lam.Ta_1_1_1.iloc[i]-273,Meteo_lam.Ta_1_1_1.iloc[i]-273.15,Meteo_lam.Rn_1_1_1.iloc[i],Meteo_lam.Pa_1_1_1.iloc[i],Meteo_lam.WS_1_1_1.iloc[i])
    #     # print(Meteo_lam.doy.iloc[i])
    #     # print(ET0_p)
    #     ET0_stat=ET0_stat.append([ET0_p])
    # ET0_stat.reset_index(inplace=True)
    # ET0_stat["date"]=meteo_lam_station.doy
    # ET0_stati_j=ET0_stat.groupby("date").sum()
    # ETO_SAFRAN=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_LAM/meteo_lam_2019.csv")
    # ETO_SAFRAN.date=pd.to_datetime(ETO_SAFRAN.date,format="%Y-%m-%d")
    # ETO_SAFRAN.set_index('date',inplace=True)
    # ETO_SAFRAN.reset_index(inplace=True)
    # ETO_SAFRAN.drop(columns=["Unnamed: 0"],inplace=True)

    # ## Plot
    # Rn_lam_jour.index=pd.to_datetime(Rn_lam_jour.index,format="%Y-%m-%d")
    # Meteo_lam.index=pd.to_datetime(Meteo_lam.index,format="%Y-%m-%d")
    # plt.figure(figsize=(12,10))
    # plt.bar(Meteo_lam.iloc[150:250].index,Meteo_lam.Prec.iloc[150:250],width=1)
    # plt.plot(Rn_lam_jour.iloc[150:250].index,Rn_lam_jour.ray.iloc[150:250],color="r")
    # #### Test détection Irrigation recherche #####
    # # Seuil Rn 150 
    # a=Meteo_lam.Prec.where((Rn_lam_jour.ray>350)&(Meteo_lam.Prec>10))
    # a.dropna(inplace=True)
# Gri
    # meteo_lam_station=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/DONNEES_RAW/DATA_PARCELLE_GRIGNON/RAW_DATA/DATA_ICOS_ECO/FR-Gri_Meteo_2019.csv")
    # Meteo=meteo_lam_station[["TIMESTAMP","P_1_1_1"]]
    # Meteo["time"]=Meteo["TIMESTAMP"].astype(str)
    # Meteo["date"]=Meteo["time"].apply(lambda x:x[0:8])
    # Meteo["date"]=pd.to_datetime(Meteo["date"],format="%Y%m%d")
    # Meteo.drop(columns=["time","TIMESTAMP"],inplace=True)
    # Meteo["Prec"]=Meteo.P_1_1_1.astype(float)
    # Meteo[Meteo.Prec<=-9998]=np.nan
    # Meteo_Gri=Meteo.groupby("date").sum()
    # Meteo_Gri.drop(columns="P_1_1_1",inplace=True)
    # Meteo_Gri.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/DATA_METEO_BV/PARCELLE_GRI/Meteo_station_2019.csv")
# =============================================================================
# Fcover_sigmo
# =============================================================================
    # for i in os.listdir("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/FCOVER_Raw/Fcover_sigmo/"):
    #     print(i)
    #     Fcover_sig=pd.read_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/FCOVER_Raw/Fcover_sigmo/"+i)
    #     Fcov_sigmo=Fcover_sig.T
    #     Fcov_sigmo=Fcov_sigmo.iloc[2:]
    #     date=pd.date_range(start="2019-01-01",end="2019-12-31")
    #     Fcov_sigmo["date"]=date
    #     Fcov_sigmo.columns=["FCOVER","date"]
    #     Fcov_sigmo.to_csv("/datalocal/vboxshare/THESE/BESOIN_EAU/TRAITEMENT/INPUT_DATA/FCOVER_parcelle/FCOVER_sigmo/"+i)
    #     Fcov_sigmo.plot()
# =============================================================================
# Data Laurent Bi
# =============================================================================
    # for file in os.listdir("/run/media/pageot/PROJET_YP/TS_SWC_NDVI_Lonzee_Yann"):
    #     print(file)
    #     if "NDVI" in file:
    #         df=pd.read_csv("/run/media/pageot/PROJET_YP/TS_SWC_NDVI_Lonzee_Yann/"+file,sep=";")
    #     else:
    #         df=pd.read_csv("/run/media/pageot/PROJET_YP/TS_SWC_NDVI_Lonzee_Yann/"+file)
    #     df["time"]=df["Date"].astype(str)
    #     df["date"]=df["time"].apply(lambda x:x[0:8])
    #     df["date"]=pd.to_datetime(df["date"],format="%d/%m/%y")
    #     df.replace("NaN !",np.nan,inplace=True)
    #     if 'TS' in file:
    #         df["Mean_TS_X-1_float"]=df["Mean_TS_X-1"].astype(float)
    #         df.drop(columns=["Mean_TS_X-1"],inplace=True)
    #     elif 'SWC' in file:
    #         df.replace("#DIV/0 !",np.nan,inplace=True)
    #         df["Mean_SWC_X-4_float"]=df["Mean_SWC_X-4"].astype(float)
    #         df.drop(columns=["Mean_SWC_X-4"],inplace=True)
    #     df=df.groupby("date").mean()
    #     print(df.columns)
    #     df.to_csv("/run/media/pageot/PROJET_YP/TS_SWC_NDVI_Lonzee_Yann/modif_V2_%s.csv"%(file[:-4]))

