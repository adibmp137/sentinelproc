# -*- coding: utf-8 -*-
"""
@author: Eric Laloy <elaloy@sckcen.be>, January 2017 / February 2018.

Adjustments by Marit Hendrickx (KU Leuven)
Last update: September 2025

Adjustments by Adib Muhammad Prawirahutama (KU Leuven)
Last update: June 2026

"""
from __future__ import print_function
import collections
import collections.abc
for name in ['Mapping', 'MutableMapping', 'MutableSequence', 'Sequence', 'Iterable']:
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

import numpy as np
import os
import numpy.matlib as matlib
try:
    import cPickle as pickle
except:
    import pickle
import time

from mcmc_func import* # This imports both all Dream_zs and inverse problem-related functions

from types import SimpleNamespace

class AttrDict(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __contains__(self, key):
        return key in self.__dict__
    def __repr__(self):
        return f"AttrDict({self.__dict__})"
from math import *

## SWB model
from SWB_model import SWB


MCMCPar=AttrDict()

MCMCVar=AttrDict()

Measurement=AttrDict()

OutDiag=AttrDict()

Extra=AttrDict()

class Sampler:
    
    def __init__(self, df_list,data_dir,case,year,ini,LB,UB,opkomst,crop_name,forecast,
                 validation_days=np.nan,
                 cal_start_serial=None,
                 CaseStudy=2,seq = 3,ndraw=10000,thin = 1,  nCR = 3, 
                 DEpairs = 3, parallelUpdate = 0.9, pCR=True,k=10,pJumpRate_one=0.2,
                 steps=100,savemodout=False, saveout=True,save_tmp_out=True,Prior='LHS',
                 DoParallel=True,eps=5e-2,BoundHandling='Reflect',
                 lik_sigma_est=False,corr_est=False,parallel_jobs=6,jr_scale=1.0,rng_seed=123,
                 cal='',cal_par_on=False,DREAM_obsdata='Sensor+stalen',prior_file=None,
                 date_YYYYMMDD=None,prior_inflation=2.0):

        self.CaseStudy=CaseStudy
        self.cal_start_serial = cal_start_serial
        if date_YYYYMMDD is None:
            date_YYYYMMDD = year + '1002'
        if not isnan(validation_days):
            validation_days = int(validation_days)
        MCMCPar.seq = seq
        MCMCPar.ndraw=ndraw
        MCMCPar.thin=thin
        MCMCPar.nCR=nCR
        MCMCPar.DEpairs=DEpairs
        MCMCPar.parallelUpdate=parallelUpdate
        MCMCPar.Do_pCR=pCR
        MCMCPar.k=k
        MCMCPar.pJumpRate_one=pJumpRate_one
        MCMCPar.steps=steps
        MCMCPar.savemodout=savemodout
        MCMCPar.saveout=saveout  
        MCMCPar.save_tmp_out=save_tmp_out  
        MCMCPar.Prior=Prior
        MCMCPar.DoParallel=DoParallel
        MCMCPar.eps = eps
        MCMCPar.BoundHandling = BoundHandling
        MCMCPar.jr_scale=jr_scale
        MCMCPar.lik_sigma_est=lik_sigma_est
        MCMCPar.corr_est=corr_est
        if not MCMCPar.lik_sigma_est: MCMCPar.corr_est==False
        Extra.n_jobs=parallel_jobs
        
        np.random.seed(rng_seed)
        MCMCPar.rng_seed=rng_seed
        
        Extra.start_from_given_values=False
        Extra.obsdata=DREAM_obsdata
        
            
        if self.CaseStudy==1: 
            ''' SOIL WATER BALANCE MODEL - CONSIDERING ERROR AUTOCOVARIANCE'''
                        
            MCMCPar.prior_file = prior_file
            MCMCPar.prior_inflation = prior_inflation
            Extra.start_from_given_values=False
            if cal_par_on:
                a=0
                b=1
                Extra.Xini=ini.extend([a,b])
                ini_full = ini
            else:
                Extra.Xini=ini
                ini_full = ini
            ini_full = list(ini_full)

            MCMCPar.lik=44 # 441 = weighted ; 44 = not weighted
            ModelName='soil_water_model'         
            Extra.x=np.zeros(seq)
            
            _sw_harmless, _, g_list_harmless, sensor_data_adj, covar_adj, df_obs, _, _ = SWB(ini_full[0],ini_full[1],ini_full[2], \
                  ini_full[3],ini_full[4],ini_full[5],ini_full[6],ini_full[7],ini_full[8],ini_full[9],ini_full[10],ini_full[11], \
                  sensor=True,cal=cal,sensor_cal=np.empty(0),CI=np.empty(0), \
                  show=[False,''],case=case,year=year,forecast=np.empty(0),df_list=df_list)
            _g0_serial = min(g_list_harmless)
            
            # --- Sentinel data replacement (before truncation) ---
            _rebuild_covar = False
            if Extra.obsdata == 'Sentinel+stalen':
                from Sensordata import sentineldata
                sentinel_data_full, _, _ = sentineldata(
                    date_YYYYMMDD, case, year, _g0_serial,
                    [], [], [], cal='gen', show=[False,''], drop_samp=np.nan
                )
                sensor_data_adj = sentinel_data_full.copy()
                _rebuild_covar = True
            
            if not isnan(validation_days):
                n_samp_ori = len(df_obs)
                N_sensor_ori = len(sensor_data_adj)
                if self.cal_start_serial is not None:
                    cal_end_serial = self.cal_start_serial + validation_days - 1
                    sensor_data_adj = sensor_data_adj[sensor_data_adj['Date'] <= cal_end_serial].copy()
                    date = sensor_data_adj['Date'].iloc[-1] if len(sensor_data_adj) > 0 else None
                    df_obs = df_obs.loc[df_obs['Date'] <= date] if date is not None else df_obs.iloc[0:0]
                    new_n_sensor = len(sensor_data_adj)
                    new_n_samp = len(df_obs)
                    Sigma_sensor = covar_adj[:new_n_sensor, :new_n_sensor].copy()
                    if new_n_samp > 0:
                        obs_stdev_list = []
                        for i in range(new_n_samp):
                            s = df_obs['Stdev30'].iloc[i] if 'Stdev30' in df_obs.columns else 0.0114
                            if s == 0 or (isinstance(s, float) and isnan(s)):
                                s = 0.0114
                            obs_stdev_list.append(float(s))
                        Sigma_samples = np.diag(np.array(obs_stdev_list)**2)
                        covar_adj = np.zeros((new_n_sensor + new_n_samp, new_n_sensor + new_n_samp))
                        covar_adj[:new_n_sensor, :new_n_sensor] = Sigma_sensor
                        covar_adj[new_n_sensor:, new_n_sensor:] = Sigma_samples
                    else:
                        covar_adj = Sigma_sensor
                else:
                    sensor_data_adj = sensor_data_adj[:validation_days]
                    date = sensor_data_adj['Date'].iloc[-1]
                    df_obs = df_obs.loc[df_obs['Date'] <= date]
                    if len(df_obs)<n_samp_ori:
                        covar_adj = np.delete(covar_adj,np.s_[len(df_obs)-n_samp_ori:],0)
                        covar_adj = np.delete(covar_adj,np.s_[len(df_obs)-n_samp_ori:],1)
                    covar_adj = np.delete(covar_adj,np.s_[validation_days:N_sensor_ori],0)
                    covar_adj = np.delete(covar_adj,np.s_[validation_days:N_sensor_ori],1)
            
            ## if only using first two samples (on top of sensor data)
            if Extra.obsdata == 'Sensor+2samp' and len(df_obs)>2:
                delete=len(df_obs)-2
                df_obs=df_obs[:2] 
                covar_adj = np.delete(covar_adj,np.s_[-delete:],0)
                covar_adj = np.delete(covar_adj,np.s_[-delete:],1)
                
            # Extra contains sensordates, sensorobs, case, year, forecast, cal_par_on
            Extra.sensordates=sensor_data_adj['Date']
            Extra.sensorobs=df_obs
            Extra.case=case
            Extra.year=year
            Extra.forecast=forecast
            Extra.cal_par_on=cal_par_on
            Extra.cal=cal
            Extra.dfs=df_list
            Extra.val_days=validation_days

            # --- Rebuild covariance for Sentinel data after truncation ---
            if _rebuild_covar:
                from covariance_analysis import build_sentinel_covariance_n, build_sample_covariance
                n_sentinel = len(sensor_data_adj)
                n_samp_rebuild = len(df_obs)
                deming_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            '..', 'output', 'deming_priors_by_N.json')
                import json as _json_cov
                with open(deming_file, 'r') as f:
                    _deming_priors_cov = _json_cov.load(f)
                N_key = str(int(Extra.val_days))
                if N_key not in _deming_priors_cov or 'alpha2' not in _deming_priors_cov[N_key]:
                    raise ValueError(
                        f"No alpha2/epsilon2 for N={N_key} in {deming_file}. "
                        f"Run compute_deming_priors.py to regenerate.")
                _alpha2 = _deming_priors_cov[N_key]['alpha2']
                _epsilon2 = _deming_priors_cov[N_key]['epsilon2']
                Sigma_sentinel = build_sentinel_covariance_n(_alpha2, _epsilon2, n_sentinel)
                if n_samp_rebuild > 0:
                    obs_stdev_list = [0.0114] * n_samp_rebuild
                    Sigma_samples = build_sample_covariance(obs_stdev_list)
                    covar_adj = np.zeros((n_sentinel + n_samp_rebuild,
                                          n_sentinel + n_samp_rebuild))
                    covar_adj[:n_sentinel, :n_sentinel] = Sigma_sentinel
                    covar_adj[n_sentinel:, n_sentinel:] = Sigma_samples
                else:
                    covar_adj = Sigma_sentinel

            # --- Sentinel calibration parameters ---
            if Extra.obsdata == 'Sentinel+stalen':
                Extra.raw_swi = np.array(sensor_data_adj['Mean'].values, dtype=float)
                Extra.n_sentinel = len(sensor_data_adj)
                Extra.sentinel_cal_on = True
                Extra.a_S_idx = 12
                Extra.b_S_idx = 13
            else:
                Extra.sentinel_cal_on = False
                Extra.raw_swi = np.array([])
                Extra.n_sentinel = 0

            if opkomst:
                samp='Mean5'
            else:
                samp='Mean30'

            
            ## if double use of soil samples:
            if cal_par_on:
                Measurement.MeasData= np.concatenate([np.array(sensor_data_adj['Mean']),np.array(df_obs[samp]),np.array(df_obs[samp])])
            ## if single use of soil samples:
            else: Measurement.MeasData= np.concatenate([np.array(sensor_data_adj['Mean']),np.array(df_obs[samp])])
            
            if Extra.obsdata == 'Sensor only': ## if only sensor data
                Measurement.MeasData= np.array(sensor_data_adj['Mean'])      
            elif Extra.obsdata == 'Samples only': ## if only soil samples
                Measurement.MeasData= np.array(df_obs['Mean30'])
            
            Measurement.N = len(Measurement.MeasData)
            Measurement.n = len(np.array(sensor_data_adj['Mean']))
            
            if MCMCPar.lik_sigma_est==True: 
                # We estimate sensor error variances in DREAM (covar=0)
                # var_samp is assumed to be known
                covar_adj[:Measurement.n, :Measurement.n] = 0
                
            if Extra.obsdata == 'Sensor only': ## if only sensor data
                Measurement.Sigma = covar_adj[:Measurement.N,:Measurement.N]
            elif Extra.obsdata == 'Samples only': ## if only soil samples
                Measurement.Sigma = covar_adj[-Measurement.N:,-Measurement.N:]
                Measurement.n = 0
            else:
                Measurement.Sigma = covar_adj

            if not MCMCPar.lik_sigma_est and not MCMCPar.corr_est:
                Measurement.Sigma_inv = np.linalg.inv(Measurement.Sigma)
                Measurement.logdet_Sigma = np.linalg.slogdet(Measurement.Sigma)[1]
            else:
                Measurement.Sigma_inv = None
                Measurement.logdet_Sigma = None

                
            ####
            
            print(Measurement.N," datapoints")
            print(Measurement.n," sensor datapoints")
            print(Measurement.N-Measurement.n," soil samples")
                  
            # parameters: Kcb_ini, Kcb_mid, Kcb_end, Lini, Ldev, Lmid, fc, K_sat_log, CN, GWT_max, Zrmax, v0 (a, b)
            
                
            MCMCPar.lb=np.array(LB).reshape((1,-1))
            MCMCPar.ub=np.array(UB).reshape((1,-1))
            
            MCMCPar.n=MCMCPar.lb.shape[1] # number of parameters

            if MCMCPar.lik_sigma_est==True:
                MCMCPar.n=MCMCPar.n+1
                MCMCPar.lb=np.hstack((MCMCPar.lb,np.log(0.00001).reshape((1,-1))))
                MCMCPar.ub=np.hstack((MCMCPar.ub,np.log(1).reshape((1,-1))))
                print('Estimate log(Sigma)')
                # Add a last dimension to x that contains log(Sigma)
                if MCMCPar.corr_est==True:
                    MCMCPar.n=MCMCPar.n+1
                    MCMCPar.lb=np.hstack((MCMCPar.lb,np.array(0).reshape((1,-1))))
                    MCMCPar.ub=np.hstack((MCMCPar.ub,np.array(1).reshape((1,-1))))
                    print('Estimate error correlation')
            
            MCMCPar.m0=10*MCMCPar.n
            
            if MCMCPar.Prior=='demo':
                MCMCPar.ngp=MCMCPar.n
                MCMCPar.pmu=np.zeros((MCMCPar.ngp))     # means of prior dist
                psd1=10
                psd2=10
                MCMCPar.psd=psd=np.ones((MCMCPar.ngp))*np.array([psd1,psd2])
                MCMCPar.pcov=np.eye(MCMCPar.n)*(psd**2)
                MCMCPar.invC=np.linalg.inv(MCMCPar.pcov)
                
            elif MCMCPar.Prior=='Normal':
                MCMCPar.ngp=MCMCPar.n
                MCMCPar.pmu=[Kcb_ini, Kcb_mid, Kcb_end, fc, zrmax, K_sat_log]    # means of prior dist
                # psd1=0.1
                # psd2=0.2
                # psd3=0.2
                # psd4=0.3
                # psd5=0.05
                # psd6=log(100)
                [psd1, psd2, psd3, psd4, psd5, psd6]=np.dot(MCMCPar.pmu,0.5)
                MCMCPar.psd=psd=np.ones((MCMCPar.ngp))*np.array([psd1,psd2,psd3,psd4,psd5,psd6])
                MCMCPar.pcov=np.eye(MCMCPar.n)*(psd**2)
                MCMCPar.invC=np.linalg.inv(MCMCPar.pcov)
            
            elif MCMCPar.Prior=='Transfer':
                import json as _json
                MZ1_ParSet = np.load(MCMCPar.prior_file)
                istart = int(np.round(0.5 * MZ1_ParSet.shape[0]))
                MZ1_ParSet50 = MZ1_ParSet[istart:, :-2]
                
                # Use MAP (maximum a posteriori) instead of posterior mean as prior centre
                log_lik = MZ1_ParSet[istart:, -1]
                map_idx = np.argmax(log_lik)
                MAP_params = MZ1_ParSet50[map_idx, :]
                
                # Inflation factor for prior covariance to account for between-site variability
                # Default 1.0 = no inflation; set via MCMCPar.prior_inflation (passed from run_pipeline)
                inflation = getattr(MCMCPar, 'prior_inflation', 2.0)

                # Compute covariance around MAP (not around mean) for consistency with MAP centre.
                # np.cov() centres around the mean, but since the prior centre is MAP,
                # we need: cov_MAP = cov_mean + outer(mean - MAP, mean - MAP)
                # This adds a rank-1 correction that widens the prior in directions
                # where MAP differs from the mean (i.e., skewed parameters near bounds).
                MZ1_mean = np.mean(MZ1_ParSet50, axis=0)
                d_full = MZ1_mean - MAP_params
                cov_around_mean_full = np.cov(MZ1_ParSet50.T)
                cov_around_MAP_full = cov_around_mean_full + np.outer(d_full, d_full)

                if MCMCPar.n == 12:
                    MCMCPar.ngp = MCMCPar.n
                    MCMCPar.pmu = MAP_params.copy()
                    MCMCPar.pcov = inflation * cov_around_MAP_full
                    MCMCPar.invC = np.linalg.inv(MCMCPar.pcov)

                elif MCMCPar.n == 14:
                    MAP_swb = MAP_params.copy()
                    swb_cov = inflation * cov_around_MAP_full
                    
                    deming_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               '..', 'output', 'deming_priors_by_N.json')
                    with open(deming_file, 'r') as f:
                        deming_priors = _json.load(f)
                    N_key = str(int(Extra.val_days))
                    
                    if N_key not in deming_priors or deming_priors[N_key].get('n_obs', 0) < 2:
                        MCMCPar.ngp = 12
                        MCMCPar.pmu = np.zeros(14)
                        MCMCPar.pmu[:12] = MAP_swb
                        MCMCPar.pcov = np.zeros((14, 14))
                        MCMCPar.pcov[:12, :12] = swb_cov
                        MCMCPar.invC = np.zeros((14, 14))
                        MCMCPar.invC[:12, :12] = np.linalg.inv(swb_cov)
                    else:
                        deming = deming_priors[N_key]
                        DEMING_INTERCEPT = deming['a']
                        DEMING_SLOPE = deming['b']
                        DEMING_INTERCEPT_SE = deming['SE_a']
                        DEMING_SLOPE_SE = deming['SE_b']
                        
                        MCMCPar.pmu = np.zeros(14)
                        MCMCPar.pmu[:12] = MAP_swb
                        MCMCPar.pmu[12] = DEMING_INTERCEPT
                        MCMCPar.pmu[13] = DEMING_SLOPE
                        
                        MCMCPar.pcov = np.zeros((14, 14))
                        MCMCPar.pcov[:12, :12] = swb_cov
                        MCMCPar.pcov[12, 12] = DEMING_INTERCEPT_SE ** 2
                        MCMCPar.pcov[13, 13] = DEMING_SLOPE_SE ** 2
                        MCMCPar.invC = np.linalg.inv(MCMCPar.pcov)
                        MCMCPar.ngp = MCMCPar.n

                elif MCMCPar.n == 8:
                    MAP_soil = MAP_params[6:12].copy()
                    d_soil = MZ1_mean[6:12] - MAP_soil
                    soil_cov_around_mean = np.cov(MZ1_ParSet50[:, 6:12].T)
                    if soil_cov_around_mean.ndim == 0:
                        soil_cov_around_mean = np.array([[soil_cov_around_mean]])
                    soil_cov_around_MAP = soil_cov_around_mean + np.outer(d_soil, d_soil)
                    soil_cov = inflation * soil_cov_around_MAP

                    deming_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                '..', 'output', 'deming_priors_by_N.json')
                    with open(deming_file, 'r') as f:
                        deming_priors = _json.load(f)
                    N_key = str(int(Extra.val_days))

                    if N_key not in deming_priors or deming_priors[N_key].get('n_obs', 0) < 2:
                        MCMCPar.ngp = 6
                        MCMCPar.pmu = np.zeros(8)
                        MCMCPar.pmu[:6] = MAP_soil
                        MCMCPar.pcov = np.zeros((8, 8))
                        MCMCPar.pcov[:6, :6] = soil_cov
                        MCMCPar.invC = np.zeros((8, 8))
                        MCMCPar.invC[:6, :6] = np.linalg.inv(soil_cov)
                    else:
                        deming = deming_priors[N_key]
                        DEMING_INTERCEPT = deming['a']
                        DEMING_SLOPE = deming['b']
                        DEMING_INTERCEPT_SE = deming['SE_a']
                        DEMING_SLOPE_SE = deming['SE_b']

                        MCMCPar.pmu = np.zeros(8)
                        MCMCPar.pmu[:6] = MAP_soil
                        MCMCPar.pmu[6] = DEMING_INTERCEPT
                        MCMCPar.pmu[7] = DEMING_SLOPE

                        MCMCPar.pcov = np.zeros((8, 8))
                        MCMCPar.pcov[:6, :6] = soil_cov
                        MCMCPar.pcov[6, 6] = DEMING_INTERCEPT_SE ** 2
                        MCMCPar.pcov[7, 7] = DEMING_SLOPE_SE ** 2
                        MCMCPar.invC = np.linalg.inv(MCMCPar.pcov)
                        MCMCPar.ngp = MCMCPar.n
                else:
                    raise ValueError(
                        f"Transfer prior not defined for n={MCMCPar.n} parameters")
            
        elif self.CaseStudy==0:
            # A theoretical multivariate normal distribution with 100 correlated dimensions 
            # (example 2 in Matlab DREAM code)
            MCMCPar.n=100 # parameters
            MCMCPar.Prior='LHS'
            MCMCPar.lb=np.zeros((1,MCMCPar.n))-5
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+15
            MCMCPar.BoundHandling='Reflect'
			
            Measurement.N=1
            ModelName='theoretical_case_mvn'
            MCMCPar.lik=0
            
            A = 0.5 * np.eye(MCMCPar.n) + 0.5 * np.ones(MCMCPar.n)
            cov=np.zeros((MCMCPar.n,MCMCPar.n))
            # Rescale to variance-covariance matrix of interest
            for i in range (0,MCMCPar.n):
                for j in range (0,MCMCPar.n):
                    cov[i,j] = A[i,j] * np.sqrt((i+1) * (j+1))
            Extra.C=cov
            Extra.invC = np.linalg.inv(cov)
            
        else: # This should not happen and is thus probably not needed
            raise NameError('Something went wrong in your case selection. Please select an existing CaseStudy.')
            MCMCPar.n=1
            MCMCPar.lb=np.zeros((1,MCMCPar.n))
            MCMCPar.ub=np.zeros((1,MCMCPar.n))+1
            MCMCPar.BoundHandling=None
            Measurement.N=1
            ModelName=None
            MCMCPar.lik=1

        Extra.ModelName=ModelName
        self.MCMCPar=MCMCPar
        self.Measurement=Measurement
        self.Extra=Extra
        self.ModelName=ModelName
       
    def _init_sampling(self):
        
        Iter=self.MCMCPar.seq
        iteration=2
        iloc=0
        T=0
        
        if self.MCMCPar.Prior=='StandardNormal':
            Zinit=np.random.randn(self.MCMCPar.m0+self.MCMCPar.seq,self.MCMCPar.n)
            if self.MCMCPar.lik_sigma_est==True: # Use log-uniform prior for sigma
                Zinit[:,0]=lhs(self.MCMCPar.lb[0][0].reshape((1,1)),self.MCMCPar.ub[0][0].reshape((1,1)),self.MCMCPar.m0+self.MCMCPar.seq).reshape((self.MCMCPar.m0+self.MCMCPar.seq))
                
        elif self.MCMCPar.Prior=='Normal':
            Zinit=np.random.multivariate_normal(self.MCMCPar.pmu+np.zeros((MCMCPar.n)),np.eye(self.MCMCPar.n)*self.MCMCPar.psd**2,MCMCPar.m0+self.MCMCPar.seq)
            if self.MCMCPar.lik_sigma_est==True: # Use log-uniform prior for sigma
                Zinit[:,0]=lhs(self.MCMCPar.lb[0][0].reshape((1,1)),self.MCMCPar.ub[0][0].reshape((1,1)),self.MCMCPar.m0+self.MCMCPar.seq).reshape((self.MCMCPar.m0+self.MCMCPar.seq))
        
        elif self.MCMCPar.Prior=='COV': # Generate initial population from multivariate normal distribution but the model returns posterior density directly
            Zinit=np.random.randn(self.MCMCPar.m0+self.MCMCPar.seq,self.MCMCPar.n)
        
        elif (self.MCMCPar.Prior=='demo') or (self.MCMCPar.Prior=='Transfer'): 
            # First draw samples from uniform distribution for all variables and replace as needed after
            Zinit=lhs(self.MCMCPar.lb,self.MCMCPar.ub,self.MCMCPar.m0+self.MCMCPar.seq) 
            # Gaussian prior for the first ngp variables (from MZ1 posterior + optionally Deming)
            Zinit[:,:self.MCMCPar.ngp]=np.random.multivariate_normal(self.MCMCPar.pmu,self.MCMCPar.pcov,self.MCMCPar.m0+self.MCMCPar.seq)

        else: # Uniform prior, LHS sampling
            Zinit=lhs(self.MCMCPar.lb,self.MCMCPar.ub,self.MCMCPar.m0+self.MCMCPar.seq)

        self.MCMCPar.CR=np.cumsum((1.0/self.MCMCPar.nCR)*np.ones((1,self.MCMCPar.nCR)))
        Nelem=np.floor(self.MCMCPar.ndraw/self.MCMCPar.seq)++self.MCMCPar.seq*2
        OutDiag.CR=np.zeros((int(np.floor(Nelem/self.MCMCPar.steps))+2,self.MCMCPar.nCR+1))
        OutDiag.AR=np.zeros((int(np.floor(Nelem/self.MCMCPar.steps))+2,2))
        OutDiag.AR[0,:] = np.array([self.MCMCPar.seq,-1])
        OutDiag.R_stat = np.zeros((int(np.floor(Nelem/self.MCMCPar.steps))+2,self.MCMCPar.n+1))
        pCR = (1.0/self.MCMCPar.nCR) * np.ones((1,self.MCMCPar.nCR))
        
        # Calculate the actual CR values based on pCR
        CR,lCR = GenCR(self.MCMCPar,pCR)  
        
        if self.MCMCPar.savemodout:
            self.fx = np.zeros((self.Measurement.N,int(np.floor(self.MCMCPar.ndraw/self.MCMCPar.thin))))
            MCMCVar.m_func = self.MCMCPar.seq     
        
        self.Sequences = np.zeros((int(np.floor(Nelem/self.MCMCPar.thin)),self.MCMCPar.n+2,self.MCMCPar.seq))
           
        self.MCMCPar.Table_JumpRate=np.zeros((self.MCMCPar.n,self.MCMCPar.DEpairs))
        for zz in range(0,self.MCMCPar.DEpairs):
            self.MCMCPar.Table_JumpRate[:,zz] = 2.38/np.sqrt(2 * (zz+1) * np.linspace(1,self.MCMCPar.n,self.MCMCPar.n).T)
        
        # Change steps to make sure to get nice iteration numbers in first loop
        self.MCMCPar.steps = self.MCMCPar.steps - 1
        
        self.Z = np.zeros((np.floor(self.MCMCPar.m0 + self.MCMCPar.seq * (self.MCMCPar.ndraw - self.MCMCPar.m0) / (self.MCMCPar.seq * self.MCMCPar.k)).astype('int64')+self.MCMCPar.seq*100,self.MCMCPar.n+2))
        self.Z[:self.MCMCPar.m0,:self.MCMCPar.n] = Zinit[:self.MCMCPar.m0,:self.MCMCPar.n]

        X = Zinit[self.MCMCPar.m0:(self.MCMCPar.m0+self.MCMCPar.seq),:self.MCMCPar.n]
        del Zinit
        
        if Extra.start_from_given_values==True:
            X=matlib.repmat(Extra.Xini,MCMCPar.seq,1)
        
        # Run forward model, if any this is done in parallel
        if  self.CaseStudy > 1:
            if self.MCMCPar.lik_sigma_est==True: # The inferred sigma must always occupy the last position in the parameter vector
                if self.MCMCPar.corr_est==True: # Last two are sigma and corr
                    fx0 = RunFoward(X[:,:-2],self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
                else:
                    fx0 = RunFoward(X[:,:-1],self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
            else:
                fx0 = RunFoward(X,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)    
        else:
            fx0 = RunFoward(X,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
        
        # Compute likelihood from simulated data    
        of,log_p = CompLikelihood(X,fx0,self.MCMCPar,self.Measurement,self.Extra)

        X = np.concatenate((X,of,log_p),axis=1)
        Xfx = fx0
        
        if self.MCMCPar.savemodout==True:
            self.fx=fx0
        else:
            self.fx=None

        self.Sequences[0,:self.MCMCPar.n+2,:self.MCMCPar.seq] = np.reshape(X.T,(1,self.MCMCPar.n+2,self.MCMCPar.seq))

        # Store N_CR
        OutDiag.CR[0,:MCMCPar.nCR+1] = np.concatenate((np.array([Iter]).reshape((1,1)),pCR),axis=1)
        delta_tot = np.zeros((1,self.MCMCPar.nCR))

        # Compute the R-statistic of Gelman and Rubin
        OutDiag.R_stat[0,:self.MCMCPar.n+1] = np.concatenate((np.array([Iter]).reshape((1,1)),GelmanRubin(self.Sequences[:1,:self.MCMCPar.n,:self.MCMCPar.seq],self.MCMCPar)),axis=1)
      
        self.OutDiag=OutDiag
        
        # Also return the necessary variable parameters
        MCMCVar.m=self.MCMCPar.m0
        MCMCVar.Iter=Iter
        MCMCVar.iteration=iteration
        MCMCVar.iloc=iloc; MCMCVar.T=T; MCMCVar.X=X
        MCMCVar.Xfx=Xfx; MCMCVar.CR=CR; MCMCVar.pCR=pCR
        MCMCVar.lCR=lCR; MCMCVar.delta_tot=delta_tot
        self.MCMCVar=MCMCVar
        
        if self.MCMCPar.save_tmp_out==True:
            with open('out_tmp'+'.pkl','wb') as f:
                 pickle.dump({'Sequences':self.Sequences,'Z':self.Z,
                 'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                 'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                 'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)
      
    def sample(self,RestartFilePath=None):
        
        if not(RestartFilePath is None):
            print('This is a restart')
            with open(RestartFilePath, 'rb') as fin:
                tmp_obj = pickle.load(fin)
            self.Sequences=tmp_obj['Sequences']
            self.Z=tmp_obj['Z']
            self.OutDiag=tmp_obj['OutDiag']
            self.fx=tmp_obj['fx']
            self.MCMCPar=tmp_obj['MCMCPar']
            self.MCMCVar=tmp_obj['MCMCVar']
            self.Measurement=tmp_obj['Measurement']
            self.ModelName=tmp_obj['ModelName']
            self.Extra=tmp_obj['Extra']
            del tmp_obj
                
            self.MCMCPar.ndraw = 2 * self.MCMCPar.ndraw
            
            # Reset rng
            np.random.seed(np.floor(time.time()).astype('int'))
            
            # Extend Sequences, Z, OutDiag.AR,OutDiag.Rstat and OutDiag.CR
            self.Sequences=np.concatenate((self.Sequences,np.zeros((self.Sequences.shape))),axis=0)
            self.Z=np.concatenate((self.Z,np.zeros((self.Z.shape))),axis=0)
            self.OutDiag.AR=np.concatenate((self.OutDiag.AR,np.zeros((self.OutDiag.AR.shape))),axis=0)
            self.OutDiag.R_stat=np.concatenate((self.OutDiag.R_stat,np.zeros((self.OutDiag.R_stat.shape))),axis=0)
            self.OutDiag.CR=np.concatenate((self.OutDiag.CR,np.zeros((self.OutDiag.CR.shape))),axis=0)
      
            
        else:
            self._init_sampling()
            
# Main sampling loop  
        while self.MCMCVar.Iter < self.MCMCPar.ndraw:
            
            # Check that exactly MCMCPar.ndraw are done (uneven numbers this is impossible, but as close as possible)
            if (self.MCMCPar.steps * self.MCMCPar.seq) > self.MCMCPar.ndraw - self.MCMCVar.Iter:
                # Change MCMCPar.steps in last iteration 
                self.MCMCPar.steps = np.ceil((self.MCMCPar.ndraw - self.MCMCVar.Iter)/float(self.MCMCPar.seq)).astype('int64')
                
            # Initialize totaccept
            totaccept = 0

#            start_time = time.time()
            
            # Loop a number of times before calculating convergence diagnostic, etc.
            for gen_number in range(0,self.MCMCPar.steps):
                
                # Update T
                self.MCMCVar.T = self.MCMCVar.T + 1
                
                # Define the current locations and associated log-densities
                xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,:self.MCMCPar.n])
                log_p_xold = np.array(self.MCMCVar.X[:self.MCMCPar.seq,self.MCMCPar.n + 2-1])

                # Without replacement draw rows from Z for proposal creation
                R=np.random.permutation(self.MCMCVar.m)
                R=R[0:2 * self.MCMCPar.DEpairs * self.MCMCPar.seq]
                Zoff = np.array(self.Z[R,:self.MCMCPar.n])
             
        
                # Determine to do parallel direction or snooker update
                if (np.random.rand(1) <= self.MCMCPar.parallelUpdate):
                    Update = 'Parallel_Direction_Update'
                else:
                    Update = 'Snooker_Update'

                # Generate candidate points (proposal) in each chain using either snooker or parallel direction update
                xnew,self.MCMCVar.CR[:,gen_number] ,alfa_s = DreamzsProp(xold,Zoff,self.MCMCVar.CR[:,gen_number],self.MCMCPar,Update)
    
    
                # Get simulated data (done in parallel)
                if  self.CaseStudy > 1:
                    if self.MCMCPar.lik_sigma_est==True: # The inferred sigma must always occupy the last position in the parameter vector
                        if self.MCMCPar.corr_est==True: # Last two are sigma and corr
                            fx_new = RunFoward(xnew[:,:-2],self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
                        else:
                            fx_new = RunFoward(xnew[:,:-1],self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
                    else:
                        fx_new = RunFoward(xnew,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)    
                else:
                    fx_new = RunFoward(xnew,self.MCMCPar,self.Measurement,self.ModelName,self.Extra)
                 
                # Compute the likelihood of each proposal in each chain
                of_xnew,log_p_xnew = CompLikelihood(xnew,fx_new,self.MCMCPar,self.Measurement,self.Extra)
    
                # Calculate the Metropolis ratio
                accept = Metrop(self.MCMCPar,xnew,log_p_xnew,xold,log_p_xold,alfa_s)

                # And update X and the model simulation
                idx_X= np.argwhere(accept==1);idx_X=idx_X[:,0]
                
                if not(idx_X.size==0):
                     
                    self.MCMCVar.X[idx_X,:] = np.concatenate((xnew[idx_X,:],of_xnew[idx_X,:],log_p_xnew[idx_X,:]),axis=1)
                    self.MCMCVar.Xfx[idx_X,:] = fx_new[idx_X,:]
                                  
                # Check whether to add the current points to the chains or not?
                if self.MCMCVar.T == self.MCMCPar.thin:
                    # Store the current sample in Sequences
                    self.MCMCVar.iloc = self.MCMCVar.iloc + 1
                    self.Sequences[self.MCMCVar.iloc,:self.MCMCPar.n+2,:self.MCMCPar.seq] = np.reshape(self.MCMCVar.X.T,(1,self.MCMCPar.n+2,self.MCMCPar.seq))
                   
                   # Check whether to store the simulation results of the function evaluations
                    if self.MCMCPar.savemodout==True:
                        self.fx=np.append(self.fx,self.MCMCVar.Xfx,axis=0)
                        # Update m_func
                        self.MCMCVar.m_func = self.MCMCVar.m_func + self.MCMCPar.seq
                    else:
                        self.MCMCVar.m_func=None
                    # And set the T to 0
                    self.MCMCVar.T = 0

                # Compute squared jumping distance for each CR value
                if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):
                   
                    # Calculate the standard deviation of each dimension of X
                    r = matlib.repmat(np.std(self.MCMCVar.X[:,:self.MCMCPar.n],axis=0),self.MCMCPar.seq,1)
                    # Compute the Euclidean distance between new X and old X
                    delta_normX = np.sum(np.power((xold[:,:self.MCMCPar.n] - self.MCMCVar.X[:,:self.MCMCPar.n])/r,2),axis=1)
                                        
                    # Use this information to update delta_tot which will be used to update the pCR values
                    self.MCMCVar.delta_tot = CalcDelta(self.MCMCPar.nCR,self.MCMCVar.delta_tot,delta_normX,self.MCMCVar.CR[:,gen_number])

                # Check whether to append X to Z
                if np.mod((gen_number+1),self.MCMCPar.k) == 0:
                   
                    ## Append X to Z
                    self.Z[self.MCMCVar.m + 0 : self.MCMCVar.m + self.MCMCPar.seq,:self.MCMCPar.n+2] = np.array(self.MCMCVar.X[:,:self.MCMCPar.n+2])
                    # Update MCMCPar.m
                    self.MCMCVar.m = self.MCMCVar.m + self.MCMCPar.seq

                # Compute number of accepted moves
                totaccept = totaccept + np.sum(accept)

                # Update total number of MCMC iterations
                self.MCMCVar.Iter = self.MCMCVar.Iter + self.MCMCPar.seq
                
            # print('Iter =',self.MCMCVar.Iter)  
            
            # Reduce MCMCPar.steps to get rounded iteration numbers
            if self.MCMCVar.iteration == 2: 
                self.MCMCPar.steps = self.MCMCPar.steps + 1

            # Store acceptance rate
            self.OutDiag.AR[self.MCMCVar.iteration-1,:] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), np.array([100 * totaccept/(self.MCMCPar.steps * self.MCMCPar.seq)]).reshape((1,1))),axis=1)
            
            # Progress print: every 25% of iterations
            ar_pct = 100 * totaccept / (self.MCMCPar.steps * self.MCMCPar.seq)
            total_iters = self.MCMCPar.ndraw
            quarter_marks = [total_iters * f for f in [0.25, 0.5, 0.75, 1.0]]
            prev_iter = self.MCMCVar.Iter - self.MCMCPar.seq * self.MCMCPar.steps
            for qm in quarter_marks:
                if prev_iter < qm <= self.MCMCVar.Iter:
                    print(f'  DREAM progress: {self.MCMCVar.Iter}/{total_iters} ({100*self.MCMCVar.Iter/total_iters:.0f}%), AR={ar_pct:.1f}%')
            
            # Store probability of individual crossover values
            self.OutDiag.CR[self.MCMCVar.iteration-1,:self.MCMCPar.nCR+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)), self.MCMCVar.pCR),axis=1)
            
            # Is pCR updating required?
            if (self.MCMCPar.Do_pCR==True and self.MCMCVar.Iter < 0.1 * self.MCMCPar.ndraw):

                # Update pCR values
                self.MCMCVar.pCR = AdaptpCR(self.MCMCPar.seq,self.MCMCVar.delta_tot,self.MCMCVar.lCR,self.MCMCVar.pCR)

            # Generate CR values from current pCR values
            self.MCMCVar.CR,lCRnew = GenCR(MCMCPar,self.MCMCVar.pCR); self.MCMCVar.lCR = self.MCMCVar.lCR + lCRnew

            # Calculate Gelman and Rubin Convergence Diagnostic
            start_idx = np.maximum(1,np.floor(0.5*self.MCMCVar.iloc)).astype('int64')-1; end_idx = self.MCMCVar.iloc
            
            current_R_stat = GelmanRubin(self.Sequences[start_idx:end_idx,:self.MCMCPar.n,:self.MCMCPar.seq],self.MCMCPar)
            
            self.OutDiag.R_stat[self.MCMCVar.iteration-1,:self.MCMCPar.n+1] = np.concatenate((np.array([self.MCMCVar.Iter]).reshape((1,1)),np.array([current_R_stat]).reshape((1,self.MCMCPar.n))),axis=1)

            # Update number of complete generation loops
            self.MCMCVar.iteration = self.MCMCVar.iteration + 1

            if self.MCMCPar.save_tmp_out==True:
                with open('out_tmp'+'.pkl','wb') as f:
                    pickle.dump({'Sequences':self.Sequences,'Z':self.Z,
                    'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,
                    'MCMCVar':self.MCMCVar,'Measurement':self.Measurement,
                    'ModelName':self.ModelName,'Extra':self.Extra},f, protocol=pickle.HIGHEST_PROTOCOL)

        # Remove zeros from pre-allocated variavbles if needed
        self.Sequences,self.Z,self.OutDiag,self.fx = Dreamzs_finalize(self.MCMCPar,self.Sequences,self.Z,self.OutDiag,self.fx,self.MCMCVar.iteration,self.MCMCVar.iloc,self.MCMCVar.pCR,self.MCMCVar.m,self.MCMCVar.m_func)
        
        if self.MCMCPar.saveout==True:
            with open('dreamzs_out'+'.pkl','wb') as f:
                pickle.dump({'Sequences':self.Sequences,'Z':self.Z,'OutDiag':self.OutDiag,'fx':self.fx,'MCMCPar':self.MCMCPar,'Measurement':self.Measurement,'Extra':self.Extra},f
                , protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.Sequences, self.Z, self.OutDiag,  self.fx, self.MCMCPar, self.MCMCVar         