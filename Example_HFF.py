#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:33:03 2019
@author: mmezache

@title: Examples for the detection and the test of hypothesis for high frequency features in a signal.
 
@comment: The documentation is in the readme file 

@author: mmezache
"""
import matplotlib.pyplot as plt
import numpy as np
import HFF_v1 as hff


#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
N=10 # length of the signal (2^N)
sigma=0.05 # standard deviation of the noise
ampl=0.1 # amplitude of the oscillations
lbda=25  # parameters of the l1 trend filtering
n_iter=200 # number of simulations for the monte carlo procedure
ex=1 # Choose example 1 or example 2 (ex=2)

if ex==1:
    
    #------------------------------------------------------------------------------
    # High-frequency features detection
    #------------------------------------------------------------------------------

    t=np.linspace(0,120,int(np.power(2,N)))
    sig=hff.SigOsc(t,ampl,sigma)
    
    delta=1
    (g1,d1)=hff.GetHFF(t,sig,delta)
    
    #------------------------------------------------------------------------------
    # Monte Carlo procedure and test of hypothesis
    #------------------------------------------------------------------------------
    trend=hff.l1(sig,lbda)
    est_sigma=hff.Est_std(sig,'sym8')
    (g_h0,d_h0)=hff.MC_COP_H0(t,trend,est_sigma,n_iter,delta)
    (_c,_nu,g_mesh,d_mesh,mesh)=hff.RejZoneAlpha(g_h0,d_h0,g1,d1,0.05)
    Pval=hff.Pvalue(g_h0,d_h0,g_mesh,d_mesh,mesh,g1,d1)

    print('First example: sigma = ', sigma, ' Estimated sigma = ', est_sigma)
    print('First example: p-value = ', Pval)
if ex==2: 
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    # Second example
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    #------------------------------------------------------------------------------
    # High-frequency features detection
    #------------------------------------------------------------------------------
    
    time=np.linspace(0,12,np.power(2,13))
    sig2=hff.SanityCheckFunct(time,sigma,ampl,30)
    delta2=hff.deltaChoice(time,sig2) # data driven choice of the smoothing parameters
    (g2,d2)=hff.GetHFF(time,sig2,delta2)

    #------------------------------------------------------------------------------
    # Monte Carlo procedure and test of hypothesis
    #------------------------------------------------------------------------------
    
    trend2=hff.l1(sig2,lbda*5)
    est_sigma2=hff.Est_std(sig2,'sym8')
    (g_h02,d_h02)=hff.MC_COP_H0(time,trend2,est_sigma2,n_iter,delta2)
    (_c,_nu,g_mesh2,d_mesh2,mesh2)=hff.RejZoneAlpha(g_h02,d_h02,g2,d2,0.05)
    Pval2=hff.Pvalue(g_h02,d_h02,g_mesh2,d_mesh2,mesh2,g2,d2)
    
    print('Second example: sigma = ', sigma, ' Estimated sigma = ', est_sigma2)
    print('Second example: p-value = ', Pval2)

plt.show()