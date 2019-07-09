#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mmezache

@title: High Frequency Features (HFF) detection

@references: "Testing high frequency features in a noisy signal" (to appear)

@acknowledgement: F.Raphel (for his precious help), 
    B.Akyildiz (for the trend filtering package https://github.com/bugra/l1)
"""
import math
import numpy as np
from numpy import linalg as LA
from itertools import chain
from cvxopt import matrix, solvers, spmatrix
from outputFigs import getFig
import pywt

solvers.options['show_progress'] = False

#------------------------------------------------------------------------------
# Procedure to obtain the HF features parameters 
#------------------------------------------------------------------------------

def get_fft(y, fs):
    """ Get the single sided amplitude spectrum of a given signal using the 
    Fast Fourier Transform.

    Parameters:
        y  - signal
        fs - sampling frequency
    Returns:
        (mag, freq) - tuple of spectrum magitude and corresponding frequencies
    """
    y=np.reshape(y,(len(y),))
    n  = len(y)      # Get the signal length
    dt = 1/float(fs) # Get time resolution

    fft_output = np.fft.rfft(y)     # Perform real fft
    rfreqs = np.fft.rfftfreq(n, dt) # Calculatel frequency bins
    fft_mag = np.abs(fft_output)    # Take only magnitude of spectrum

    # Normalize the amplitude by number of bins and multiply by 2
    # because we removed second half of spectrum above the Nyqist frequency 
    # and energy must be preserved
    fft_mag = fft_mag * 2 / n           

    return np.array(fft_mag), np.array(rfreqs)

def EnergyBand(f,P1,sb,FixorRoll):
    """ Compute the L2 norm of the Fourier transform of a signal on specifics bandwidth
    Parameters:
        f - frequency scale of a signal
        P1 - Amplitude coefficient of the the Fourier transform of a signal
        sb - length of the band 
        FixorRoll - Either 'Fix' or 'Roll', choice between the way to compute the Localised L2 norm
    Returns:
        fb - Frequency scale of the localised L2 norm of the signal
        En - Localised L2 norm of the signal on each band
    
    """
    
    if FixorRoll .startswith('Fix'):
        k=math.modf(len(f)/sb)[1] # lower integer part 
        fb=np.zeros((k+1,))
        En=np.zeros((k+1,))
        fb[0]=f[sb//2]
        En[0]=LA.norm(P1[:sb-1],2);
        for i in range(1,k):
            fb[i]=f[sb*(i-1)+(sb)//2]
            En[i]=LA.norm(P1[sb*(i-1):sb*i-1],2)
        fb[-1]=f[sb*k+math.floor((len(fb)-sb*k)//2)]
        En[-1]=LA.norm(P1[sb*k:],2)
    elif FixorRoll .startswith('Roll'):
        En=np.zeros((len(P1),1))
        for i in range(len(P1)):
            if i>sb//2 and i<(len(P1)-sb//2):
                if sb<=1:
                    En[i]=P1[i]
                else:
                    En[i]=LA.norm(P1[(i-sb//2):(i+sb//2)],2)/np.sqrt(sb)
            elif(i<=sb//2):
                En[i]=LA.norm(P1[:sb],2)/np.sqrt(sb)
            if (i>=len(P1)-sb//2):
                En[i]=LA.norm(P1[len(P1)-sb:len(P1)],2)/np.sqrt(sb)
        fb=f
    return(fb,En)

def HFF_Parameters(f,osc):
    """ Numerical procedure to obtain the relative amplitude of the 
    oscillations and the localization in frequency
    Parameters:
        f - frequency scale of a signal Z
        osc - Amplitude of the Fourier coefficients corresponding to f
        
    Returns:
        nu_star - relative amplitude of the oscillations
        index_n_star - Index of the peak corresponding to the oscillations
        n_1_star - index of the min between peak and low-frequency components
        n_2_star - index on the low-frequency components for the same amplitude
            as the oscillations
        n_3_star - index of the peak corresponding to the oscillations
    
    Remarks: see "Testing high-frequency features in a noisy signal" for 
    references, g(f)=n_3_star-n_2_star, d(f)=nu_star.
    """
    sig=np.reshape(np.array(osc),(len(f),))
    mu=np.array(sorted(sig)) # Statistics of order
    n=len(mu)
    a=np.zeros((n,),dtype=int)
    b=np.zeros((n,),dtype=int)
    for i in range(n):
        a[i]=np.where(sig<=mu[i])[0][0] # Smallest index where sig<=mu
        sg = sig[int(a[i]):]
        b[i] = len(sg) - np.argmax(sg[::-1]) -1 + a[i] # index of the peak after index a[i]
    ib=np.unique(b) # index set of all peaks
    ib_tilt=ib[1:]
    if len(ib_tilt)>0:
        l=len(ib_tilt)
        index_n_3=np.zeros((l,),dtype=int) # initialization peak 
        index_n_1=np.zeros((l,),dtype=int) # initialization min between trend and peak
        for i in range(l):
            index_n_3[i]=ib_tilt[i]
            index_n_1[i]=max(np.where(sig==min(sig[:int(index_n_3[i])]))[0])
        ampl=sig[index_n_3]-sig[index_n_1] # set of all relative amplitudes
        (nu_star,index_n_star)=(max(ampl),np.argmax(ampl))   # max of relative amplitudes
        n_3_star=int(index_n_3[index_n_star])   # index of peak
        n_1_star=int(index_n_1[index_n_star])   # index of the min between peak and trend
        n_2_star=np.where(sig<=sig[n_3_star])[0][0]  # index of the value on the trend
    else:
        nu_star=0
        index_n_star=len(sig)
        n_1_star=len(sig)
        n_2_star=len(sig)
        n_3_star=len(sig)
    return(nu_star,index_n_star,n_1_star,n_2_star,n_3_star)


def deltaChoice(x,y):
    """ Get the smoothing parameter delta    
    Parameters: 
        x - Time
        y - Signal
       
    Returns:
        delta - smoothing parameter

    Remarks: see "Testing high-frequency features in a noisy signal" for 
    references. Modify the variable delta_var in order to increase or decrease 
    the set of smoothing parameters.
    """
    dt=x[int(len(x)/2)]-x[int(len(x)/2-1)]
    Fs=1/(dt*3600)
    (P1,f)=get_fft(y,Fs)
    delta_var=(np.arange(int(np.sqrt(len(x)/8)))+1)*2
    n=len(delta_var)
    G=[] # vector of relative amplitude
    for di in delta_var:
        (fb,En)=EnergyBand(f,P1,di,'Roll')
        (nu_star,unused,n_1_star,n_2_star,n_3_star)=HFF_Parameters(fb,np.power(En,2))
        G.append(np.array(n_3_star-n_2_star))#G.append(np.array(n_3_star-n_2_star)[0])
    U=[] # range 2 by 2
    for ie in range(n-1):
        U.append(np.abs(G[ie+1]-G[ie]))
    tau=np.argmax(U)
    g_star=max(G[tau+1],G[tau])
    if g_star==G[tau+1]:
        delta=delta_var[tau+1]
    else:
        delta=delta_var[tau]
    return(delta)

#------------------------------------------------------------------------------
# Procedure to simulate the null
#------------------------------------------------------------------------------

def _second_order_derivative_matrix(size_of_matrix):
    """ Return a second order derivative matrix
    for a given signal size
    Parameters:
        size_of_matrix(int) - Size of matrix
    Returns:
        second_order(cvxopt.spmatrix) - Sparse matrix
        that has the second order derivative matrix
    """
    temp = size_of_matrix - 2
    first = [1, -2, 1] * temp
    second = list(chain.from_iterable([[ii] * 3 for ii in range(temp)]))
    third = list(chain.from_iterable([[ii, ii + 1, ii + 2] for ii in range(temp)]))
    second_order = spmatrix(first, second, third)

    return second_order


def _l1(signal, regularizer):
    """
    Parameters:
        signal(np.ndarray) - Signal
        regularizer(float) - regularizer to keep the balance between smoothing
            and 'truthfulness' of the signal
    Returns:
        trend(np.ndarray) - Trend of the signal extracted from l1 
        regularization

    Problem Formulation:
        minimize    (1/2) * ||x - signal||_2^2 + regularizer * sum(y)
        subject to  | D*x | <= y

    """

    signal_size = signal.size[0]
    temp = signal_size - 2
    temp_ls = range(temp)

    D = _second_order_derivative_matrix(signal_size)
    P = D * D.T
    q = -D * signal

    G = spmatrix([], [], [], (2 * temp, temp))
    G[:temp, :temp] = spmatrix(1.0, temp_ls, temp_ls)
    G[temp:, :temp] = -spmatrix(1.0, temp_ls, temp_ls)
    h = matrix(regularizer, (2 * temp, 1), tc='d')
    residual = solvers.qp(P, q, G, h)
    trend =  signal - D.T * residual['x']

    return trend    
 
def l1(signal, regularizer):
    """
    Fits the l1 trend on top of the `signal` with a particular
    `regularizer`
    Parameters:
            signal(np.ndarray) - Original Signal that we want to fit l1
                trend
            regularizer(float) - Regularizer which provides a balance between
                smoothing of a signal and truthfulness of signal
    Returns:
        values(np.array) - l1 Trend of a signal that is extracted 
        from the signal
    """

    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal Needs to be a numpy array")

    m = float(signal.min())
    M = float(signal.max())
    difference = M - m
    if not difference: # If signal is constant
        difference = 1
    t = (signal - m) / difference

    values = matrix(t)
    values = _l1(values, regularizer)
    values = values * difference + m
    values = np.asarray(values).squeeze()

    fig, ax = getFig(1,'Trend Estimation')
    x=np.linspace(0,1,len(signal))
    ax.plot(x, signal, label='Signal')
    ax.plot(x, values, label='Trend')
    fig.legend()
    return values  

def mad(a, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    Parameters:
        a - input sequence
    Returns:
        mad - value of the median absolute deviation
    """
    # Median along given axis
    k=1.4826 # 1/ phi^{-1}(3/4) 
    med = np.median(a, axis=axis, keepdims=True)
    mad = k*np.median(np.absolute(a - med), axis=axis)
    return(mad)

def Est_std(sig,wlt):
    """Return the median absolute deviation estimator of the standard deviation
    of the noise of a 1-D signal. The MAD estimator is computed on the wavelet 
    coefficients at maximum resolution.  
    
    Parameters: 
        sig - 1-D signal
        wlt - name of the discrete wavelet choice (example: 'sym8')
    Returns: 
        sigma - standard deviation estimator
    """
    w=pywt.Wavelet(wlt)
    J=pywt.dwt_max_level(len(sig),w)
    toto=pywt.wavedec(sig, 'sym8', mode='symmetric', level=J, axis=-1)
    sigma=mad(toto[len(toto)-1])
    return(sigma)
#------------------------------------------------------------------------------
# Monte Carlo procedure for the computation of the p-value
#------------------------------------------------------------------------------

def MC_COP_H0(t,trend,sigma,n_iter,delta):
    """Compute the cloud of points (corresponding of the HFF parameters of the 
    simulated signals corresponding to the null hypothesis)
    Parameters:
        t - time vector
        trend - trend vector of the signal
        sigma - standard deviation of the signal
        n_iter - number of iteration (size of the cloud of points)
        delta - smoothing parameter of the signal
    Return:
        g - vector of the gap in frequency between the trend and the 
        oscillations (null hypothesis)
        d - vector of the relative amplitudes of the oscillations (null 
        hypothesis)
    """
    g0=[]
    d0=[]
    dt=t[3]-t[2]
    fs_sig=1/(3600*dt)
    for i in range(n_iter):
        sig=trend+np.random.normal(0,sigma,len(t))
        P1,f=get_fft(sig,fs_sig)
        (fb,En)=EnergyBand(f,P1,delta,'Roll')
        (nu_star,unused,n_1_star,n_2_star,n_3_star)=HFF_Parameters(fb,np.power(En,2))
        d0.append(np.reshape(nu_star,1))
        g0.append(n_3_star-n_2_star)
    g=np.reshape(np.array(g0),(n_iter,))
    d=np.reshape(np.array(d0),(n_iter,))
    return(g,d)

def RejZoneAlpha(g,d,gmu,dmu,alpha):
    """Compute for each point of the cloud of the points {(g,d)} the proportion
    of points in the North-East quarter of the plane.
    Parameters:
        g - vector of the gap in frequency between the trend and the 
        oscillations (null hypothesis)
        d - vector of the relative amplitudes of the oscillations (null 
        hypothesis)
        gmu - gap in frequency between the trend and the 
        oscillations (signal subject to the test)
        dmu - relative amplitude of the oscillations (signal subject to the 
        test)
        alpha - level of the test (number of false negative acceptable in %)
    Returns:
        c_alpha - set of threshold for the gap in frequency of level alpha 
        nu_alpha - set of threshold for the relative amplitude of level alpha
        g_mesh - x-axis for the grid of the mesh (frequency gap)
        d_mesh - y-axis for the grid of the mesh (relative amplitude)
        mesh - necessary grid to compute threshold of any level alpha
    """
    
    n=len(g)
    K=int(alpha*n)
    g_mesh=np.unique(g) # ordered vector of gap in frequency  
    d_mesh=np.unique(d)  # ordered vector of amplitude 
    x=np.vstack((g,d))
    x=np.transpose(x)
    mesh=np.zeros(n)
    for i in range(n):
        xtest=np.concatenate((np.ones((n,1))*g[i],np.ones((n,1))*d[i]),axis=1) # vector for the 2D order relation
        xtilde=(x>=xtest).astype(int) # 2D order relation
        out=xtilde[np.all(xtilde!=0,axis=1)] # get rid of rows with 0 in it
        mesh[i]=len(out[:,1]) # number of couple in the north east corner of the plane
    index_alpha=np.where(abs(mesh-K)<(1/n))
    c_alpha=g[index_alpha]
    nu_alpha=d[index_alpha]

    fig, ax = getFig(1,'Cloud of points and HFF parameters')
    ax.scatter(g,d, c='b', s=10)
    ax.set_xlabel('Frequency gap g')
    ax.set_ylabel('Relative Amplitude d')
    ax.scatter(gmu, dmu, c='r', s=40)
    
    ax.set_ylim([min(np.hstack((d,dmu))),max(np.hstack((d,dmu)))])
    
    return(c_alpha,nu_alpha,g_mesh,d_mesh,mesh)    
    
def Pvalue(g,d,g_mesh,d_mesh,mesh,gmu,dmu):
    """ Theoretical p-value is the smallest level for which we reject H0
    pval=inf{\alpha, Z\in R_{\alpha}}, Z signal which undergoes the test
    Parameters:
        gmu - gap in frequency between the oscillations and the trend for Z
        dmu - relative amplitude of the oscillations for Z
        mesh - grid to get the Reject Region for levels alpha 
        (g,d) - tuple of the MC simulation for signals in H0
        (gmu,dmu) - Relative amplitude and Frequency gap for observations Z
    Returns:
        Pval - P-value of the test for the tuple (g,d)
    """
    n=len(g)
    x=np.vstack((g,d))
    x=np.transpose(x) # vector to compute the 2D order relation
    xtest=np.concatenate((np.ones((n,1))*gmu,np.ones((n,1))*dmu),axis=1) # vector for the 2D order relation
    xtilde=(x<=xtest).astype(int) # 2D order relation calpha<=gmu and nualpha<=dmu
    xtemp=np.concatenate((np.ones((n,1)),np.ones((n,1))),axis=1)
    for i in range(n): # Loop to remove when only one order relation is satisfied
        if xtilde[i,0]==0:
            xtemp[i,1]=0
            xtemp[i,0]=0
        if xtilde[i,1]==0:
            xtemp[i,0]=0
            xtemp[i,1]=0
    index=np.where(xtemp[:,0]==1) # Set of indexes where the order relation is fulfilled
    D=mesh[index]/n # alpha associated to these couples (calpha,nualpha)
    if D.size:
        Pval=min(D)
    else:
        Pval=1
        print("The statistical test can not conclude")
    return(Pval)

#------------------------------------------------------------------------------
# Construction of the test signals 
#------------------------------------------------------------------------------
def SigOsc(x,ampl,sigma):
    """Return an oscillating signal with decreasing trend.
    Parameters:
        x - time vector
        ampl - amplitude of the oscillations
    Return:
        y - Oscillating signal
    """
    y1=1/np.sqrt((x+1))
    t1=np.zeros((int(len(x)/4),))
    t2=x[int(len(x)/4):int(3*len(x)/4)]
    t3=np.hstack((t1,t2))
    t=np.hstack((t3,t1))
    y2=ampl*np.sin(2*np.pi*t/6)
    y=y1+y2+5
    sig=y+np.random.normal(0,sigma,len(t))
    
    fig, ax = getFig(2, 'Test signal')
    ax[0].plot(x,y)
    ax[0].set_xlabel('time axis')
    ax[0].set_ylabel('real signal')
    ax[1].plot(x,sig)
    ax[1].set_xlabel('time axis')
    ax[1].set_ylabel('noisy signal')
    fig.tight_layout()    
    return(sig)

def GetHFF(x,y,delta):
    """Compute and display the HF parameters of a signal on its single sided amplitude spectrum.
    
    Parameters:
        x - time vector
        y - signal
        delta - smoothing parameter
    Return:
        g - vector of the gap in frequency between the trend and the oscillations
        d - relative amplitude of the oscillations
    """
    # Fourier Transform frequency sampling
    dt=x[int(len(x)/2)]-x[int(len(x)/2-1)]
    Fs=1/(dt)
    (P1,f)=get_fft(y,Fs)
    # Smoothing the Fourier transform
    (fb,En)=EnergyBand(f,P1,delta,'Roll')
    #(fb,En)=(f,P1)
    # HFF parameters detection
    (nu_star,unused,n_1_star,n_2_star,n_3_star)=HFF_Parameters(fb,np.power(En,2))
    d=nu_star # relative amplitude of the HFF
    g=n_3_star-n_2_star # frequency interval between HFF and trend 
    # plot of the smoothed Fourier transfor and the HFF parameters
    sig=En
    
    fig, ax = getFig(1, 'Amplitude spectrum and HFF parameters')
    ax.plot(fb,sig)
    ax.plot(fb[n_1_star],sig[n_1_star],'rD')
    ax.plot(fb[n_2_star],sig[n_2_star],'rD')
    ax.plot(fb[n_3_star],sig[n_3_star],'bD')
    ax.set_yscale('log')
    ax.set_xlabel('Freq (Hz)')
    ax.set_ylabel('|Y(freq)|')
    return(g,d)
def SanityCheckFunct(time,sigma,ca,cf):
    """ Build an oscillating signal with a trend like the Lennard John potential
    
    Parameters:
        time - discretisation x-axis
        sigma - standard deviation of the noise
        ca   - amplitude of the oscillations
        cf   - frenquency of the oscillations in 3600*Hz
    Returns:
        t - time 
        sc - oscillating signal   
    """
    # Parameters for the trend of the signal
    c1=0.4
    c2=2
    c3=4
    c4=5
    n=len(time)
    
    # Trend construction
    t_temp1=time[int(n/8):]
    trend2 = c1*(np.power(c2*np.true_divide(1,t_temp1),8) \
                 - c3*np.power(c2*np.true_divide(1,t_temp1),4))+c4
    t_temp2=time[:int(n/8)]
    a=time[int(n/8-10)]
    b=time[int(n/8)]
    f_a=c1*((c2/a)**8 - c3*(c2/a)**3)+c4
    f_b=c1*((c2/b)**8 - c3*(c2/b)**3)+c4
    trend1=((f_b-f_a)/b)*t_temp2+f_a;
    trend=np.concatenate((trend1,trend2))
    
    # oscillations construction
    t2=np.concatenate((np.ones(int(n/8-1))*time[int(n/8)],time[int(n/8):int(2*n/8)+1],\
                       np.ones(len(time)-int(2*n/8))*time[int(n/8)]));
    osc=ca*np.multiply(np.multiply((t2-time[int(int(n/8))]),(time[int(2*n/8)]-t2)),np.sin(t2*2*np.pi*cf))\
                       *(4/(time[int(2*n/8)]-time[int(n/8)])**2)
                       
    # Noise construction Gaussian distributed
    noise=np.random.normal(0,sigma,len(time))
    
    sc=trend+osc+noise
    fig, ax = getFig(2, 'Test signal')
    ax[0].plot(time,trend+osc)
    ax[0].set_xlabel('time axis')
    ax[0].set_ylabel('real signal')
    ax[1].plot(time,sc)
    ax[1].set_xlabel('time axis')
    ax[1].set_ylabel('noisy signal')
    fig.tight_layout() 
    return(sc)
