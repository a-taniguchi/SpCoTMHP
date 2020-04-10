#coding:utf-8
#This file for general modules (一般性の高い関数はこちらへ集約)
#Akira Taniguchi 2018/11/26-2018/12/17-
import os
import numpy as np
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2,gamma,lgamma
from __init__ import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass
        
def fill_param(param, default):   ##パラメータをNone の場合のみデフォルト値に差し替える関数
    if (param == None): return default
    else: return param
        
def multivariate_t_distribution(x, mu, Sigma, df):
    """
    Multivariate t-student density. Returns the density
    of the function at points specified by x.
    
    input:
        x = parameter (n-d numpy array; will be forced to 2d)
        mu = mean (d dimensional numpy array)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        
    Edited from: http://stackoverflow.com/a/29804411/3521179
    """
    
    x = np.atleast_2d(x) # requires x as 2d
    nD = Sigma.shape[0] # dimensionality
    
    numerator = gamma(1.0 * (nD + df) / 2.0)
    denominator = (
            gamma(1.0 * df / 2.0) * 
            np.power(df * PI, 1.0 * nD / 2.0) *  
            np.power(np.linalg.det(Sigma), 1.0 / 2.0) * 
            np.power(
                1.0 + (1.0 / df) *
                np.diagonal(
                    np.dot( np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)
                ), 
                1.0 * (nD + df) / 2.0
                )
            )
            
    return 1.0 * numerator / denominator 

def log_multivariate_t_distribution(x, mu, Sigma, df):
    x = np.atleast_2d(x) # requires x as 2d
    nD = Sigma.shape[0] # dimensionality
    
    lnumerator = lgamma( (nD + df) / 2.0 )
    ldenominator = (
            lgamma(0.5 * df) + 
            (0.5 * nD) * ( log(df) + log(PI) ) + 
            0.5 * log(np.linalg.det(Sigma))  + 
            (0.5 * (nD + df)) * 
            log(1.0 + (1.0 / df) * np.diagonal(np.dot( np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)))
            )
            
    return lnumerator - ldenominator 

# 計算された共分散行列のパラメータが正定値性を満たすか簡易チェック
def Check_VN(VN):
  if (VN[0][0] <= 0 or VN[1][1] <= 0 ):
    print "ERROR!!!! Posterior parameter VN is negative."
    print VN
    VN = V0
  return VN

# ガウス-逆ウィシャート分布(NIW)の事後分布のパラメータ推定の計算
def PosteriorParameterGIW(k,nk,step,IT,XT,icitems_k0):
  ###kについて、ITが同じものを集める
  if nk != 0 :  #もしzaの中にkがあれば(計算短縮処理)        ##0ワリ回避
    xk = []
    for s in xrange(step) : 
      if IT[s] == icitems_k0 : 
        xk = xk + [ np.array([XT[s].x, XT[s].y]) ]
    m_ML = sum(xk) / float(nk) #fsumではダメ
    print "K%d n:%d m_ML:%s" % (k,nk,str(m_ML))
    
    ##ハイパーパラメータ更新
    kN = k0 + nk
    mN = ( k0*m0 + nk*m_ML ) / kN  #dim 次元横ベクトル
    nN = n0 + nk
    #VN = V0 + sum([np.dot(np.array([xk[j]-m_ML]).T,np.array([xk[j]-m_ML])) for j in xrange(nk)]) + (k0*nk/kN)*np.dot(np.array([m_ML-m0]).T,np.array([m_ML-m0])) #旧バージョン
    VN = V0 + sum([np.dot(np.array([xk[j]]).T,np.array([xk[j]])) for j in xrange(nk)]) + k0m0m0 - kN*np.dot(np.array([mN]).T,np.array([mN]))  #speed up? #NIWを仮定した場合、V0は逆行列にしなくてよい
    VN = Check_VN(VN)
    
  else:  #データがないとき
    print "nk["+str(k)+"]="+str(nk)
    kN = k0
    mN = m0
    nN = n0
    VN = V0
  
  return kN,mN,nN,VN

#http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section5_2-Dirichlet-Processes.ipynb
def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()


def levenshtein_distance(a, b):
    m = [ [0] * (len(b) + 1) for i in range(len(a) + 1) ]

    for i in xrange(len(a) + 1):
        m[i][0] = i

    for j in xrange(len(b) + 1):
        m[0][j] = j

    for i in xrange(1, len(a) + 1):
        for j in xrange(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                x = 0
            else:
                x = 1
            m[i][j] = min(m[i - 1][j] + 1, m[i][ j - 1] + 1, m[i - 1][j - 1] + x)
    # print m
    return m[-1][-1]
