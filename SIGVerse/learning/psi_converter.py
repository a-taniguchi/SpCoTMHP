#coding:utf-8

##############################################
## psi.npyを変換するためのプログラム
## Akira Taniguchi 2021/77
##############################################
# [command] $ python psi_converter.py <trialname>

import glob
import os
import os.path
import sys
import random
import shutil
import time
import numpy as np
import scipy as sp
from numpy.random import uniform,dirichlet
from scipy.stats import multivariate_normal,invwishart,multinomial
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum #,degrees,radians,atan2,gamma,lgamma
from __init__ import *
#from JuliusLattice_dec import *
from submodules import *

iteration = 0 # if (ITERATION == 1 )
sample_max = 0
datasetfolder = inputfolder

# 学習済みパラメータフォルダ名 trialname を得る
trialname = sys.argv[1]

file_trialname   = outputfolder + trialname +'/' + trialname
iteration_sample = str(iteration+1) + "_" + str(sample_max) 

# Spatial concept の Transition probability parameter (psi) を読み込む
psi = np.loadtxt(file_trialname + '_psi_setting.csv', delimiter=',')

temp = np.ones((K,K)) * omega0 + psi

psi_convert = [ np.mean(dirichlet(temp[k],Robust_psi),0) for k in range(K) ]
          

# 変換処理
#for k1 in range(K):
#  for k2 in range(K):
#    psi_convert[k1][k2] = psi_convert[k1]


# 保存
np.save(file_trialname + '_psi_'   + iteration_sample, psi_convert)


