#coding:utf-8

##############################################
## Spatial concept formation model (SpCoA++ with lexical acquisition)
## For SpCoTMHP (on SIGVerse)
## Learning algorithm is Gibbs sampling.
## Akira Taniguchi 2020/04/14-
##############################################
# python ./learnSpCoTMHP_SIGVerse.py trialname ** (3LDK_**)

import glob
import codecs
import re
import os
import os.path
import sys
import random
import string
import numpy as np
import scipy as sp
from numpy.random import uniform,dirichlet #multinomial
from scipy.stats import multivariate_normal,invwishart,multinomial #,rv_discrete
#from numpy.linalg import inv, cholesky
#from scipy.stats import chi2
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum #,degrees,radians,atan2,gamma,lgamma
from initSpCoSLAMSIGVerse import *
from JuliusLattice_dec_SIGVerse import *
from submodules import *
import time
from hmmlearn import hmm
import pickle
#import joblib

#Mutual information (binary variable): word_index, W, π, Ct
def MI_binary(b,W,pi,c):
    #相互情報量の計算
    POC = W[c][b] * pi[c]    #場所の名前の多項分布と場所概念の多項分布の積
    PO = sum([W[ct][b] * pi[ct] for ct in xrange(L)]) 
    PC = pi[c]
    POb = 1.0 - PO
    PCb = 1.0 - PC
    PObCb = PCb - PO + POC
    POCb = PO - POC
    PObC = PC - POC
    
    # Calculate each term for MI 
    temp1 = POC * log(POC/(PO*PC), 2)
    temp2 = POCb * log(POCb/(PO*PCb), 2)
    temp3 = PObC * log(PObC/(POb*PC), 2)
    temp4 = PObCb * log(PObCb/(POb*PCb), 2)
    score = temp1 + temp2 + temp3 + temp4
    return score

def Mutual_Info(W,pi):  #Mutual information: W, π 
    MI = 0
    for c in xrange(len(pi)):
      PC = pi[c]
      for j in xrange(len(W[c])):
        PO = fsum([W[ct][j] * pi[ct] for ct in xrange(len(pi))]) 
        POC = W[c][j] * pi[c]   #場所の名前の多項分布と場所概念の多項分布の積
        
        # Calculate each term for MI
        MI = MI + POC * ( log((POC/(PO*PC)), 2) )
    
    return MI

#All parameters and initial values are output
def SaveParameters_init(filename, trialname, iteration, sample, THETA_init, Ct_init, It_init, N, TN):
  phi_init, pi_init, W_init, theta_init, Mu_init, S_init = THETA_init  #THETA = [phi, pi, W, theta, Mu, S]

  fp_init = open( filename + '/' + trialname + '_init_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
  fp_init.write('init_data\n')  #num_iter = 10  #The number of iterations
  fp_init.write('L,'+repr(L)+'\n')
  fp_init.write('K,'+repr(K)+'\n')
  fp_init.write('alpha0,'+repr(alpha0)+'\n')
  fp_init.write('gamma0,'+repr(gamma0)+'\n')
  fp_init.write('bata0,'+repr(beta0)+'\n')
  fp_init.write('k0,'+repr(k0)+'\n')
  fp_init.write('m0,'+repr(m0)+'\n')
  fp_init.write('V0,'+repr(V0)+'\n')
  fp_init.write('n0,'+repr(n0)+'\n')
  fp_init.write('sigma_init,'+repr(sig_init)+'\n')
  #fp_init.write('M,'+repr(M)+'\n')
  fp_init.write('N,'+repr(N)+'\n')
  fp_init.write('TN,'+repr(TN)+'\n')
  fp_init.write('Ct_init\n')
  for i in xrange(N):
    fp_init.write(repr(i)+',')
  fp_init.write('\n')
  for i in xrange(N):
    fp_init.write(repr(Ct_init[i])+',')
  fp_init.write('\n')
  fp_init.write('It_init\n')
  for i in xrange(N):
    fp_init.write(repr(i)+',')
  fp_init.write('\n')
  for i in xrange(N):
    fp_init.write(repr(It_init[i])+',')
  fp_init.write('\n')
  fp_init.write('Position distribution_init\n')
  for k in xrange(K):
    fp_init.write('Mu_init'+repr(k)+',')
    for dim in xrange(dimx):
      fp_init.write(repr(float(Mu_init[k][dim]))+',')
    fp_init.write('\n')
  for k in xrange(K):
    fp_init.write('Sig_init'+repr(k)+'\n')
    fp_init.write(repr(S_init[k])+'\n')
  for c in xrange(L):
    fp_init.write('W_init'+repr(c)+','+repr(W_init[c])+'\n')
  for c in xrange(L):
    fp_init.write('theta_init'+repr(c)+','+repr(theta_init[c])+'\n')
  for c in xrange(L):
    fp_init.write(',')
    for k in xrange(K):
      fp_init.write(repr(k)+',')
    fp_init.write('\n')
    fp_init.write('phi_init'+repr(c)+',')
    for k in xrange(K):
      fp_init.write(repr(phi_init[c][k])+',')
    fp_init.write('\n')
  fp_init.write(',')
  for c in xrange(L):
    fp_init.write(repr(c)+',')
  fp_init.write('\n')
  fp_init.write('pi_init'+',')
  for c in xrange(L):
    fp_init.write(repr(pi_init[c])+',')
  fp_init.write('\n')
  fp_init.close()

#Samplingごとに各paramters値をoutput
def SaveParameters_all(filename, trialname, iteration, sample, THETA, Ct, It, W_index):
  phi, pi, W, theta, Mu, S = THETA  #THETA = [phi, pi, W, theta, Mu, S]
  N = len(Ct)

  fp = open( filename + '/' + trialname +'_kekka_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
  fp.write('sampling_data,'+repr(num_iter)+'\n')  #num_iter = 10  #The number of iterations
  fp.write('Ct\n')
  for i in xrange(N):
    fp.write(repr(i)+',')
  fp.write('\n')
  for i in xrange(N):
    fp.write(repr(Ct[i])+',')
  fp.write('\n')
  fp.write('It\n')
  for i in xrange(N):
    fp.write(repr(i)+',')
  fp.write('\n')
  for i in xrange(N):
    fp.write(repr(It[i])+',')
  fp.write('\n')
  fp.write('Position distribution\n')
  for k in xrange(K):
    fp.write('Mu'+repr(k)+',')
    for dim in xrange(dimx):
      fp.write(repr(float(Mu[k][dim]))+',')
    fp.write('\n')
  for k in xrange(K):
    fp.write('Sig'+repr(k)+'\n')
    fp.write(repr(S[k])+'\n')
  
  for c in xrange(L):
    fp.write(',')
    for i in xrange(len(W_index)):
      fp.write(W_index[i] + ',')   #####空白が入っているものがあるので注意(', ')
    fp.write('\n')
    fp.write('W'+repr(c)+',')
    for i in xrange(len(W_index)):
      fp.write(repr(W[c][i])+',')
    fp.write('\n')
    
  for c in xrange(L):
    fp.write(',')
    for i in xrange(DimImg):
      fp.write(repr(i) + ',')   #####空白が入っているものがあるので注意(', ')
    fp.write('\n')
    fp.write('theta'+repr(c)+',')
    for i in xrange(DimImg):
      fp.write(repr(theta[c][i])+',')
    fp.write('\n')

  for c in xrange(L):
    fp.write(',')
    for k in xrange(K):
      fp.write(repr(k)+',')
    fp.write('\n')
    fp.write('phi'+repr(c)+',')
    for k in xrange(K):
      fp.write(repr(phi[c][k])+',')
    fp.write('\n')
  fp.write(',')
  for c in xrange(L):
    fp.write(repr(c)+',')
  fp.write('\n')
  fp.write('pi'+',')
  for c in xrange(L):
    fp.write(repr(pi[c])+',')
  fp.write('\n')
  fp.close()

  #fp_x = open( filename + '/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
  #for t in xrange(len(Xt[t])) : 
  #  fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
  #fp_x.close()
        

# Saving data for parameters Θ of spatial concepts
def SaveParameter_EachFile(filename, trialname, iteration, sample, THETA, Ct, It):
  phi, pi, W, theta, Mu, S = THETA  #THETA = [phi, pi, W, theta, Mu, S]
  file_trialname   = filename + '/' + trialname
  iteration_sample = str(iteration) + "_" + str(sample) 
  N = DATA_NUM

  fp = open( file_trialname + '_Mu_'+ iteration_sample + '.csv', 'w')
  for k in xrange(K):
    for dim in xrange(dimx):
      fp.write(repr(float(Mu[k][dim]))+',')
    fp.write('\n')
  fp.close()

  fp = open( file_trialname + '_S_'+ iteration_sample + '.csv', 'w')
  for k in xrange(K):
    for dim in xrange(dimx):
      for dim2 in xrange(dimx):
        fp.write(repr(S[k][dim][dim2])+',')
    fp.write('\n')
  fp.close()

  fp = open( file_trialname + '_W_'+ iteration_sample + '.csv', 'w')
  for c in xrange(L):
    for i in xrange(len(W[c])): #len(W_index)
      fp.write(repr(W[c][i])+',')
    fp.write('\n')
  fp.close()

  fp = open( file_trialname + '_theta_'+ iteration_sample + '.csv', 'w')
  for c in xrange(L):
    for i in xrange(len(theta[c])): 
      fp.write(repr(theta[c][i])+',')
    fp.write('\n')
  fp.close()

  fp = open( file_trialname + '_phi_'+ iteration_sample + '.csv', 'w')
  for c in xrange(L):
    for k in xrange(K):
      fp.write(repr(phi[c][k])+',')
    fp.write('\n')
  fp.close()

  fp = open( file_trialname + '_pi_'+ iteration_sample + '.csv', 'w')
  for c in xrange(L):
    fp.write(repr(pi[c])+',')
  fp.write('\n')
  fp.close()
  
  fp = open( file_trialname + '_Ct_'+ iteration_sample + '.csv', 'w')
  for t in xrange(N):
    fp.write(repr(Ct[t])+',')
  fp.write('\n')
  fp.close()
  
  fp = open( file_trialname + '_It_'+ iteration_sample + '.csv', 'w')
  for t in xrange(N):
    fp.write(repr(It[t])+',')
  fp.write('\n')
  fp.close()

  #fp = open( filename + "/W_list.csv", 'w')
  #for w in xrange(len(W_index)):
  #  fp.write(W_index[w]+",")
  #fp.close()

##Output to file: the set of word recognition results
def SaveWordData(filename, trialname, iteration, sample, W_index, Otb):
  N = DATA_NUM
  #filename_ot = raw_input("Otb:filename?(.csv) >")  #ファイル名を個別に指定する場合
  #filename_ot = trialname
  fp = open(filename + '/' + trialname + '_ot_'+ str(iteration) + "_" + str(sample) + '.csv', 'w')
  fp2 = open(filename + '/' + trialname + '_w_index_'+ str(iteration) + "_" + str(sample) + '.csv', 'w')
  for n in xrange(N) : 
      for j in xrange(len(Otb[n])):
          fp.write(Otb[n][j] + ',')
      fp.write('\n')
  for i in xrange(len(W_index)):
      fp2.write(repr(i) + ',')
  fp2.write('\n')
  for i in xrange(len(W_index)):
      fp2.write(W_index[i] + ',')
  fp.close()
  fp2.close()

"""
# Read Position data(SpCoNaviのSIGVerse実験で使用したバージョン)
def position_data_read_pass(directory,DATA_NUM):
    all_position=[] 
    hosei = 1  # 04 is *2, 06 is -1, 10 is *1.5.
    
    ##### 座標の補正 #####
    if ("04" in directory):
      hosei *= 2
      print "hosei",hosei
    elif ("06" in directory):
      hosei *= -1
      print "hosei",hosei
    elif ("10" in directory):
      hosei *= 1.5
      print "hosei",hosei
    ######################

    for i in range(DATA_NUM):
      #if  (i in test_num)==False:
      f=directory+"/position/"+repr(i)+".txt"
      position=[] #(x,y,sin,cos)
      itigyoume = 1
      for line in open(f, 'r').readlines():
          if (itigyoume == 1):
            data=line[:-1].split('	')
            #print data
            position +=[float(data[0])*(-1) + float(origin[0]*resolution)*hosei]
            position +=[float(data[1])]
            itigyoume = 0
      all_position.append(position)
    
    return np.array(all_position)
"""
# Read Position data
def ReadPositionData():
  N  = 0
  Xt = []  #Xt = [(0.0,0.0) for n in xrange(len(HTW)) ]
  TN = []
  for line in open(datasetfolder + datasetname + PositionDataFile, 'r'):
    itemList = line[:-1].split(',')
    Xt = Xt + [(float(itemList[0]), float(itemList[1]))]
    TN = TN + [N]
    N  = N + 1    
    #print TN

  if (N != DATA_NUM):   ##教示フェーズの教示数と読み込んだ発話文データ数が違う場合
    print "DATA_NUM" + str(DATA_NUM) + ":KYOUJI error!! N:" + str(N)
    #DATA_NUM = N
    #exit()
  return Xt,TN

# Read Image feature data (フォルダパス設定が必要)
def ReadImageData():
  FT = []
  for s in xrange(DATA_NUM):
    for line in open( datasetfolder + datasetname + 'img/ft' + str(s+1) + '.csv', 'r'):
      itemList = line[:].split(',')
    FT.append( [float(itemList[i]) for i in xrange(DimImg)] )
  return FT

# Reading data for image feature (For SIGVerse)
def ReadImageData_SIGVerse():
  FT = []
  files = glob.glob(datasetfolder + datasetname + '/' + Descriptor + "/*.csv")
  files.sort()

  for s in xrange(DATA_NUM):
    FT_temp = []
    i = 0
    for line in open( files[s], 'r'):
      #itemList = line[:].split(',')
      if (i < DimImg):
        FT_temp.append(float(line)*Feture_times)
      i += 1
    if (Feture_noize > 0.0):
      FT_temp = [FT_temp[i]+(Feture_noize/float(DimImg)) for i in range(DimImg)] 
    if (Feture_sum_1 == 1):
      Ft_sum = sum(FT_temp)
      FT_temp = [FT_temp[i]/float(Ft_sum) for i in range(DimImg)] 
    FT.append( FT_temp )
    #print files[s]

  if( DATA_NUM != len(FT) ):
    print "ERROR FT", DATA_NUM, len(FT)
  else:
    print "READ FT", DATA_NUM, len(FT[0])
  #print FT
  return FT

# Read Word data (segmenated word sequences)
def ReadWordData(iteration):
  Otb_Samp     = [ [ [] for _ in xrange(DATA_NUM)] for _ in range(sample_num) ]
  W_index_Samp = [ [ [] for _ in xrange(DATA_NUM)] for _ in range(sample_num) ]
  N = DATA_NUM

  ##発話認識文(単語)データを読み込む
  ##空白またはカンマで区切られた単語を行ごとに読み込むことを想定する
  for sample in xrange(sample_num):
    #N = 0
    Otb = []
    #Read text file
    for line in open(filename + '/out_gmm_' + str(iteration) + '/' + str(sample) + '_samp.100', 'r'):   ##*_samp.100を順番に読み込む
      itemList = line[:-1].split(' ')
      
      #remove <s>,<sp>,</s> and "\r", "": if its were segmented to words.
      itemList = Ignore_SP_Tags(itemList)
      
      #Otb[sample] = Otb[sample] + [itemList]
      Otb = Otb + [itemList]
      #N = N + 1  #count

      #for j in xrange(len(itemList)):
      #    print "%s " % (str(itemList[j])),
      #print ""  #改行用
    
    ##For index of multinominal distribution of place names
    W_index = []
    for n in xrange(N):
      for j in xrange(len(Otb[n])):
        if ( (Otb[n][j] in W_index) == False ):
          W_index.append(Otb[n][j])
          #print str(W_index),len(W_index)
    
    print "[",
    for i in xrange(len(W_index)):
      print "\""+ str(i) + ":" + str(W_index[i]) + "\",",
    print "]"
    
    ##Vectorize: Bag-of-Words for each time-step n (=t)
    Otb_B = [ [0 for i in xrange(len(W_index))] for n in xrange(N) ]
    
    for n in xrange(N):
      for j in xrange(len(Otb[n])):
        for i in xrange(len(W_index)):
          if ( W_index[i] == Otb[n][j] ):
            Otb_B[n][i] += word_increment
    #print Otb_B
    
    Otb_Samp[sample], W_index_Samp[sample] = Otb_B, W_index
    SaveWordData(filename, trialname, iteration, sample, W_index, Otb)

  return Otb_Samp, W_index_Samp

######################################################
# Gibbs sampling
######################################################
def Gibbs_Sampling(iteration, Otb_Samp, W_index_Samp, Xt, TN, Ft):
    Ct_Samp     = [ [ 0 for _ in xrange(DATA_NUM)] for _ in range(sample_num) ]
    THETA_Samp  = [ [] for _ in range(sample_num) ]
    N = DATA_NUM

    for sample in xrange(sample_num):
      #TN = [i for i in xrange(N)]   #TN[N]: teaching time-step #テスト用教示時刻(step)集合
      #Otb_B[N][W_index]：時刻tごとの発話文をBOWにしたものの集合
      Otb_B   = Otb_Samp[sample]
      W_index = W_index_Samp[sample]

      ##Initialization of all parameters
      print u"Initialize Parameters..."

      # index of spatial concepts [N]
      Ct = [ int(random.uniform(0,L)) for _ in xrange(N) ] #[ int(n/15) for n in xrange(N)]  
      # index of position distributions [N]
      It = [ int(random.uniform(0,K)) for _ in xrange(N) ] #[ int(n/15) for n in xrange(N)]  
      ## Uniform random numbers within the range
      # the position distribution (Gaussian)の平均(x,y)[K]
      Mu = [ np.array([ int( random.uniform(WallXmin,WallXmax) ) ,
                        int( random.uniform(WallYmin,WallYmax) ) ]) for _ in xrange(K) ]      
      # the position distribution (Gaussian)の共分散(2×2-dimension)[K]
      S  = [ np.eye(dimx) * sig_init for _ in xrange(K) ] 
      # the name of place (multinomial distribution: W_index-dimension)[L]
      W  = [ [ 1.0/len(W_index) for _ in xrange(len(W_index)) ] for _ in xrange(L) ]
      # the image feature (multinomial distribution: DimImg-dimension)[L]
      theta  = [ [ 1.0/DimImg for _ in xrange(DimImg) ] for _ in xrange(L) ]
      if (nonpara == 1):  
        # index of spatial conceptのmultinomial distribution(L-dimension)
        pi  = stick_breaking(alpha0, L)     
        # index of position distributionのmultinomial distribution(K-dimension)[L]
        phi = [ stick_breaking(gamma0, K) for _ in xrange(L) ] 
      elif (nonpara == 0):
        # index of spatial conceptのmultinomial distribution(L-dimension)
        pi  = [ 1.0/L for _ in xrange(L) ]     
        # index of position distributionのmultinomial distribution(K-dimension)[L]
        phi = [ [ 1.0/K for _ in xrange(K) ] for _ in xrange(L) ]        
      

      if (IT_mode == "HMM"):
        # Initial setting for GaussianHMM
        model = hmm.GaussianHMM(
          n_components=K, covariance_type="full", 
          startprob_prior=gamma0, transmat_prior=np.ones((K,K))*omega0, 
          means_weight=m0, means_prior=k0, # Mean and precision
          covars_weight=V0, covars_prior=n0,
          algorithm="viterbi", n_iter=100,
          params="stmc", init_params="stmc"  # update, initialized 
          #‘s’ for startprob, ‘t’ for transmat, ‘m’ for means and ‘c’ for covars. 
        )
        
        # HMM transition distribution (multinomial distribution [K][K])
        if (nonpara == 1):  
          psi = np.array([ stick_breaking(omega0, K) for _ in xrange(K) ])
        elif (nonpara == 0):
          psi = np.array([ [ 1.0/K for _ in xrange(K) ] for _ in xrange(K) ])

        # Initial parameters setting
        model.startprob_ = np.mean(phi, 0) #startprob
        model.transmat_  = psi #transmat
        model.means_     = Mu  #means
        model.covars_    = S   #covars

      print ">> Mu\n", Mu
      print ">> Sig\n", S
      print ">> W\n", W
      print ">> theta\n", theta
      print ">> pi\n", pi
      print ">> phi\n", phi

      THETA_init = [phi, pi, W, theta, Mu, S]
      #All parameters and initial values are output
      SaveParameters_init(filename, trialname, iteration, sample, THETA_init, Ct, It, N, TN)

      #############################################################################
      ####                 ↓Learning phase of spatial concept↓                 ####
      #############################################################################
      ##Start learning of spatial concepts
      print u"- <START> Learning of Spatial Concepts ver. NEW MODEL. -"
      
      for iter in xrange(num_iter):   #Iteration of Gibbs sampling
        print 'Iter.'+repr(iter+1)+'\n'
        
        ########## ↓ ##### it(index of position distribution) is samplied ##### ↓ ##########
        print u"Sampling it...", IT_mode
        if (IT_mode == "HMM"):
          # psi (model.transmat_) のパラメータをphi_cでリスケーリングする (データごとに指定できない)
          psi_rescaling = psi * (phi[1] / np.sum(phi,0))
          psi_rescaling = psi_rescaling / np.array([np.sum(psi_rescaling,1)]).T
          model.transmat_ = psi_rescaling

          It = model.predict(Xt) #本当はサンプリングにしたい
          model.fit(Xt)  # fitting HMM parameters
          print It
          print u"Sampling psi_k...", 
          psi = model.transmat_

        elif (IT_mode == "GMM"):
          #itと同じtのCtの値c番目のφc  の要素kごとに事後multinomial distributionの値を計算
          temp = np.zeros(K)
          for t in xrange(N):    #時刻tごとのdata
            phi_c = phi[int(Ct[t])]
            
            for k in xrange(K):
              #it=k番目のμΣについての2-dimension Gaussian distributionをitと同じtのxtから計算
              temp[k] = multivariate_normal.pdf(Xt[TN[t]], mean=Mu[k], cov=S[k]) * phi_c[k]
              
            temp = temp / np.sum(temp)  #Normalization
            It[t] = list(multinomial.rvs(1,temp)).index(1)
          print It
        ########## ↑ ##### it(index of position distribution) is samplied ##### ↑ ##########
        
        ########## ↓ ##### Ct(index of spatial concept) is samplied ##### ↓ ##########
        print u"Sampling Ct..."
        #Ct～多項値P(Ot|Wc)*多項値P(it|φc)*多項P(c|π)  N個
        
        temp = np.zeros(L)
        for t in xrange(N):    #時刻tごとのdata
          for c in xrange(L):  #index of spatial conceptのmultinomial distributionそれぞれについて
            temp[c] = pi[c] * phi[c][It[t]] * multinomial.pmf(Otb_B[t], sum(Otb_B[t]), W[c]) * multinomial.pmf(Ft[t], sum(Ft[t]), theta[c])
          
          temp = temp / np.sum(temp)  #Normalization
          Ct[t] = list(multinomial.rvs(1,temp)).index(1)

        print Ct
        ########## ↑ ##### Ct(index of spatial concept) is samplied ##### ↑ ##########
        
        ########## ↓ ##### W(the name of place: multinomial distribution) is samplied ##### ↓ ##########
        print u"Sampling Wc..."
        ##Dirichlet multinomial distributionからDirichlet Posterior distributionを計算しSamplingする
        
        temp = [ np.ones(len(W_index))*beta0 for c in xrange(L) ]  #集めて加算するための array :paramtersで初期化しておけばよい
        #Ctがcであるときのdataを集める
        for c in xrange(L) :   #ctごとにL個分計算
          nc = 0
          ##Posterior distributionのためのparamters計算
          if c in Ct : 
            for t in xrange(N) : 
              if Ct[t] == c : 
                #dataを集めるたびに値を加算
                temp[c] = temp[c] + np.array(Otb_B[t])
                nc = nc + 1  #counting the number of data
            print "%d n:%d %s" % (c,nc,temp[c])
          
        #加算したdataとparamtersからPosterior distributionを計算しSampling
        W = [ np.mean(dirichlet(temp[c],Robust_W),0) for c in xrange(L) ] 
        
        print W
        ########## ↑ ##### W(the name of place: multinomial distribution) is samplied ##### ↑ ##########

        ########## ↓ ##### theta(the image feature: multinomial distribution) is samplied ##### ↓ ##########
        print u"Sampling theta_c..."
        ##Dirichlet multinomial distributionからDirichlet Posterior distributionを計算しSamplingする
        
        temp = [ np.ones(DimImg)*chi0 for c in xrange(L) ]  #集めて加算するための array :paramtersで初期化しておけばよい
        #Ctがcであるときのdataを集める
        for c in xrange(L) :   #ctごとにL個分計算
          nc = 0
          ##Posterior distributionのためのparamters計算
          if c in Ct : 
            for t in xrange(N) : 
              if Ct[t] == c : 
                #dataを集めるたびに値を加算
                temp[c] = temp[c] + np.array(Ft[t])
                nc = nc + 1  #counting the number of data
            print "%d n:%d %s" % (c,nc,temp[c])
          
        #加算したdataとparamtersからPosterior distributionを計算しSampling
        theta = [ np.mean(dirichlet(temp[c],Robust_theta),0) for c in xrange(L) ] 
        
        print theta
        ########## ↑ ##### W(the name of place: multinomial distribution) is samplied ##### ↑ ##########
               
        ########## ↓ ##### μ, Σ (the position distribution (Gaussian distribution: mean and covariance matrix) is samplied ##### ↓ ##########
        print u"Sampling Mu_i,Sigma_i..."
        np.random.seed()

        for k in xrange(K) : 
          nk = It.count(k) #cc[k]
          kN,mN,nN,VN = PosteriorParameterGIW2(k,nk,N,It,Xt,k)
          
          ##3.1##ΣをInv-WishartからSampling
          S[k] = np.mean([invwishart.rvs(df=nN, scale=VN) for _ in xrange(Robust_Sig)],0) #サンプリングをロバストに

          if np.linalg.det(S[k]) < -0.0: #半正定値を満たさない場合；エラー処理
            S[k] = invwishart.rvs(df=nN, scale=VN)
          
          ##3.2##μをGaussianからSampling
          Mu[k] = np.mean([multivariate_normal.rvs(mean=mN, cov=S[k]/kN) for _ in xrange(Robust_Mu)],0) #サンプリングをロバストに
          
        for k in xrange(K) : 
          if (It.count(k) != 0):  #dataなしは表示しない
            print 'Mu'+str(k)+':'+str(Mu[k]),
        print ''
        
        for k in xrange(K):
          if (It.count(k) != 0):  #dataなしは表示しない
            print 'sig'+str(k)+':'+str(S[k])
        ########## ↑ ##### μ, Σ (the position distribution (Gaussian distribution: mean and covariance matrix) is samplied ##### ↑ ##########
        
       ########## ↓ ##### π(index of spatial conceptのmultinomial distribution) is samplied ##### ↓ ##########
        print u"Sampling PI..."
        temp = np.array([Ct.count(c) + alpha0 for c in xrange(L)])

        #加算したdataとparamtersからPosterior distributionを計算しSampling
        pi = np.mean(dirichlet(temp,Robust_pi),0)
        print pi
        ########## ↑ ##### π(index of spatial conceptのmultinomial distribution) is samplied ##### ↑ ##########
        
        ########## ↓ ##### φ(index of position distributionのmultinomial distribution) is samplied ##### ↓ ##########
        print u"Sampling PHI_c..."
        for c in xrange(L):  #L個分
          temp = np.ones(K) * gamma0
          #Ctとcが一致するdataを集める
          if c in Ct :
            for t in xrange(N):
              if Ct[t] == c:  #Ctとcが一致したdataで
                temp = temp + [int(It[t] == k) for k in range(K)] ## 変更、要確認
                #index kごとに, dataとindex番号が一致したとき, 集めたdataを元にindex of position distributionごとに加算
          
          #加算したdataとparamtersからPosterior distributionを計算しSampling
          phi[c] = np.mean(dirichlet(temp,Robust_phi),0) 
          
          if c in Ct:
            print c,phi[c]
        ########## ↑ ##### φ(index of position distributionのmultinomial distribution) is samplied ##### ↑ ##########

        
      #############################################################################
      ####                 ↑Learning phase of spatial concept↑                 ####
      ############################################################################# 
      
      #theta = []  #仮置き
      THETA = [phi, pi, W, theta, Mu, S]

      ########  ↓File output↓  ########
      print "--------------------"
      #最終学習結果をoutput
      print u"\n- <COMPLETED> Learning of Spatial Concepts ver. NEW MODEL. -"
      print 'Sample: ' + str(sample)
      print 'Ct: ' + str(Ct)
      print 'It: ' + str(It)
      for c in xrange(L):
        print "W%d: %s" % (c,W[c])
      for c in xrange(L):
        print "theta%d: %s" % (c,theta[c])
      for k in xrange(K):
        print "Mu%d: %s" % (k, str(Mu[k].T))
      for k in xrange(K):
        print "sig%d: \n%s" % (k, str(S[k]))
      print 'pi: ' + str(pi)
      for c in xrange(L):
        print 'phi' + str(c) + ':',
        print str(phi[c])
      print "--------------------"
      
      #Samplingごとに各paramters値をoutput
      SaveParameters_all(filename, trialname, iteration, sample, THETA, Ct, It, W_index)

      ##paramtersそれぞれをそれぞれのファイルとしてはく
      ##Output to file: the set of word recognition results
      SaveParameter_EachFile(filename, trialname, iteration, sample, THETA, Ct, It)
      
      if (IT_mode == "HMM"):
        with open(filename + '/' + trialname + "_HMM_" + str(iteration) + "_" + str(sample)  + ".pkl", "wb") as file: pickle.dump(model, file)
        #joblib.dump(model, "filename.pkl")


      print 'File Output Successful!(filename:'+filename+ "_" +str(iteration) + "_" + str(sample) + ')\n'
      ########  ↑File output↑  ########

      Ct_Samp[sample]    = Ct
      THETA_Samp[sample] = THETA

    return Ct_Samp, THETA_Samp

##発話した文章ごとに相互情報量を計算し、サンプリング結果を選ぶ
def SelectMaxWordDict(iteration, THETA, W_index):
  filename = outputfolder + trialname
  file_trialname   = filename + '/' + trialname
  MI_Samp2 = [0.0 for sample in xrange(sample_num)]  ##サンプルの数だけMIを求める
  
  for sample in xrange(sample_num):
    #####↓##場所概念ごとに単語ごとに相互情報量を計算、高いものから表示##↓######
    ##相互情報量による単語のセレクション
    MI   = [[] for c in xrange(L)]
    W_in = []    #閾値以上の単語集合
    #i_best = len(W_index)    ##相互情報量上位の単語をどれだけ使うか
    _, pi, W, _, _, _ = THETA[sample]  #THETA = [phi, pi, W, theta, Mu, S]
    ###相互情報量を計算
    for c in xrange(L):
      #print "Concept:%d" % c
      for o in xrange(len(W_index[sample])):
        word  = W_index[sample][o]
        score = MI_binary(o,W,pi,c)  ##相互情報量計算
        MI[c].append( (score, word) )
        
        if (score >= threshold):  ##閾値以上の単語をすべて保持
          #print score , threshold ,word in W_in
          if ((word in W_in) == False):  #リストに単語が存在しないとき
            #print word
            W_in = W_in + [word]
        
      MI[c].sort(reverse=True)
      #for score, word in MI[c]:
      #  print score, word
    
    ##ファイル出力
    fp = open(file_trialname + '_sougo_C_' + str(iteration) + '_' + str(sample) + '.csv', 'w')
    for c in xrange(L):
      fp.write("Concept:" + str(c) + '\n')
      for score, word in MI[c]:
        fp.write(str(score) + "," + word + '\n')
      fp.write('\n')
    fp.close()
    #####↑##場所概念ごとに単語ごとに相互情報量を計算、高いものから表示##↑######
    
    if ( len(W_in) == 0 ):
      print "W_in is empty."
      W_in = W_index[sample] ##選ばれる単語がなかった場合、W_indexをそのままいれる
    #else:
    #  print W_in
    
    ##場所の名前W（多項分布）をW_inに含まれる単語のみにする
    W_reco = [ [0.0 for j in xrange(len(W_in))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
    for c in xrange(L):
      for j in xrange(len(W_index[sample])):
        for i in xrange(len(W_in)):
          if ((W_in[i] in W_index[sample][j]) == True):
            W_reco[c][i] = float(W[c][j])
      
      #正規化処理(全要素が0になるのを回避する処理入り)
      W_reco_sum    = fsum(W_reco[c])
      W_reco_max    = max(W_reco[c])
      W_reco_summax = float(W_reco_sum) / W_reco_max
      W_reco[c] = [float(float(W_reco[c][i])/W_reco_max) / W_reco_summax for i in xrange(len(W_in))]
    #print W_reco
    
    ###相互情報量を計算(それぞれの単語とCtとの共起性を見る)
    MI_Samp2[sample] = Mutual_Info(W_reco,pi)
    print "sample:",sample," MI:",MI_Samp2[sample]
    
  MAX_Samp = MI_Samp2.index(max(MI_Samp2))  #相互情報量が最大のサンプル番号
  
  ##ファイル出力
  fp = open(file_trialname + '_sougo_MI_' + str(iteration) + '.csv', 'w')
  for sample in xrange(sample_num):
      fp.write(str(sample) + ',' + str(MI_Samp2[sample]) + '\n') 
  fp.close()

  return W_index[MAX_Samp]
  
###↓###単語辞書読み込み書き込み追加############################################
#MAX_Samp : 重みが最大のパーティクル
def WordDictionaryUpdate(iteration, W_index):
  LIST      = []
  LIST_plus = []
  #i_best    = len(W_index[MAX_Samp])    ##相互情報量上位の単語をどれだけ使うか（len(W_index)：すべて）
  i_best = len(W_index)
  #W_index   = W_index[MAX_Samp]
  hatsuon   = [ "" for i in xrange(i_best) ]
  TANGO     = []
  ##単語辞書の読み込み
  for line in open(lmfolder + lang_init, 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]   
  #print TANGO
  
  ##W_indexの単語を順番に処理していく
  for c in xrange(i_best):    # i_best = len(W_index)
          #W_list_sj = unicode(MI_best[c][i], encoding='shift_jis')
          W_list_sj = unicode(W_index[c], encoding='shift_jis')
          if len(W_list_sj) != 1:  ##１文字は除外
            #for moji in xrange(len(W_list_sj)):
            moji = 0
            while (moji < len(W_list_sj)):
              flag_moji = 0
              #print len(W_list_sj),str(W_list_sj),moji,W_list_sj[moji]#,len(unicode(W_index[i], encoding='shift_jis'))
              
              for j in xrange(len(TANGO)):
                if (len(W_list_sj)-2 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]+"_"+W_list_sj[moji+2]) and (W_list_sj[moji+1] == "_"): 
                    ###print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_list_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]+W_list_sj[moji+1]):
                    ###print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_list_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_list_sj) > moji) and (flag_moji == 0):
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]):
                      ###print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
            print W_list_sj,hatsuon[c]
          else:
            print W_list_sj,W_index[c] + " (one name)"
        
  print JuliusVer,HMMtype
  if (JuliusVer == "v4.4" and HMMtype == "DNN"):
      #hatsuonのすべての単語の音素表記を"*_I"にする
      for i in range(len(hatsuon)):
        hatsuon[i] = hatsuon[i].replace("_S","_I")
        hatsuon[i] = hatsuon[i].replace("_B","_I")
        hatsuon[i] = hatsuon[i].replace("_E","_I")
      
      #hatsuonの単語の先頭の音素を"*_B"にする
      for i in range(len(hatsuon)):
        #onsohyoki_index = onsohyoki.find(target)
        hatsuon[i] = hatsuon[i].replace("_I","_B", 1)
        
        #hatsuonの単語の最後の音素を"*_E"にする
        hatsuon[i] = hatsuon[i][0:-2] + "E "
        
        #hatsuonの単語の音素の例外処理（N,q）
        hatsuon[i] = hatsuon[i].replace("q_S","q_I")
        hatsuon[i] = hatsuon[i].replace("q_B","q_I")
        hatsuon[i] = hatsuon[i].replace("N_S","N_I")
        #print type(hatsuon),hatsuon,type("N_S"),"N_S"
  
  ##各場所の名前の単語ごとに
  meishi = u'名詞'
  meishi = meishi.encode('shift-jis')
  
  ##単語辞書ファイル生成
  fp = open(filename + '/WD_' + str(iteration) + '.htkdic', 'w')
  for list in xrange(len(LIST)):
        fp.write(LIST[list])
  ##新しい単語を追加
  c = 0
  for mi in xrange(i_best):    # i_best = len(W_index)
    if hatsuon[mi] != "":
        if ((W_index[mi] in LIST_plus) == False):  #同一単語を除外
          flag_tango = 0
          for j in xrange(len(TANGO)):
            if(W_index[mi] == TANGO[j][0]):
              flag_tango = -1
          if flag_tango == 0:
            LIST_plus = LIST_plus + [W_index[mi]]
            
            fp.write(LIST_plus[c] + "+" + meishi +"	[" + LIST_plus[c] + "]	" + hatsuon[mi])
            fp.write('\n')
            c += 1

  fp.close()
  ###↑###単語辞書読み込み書き込み追加############################################



if __name__ == '__main__':
    #Request a file name for output
    #trialname = raw_input("trialname?(folder) >")
    trialname = sys.argv[1]
    print trialname

    ########## SIGVerse ##########
    datasetNUM = sys.argv[2] #0
    datasetname = "3LDK_" + datasets[int(datasetNUM)]
    print "ROOM:", datasetname #datasetPATH
    #print trialname
    ########## SIGVerse ##########
    
    start_time = time.time()
    iteration_time = [0.0 for i in range(ITERATION)]
    filename = outputfolder + trialname
    Makedir( filename )

    # DATA read
    Xt, TN = ReadPositionData()  # Reading Position data 
    Ft     = ReadImageData()     # Reading Image feature data  予約（未完成）

    for i in xrange(ITERATION):
      print "--------------------------------------------------"
      print "ITERATION:",i+1
      start_iter_time = time.time()
      
      Julius_lattice(i,trialname)    ##音声認識、ラティス形式出力、opemFST形式へ変換
      
      FST_PATH = filename + "/fst_gmm_" + str(i+1) + "/" + str(DATA_NUM-1).zfill(3) +".fst"
      while (os.path.exists( FST_PATH ) != True):
        print FST_PATH, os.path.exists( FST_PATH ), "wait(30s)... or ERROR?"
        time.sleep(30.0) #sleep(秒指定)
      print "ITERATION:",i+1," Julius complete!"
      
      sample = 0  ##latticelmのパラメータ通りだけサンプルする
      for p1 in xrange(len(knownn)):
        for p2 in xrange(len(unkn)):
          #if sample < sample_num:
          print "latticelm run. sample_num:" + str(sample)
          latticelm_CMD = "latticelm -input fst -filelist " + filename + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix " + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile " + filename + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2])
          ##latticelm  ## -annealsteps 10 -anneallength 15
          OUT_PATH = filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100"

          p = os.popen( latticelm_CMD ) 
          p.close()  
          time.sleep(1.0) #sleep(秒指定)
          while (os.path.exists( OUT_PATH ) != True):
            print OUT_PATH, os.path.exists( OUT_PATH ),"wait(3.0s)... or ERROR?"
            p = os.popen( latticelm_CMD ) 
            p.close() 
            time.sleep(3.0) #sleep(秒指定)
          sample = sample + 1
      print "ITERATION:",i+1," latticelm complete!"
      
      Otb_Samp, W_index_Samp = ReadWordData(i+1)   # Reading word data and Making word list
      Ct_Samp, THETA_Samp = Gibbs_Sampling(i+1, Otb_Samp, W_index_Samp, Xt, TN, Ft) #         ##場所概念の学習
      print "ITERATION:",i+1," Learning complete!"

      W_index_MAX = SelectMaxWordDict(i+1, THETA_Samp, W_index_Samp)  ##相互情報量計算
      WordDictionaryUpdate(i+1, W_index_MAX)  ##単語辞書登録
      print "ITERATION:",i+1," Language Model update!"

      end_iter_time = time.time()
      iteration_time[i] = end_iter_time - start_iter_time
    
    ##ループ後処理
    end_time = time.time()
    time_cost = end_time - start_time

    fp = open(filename + '/time.txt', 'w')
    fp.write(str(time_cost)+"\n")
    fp.write(str(start_time)+","+str(end_time)+"\n")
    for i in range(ITERATION):
      fp.write(str(i+1)+","+str(iteration_time[i])+"\n")
    fp.close()

########################################
