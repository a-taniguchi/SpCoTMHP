#coding:utf-8

##############################################
## Spatial concept formation model (SpCoA without lexical acquisition)
## For SpCoNavi -> TMHP (on SIGVerse for /3LDK/ dataset)
## Learning algorithm is Gibbs sampling.
## Akira Taniguchi 2020/04/12--2021/7/6
##############################################
# python ./learnSpCoTMHP.py 3LDK_00

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

# Mutual information (binary variable): word_index, W, π, Ct
def MI_binary(b,W,pi,c):
  #相互情報量の計算
  POC   = W[c][b] * pi[c]    #場所の名前の多項分布と場所概念の多項分布の積
  PO    = sum([W[ct][b] * pi[ct] for ct in xrange(L)]) 
  PC    = pi[c]
  POb   = 1.0 - PO
  PCb   = 1.0 - PC
  PObCb = PCb - PO + POC
  POCb  = PO - POC
  PObC  = PC - POC
  
  # Calculate each term for MI 
  temp1 = POC   * log( POC   / ( PO  * PC  ), 2)
  temp2 = POCb  * log( POCb  / ( PO  * PCb ), 2)
  temp3 = PObC  * log( PObC  / ( POb * PC  ), 2)
  temp4 = PObCb * log( PObCb / ( POb * PCb ), 2)
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
      MI = MI + ( POC * log( POC/(PO*PC), 2) )
  return MI



#All parameters and initial values are output
def SaveParameters_init(filename, trialname, iteration, sample, THETA_init, Ct_init, It_init, N, TN):
  phi_init, pi_init, W_init, theta_init, Mu_init, S_init, psi_init = THETA_init  #THETA = [phi, pi, W, theta, Mu, S]

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
  phi, pi, W, theta, Mu, S, psi = THETA  #THETA = [phi, pi, W, theta, Mu, S, psi]
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
    for dim1 in range(dimx):
      for dim2 in range(dimx):
        fp.write(repr(S[k][dim1][dim2])+',')
      fp.write('\n')

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

  for i in xrange(K):
    ##fp.write(',')
    #for k in xrange(K):
    #  fp.write(',' + repr(k))
    #fp.write('\n')
    fp.write('psi'+repr(i)+',')
    for k in xrange(K):
      fp.write(repr(psi[i][k])+',')
    fp.write('\n')

  fp.close()

  #fp_x = open( filename + '/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
  #for t in xrange(len(Xt[t])) : 
  #  fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
  #fp_x.close()
        

# Saving data for parameters Θ of spatial concepts
def SaveParameter_EachFile(filename, trialname, iteration, sample, THETA, Ct, It, W_index, Otb):
  phi, pi, W, theta, Mu, S, psi = THETA  #THETA = [phi, pi, W, theta, Mu, S, psi]
  file_trialname   = filename + '/' + trialname
  iteration_sample = str(iteration) + "_" + str(sample) 

  np.savetxt(file_trialname + '_Mu_'    + iteration_sample + '.csv', Mu,    delimiter=',')
  np.savetxt(file_trialname + '_W_'     + iteration_sample + '.csv', W,     delimiter=',')
  np.savetxt(file_trialname + '_phi_'   + iteration_sample + '.csv', phi,   delimiter=',')
  np.savetxt(file_trialname + '_pi_'    + iteration_sample + '.csv', pi,    delimiter=',')
  np.savetxt(file_trialname + '_Ct_'    + iteration_sample + '.csv', Ct,    delimiter=',', fmt='%d')
  np.savetxt(file_trialname + '_It_'    + iteration_sample + '.csv', It,    delimiter=',', fmt='%d')

  np.save(file_trialname + '_Sig_'   + iteration_sample, S)
  np.save(file_trialname + '_theta_' + iteration_sample, theta)
  np.save(file_trialname + '_psi_'   + iteration_sample, psi)

 
  ##Output to file: the set of word recognition results
  N = len(Ct)
  #filename_ot = raw_input("Otb:filename?(.csv) >")  #ファイル名を個別に指定する場合
  #filename_ot = trialname
  fp  = open(filename + '/' + trialname + '_ot_' + str(iteration) + "_" + str(sample) + '.csv', 'w')
  fp2 = open(filename + '/' + trialname + '_w_index_' + str(iteration) + "_" + str(sample) + '.csv', 'w')
  for n in xrange(N): 
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
def Name_data_read(directory,word_increment,DATA_NUM):
    name_data_set=[]
    
    for i in range(DATA_NUM):
        name_data=[0 for w in range(len(name_list))]

        if  (i in test_num)==False:
            try:
                file=directory+Name_data_dir+repr(i)+".txt"
                data=np.genfromtxt(file, delimiter="\n", dtype='S' )
                #print file

                try:
                    for d in data:
                        #print d
                        for w,dictionry in enumerate(name_list):
                            if d == dictionry:
                                name_data[w]+=word_increment


                except TypeError:
                    #print d
                    for w,dictionry in enumerate(name_list):
                        if data == dictionry:
                            name_data[w]+=word_increment
            except IOError:
                pass
            name_data=np.array(name_data)
            name_data_set.append(name_data)
        else:
            print i
        #else:
            #print i,"is test data."
    return np.array(name_data_set)
"""


######################################################
# Gibbs sampling
######################################################
def Gibbs_Sampling(iteration):
    DataSetFolder = inputfolder + trialname
    filename  = outputfolder + trialname
    
    ##S## ##### Ishibushi's code #####
    env_para = np.genfromtxt(DataSetFolder+"/Environment_parameter.txt",dtype= None,delimiter =" ")

    #MAP_X = float(env_para[0][1])  #Max x value of the map
    #MAP_Y = float(env_para[1][1])  #Max y value of the map
    #map_x = float(env_para[2][1])  #Min x value of the map
    #map_y = float(env_para[3][1])  #Min y value of the map

    #map_center_x = ((MAP_X - map_x)/2)+map_x
    #map_center_y = ((MAP_Y - map_x)/2)+map_y
    #mu_0 = np.array([map_center_x,map_center_y,0,0])
    DATA_initial_index = int(env_para[5][1]) #Initial data num
    DATA_last_index    = int(env_para[6][1]) #Last data num
    DATA_NUM = DATA_last_index - DATA_initial_index + 1
    ##E## ##### Ishibushi's code ######
    
    # DATA read
    Xt = position_data_read_pass(DataSetFolder,DATA_NUM)
    #name = Name_data_read(DataSetFolder,word_increment,DATA_NUM)
    
    for sample in xrange(sample_num):
      N = 0
      Otb = []
      #Read text file
      for word_data_num in range(DATA_NUM):
        f = open(DataSetFolder + word_folder + str(word_data_num) + ".txt", "r")
        line = f.read()
        itemList = line[:].split(' ')
        
        #remove <s>,<sp>,</s> and "\r", "": if its were segmented to words.
        itemList = Ignore_SP_Tags(itemList)
        
        #Otb[sample] = Otb[sample] + [itemList]
        Otb = Otb + [itemList]
        N = N + 1  #count
        
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
      
      if (DATA_NUM != N):
         print "DATA_NUM" + str(DATA_NUM) + ":KYOUJI error!! N:" + str(N)   ##教示フェーズの教示数と読み込んだ発話文データ数が違う場合
         #exit()
      
      TN = [i for i in xrange(N)]   #TN[N]: teaching time-step
      
      
      #############################################################################
      ####                 ↓Learning phase of spatial concept↓                 ####
      #############################################################################
      ##Initialization of all parameters
      print u"Initialize Parameters..."
      if (LEARN_MODE == "GT"):
        # index of spatial concepts [N]
        Ct = [ int(n/15) for n in xrange(N)]  
        # index of position distributions [N]
        It = [ int(n/15) for n in xrange(N)]  
      else:
        # index of spatial concepts [N]
        Ct = [ int(random.uniform(0,L)) for n in xrange(N) ] #[ int(n/15) for n in xrange(N)]  
        # index of position distributions [N]
        It = [ int(random.uniform(0,K)) for n in xrange(N) ] #[ int(n/15) for n in xrange(N)]    
      ## Uniform random numbers within the range
      # the position distribution (Gaussian)の平均(x,y)[K]
      Mu = [ np.array([ int( random.uniform(WallXmin,WallXmax) ) ,
                        int( random.uniform(WallYmin,WallYmax) ) ]) for i in xrange(K) ]      
      # the position distribution (Gaussian)の共分散(2×2-dimension)[K]
      S  = [ np.eye(dimx) * sig_init for i in xrange(K) ] 
      # the name of place(multinomial distribution: W_index-dimension)[L]
      W  = [ [ 1.0/len(W_index) for j in xrange(len(W_index)) ] for c in xrange(L) ]
      # the image feature (multinomial distribution: DimImg-dimension)[L]
      theta = [ [ 1.0/DimImg for _ in xrange(DimImg) ] for _ in xrange(L) ]
      if (nonpara == 1):  
        # index of spatial conceptのmultinomial distribution(L-dimension)
        pi  = stick_breaking(alpha0, L)     
        # index of position distributionのmultinomial distribution(K-dimension)[L]
        phi = [ stick_breaking(gamma0, K) for c in xrange(L) ] 
      elif (nonpara == 0):
        # index of spatial conceptのmultinomial distribution(L-dimension)
        pi  = [ 1.0/L for c in xrange(L) ]     
        # index of position distributionのmultinomial distribution(K-dimension)[L]
        phi = [ [ 1.0/K for i in xrange(K) ] for c in xrange(L) ]  
      
      if (IT_mode == "HMM"):
        # HMM transition distribution (multinomial distribution)[K][K]
        if (nonpara == 1):  
          psi = np.array([ stick_breaking(omega0*K, K) for _ in xrange(K) ])
        elif (nonpara == 0):
          psi = np.array([ [ 1.0/K for _ in xrange(K) ] for _ in xrange(K) ])
        #print ">> psi init\n", psi
      else:
        psi = np.ones((K,K)) #dummy

      if (terminal_output_prams == 1):
        print ">> Mu init\n", Mu
        print ">> Sig init\n", S
        print ">> W init\n", W
        print ">> pi init\n", pi
        print ">> phi init\n", phi
        if (UseFT == 1):
          print ">> theta init\n", theta
        if (IT_mode == "HMM"):
          print ">> psi init\n", psi

      theta = []  #仮置き
      THETA_init = [phi, pi, W, theta, Mu, S, psi]
      # All parameters and initial values are output
      SaveParameters_init(filename, trialname, iteration, sample, THETA_init, Ct, It, N, TN)
      
      ##Start learning of spatial concepts
      print u"- <START> Learning of Spatial Concepts ver. NEW MODEL. -"
      
      for iter in xrange(num_iter):   # Iteration of Gibbs sampling
        print ' ----- Iter.' + repr(iter+1) + ' ----- '
        
        if (LEARN_MODE != "GT"): # GTのときは実行しない
          ########## ↓ ##### it(index of position distribution) is samplied ##### ↓ ##########
          print u"Sampling it... model:", IT_mode
          # itと同じtのCtの値c番目のφc の要素kごとに事後multinomial distributionの値を計算
          temp = np.ones(K)
          for t in xrange(N):    # 時刻tごとのdata
            # it=k番目のμΣについての2-dimension Gaussian distributionをitと同じtのxtから計算
            temp = np.array([ multivariate_normal.logpdf(Xt[TN[t]], mean=Mu[k], cov=S[k]) 
                              + np.log(phi[int(Ct[t])][k]) for k in xrange(K) ])
            
            if (IT_mode == "HMM") and (t > 0):
              temp = np.log(log2prob(temp))
              temp += [ psi[k][It[t-1]] - np.log(np.sum(phi,0)[k]) for k in xrange(K) ]  # Direct assignment sampling
              
            It[t] = list(multinomial.rvs(1,log2prob(temp))).index(1)
          print It
          ########## ↑ ##### it(index of position distribution) is samplied ##### ↑ ##########
          
          ########## ↓ ##### Ct(index of spatial concept) is samplied ##### ↓ ##########
          print u"Sampling Ct..."
          # Ct～多項値P(Ot|Wc)*多項値P(it|φc)*多項P(c|π)  N個
          temp = np.ones(L)
          for t in xrange(N):    # 時刻tごとのdata
            # For each multinomial distribution (index of spatial concept)
            temp = np.array([ multinomial.logpmf(Otb_B[t], sum(Otb_B[t]), W[c]) for c in xrange(L) ])
            count_nan = 0
            while (True in np.isnan(temp)): # nan があったときのエラー対処処理
              nanind = np.where(np.isnan(temp))[0]
              W_refine = (W[nanind[0]]+approx_zero)/np.sum((W[nanind[0]]+approx_zero))
              temp[nanind[0]] = multinomial.logpmf(Otb_B[t], sum(Otb_B[t]), W_refine)
              print "[nan] Wc", nanind[0], temp[nanind[0]]
              count_nan += 1
              if (True in [ temp[nanind[0]] ]):
                temp[nanind[0]] = approx_zero
              if (count_nan >= len(temp)):
                temp = log2prob(temp)
            temp += np.array([ np.log(pi[c]) + np.log(phi[c][It[t]]) for c in xrange(L) ])

            if (UseFT == 1):
              temp = np.log(log2prob(temp))
              temp_FT = np.array([ multinomial.logpmf(Ft[t], sum(Ft[t]), theta[c]) for c in xrange(L) ])
              count_nan = 0
              while (True in np.isnan(temp_FT)): # nan があったときのエラー対処処理
                nanind = np.where(np.isnan(temp_FT))[0]
                theta_refine = (theta[nanind[0]]+approx_zero)/np.sum((theta[nanind[0]]+approx_zero))
                temp_FT[nanind[0]] = multinomial.logpmf(Ft[t], sum(Ft[t]), theta_refine)
                print "[nan] theta_c", temp_FT[nanind[0]]
                count_nan += 1
                if (True in [ temp_FT[nanind[0]] ]):
                  temp_FT[nanind[0]] = approx_zero
                if (count_nan >= len(temp_FT)):
                  temp_FT = log2prob(temp_FT)
              temp += np.log(log2prob(temp_FT)) #temp_FT

            Ct[t] = list(multinomial.rvs(1,log2prob(temp))).index(1)
          print Ct
          ########## ↑ ##### Ct(index of spatial concept) is samplied ##### ↑ ##########
        
        ########## ↓ ##### W(the name of place: multinomial distribution) is samplied ##### ↓ ##########
        print u"Sampling Wc..."
        ## Dirichlet multinomial distributionからDirichlet Posterior distributionを計算しSamplingする
        temp = [ np.sum([int(Ct[t] == c)*np.array(Otb_B[t]) for t in range(N)],0)+beta0 for c in range(L) ]
          
        # 加算したdataとparamtersからPosterior distributionを計算しSampling
        W = [ np.mean(dirichlet(temp[c],Robust_W),0) for c in xrange(L) ] 
        
        print W
        ########## ↑ ##### W(the name of place: multinomial distribution) is samplied ##### ↑ ##########
        
        ########## ↓ ##### theta(the image feature: multinomial distribution) is samplied ##### ↓ ##########
        if (UseFT == 1):
          print u"Sampling theta_c..."
          ## Dirichlet multinomial distributionからDirichlet Posterior distributionを計算しSamplingする
          temp = [ np.sum([int(Ct[t] == c)*np.array(Ft[t]) for t in range(N)],0)+chi0 for c in range(L) ]

          # 加算したdataとparamtersからPosterior distributionを計算しSampling
          theta = [ np.mean(dirichlet(temp[c],Robust_theta),0) for c in xrange(L) ] 
          
          if(terminal_output_prams == 1):
            print theta
        ########## ↑ ##### W(the name of place: multinomial distribution) is samplied ##### ↑ ##########

        ########## ↓ ##### μ, Σ (the position distribution (Gaussian distribution: mean and covariance matrix) is samplied ##### ↓ ##########
        print u"Sampling Mu_i,Sigma_i..."
        np.random.seed()
        for k in xrange(K): 
          nk = It.count(k) #cc[k]
          kN,mN,nN,VN = PosteriorParameterGIW2(k,nk,N,It,Xt,k)
          
          ##3.1## ΣをInv-WishartからSampling
          S[k] = np.mean([invwishart.rvs(df=nN, scale=VN) for _ in xrange(Robust_Sig)],0) # サンプリングをロバストに

          if np.linalg.det(S[k]) < -0.0: # 半正定値を満たさない場合；エラー処理
            S[k] = invwishart.rvs(df=nN, scale=VN)
            print "Robust_Sig is NOT", Robust_Sig
          
          ##3.2## μをGaussianからSampling
          Mu[k] = np.mean([multivariate_normal.rvs(mean=mN, cov=S[k]/kN) for _ in xrange(Robust_Mu)],0) # サンプリングをロバストに

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
        temp = [ np.sum([int(Ct[t] == c)*np.array([int(It[t] == k) for k in range(K)]) for t in range(N)],0)
                 + gamma0 for c in xrange(L) ]

        # 加算したdataとparamtersからPosterior distributionを計算しSampling
        phi = [ np.mean(dirichlet(temp[c],Robust_phi),0) for c in range(L) ]
          
        if(terminal_output_prams == 1):
          for c in xrange(L):  #L個分
            if c in Ct:
              print c,phi[c]
        ########## ↑ ##### φ(index of position distributionのmultinomial distribution) is samplied ##### ↑ ##########
        
        ########## ↓ ##### ψ(HMM transition distribution[K][K]) is samplied ##### ↓ ##########
        if (IT_mode == "HMM"):
          print u"Sampling psi_k..." 
          temp = np.ones((K,K)) * omega0
          for t in xrange(1,N):    # 時刻tごとのdata
            temp[It[t]][It[t-1]] += 1.0
          if (transition_type == "sym"):
            temp = (temp + temp.T) #/ 2.0
          
          psi = [ np.mean(dirichlet(temp[k],Robust_psi),0) for k in range(K) ]
          
          if(terminal_output_prams == 1):
            print psi
        ########## ↑ ##### ψ(HMM transition distribution[K][K]) is samplied ##### ↑ ##########
        
        
      #############################################################################
      ####                 ↑Learning phase of spatial concept↑                 ####
      ############################################################################# 
      THETA = [phi, pi, W, theta, Mu, S, psi]

      ########  ↓File output↓  ########
      print "--------------------"
      #最終学習結果をoutput
      print u"\n- <COMPLETED> Learning of Spatial Concepts ver. NEW MODEL. -"
      print 'Sample: ' + str(sample)
      print 'Ct: ' + str(Ct)
      print 'It: ' + str(It)
      for c in xrange(L):
        print "W%d: %s" % (c,W[c])
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
      SaveParameter_EachFile(filename, trialname, iteration, sample, THETA, Ct, It, W_index, Otb)
      
      print 'File Output Successful!(filename:'+filename+ "_" +str(iteration) + "_" + str(sample) + ')\n'
      ########  ↑File output↑  ########
      
     
if __name__ == '__main__':
    #Request a file name for output
    #trialname = raw_input("trialname?(folder) >")
    trialname = sys.argv[1]
    print trialname

    #start_time = time.time()
    #iteration_time = [0.0 for i in range(ITERATION)]
    filename = outputfolder + trialname
    Makedir( filename )

    # init.pyをコピー
    shutil.copy( "./__init__.py", filename )


    print "--------------------------------------------------"
    print "ITERATION:",1
    Gibbs_Sampling(1)          ##Learning of spatial concepts
    print "ITERATION:",1," Learning complete!"
    

    """
    for i in xrange(ITERATION):
      print "--------------------------------------------------"
      print "ITERATION:",i+1

      #start_iter_time = time.time()
      
      #Julius_lattice(i,filename)    ##speech recognition, ラティス形式output, opemFST形式へ変換
      #p = os.popen( "python JuliusLattice_gmm.py " + str(i+1) +  " " + filename )
      
      #while (os.path.exists("./data/" + filename + "/fst_gmm_" + str(i+1) + "/" + str(kyouji_count-1).zfill(3) +".fst" ) != True):
      #  print "./data/" + filename + "/fst_gmm_" + str(i+1) + "/" + str(kyouji_count-1).zfill(3) + ".fst",os.path.exists("./data/" + filename + "/fst_gmm_" + str(i+1).zfill(3) + "/" + str(kyouji_count-1) +".fst" ),"wait(60s)... or ERROR?"
      #  time.sleep(60.0) #sleep(秒指定)
      #print "ITERATION:",i+1," Julius complete!"

      #for sample in xrange(sample_num):
      sample = 0  ##latticelmのparamters通りだけサンプルする
      for p1 in xrange(len(knownn)):
        for p2 in xrange(len(unkn)):
          if sample < sample_num:
            print "latticelm run. sample_num:" + str(sample)
            p = os.popen( "latticelm -input fst -filelist data/" + filename + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile data/" + filename + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2]) )   ##latticelm  ## -annealsteps 10 -anneallength 15
            time.sleep(1.0) #sleep(秒指定)
            while (os.path.exists("./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100" ) != True):
              print "./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100",os.path.exists("./data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100" ),"wait(30s)... or ERROR?"
              p.close()
              p = os.popen( "latticelm -input fst -filelist data/" + filename + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix data/" + filename + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile data/" + filename + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2]) )   ##latticelm  ## -annealsteps 10 -anneallength 15
              
              time.sleep(3.0) #sleep(秒指定)
            sample = sample + 1
            p.close()
      print "ITERATION:",i+1," latticelm complete!"

      #Gibbs_Sampling(i+1,trialname)          ##Learning of spatial concepts
      
      #print "ITERATION:",i+1," Learning complete!"
      #sougo(i+1)             ##相互情報量計算+##Make the word dictionary
      #print "ITERATION:",i+1," Language Model update!"
      #Language_model_update(i+1)  ##Make the word dictionary
      #end_iter_time = time.time()
      #iteration_time[i] = end_iter_time - start_iter_time
   
    ##ループ後処理
    
    #p0.close()
    #end_time = time.time()
    #time_cost = end_time - start_time

    fp = open('./data/' + filename + '/time.txt', 'w')
    fp.write(str(time_cost)+"\n")
    fp.write(str(start_time)+","+str(end_time)+"\n")
    for i in range(ITERATION):
      fp.write(str(i+1)+","+str(iteration_time[i])+"\n")
    """
    #import sys
    #import os.path
    #from __init__ import *
    ##from JuliusLattice_dec import *
    ##import time
########################################
