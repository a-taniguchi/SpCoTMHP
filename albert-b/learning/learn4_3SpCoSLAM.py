#coding:utf-8

##############################################
## Spatial concept formation model (SpCoA++ with lexical acquisition)
## For SpCoTMHP (on albert-b)
## Learning algorithm is Gibbs sampling.
## Akira Taniguchi 2020/04/14-
##############################################
# python ./learnSpCoTMHP.py 3LDK_00 (要変更)

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
from initSpCoSLAM import *
from JuliusLattice_dec import *
from submodules import *
import time

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

def Mutual_Info(W,pi):  #Mutual information:W、π 
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
  #for t in xrange(EndStep) : 
  #  fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
  #fp_x.close()
        

# Saving data for parameters Θ of spatial concepts
def SaveParameter_EachFile(filename, trialname, iteration, sample, THETA, Ct, It):
  phi, pi, W, theta, Mu, S = THETA  #THETA = [phi, pi, W, theta, Mu, S]
  file_trialname   = filename + '/' + trialname
  iteration_sample = str(iteration) + "_" + str(sample) 

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
  
  N = len(Ct)
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


#位置推定の教示データファイル名を要求
#data_name = raw_input("Read_XTo_filename?(.csv) >")

######################################################
# Gibbs sampling
######################################################
def Gibbs_Sampling(iteration):
    DataSetFolder = inputfolder  + trialname
    filename  = outputfolder + trialname

    ##発話認識文(単語)データを読み込む
    ##空白またはカンマで区切られた単語を行ごとに読み込むことを想定する
    for sample in xrange(sample_num):
      N = 0
      Otb = []
      #Read text file
      for line in open(DataSetFolder + '/out_gmm_' + str(iteration) + '/' + str(sample) + '_samp.100', 'r'):   ##*_samp.100を順番に読み込む
        itemList = line[:-1].split(' ')
        
        #<s>,<sp>,</s>を除く処理：単語に区切られていた場合
        for b in xrange(5):
          if ("<s><s>" in itemList):
            itemList.pop(itemList.index("<s><s>"))
          if ("<s><sp>" in itemList):
            itemList.pop(itemList.index("<s><sp>"))
          if ("<s>" in itemList):
            itemList.pop(itemList.index("<s>"))
          if ("<sp>" in itemList):
            itemList.pop(itemList.index("<sp>"))
          if ("<sp><sp>" in itemList):
            itemList.pop(itemList.index("<sp><sp>"))
          if ("</s>" in itemList):
            itemList.pop(itemList.index("</s>"))
          if ("<sp></s>" in itemList):
            itemList.pop(itemList.index("<sp></s>"))
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        #<s>,<sp>,</s>を除く処理：単語中に存在している場合
        for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("<s><s>", "")
          itemList[j] = itemList[j].replace("<s>", "")
          itemList[j] = itemList[j].replace("<sp>", "")
          itemList[j] = itemList[j].replace("</s>", "")
        for b in xrange(5):
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        
        #Otb[sample] = Otb[sample] + [itemList]
        Otb = Otb + [itemList]
        #if sample == 0:  #最初だけデータ数Nを数える
        N = N + 1  #count
        #else:
        #  Otb[] = Otb[NN] + itemList
        #  NN = NN + 1
        
        for j in xrange(len(itemList)):
            print "%s " % (str(itemList[j])),
        print ""  #改行用
      
      
      ##場所の名前の多項分布のインデックス用
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
      
      ##時刻tデータごとにBOW化(?)する、ベクトルとする
      Otb_B = [ [0 for i in xrange(len(W_index))] for n in xrange(N) ]
      
      
      for n in xrange(N):
        for j in xrange(len(Otb[n])):
          for i in xrange(len(W_index)):
            if (W_index[i] == Otb[n][j] ):
              Otb_B[n][i] = Otb_B[n][i] + 1
      #print Otb_B
      
      
      if DATA_NUM != N:
         print "N:KYOUJI error!!" + str(N)   ##教示フェーズの教示数と読み込んだ発話文データ数が違う場合
         #exit()
      
      #TN = [i for i in xrange(N)]#[0,1,2,3,4,5]  #テスト用
      
      ##教示位置をプロットするための処理
      #x_temp = []
      #y_temp = []
      #for t in xrange(len(TN)):
      #  x_temp = x_temp + [Xt[int(TN[t])][0]]  #設定は実際の教示時刻に対応できるようになっている。
      #  y_temp = y_temp + [Xt[int(TN[t])][1]]  #以前の設定のままで、動かせるようにしている。
      
      if (1):
        i = 0
        Xt = []
        #Xt = [(0.0,0.0) for n in xrange(len(HTW)) ]
        TN = []
        for line3 in open(DataSetFolder + PositionDataFile, 'r'):
          itemList3 = line3[:-1].split(',')
          Xt = Xt + [(float(itemList3[0]), float(itemList3[1]))]
          TN = TN + [i]
          print TN
          i = i + 1
        
        #Xt = Xt_temp
        EndStep = len(Xt)-1
      

  ######################################################################
  ####                   ↓場所概念学習フェーズ↓                   ####
  ######################################################################
      #TN[N]：教示時刻(step)集合
      
      #Otb_B[N][W_index]：時刻tごとの発話文をBOWにしたものの集合
      
      ##各パラメータ初期化処理
      print u"Initialize Parameters..."
      #xtは既にある、ct,it,Myu,S,Wは事前分布からサンプリングにする？(要相談)
      Ct = [ int(random.uniform(0,L)) for n in xrange(N)] #[0,0,1,1,2,3]     #物体概念のindex[N]
      It = [ int(random.uniform(0,K)) for n in xrange(N)]#[1,1,2,2,3,2]     #位置分布のindex[N]
      ##領域範囲内に一様乱数
      #if (data_name == "test000"):
      Myu = [ np.array([[ int( random.uniform(WallXmin,WallXmax) ) ],[ int( random.uniform(WallYmin,WallYmax) ) ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
      #else:
      #  Myu = [ np.array([[ random.uniform(-37.8+5,-37.8+80-10) ],[ random.uniform(-34.6+5,-34.6+57.6-10) ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
      S = [ np.array([ [sig_init, 0.0],[0.0, sig_init] ]) for i in xrange(K) ]      #位置分布の共分散(2×2次元)[K]
      W = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
      pi = stick_breaking(gamma, L)#[ 0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
      phi_l = [ stick_breaking(alpha, K) for c in xrange(L) ]#[ [0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K次元)[L]
      
      
      print Myu
      print S
      print W
      print pi
      print phi_l
      
      ###初期値を保存(このやり方でないと値が変わってしまう)
      Ct_init = [Ct[n] for n in xrange(N)]
      It_init = [It[n] for n in xrange(N)]
      Myu_init = [Myu[i] for i in xrange(K)]
      S_init = [ np.array([ [S[i][0][0], S[i][0][1]],[S[i][1][0], S[i][1][1]] ]) for i in xrange(K) ]
      W_init = [W[c] for c in xrange(L)]
      pi_init = [pi[c] for c in xrange(L)]
      phi_l_init = [phi_l[c] for c in xrange(L)]
      
      
      
      
      ##場所概念の学習
      #関数にとばす->のは後にする
      print u"- <START> Learning of Spatial Concepts ver. NEW MODEL. -"
      
      for iter in xrange(num_iter):   #イテレーションを行う
        print 'Iter.'+repr(iter+1)+'\n'
        
        
        ########## ↓ ##### it(位置分布のindex)のサンプリング ##### ↓ ##########
        print u"Sampling it..."
        
        #It_B = [0 for k in xrange(K)] #[ [0 for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #itと同じtのCtの値c番目のφc  の要素kごとに事後多項分布の値を計算
        temp = np.zeros(K)
        for t in xrange(N):    #時刻tごとのデータ
          phi_c = phi_l[int(Ct[t])]
          #np.array([ 0.0 for k in xrange(K) ])   #多項分布のパラメータ
          
          for k in xrange(K):
            #phi_temp = Multinomial(phi_c)
            #phi_temp.pmf([kのとき1のベクトル]) #パラメータと値は一致するのでphi_c[k]のままで良い
            
            #it=k番目のμΣについてのガウス分布をitと同じtのxtから計算
            xt_To = TN[t]
            g2 = gaussian2d(Xt[xt_To][0],Xt[xt_To][1],Myu[k][0],Myu[k][1],S[k])  #2次元ガウス分布を計算
            
            temp[k] = g2 * phi_c[k]
            #print g2,phi_c[k]  ###Xtとμが遠いとg2の値がアンダーフローする可能性がある
            
          temp = temp / np.sum(temp)  #正規化
          #print temp
          #Mult_samp = np.random.multinomial(1,temp)
          
          #print Mult_samp
          It_B = np.random.multinomial(1,temp) #Mult_samp [t]
          #print It_B[t]
          It[t] = np.where(It_B == 1)[0][0] #It_B.index(1)
          #for k in xrange(K):
          #  if (It_B[k] == 1):
          #    It[t] = k
          #    #print k
          
        #gaussian2d(Xx,Xy,myux,myuy,sigma)
        
        print It
        
        #多項分布からのサンプリング(1点)
        #http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html#numpy.random.multinomial
        #Mult_samp = np.random.multinomial(1,[確率の配列])
        ########## ↑ ##### it(位置分布のindex)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### Ct(場所概念のindex)のサンプリング ##### ↓ ##########
        print u"Sampling Ct..."
        #Ct～多項値P(Ot|Wc)*多項値P(it|φc)*多項P(c|π)  N個
        
        #It_B = [ [int(k == It[n]) for k in xrange(K)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][k]
        #Ct_B = [0 for c in xrange(L)] #[ [0 for c in xrange(L)] for n in xrange(N) ]   #多項分布のための出現回数ベクトル[t][l]
        
        temp = np.zeros(L)
        for t in xrange(N):    #時刻tごとのデータ
          #for k in xrange(K):
          #  if (k == It[t]):
          #    It_B[t][k] = 1
          
          #print It_B[t] #ok
          
          #np.array([ 0.0 for c in xrange(L) ])   #多項分布のパラメータ
          for c in xrange(L):  #場所概念のindexの多項分布それぞれについて
            #phi_temp = Multinomial(phi_l[c])
            W_temp = Multinomial(W[c])
            #print pi[c], phi_temp.pmf(It_B[t]), W_temp.pmf(Otb_B[t])
            temp[c] = pi[c] * phi_l[c][It[t]] * W_temp.pmf(Otb_B[t])    # phi_temp.pmf(It_B[t])各要素について計算
          
          temp = temp / np.sum(temp)  #正規化
          #print temp
          #Mult_samp = np.random.multinomial(1,temp)
          
          #print Mult_samp
          Ct_B = np.random.multinomial(1,temp) #Mult_samp
          #print Ct_B[t]
          
          Ct[t] = np.where(Ct_B == 1)[0][0] #Ct_B.index(1)
          #for c in xrange(L):
          #  if (Ct_B[c] == 1):
          #    Ct[t] = c
          #    #print c
          
        print Ct
        ########## ↑ ##### Ct(場所概念のindex)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### W(場所の名前：多項分布)のサンプリング ##### ↓ ##########
        ##ディリクレ多項からディリクレ事後分布を計算しサンプリングする
        ##ディリクレサンプリング関数へ入れ込む配列を作ればよい
        ##ディリクレ事前分布をサンプリングする必要はない->共役
        print u"Sampling Wc..."
        
        #data = [Otb_B[1],Otb_B[3],Otb_B[7],Otb_B[8]]  #仮データ
        
        #temp = np.ones((len(W_index),L))*beta0 #
        temp = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #集めて加算するための配列:パラメータで初期化しておけばよい
        #temp = [ np.ones(len(W_index))*beta0 for c in xrange(L)]
        #Ctがcであるときのデータを集める
        for c in xrange(L) :   #ctごとにL個分計算
          #temp = np.ones(len(W_index))*beta0
          nc = 0
          ##事後分布のためのパラメータ計算
          if c in Ct : 
            for t in xrange(N) : 
              if Ct[t] == c : 
                #データを集めるたびに値を加算
                for j in xrange(len(W_index)):    #ベクトル加算？頻度
                  temp[c][j] = temp[c][j] + Otb_B[t][j]
                nc = nc + 1  #データが何回加算されたか
              
          if (nc != 0):  #データなしのcは表示しない
            print "%d n:%d %s" % (c,nc,temp[c])
          
          #加算したデータとパラメータから事後分布を計算しサンプリング
          sumn = sum(np.random.dirichlet(temp[c],1000)) #fsumではダメ
          W[c] = sumn / sum(sumn)
          #print W[c]
        
        #Dir_0 = np.random.dirichlet(np.ones(L)*jp)
        #print Dir_0
        
        #ロバストなサンプリング結果を得るために
        #sumn = sum(np.random.dirichlet([0.1,0.2,0.5,0.1,0.1],10000))
        #multi = sumn / fsum(sumn)
        
        ########## ↑ ##### W(場所の名前：多項分布)のサンプリング ##### ↑ ##########
        
        ########## ↓ ##### μΣ(位置分布：ガウス分布の平均、共分散行列)のサンプリング ##### ↓ ##########
        print u"Sampling myu_i,Sigma_i..."
        #myuC = [ np.zeros((2,1)) for k in xrange(K) ] #np.array([[ 0.0 ],[ 0.0 ]])
        #sigmaC = [ np.zeros((2,2)) for k in xrange(K) ] #np.array([ [0,0],[0,0] ])
        np.random.seed()
        nk = [0 for j in xrange(K)]
        for j in xrange(K) : 
          ###jについて、Ctが同じものを集める
          #n = 0
          
          xt = []
          if j in It : 
            for t in xrange(N) : 
              if It[t] == j : 
                xt_To = TN[t]
                xt = xt + [ np.array([ [Xt[xt_To][0]], [Xt[xt_To][1]] ]) ]
                nk[j] = nk[j] + 1
          
          m_ML = np.array([[0.0],[0.0]])
          if nk[j] != 0 :        ##0ワリ回避
            m_ML = sum(xt) / float(nk[j]) #fsumではダメ
            print "n:%d m_ML.T:%s" % (nk[j],str(m_ML.T))
          
          #m0 = np.array([[0],[0]])   ##m0を元に戻す
          
          ##ハイパーパラメータ更新
          kappaN = kappa0 + nk[j]
          mN = ( (kappa0*m0) + (nk[j]*m_ML) ) / kappaN
          nuN = nu0 + nk[j]
          
          dist_sum = 0.0
          for k in xrange(nk[j]) : 
            dist_sum = dist_sum + np.dot((xt[k] - m_ML),(xt[k] - m_ML).T)
          VN = V0 + dist_sum + ( float(kappa0*nk[j])/(kappa0+nk[j]) ) * np.dot((m_ML - m0),(m_ML - m0).T)
          
          #if nk[j] == 0 :        ##0ワリ回避
          #  #nuN = nu0# + 1  ##nu0=nuN=1だと何故かエラーのため
          #  #kappaN = kappaN# + 1
          #  mN = np.array([[ int( random.uniform(1,WallX-1) ) ],[ int( random.uniform(1,WallY-1) ) ]])   ###領域内に一様
          
          ##3.1##Σを逆ウィシャートからサンプリング
          
          samp_sig_rand = np.array([ invwishartrand(nuN,VN) for i in xrange(100)])    ######
          samp_sig = np.mean(samp_sig_rand,0)
          #print samp_sig
          
          if np.linalg.det(samp_sig) < -0.0:
            samp_sig = np.mean(np.array([ invwishartrand(nuN,VN)]),0)
          
          ##3.2##μを多変量ガウスからサンプリング
          #print mN.T,mN[0][0],mN[1][0]
          x1,y1 = np.random.multivariate_normal([mN[0][0],mN[1][0]],samp_sig / kappaN,1).T
          #print x1,y1
          
          Myu[j] = np.array([[x1],[y1]])
          S[j] = samp_sig
          
        
        for j in xrange(K) : 
          if (nk[j] != 0):  #データなしは表示しない
            print 'myu'+str(j)+':'+str(Myu[j].T),
        print ''
        
        for j in xrange(K):
          if (nk[j] != 0):  #データなしは表示しない
            print 'sig'+str(j)+':'+str(S[j])
          
          
        """
        #データのあるKのみをプリントする？(未実装)
        print "myu1:%s myu2:%s myu3:%s myu4:%s myu5:%s" % (str(myuC[0].T), str(myuC[1].T), str(myuC[2].T),str(myuC[3].T), str(myuC[4].T))
        print "sig1:\n%s \nsig2:\n%s \nsig3:\n%s" % (str(sigmaC[0]), str(sigmaC[1]), str(sigmaC[2]))
        """
        #Myu = myuC
        #S = sigmaC
        
        ########## ↑ ##### μΣ(位置分布：ガウス分布の平均、共分散行列)のサンプリング ##### ↑ ##########
        
        
       ########## ↓ ##### π(場所概念のindexの多項分布)のサンプリング ##### ↓ ##########
        print u"Sampling PI..."
        
        #GEM = stick_breaking(gamma, L)
        #print GEM
        
        temp = np.ones(L) * (gamma / float(L)) #np.array([ gamma / float(L) for c in xrange(L) ])   #よくわからないので一応定義
        for c in xrange(L):
          temp[c] = temp[c] + Ct.count(c)
        #for t in xrange(N):    #Ct全データに対して
        #  for c in xrange(L):  #index cごとに
        #    if Ct[t] == c :      #データとindex番号が一致したとき
        #      temp[c] = temp[c] + 1
        #print temp  #確認済み
        
        #とりあえずGEMをパラメータとして加算してみる->桁落ちが発生していて意味があるのかわからない->パラメータ値を上げてみる&tempを正規化して足し合わせてみる(やめた)
        #print fsum(GEM),fsum(temp)
        #temp = temp / fsum(temp)
        #temp =  temp + GEM
        
        #持橋さんのスライドのやり方の方が正しい？ibis2008-npbayes-tutorial.pdf
        
        #print temp
        #加算したデータとパラメータから事後分布を計算しサンプリング
        sumn = sum(np.random.dirichlet(temp,1000)) #fsumではダメ
        pi = sumn / np.sum(sumn)
        print pi
        
        ########## ↑ ##### π(場所概念のindexの多項分布)のサンプリング ##### ↑ ##########
        
        
        ########## ↓ ##### φ(位置分布のindexの多項分布)のサンプリング ##### ↓ ##########
        print u"Sampling PHI_c..."
        
        #GEM = [ stick_breaking(alpha, K) for c in xrange(L) ]
        #print GEM
        
        for c in xrange(L):  #L個分
          temp = np.ones(K) * (alpha / float(K)) #np.array([ alpha / float(K) for k in xrange(K) ])   #よくわからないので一応定義
          #Ctとcが一致するデータを集める
          if c in Ct :
            for t in xrange(N):
              if Ct[t] == c:  #Ctとcが一致したデータで
                for k in xrange(K):  #index kごとに
                  if It[t] == k :      #データとindex番号が一致したとき
                    temp[k] = temp[k] + 1  #集めたデータを元に位置分布のindexごとに加算
            
          
          #ここからは一個分の事後GEM分布計算(πのとき)と同様
          #print fsum(GEM[c]),fsum(temp)
          #temp = temp / fsum(temp)
          #temp =  temp + GEM[c]
          
          #加算したデータとパラメータから事後分布を計算しサンプリング
          sumn = sum(np.random.dirichlet(temp,1000)) #fsumではダメ
          phi_l[c] = sumn / np.sum(sumn)
          
          if c in Ct:
            print c,phi_l[c]
          
          
        ########## ↑ ##### φ(位置分布のindexの多項分布)のサンプリング ##### ↑ ##########
        
        
  ######################################################################
  ####                   ↑場所概念学習フェーズ↑                   ####
  ######################################################################
      
      
      loop = 1
      ########  ↓ファイル出力フェーズ↓  ########
      if loop == 1:
        print "--------------------"
        #最終学習結果を出力
        print u"\n- <COMPLETED> Learning of Spatial Concepts ver. NEW MODEL. -"
        print 'Sample: ' + str(sample)
        print 'Ct: ' + str(Ct)
        print 'It: ' + str(It)
        for c in xrange(L):
          print "W%d: %s" % (c,W[c])
        for k in xrange(K):
          print "myu%d: %s" % (k, str(Myu[k].T))
        for k in xrange(K):
          print "sig%d: \n%s" % (k, str(S[k]))
        print 'pi: ' + str(pi)
        for c in xrange(L):
          print 'phi' + str(c) + ':',
          print str(phi_l[c])
        
        print "--------------------"
        
        #サンプリングごとに各パラメータ値を出力
        if loop == 1:
          fp = open(DataSetFolder +'/' + filename +'_kekka_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
          fp.write('sampling_data,'+repr(iter+1)+'\n')  #num_iter = 10  #イテレーション回数
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
            fp.write('Myu'+repr(k)+','+repr(Myu[k][0][0])+','+repr(Myu[k][1][0])+'\n')
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
            for k in xrange(K):
              fp.write(repr(k)+',')
            fp.write('\n')
            fp.write('phi_l'+repr(c)+',')
            for k in xrange(K):
              fp.write(repr(phi_l[c][k])+',')
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
          #fp_x = open(DataSetFolder +'/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
          #for t in xrange(EndStep) : 
          #  fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
          #fp_x.close()
        
        
        
        
        #各パラメータ値、初期値を出力
        fp_init = open(DataSetFolder +'/' + filename + '_init_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp_init.write('init_data\n')  #num_iter = 10  #イテレーション回数
        fp_init.write('L,'+repr(L)+'\n')
        fp_init.write('K,'+repr(K)+'\n')
        fp_init.write('alpha,'+repr(alpha)+'\n')
        fp_init.write('gamma,'+repr(gamma)+'\n')
        fp_init.write('bata0,'+repr(beta0)+'\n')
        fp_init.write('kappa0,'+repr(kappa0)+'\n')
        fp_init.write('m0,'+repr(m0)+'\n')
        fp_init.write('V0,'+repr(V0)+'\n')
        fp_init.write('nu0,'+repr(nu0)+'\n')
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
          fp_init.write('Myu_init'+repr(k)+','+repr(Myu_init[k][0])+','+repr(Myu_init[k][1])+'\n')
        for k in xrange(K):
          fp_init.write('Sig_init'+repr(k)+'\n')
          fp_init.write(repr(S_init[k])+'\n')
        for c in xrange(L):
          fp_init.write('W_init'+repr(c)+','+repr(W_init[c])+'\n')
        #for c in xrange(L):
        #  fp_init.write('phi_l_init'+repr(c)+','+repr(phi_l_init[c])+'\n')
        #fp_init.write('pi_init'+','+repr(pi_init)+'\n')
        for c in xrange(L):
          fp_init.write(',')
          for k in xrange(K):
            fp_init.write(repr(k)+',')
          fp_init.write('\n')
          fp_init.write('phi_l_init'+repr(c)+',')
          for k in xrange(K):
            fp_init.write(repr(phi_l_init[c][k])+',')
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
        
        ##自己位置推定結果をファイルへ出力
        #filename_xt = raw_input("Xt:filename?(.csv) >")  #ファイル名を個別に指定する場合
        #filename_xt = filename
        #fp = open(DataSetFolder +'/' + filename_xt + '_xt_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        #fp2 = open('./data/' + filename_xt + '_xt_true.csv', 'w')
        #fp3 = open('./data/' + filename_xt + '_xt_heikatsu.csv', 'w')
        #fp.write(Xt)
        #for t in xrange(EndStep) : 
        #    fp.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
        #    #fp2.write(repr(Xt_true[t][0]) + ', ' + repr(Xt_true[t][1]) + '\n')
        #    #fp2.write(repr(Xt_heikatsu[t][0]) + ', ' + repr(Xt_heikatsu[t][1]) + '\n')
        #fp.writelines(repr(Xt))
        #fp.close()
        #fp2.close()
        #fp3.close()
        
        ##認識発話単語集合をファイルへ出力
        #filename_ot = raw_input("Otb:filename?(.csv) >")  #ファイル名を個別に指定する場合
        filename_ot = filename
        fp = open(DataSetFolder +'/' + filename_ot + '_ot_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp2 = open(DataSetFolder +'/' + filename_ot + '_w_index_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
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
        
        print 'File Output Successful!(filename:'+filename+ "_" +str(iteration) + "_" + str(sample) + ')\n'
      
      
      ##パラメータそれぞれをそれぞれのファイルとしてはく
      if loop == 1:
        fp = open(DataSetFolder +'/' + filename + '_Myu_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(float(Myu[k][0][0]))+','+repr(float(Myu[k][1][0])) + '\n')
        fp.close()
        fp = open(DataSetFolder +'/' + filename + '_S_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(S[k][0][0])+','+repr(S[k][0][1])+','+repr(S[k][1][0]) + ','+repr(S[k][1][1])+'\n')
        fp.close()
        fp = open(DataSetFolder +'/' + filename + '_W_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for i in xrange(len(W_index)):
            fp.write(repr(W[c][i])+',')
          fp.write('\n')
          #fp.write(repr(W[l][0])+','+repr(W[l][1])+'\n')
        fp.close()
        fp = open(DataSetFolder +'/' + filename + '_phi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for k in xrange(K):
            fp.write(repr(phi_l[c][k])+',')
          fp.write('\n')
        fp.close()
        fp = open(DataSetFolder +'/' + filename + '_pi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          fp.write(repr(pi[c])+',')
        fp.write('\n')
        fp.close()
        
        fp = open(DataSetFolder +'/' + filename + '_Ct_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(Ct[t])+',')
        fp.write('\n')
        fp.close()
        
        fp = open(DataSetFolder +'/' + filename + '_It_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(It[t])+',')
        fp.write('\n')
        fp.close()
      
      ########  ↑ファイル出力フェーズ↑  ########
      
      """
      ##学習後の描画用処理
      iti = []    #位置分布からサンプリングした点(x,y)を保存する
      #Plot = 500  #プロット数
      
      K_yes = 0
      ###全てのパーティクルに対し
      for j in range(K) : 
        yes = 0
        for t in xrange(N):  #jが推定された位置分布のindexにあるか判定
          if j == It[t]:
            yes = 0 #1
        if yes == 1:
          K_yes = K_yes + 1
          for i in xrange(Plot):
            if (data_name != "test000"):
              S_temp = [[ S[j][0][0]/(0.05*0.05) , S[j][0][1]/(0.05*0.05) ] , [ S[j][1][0]/(0.05*0.05) , S[j][1][1]/(0.05*0.05) ]]
              x1,y1 = np.random.multivariate_normal( [(Myu[j][0][0][0]+37.8)/0.05, (Myu[j][1][0][0]+34.6)/0.05] , S_temp , 1).T
            else:
              x1,y1 = np.random.multivariate_normal([Myu[j][0][0][0],Myu[j][1][0][0]],S[j],1).T
            #print x1,y1
            iti = iti + [[x1,y1]]
      
      #iti = iti + [[K_yes,Plot]]  #最後の要素[[位置分布の数],[位置分布ごとのプロット数]]
      #print iti
      filename2 = str(iteration) + "_" + str(sample)
      
      #loop = 0 #メインループ用フラグ
      #while loop:
      #  #MAINCLOCK.tick(FPS)
      #  events = pygame.event.get()
      #  for event in events:
      #      if event.type == KEYDOWN:
      #          if event.key  == K_ESCAPE: exit()
      #  viewer.show(world,iti,0,[filename],[filename2])
      #  loop = 0
      """
      
      
      

def sougo(iteration):
  #MI_Samp = [0.0 for sample in xrange(sample_num)]  ##サンプルの数だけMIを求める
  MI_Samp2 = [0.0 for sample in xrange(sample_num)]  ##サンプルの数だけMIを求める
  #tanjyun_log = [0.0 for sample in xrange(sample_num)]
  #tanjyun_log2 = [0.0 for sample in xrange(sample_num)]
  #N = 0      #データ個数用
  #sample_num = 1  #取得するサンプル数
  Otb_Samp = [[] for sample in xrange(sample_num)]   #単語分割結果：教示データ
  W_index = [[] for sample in xrange(sample_num)]
  
  for sample in xrange(sample_num):
    
    #####↓##発話した文章ごとに相互情報量を計算し、サンプリング結果を選ぶ##↓######
    
    ##発話認識文データを読み込む
    ##空白またはカンマで区切られた単語を行ごとに読み込むことを想定する
    
    N = 0
    #for sample in xrange(sample_num):
    #Read text file
    for line in open(DataSetFolder + '/out_gmm_' + str(iteration) + '/' + str(sample) + '_samp.100', 'r'):   ##*_samp.100を順番に読み込む
        itemList = line[:-1].split(' ')
        
        #<s>,<sp>,</s>を除く処理：単語に区切られていた場合
        for b in xrange(5):
          if ("<s><s>" in itemList):
            itemList.pop(itemList.index("<s><s>"))
          if ("<s><sp>" in itemList):
            itemList.pop(itemList.index("<s><sp>"))
          if ("<s>" in itemList):
            itemList.pop(itemList.index("<s>"))
          if ("<sp>" in itemList):
            itemList.pop(itemList.index("<sp>"))
          if ("<sp><sp>" in itemList):
            itemList.pop(itemList.index("<sp><sp>"))
          if ("</s>" in itemList):
            itemList.pop(itemList.index("</s>"))
          if ("<sp></s>" in itemList):
            itemList.pop(itemList.index("<sp></s>"))
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        #<s>,<sp>,</s>を除く処理：単語中に存在している場合
        for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("<s><s>", "")
          itemList[j] = itemList[j].replace("<s>", "")
          itemList[j] = itemList[j].replace("<sp>", "")
          itemList[j] = itemList[j].replace("</s>", "")
        for b in xrange(5):
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        
        Otb_Samp[sample] = Otb_Samp[sample] + [itemList]
        #if sample == 0:
        N = N + 1  #count
        
        #for j in xrange(len(itemList)):
        #    print u"%s " % (str(itemList[j])),
        #print u""  #改行用
        
        
    
    ##場所の名前の多項分布のインデックス用
    #W_index = []
    #for sample in xrange(sample_num):    #サンプル個分
    for n in xrange(N):                #発話文数分
        for j in xrange(len(Otb_Samp[sample][n])):   #一文における単語数分
          if ( (Otb_Samp[sample][n][j] in W_index[sample]) == False ):
            W_index[sample].append(Otb_Samp[sample][n][j])
            #print str(W_index),len(W_index)
    
    print "[",
    for i in xrange(len(W_index[sample])):
      print "\""+ str(i) + ":" + str(W_index[sample][i]) + "\",",  #unicode(W_index[sample][i], 'shift-jis').encode('utf-8')
    print "]"
    
    
    #print type(W_index[sample][i])
    #print type(unicode(W_index[sample][i], 'shift-jis').encode('utf-8'))
    #print type(unicode(W_index[sample][i], 'utf-8'))
    
  ##サンプリングごとに、時刻tデータごとにBOW化(?)する、ベクトルとする
  Otb_B_Samp = [ [ [] for n in xrange(N) ] for ssss in xrange(sample_num) ]
  for sample in xrange(sample_num):
    for n in xrange(N):
      Otb_B_Samp[sample][n] = [0 for i in xrange(len(W_index[sample]))]
  
  for sample in xrange(sample_num):
    #for sample in xrange(sample_num):
    for n in xrange(N):
      for j in xrange(len(Otb_Samp[sample][n])):
          #print n,j,len(Otb_Samp[sample][n])
          for i in xrange(len(W_index[sample])):
            if (W_index[sample][i] == Otb_Samp[sample][n][j] ):
              Otb_B_Samp[sample][n][i] = Otb_B_Samp[sample][n][i] + 1
    #print Otb_B
    
    
    
    W = [ [beta0 for j in xrange(len(W_index[sample]))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
    pi = [ 0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
    #Ct = [ int(random.uniform(0,L)) for n in xrange(N)]
    Ct = []
    
    ##piの読み込み
    for line in open(DataSetFolder +'/' + filename + '_pi_'+str(iteration) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            pi[i] = float(itemList[i])
        
    ##Ctの読み込み
    for line in open(DataSetFolder +'/' + filename + '_Ct_'+str(iteration) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            Ct = Ct + [int(itemList[i])]
        
    ##Wの読み込み
    c = 0
    #Read text file
    for line in open(DataSetFolder +'/' + filename + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              W[c][i] = float(itemList[i])
              
              #print itemList
        c = c + 1
    
    
    
    #####↓##場所概念ごとに単語ごとに相互情報量を計算、高いものから表示##↓######
    ##相互情報量による単語のセレクション
    MI = [[] for c in xrange(L)]
    W_in = []    #閾値以上の単語集合
    #W_out = []   #W_in以外の単語
    #i_best = len(W_index)    ##相互情報量上位の単語をどれだけ使うか
    #MI_best = [ ['' for c in xrange(L)] for i in xrange(i_best) ]
    ###相互情報量を計算
    for c in xrange(L):
      #print "Concept:%d" % c
      #W_temp = Multinomial(W[c])
      for o in xrange(len(W_index[sample])):
        word = W_index[sample][o]
        
        ##BOW化(?)する、ベクトルとする
        #Otb_B = [0 for i in xrange(len(W_index[sample]))]
        #Otb_B[o] = 1
        
        #print W[c]
        #print Otb_B
        
        score = MI_binary(o,W,pi,c)
        
        MI[c].append( (score, word) )
        
        if (score >= threshold):  ##閾値以上の単語をすべて保持
          #print score , threshold ,word in W_in
          if ((word in W_in) == False):  #リストに単語が存在しないとき
            #print word
            W_in = W_in + [word]
        #else:
        #  W_out = W_out + [word]
        
      MI[c].sort(reverse=True)
      #for i in xrange(i_best):
      #  MI_best[i][c] = MI[c][i][1]
      
      #for score, word in MI[c]:
      #  print score, word
    
    ##ファイル出力
    fp = open(DataSetFolder + '/' + filename + '_sougo_C_' + str(iteration) + '_' + str(sample) + '.csv', 'w')
    for c in xrange(L):
      fp.write("Concept:" + str(c) + '\n')
      for score, word in MI[c]:
        fp.write(str(score) + "," + word + '\n')
      fp.write('\n')
    #for c in xrange(len(W_index)):
    fp.close()
    
    #####↑##場所概念ごとに単語ごとに相互情報量を計算、高いものから表示##↑######
    
    if (len(W_in) == 0 ):
      print "W_in is empty."
      W_in = W_index[sample] ##選ばれる単語がなかった場合、W_indexをそのままいれる
    
    print W_in
    
    ##場所の名前W（多項分布）をW_inに含まれる単語のみにする
    """
    for j in xrange(len(W_index[sample])):
      for i in xrange(len(W_in)):
        if (W_index[sample][j] in W_in == False):
          W_out = W_out + [W_index[sample][j]]
    """
    
    W_reco = [ [0.0 for j in xrange(len(W_in))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
    #W_index_reco = ["" for j in xrange(len(W_in))]
    #Otb_B_Samp_reco = [ [0 for j in xrange(len(W_in))] for n in xrange(N) ]
    #print L,N
    #print W_reco
    for c in xrange(L):
      for j in xrange(len(W_index[sample])):
        for i in xrange(len(W_in)):
          if ((W_in[i] in W_index[sample][j]) == True):
            W_reco[c][i] = float(W[c][j])
            #for t in xrange(N):
            #  Otb_B_Samp_reco[t][i] = Otb_B_Samp[sample][t][j]
      
      #正規化処理
      W_reco_sum = fsum(W_reco[c])
      W_reco_max = max(W_reco[c])
      W_reco_summax = float(W_reco_sum) / W_reco_max
      for i in xrange(len(W_in)):
        W_reco[c][i] = float(float(W_reco[c][i])/W_reco_max) / W_reco_summax
    
    #print W_reco
    
    ###相互情報量を計算(それぞれの単語とCtとの共起性を見る)
    MI_Samp2[sample] = Mutual_Info(W_reco,pi)
    
    print "sample:",sample," MI:",MI_Samp2[sample]
    
    
  MAX_Samp = MI_Samp2.index(max(MI_Samp2))  #相互情報量が最大のサンプル番号
  
  ##ファイル出力
  fp = open(DataSetFolder + '/' + filename + '_sougo_MI_' + str(iteration) + '.csv', 'w')
  #fp.write(',Samp,Samp2,tanjyun_log,tanjyun_log2,' +  '\n')
  for sample in xrange(sample_num):
      fp.write(str(sample) + ',' + str(MI_Samp2[sample]) + '\n') 
      #fp.write(str(sample) + ',' + str(MI_Samp[sample]) + ',' + str(MI_Samp2[sample]) + ',' + str(tanjyun_log[sample])  + ',' + str(tanjyun_log2[sample]) + '\n')  #文章ごとに計算
  fp.close()
  
  #  #####↑##発話した文章ごとに相互情報量を計算し、サンプリング結果を選ぶ##↑######
  
  #def Language_model_update(iteration):
  """
    ###推定された場所概念番号を調べる
    L_dd = [0 for c in xrange(L)]
    for t in xrange(len(Ct)):
      for c in xrange(L):
        if Ct[t] == c:
          L_dd[c] = 1
    ##print L_dd #ok
  """
  
  ###↓###単語辞書読み込み書き込み追加############################################
  LIST = []
  LIST_plus = []
  i_best = len(W_index[MAX_Samp])    ##相互情報量上位の単語をどれだけ使うか（len(W_index)：すべて）
  W_index = W_index[MAX_Samp]
  hatsuon = [ "" for i in xrange(i_best) ]
  TANGO = []
  ##単語辞書の読み込み
  for line in open('./lang_m/' + lang_init, 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
      
      
  #print TANGO
  
  #dd_num = 0
  ##W_indexの単語を順番に処理していく
  #for i in xrange(len(W_index)):
  #  W_index_sj = unicode(W_index[i], encoding='shift_jis')
  #for i in xrange(L):
  #  if L_dd[i] == 1:
  for c in xrange(i_best):    # i_best = len(W_index)
          #W_index_sj = unicode(MI_best[c][i], encoding='shift_jis')
          W_index_sj = unicode(W_index[c], encoding='shift_jis')
          if len(W_index_sj) != 1:  ##１文字は除外
            #for moji in xrange(len(W_index_sj)):
            moji = 0
            while (moji < len(W_index_sj)):
              flag_moji = 0
              #print len(W_index_sj),str(W_index_sj),moji,W_index_sj[moji]#,len(unicode(W_index[i], encoding='shift_jis'))
              
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-2 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+"_"+W_index_sj[moji+2]) and (W_index_sj[moji+1] == "_"): 
                    print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+W_index_sj[moji+1]):
                    print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_index_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_index_sj) > moji) and (flag_moji == 0):
                  #else:
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                      print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
            print hatsuon[c]
          else:
            print W_index[c] + " (one name)"
        
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
  fp = open(DataSetFolder + '/web.000s_' + str(iteration) + '.htkdic', 'w')
  for list in xrange(len(LIST)):
        fp.write(LIST[list])
  #fp.write('\n')
  #for c in xrange(len(W_index)):
  ##新しい単語を追加
  #i = 0
  c = 0
  #while i < L:
  #  #if L_dd[i] == 1:
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
                c = c+1
  #i = i+1
  fp.close()
  
  ###↑###単語辞書読み込み書き込み追加############################################
  





if __name__ == '__main__':
    #Request a file name for output
    #trialname = raw_input("trialname?(folder) >")
    trialname = sys.argv[1]
    print trialname
    
    start_time = time.time()
    iteration_time = [0.0 for i in range(ITERATION)]
    filename = outputfolder + trialname
    Makedir( filename )
    
    for i in xrange(ITERATION):
      print "--------------------------------------------------"
      print "ITERATION:",i+1
      start_iter_time = time.time()
      
      Julius_lattice(i,trialname)    ##音声認識、ラティス形式出力、opemFST形式へ変換
      
      FST_PATH = "./data/" + trialname + "/fst_gmm_" + str(i+1) + "/" + str(DATA_NUM-1).zfill(3) +".fst"
      while (os.path.exists( FST_PATH ) != True):
        print FST_PATH, os.path.exists( FST_PATH ), "wait(30s)... or ERROR?"
        time.sleep(30.0) #sleep(秒指定)
      print "ITERATION:",i+1," Julius complete!"
      
      sample = 0  ##latticelmのパラメータ通りだけサンプルする
      for p1 in xrange(len(knownn)):
        for p2 in xrange(len(unkn)):
          if sample < sample_num:
            print "latticelm run. sample_num:" + str(sample)
            latticelm_CMD = "latticelm -input fst -filelist data/" + trialname + "/fst_gmm_" + str(i+1) + "/fstlist.txt -prefix data/" + trialname + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_ -symbolfile data/" + trialname + "/fst_gmm_" + str(i+1) + "/isyms.txt -burnin 100 -samps 100 -samprate 100 -knownn " + str(knownn[p1]) + " -unkn " + str(unkn[p2])
            ##latticelm  ## -annealsteps 10 -anneallength 15
            OUT_PATH = "./data/" + trialname + "/out_gmm_" + str(i+1) + "/" + str(sample) + "_samp.100"

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
      
      Gibbs_Sampling(i+1)          ##場所概念の学習
      
      print "ITERATION:",i+1," Learning complete!"
      sougo(i+1)             ##相互情報量計算+##単語辞書登録
      print "ITERATION:",i+1," Language Model update!"
      #Language_model_update(i+1)  ##単語辞書登録
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

