#coding:utf-8

##############################################
##Spatial concept formation model (SpCoA without lexical acquisition)
##For SpCoNavi (on SIGVerse)
##Learning algorithm is Gibbs sampling.
##Akira Taniguchi -2019/07/25
##############################################

# python ./learn4_3SpCoA_GT.py 3LDK_00

import glob
import codecs
import re
import os
import os.path
import sys
import random
import string
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
from __init__ import *
from submodules import *

def gaussian2d(Xx,Xy,myux,myuy,sigma):
    ###Gaussian distribution(2-dimension)
    sqrt_inb = float(1) / ( 2.0 * PI * sqrt( np.linalg.det(sigma)) )
    xy_myu = np.array( [ [float(Xx - myux)],[float(Xy - myuy)] ] )
    dist = np.dot(np.transpose(xy_myu),np.linalg.solve(sigma,xy_myu))
    gauss2d = (sqrt_inb) * exp( float(-1/2) * dist )
    return gauss2d
    
#def invwishartrand_prec(nu,W):
#    return inv(wishartrand(nu,W))

def invwishartrand(nu, W):
    return inv(wishartrand(nu, inv(W)))

def wishartrand(nu, W):
    dim = W.shape[0]
    chol = cholesky(W)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.axrange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in xrange(dim):
        for j in xrange(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = np.random.normal(0,1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

#http://stackoverflow.com/questions/13903922/multinomial-pmf-in-python-scipy-numpy
class Multinomial(object):
  def __init__(self, params):
    self._params = params

  def pmf(self, counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 1.
    for i,c in enumerate(counts):
      prob *= self._params[i]**counts[i]

    return prob * exp(self._log_multinomial_coeff(counts))

  def log_pmf(self,counts):
    if not(len(counts)==len(self._params)):
      raise ValueError("Dimensionality of count vector is incorrect")

    prob = 0.
    for i,c in enumerate(counts):
      prob += counts[i]*log(self._params[i])

    return prob + self._log_multinomial_coeff(counts)

  def _log_multinomial_coeff(self, counts):
    return self._log_factorial(sum(counts)) - sum(self._log_factorial(c)
                                                    for c in counts)

  def _log_factorial(self, num):
    if not round(num)==num and num > 0:
      raise ValueError("Can only compute the factorial of positive ints")
    return sum(log(n) for n in range(1,num+1))

"""
def MI_binary(b,W,pi,c):  #Mutual information (binary variable): word_index, W, π, Ct
    POC = W[c][b] * pi[c] #Multinomial(W[c]).pmf(B) * pi[c]  
    PO = sum([W[ct][b] * pi[ct] for ct in xrange(L)]) #Multinomial(W[ct]).pmf(B)
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
        #B = [int(i==j) for i in xrange(len(W[c]))]
        PO = fsum([W[ct][j] * pi[ct] for ct in xrange(len(pi))])  #Multinomial(W[ct]).pmf(B)
        POC = W[c][j] * pi[c]   
        
        # Calculate each term for MI
        MI = MI + POC * ( log((POC/(PO*PC)), 2) )
    
    return MI
"""

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
    
    #Xt = (np.array(all_position) + origin[0] ) / resolution #* 10
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

# Gibbs sampling
def Gibbs_Sampling(iteration,filename):
    sample_num = 1  #the sample number
    #N = 0      #the number of data
    #Otb = [[] for sample in xrange(sample_num)]   #word information

    inputfile = inputfolder_SIG  + trialname
    filename  = outputfolder_SIG + trialname
    
    ##S## ##### Ishibushi's code #####
    env_para = np.genfromtxt(inputfile+"/Environment_parameter.txt",dtype= None,delimiter =" ")

    MAP_X = float(env_para[0][1])  #Max x value of the map
    MAP_Y = float(env_para[1][1])  #Max y value of the map
    map_x = float(env_para[2][1])  #Min x value of the map
    map_y = float(env_para[3][1])  #Max y value of the map

    map_center_x = ((MAP_X - map_x)/2)+map_x
    map_center_y = ((MAP_Y - map_x)/2)+map_y
    mu_0 = np.array([map_center_x,map_center_y,0,0])
    #mu_0_set.append(mu_0)
    DATA_initial_index = int(env_para[5][1]) #Initial data num
    DATA_last_index = int(env_para[6][1]) #Last data num
    DATA_NUM = DATA_last_index - DATA_initial_index +1
    ##E## ##### Ishibushi's code ######
    
    #DATA read
    pose = position_data_read_pass(inputfile,DATA_NUM)
    #name = Name_data_read(inputfile,word_increment,DATA_NUM)
    
    for sample in xrange(sample_num):
      N = 0
      Otb = []
      #Read text file
      for word_data_num in range(DATA_NUM):
        f = open(inputfile + word_folder + str(word_data_num) + ".txt", "r")
        line = f.read()
        itemList = line[:].split(' ')
        
        """
        #remove <s>,<sp>,</s>: if its were segmented to words.
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
        #remove <s>,<sp>,</s>: if its exist in words.
        for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("<s><s>", "")
          itemList[j] = itemList[j].replace("<s>", "")
          itemList[j] = itemList[j].replace("<sp>", "")
          itemList[j] = itemList[j].replace("</s>", "")
        """
        
        for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("\r", "")  

        for b in xrange(5):
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        
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
            if (W_index[i] == Otb[n][j] ):
              Otb_B[n][i] = Otb_B[n][i] + word_increment
      #print Otb_B
      
      #N = DATA_NUM
      if N != DATA_NUM:
         print "DATA_NUM" + str(DATA_NUM) + ":KYOUJI error!! N:" + str(N)  ##教示フェーズの教示数と読み込んだ発話文data数が違う場合
         #exit()
      
      Xt = pose
      TN = [i for i in range(DATA_NUM)]
      
      
  #############################################################################
  ####                 ↓Learning phase of spatial concept↓                 ####
  #############################################################################
      #TN[N]: teaching time-step
      
      ##Initialization of all parameters
      print u"Initialize Parameters..."
      Ct = [ int(n/15) for n in xrange(N)]    #[ int(random.uniform(0,L)) for n in xrange(N)] #index of spatial concepts [N]
      It = [ int(n/15) for n in xrange(N)]    #[ int(random.uniform(0,K)) for n in xrange(N)] #index of position distributions [N]
      ##Uniform random numbers within the range
      Myu   = [ np.array([[ int( random.uniform(WallXmin,WallXmax) ) ],[ int( random.uniform(WallYmin,WallYmax) ) ]]) for i in xrange(K) ]      #the position distribution (Gaussian)の平均(x,y)[K]
      S     = [ np.array([ [sig_init, 0.0],[0.0, sig_init] ]) for i in xrange(K) ]      #the position distribution (Gaussian)の共分散(2×2-dimension)[K]
      W     = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #the name of place(multinomial distribution: W_index-dimension)[L]
      pi    = stick_breaking(gamma, L) #[ 0 for c in xrange(L)]     #index of spatial conceptのmultinomial distribution(L-dimension)
      phi_l = [ stick_breaking(alpha, K) for c in xrange(L) ] #[ [0 for i in xrange(K)] for c in xrange(L) ]  #index of position distributionのmultinomial distribution(K-dimension)[L]
      
      
      print Myu
      print S
      print W
      print pi
      print phi_l
      
      ###Copy initial values
      Ct_init    = [Ct[n] for n in xrange(N)]
      It_init    = [It[n] for n in xrange(N)]
      Myu_init   = [Myu[i] for i in xrange(K)]
      S_init     = [ np.array([ [S[i][0][0], S[i][0][1]],[S[i][1][0], S[i][1][1]] ]) for i in xrange(K) ]
      W_init     = [W[c] for c in xrange(L)]
      pi_init    = [pi[c] for c in xrange(L)]
      phi_l_init = [phi_l[c] for c in xrange(L)]
      
      
      ##Start learning of spatial concepts
      print u"- <START> Learning of Location Concepts ver. NEW MODEL. -"
      
      for iter in xrange(num_iter):   #Iteration of Gibbs sampling
        print 'Iter.'+repr(iter+1)+'\n'
        
        ########## ↓ ##### W(the name of place: multinomial distribution) is samplied ##### ↓ ##########
        ##Dirichlet multinomial distributionからDirichlet Posterior distributionを計算しSampingする
        print u"Sampling Wc..."
        
        temp = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #集めて加算するための array :paramtersで初期化しておけばよい
        #Ctがcであるときのdataを集める
        for c in xrange(L) :   #ctごとにL個分計算
          nc = 0
          ##Posterior distributionのためのparamters計算
          if c in Ct : 
            for t in xrange(N) : 
              if Ct[t] == c : 
                #dataを集めるたびに値を加算
                for j in xrange(len(W_index)):    #ベクトル加算？頻度
                  temp[c][j] = temp[c][j] + Otb_B[t][j]
                nc = nc + 1  #counting the number of data
              
          if (nc != 0):  #dataなしのcは表示しない
            print "%d n:%d %s" % (c,nc,temp[c])
          
          #加算したdataとparamtersからPosterior distributionを計算しSamping
          sumn = sum(np.random.dirichlet(temp[c],1000)) #NO use fsum
          W[c] = sumn / sum(sumn)
          #print W[c]
        
        ########## ↑ ##### W(the name of place: multinomial distribution) is samplied ##### ↑ ##########
        
        ########## ↓ ##### μΣ(the position distribution (Gaussian): Gaussian distributionの平均, 共分散行列) is samplied ##### ↓ ##########
        print u"Sampling myu_i,Sigma_i..."
        np.random.seed()
        nk = [0 for j in xrange(K)]
        for j in xrange(K) : 
          ###jについて, Ctが同じものを集める
          #n = 0
          
          xt = []
          if j in It : 
            for t in xrange(N) : 
              if It[t] == j : 
                xt_To = TN[t]
                xt = xt + [ np.array([ [Xt[xt_To][0]], [Xt[xt_To][1]] ]) ]
                nk[j] = nk[j] + 1
          
          m_ML = np.array([[0.0],[0.0]])
          if nk[j] != 0 :        ##Avoid divide by zero
            m_ML = sum(xt) / float(nk[j]) #NO use fsum
            print "n:%d m_ML.T:%s" % (nk[j],str(m_ML.T))
          
          ##hyper-paramters update
          kappaN = kappa0 + nk[j]
          mN = ( (kappa0*m0) + (nk[j]*m_ML) ) / kappaN
          nuN = nu0 + nk[j]
          
          dist_sum = 0.0
          for k in xrange(nk[j]) : 
            dist_sum = dist_sum + np.dot((xt[k] - m_ML),(xt[k] - m_ML).T)
          VN = V0 + dist_sum + ( float(kappa0*nk[j])/(kappa0+nk[j]) ) * np.dot((m_ML - m0),(m_ML - m0).T)
          
          #if nk[j] == 0 :        ##Avoid divide by zero
          #  #nuN = nu0# + 1  ##nu0=nuN=1だと何故かエラーのため
          #  #kappaN = kappaN# + 1
          #  mN = np.array([[ int( random.uniform(1,WallX-1) ) ],[ int( random.uniform(1,WallY-1) ) ]])   ###領域内に一様
          
          ##3.1##ΣをInv-WishartからSamping
          samp_sig_rand = np.array([ invwishartrand(nuN,VN) for i in xrange(100)])    ######
          samp_sig = np.mean(samp_sig_rand,0)
          #print samp_sig
          
          if np.linalg.det(samp_sig) < -0.0:
            samp_sig = np.mean(np.array([ invwishartrand(nuN,VN)]),0)
          
          ##3.2##μをGaussianからSamping
          #print mN.T,mN[0][0],mN[1][0]
          x1,y1 = np.random.multivariate_normal([mN[0][0],mN[1][0]],samp_sig / kappaN,1).T
          #print x1,y1
          
          Myu[j] = np.array([[x1],[y1]])
          S[j] = samp_sig
          
        
        for j in xrange(K) : 
          if (nk[j] != 0):  #dataなしは表示しない
            print 'myu'+str(j)+':'+str(Myu[j].T),
        print ''
        
        for j in xrange(K):
          if (nk[j] != 0):  #dataなしは表示しない
            print 'sig'+str(j)+':'+str(S[j])
          
        
        ########## ↑ ##### μΣ(the position distribution (Gaussian): Gaussian distributionの平均, 共分散行列) is samplied ##### ↑ ##########
        
        
       ########## ↓ ##### π(index of spatial conceptのmultinomial distribution) is samplied ##### ↓ ##########
        print u"Sampling PI..."
        
        temp = np.ones(L) * (gamma / float(L)) #np.array([ gamma / float(L) for c in xrange(L) ])   #よくわからないので一応定義
        for c in xrange(L):
          temp[c] = temp[c] + Ct.count(c)

        #print temp
        #加算したdataとparamtersからPosterior distributionを計算しSamping
        sumn = sum(np.random.dirichlet(temp,1000)) #NO use fsum
        pi = sumn / np.sum(sumn)
        print pi
        
        ########## ↑ ##### π(index of spatial conceptのmultinomial distribution) is samplied ##### ↑ ##########
        
        
        ########## ↓ ##### φ(index of position distributionのmultinomial distribution) is samplied ##### ↓ ##########
        print u"Sampling PHI_c..."

        for c in xrange(L):  #L個分
          temp = np.ones(K) * (alpha / float(K)) #np.array([ alpha / float(K) for k in xrange(K) ])   #よくわからないので一応定義
          #Ctとcが一致するdataを集める
          if c in Ct :
            for t in xrange(N):
              if Ct[t] == c:  #Ctとcが一致したdataで
                for k in xrange(K):  #index kごとに
                  if It[t] == k :      #dataとindex番号が一致したとき
                    temp[k] = temp[k] + 1  #集めたdataを元にindex of position distributionごとに加算
          
          #加算したdataとparamtersからPosterior distributionを計算しSamping
          sumn = sum(np.random.dirichlet(temp,1000)) #NO use fsum
          phi_l[c] = sumn / np.sum(sumn)
          
          if c in Ct:
            print c,phi_l[c]
          
        ########## ↑ ##### φ(index of position distributionのmultinomial distribution) is samplied ##### ↑ ##########
        
        ########## ↓ ##### it(index of position distribution) is samplied ##### ↓ ##########
        print u"Sampling it..."
        
        #itと同じtのCtの値c番目のφc  の要素kごとに事後multinomial distributionの値を計算
        temp = np.zeros(K)
        for t in xrange(N):    #時刻tごとのdata
          phi_c = phi_l[int(Ct[t])]
          
          for k in xrange(K):
            #it=k番目のμΣについてのGaussian distributionをitと同じtのxtから計算
            xt_To = TN[t]
            g2 = gaussian2d(Xt[xt_To][0],Xt[xt_To][1],Myu[k][0],Myu[k][1],S[k])  #2-dimensionGaussian distributionを計算
            
            temp[k] = g2 * phi_c[k]
            #print g2,phi_c[k]  ###Xtとμが遠いとg2の値がアンダーフローする可能性がある
            
          temp = temp / np.sum(temp)  #Normalization
          
          #print Mult_samp
          It_B = np.random.multinomial(1,temp) #Mult_samp [t]
          #print It_B[t]
          It[t] = np.where(It_B == 1)[0][0] #It_B.index(1)
        
        print It
        
        ########## ↑ ##### it(index of position distribution) is samplied ##### ↑ ##########
        
        
        ########## ↓ ##### Ct(index of spatial concept) is samplied ##### ↓ ##########
        print u"Sampling Ct..."
        #Ct～多項値P(Ot|Wc)*多項値P(it|φc)*多項P(c|π)  N個
        
        temp = np.zeros(L)
        for t in xrange(N):    #時刻tごとのdata
          for c in xrange(L):  #index of spatial conceptのmultinomial distributionそれぞれについて
            W_temp = Multinomial(W[c])
            #print pi[c], phi_temp.pmf(It_B[t]), W_temp.pmf(Otb_B[t])
            temp[c] = pi[c] * phi_l[c][It[t]] * W_temp.pmf(Otb_B[t])    # phi_temp.pmf(It_B[t])各要素について計算
          
          temp = temp / np.sum(temp)  #Normalization
          #print temp

          Ct_B = np.random.multinomial(1,temp) #Mult_samp
          #print Ct_B[t]
          
          Ct[t] = np.where(Ct_B == 1)[0][0] #Ct_B.index(1)
          
        print Ct
        ########## ↑ ##### Ct(index of spatial concept) is samplied ##### ↑ ##########
        
        
        """
        loop = 0
        if loop == 1:
          #Sampingごとに各paramters値をoutput
          fp = open('./data/' + filename + '/' + filename +'_samp'+ repr(iter)+'.csv', 'w')
          fp.write('sampling_data,'+repr(iter)+'\n')  #num_iter = 10  #The number of iterations
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
            fp.write('Myu'+repr(k)+','+repr(Myu[k][0])+','+repr(Myu[k][1])+'\n')
          for k in xrange(K):
            fp.write('Sig'+repr(k)+'\n')
            fp.write(repr(S[k])+'\n')
          for c in xrange(L):
            fp.write('W'+repr(c)+','+repr(W[c])+'\n')
          for c in xrange(L):
            fp.write('phi_l'+repr(c)+','+repr(phi_l[c])+'\n')
          fp.write('pi'+','+repr(pi)+'\n')
          fp.close()
          fp_x = open( filename + '/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
          for t in xrange(EndStep) : 
            fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
          fp_x.close()
        """
      
  #############################################################################
  ####                 ↑Learning phase of spatial concept↑                 ####
  ############################################################################# 
      
      loop = 1
      ########  ↓File output↓  ########
      if loop == 1:
        print "--------------------"
        #最終学習結果をoutput
        print u"\n- <COMPLETED> Learning of Location Concepts ver. NEW MODEL. -"
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
        
        #Sampingごとに各paramters値をoutput
        if loop == 1:
          fp = open( filename + '/' + trialname +'_kekka_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
          fp.write('sampling_data,'+repr(iter+1)+'\n')  #num_iter = 10  #The number of iterations
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
          #fp_x = open( filename + '/' + filename +'_xt'+ repr(iter)+'.csv', 'w')
          #for t in xrange(EndStep) : 
          #  fp_x.write(repr(Xt[t][0]) + ', ' + repr(Xt[t][1]) + '\n')
          #fp_x.close()
        
        

        #All parameters and initial values are output
        fp_init = open( filename + '/' + trialname + '_init_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp_init.write('init_data\n')  #num_iter = 10  #The number of iterations
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
        
        
        ##Output the set of recognition results of words to file
        #filename_ot = raw_input("Otb:filename?(.csv) >")  #ファイル名を個別に指定する場合
        filename_ot = trialname
        fp = open(filename + '/' + filename_ot + '_ot_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        fp2 = open(filename + '/' + filename_ot + '_w_index_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
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
      
      
      ##paramtersそれぞれをそれぞれのファイルとしてはく
      if loop == 1:
        fp = open( filename + '/' + trialname + '_Myu_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(float(Myu[k][0][0]))+','+repr(float(Myu[k][1][0])) + '\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_S_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for k in xrange(K):
          fp.write(repr(S[k][0][0])+','+repr(S[k][0][1])+','+repr(S[k][1][0]) + ','+repr(S[k][1][1])+'\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_W_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for i in xrange(len(W_index)):
            fp.write(repr(W[c][i])+',')
          fp.write('\n')
          #fp.write(repr(W[l][0])+','+repr(W[l][1])+'\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_phi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          for k in xrange(K):
            fp.write(repr(phi_l[c][k])+',')
          fp.write('\n')
        fp.close()
        fp = open( filename + '/' + trialname + '_pi_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for c in xrange(L):
          fp.write(repr(pi[c])+',')
        fp.write('\n')
        fp.close()
        
        fp = open( filename + '/' + trialname + '_Ct_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(Ct[t])+',')
        fp.write('\n')
        fp.close()
        
        fp = open( filename + '/' + trialname + '_It_'+str(iteration) + "_" + str(sample) + '.csv', 'w')
        for t in xrange(N):
          fp.write(repr(It[t])+',')
        fp.write('\n')
        fp.close()

        #fp = open( filename + "/W_list.csv", 'w')
        #for w in xrange(len(W_index)):
        #  fp.write(W_index[w]+",")
        #fp.close()

      ########  ↑File output↑  ########
      
     
if __name__ == '__main__':
    
    trialname = sys.argv[1]
    print trialname
    
    #Request a file name for output
    #filename = raw_input("trialname?(folder) >")
    #start_time = time.time()
    #iteration_time = [0.0 for i in range(ITERATION)]
    filename = outputfolder_SIG + trialname
    Makedir( filename )

    print "--------------------------------------------------"
    print "ITERATION:",1
    Gibbs_Sampling(1,trialname)          ##Learning of spatial concepts
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
