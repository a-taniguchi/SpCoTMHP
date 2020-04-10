#coding:utf-8
#PRR評価用プログラム（範囲指定版）
#Akira Taniguchi (2017/02/27)
import sys
import os.path
import random
import string
import collections
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import chi2
from math import pi as PI
from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from __init__ import *
import os.path
from Julius1best_gmmSpCoSLAM import *
#import time

##追加の評価指標のプログラム
##発話→位置の評価：p(xt|Ot)
##turtlebot用。例外処理未対応
step = 50

#相互推定のプログラムにimportして使う。
#プログラムが単体でも動くようにつくる。

#各関数、編集途中。

def gaussian(x,myu,sig):
    ###1次元ガウス分布
    gauss = (1.0 / sqrt(2.0*PI*sig*sig)) * exp(-1.0*(float((x-myu)*(x-myu))/(2.0*sig*sig)))
    return gauss
    
def gaussian2d(Xx,Xy,myux,myuy,sigma):
    ###ガウス分布(2次元)
    sqrt_inb = float(1) / ( 2.0 * PI * sqrt( np.linalg.det(sigma)) )
    xy_myu = np.array( [ [float(Xx - myux)],[float(Xy - myuy)] ] )
    dist = np.dot(np.transpose(xy_myu),np.linalg.solve(sigma,xy_myu))
    gauss2d = (sqrt_inb) * exp( float(-1/2) * dist )
    return gauss2d

def fill_param(param, default):   ##パラメータをNone の場合のみデフォルト値に差し替える関数
    if (param == None): return default
    else: return param

def invwishartrand_prec(nu,W):
    return inv(wishartrand(nu,W))

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
    
class NormalInverseWishartDistribution(object):
#http://stats.stackexchange.com/questions/78177/posterior-covariance-of-normal-inverse-wishart-not-converging-properly
    def __init__(self, mu, lmbda, nu, psi):
        self.mu = mu
        self.lmbda = float(lmbda)
        self.nu = nu
        self.psi = psi
        self.inv_psi = np.linalg.inv(psi)

    def sample(self):
        sigma = np.linalg.inv(self.wishartrand())
        return (np.random.multivariate_normal(self.mu, sigma / self.lmbda), sigma)

    def wishartrand(self):
        dim = self.inv_psi.shape[0]
        chol = np.linalg.cholesky(self.inv_psi)
        foo = np.zeros((dim,dim))
        
        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    foo[i,j] = np.sqrt(chi2.rvs(self.nu-(i+1)+1))
                else:
                    foo[i,j]  = np.random.normal(0,1)
        return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

    def posterior(self, data):
        n = len(data)
        mean_data = np.mean(data, axis=0)
        sum_squares = np.sum([np.array(np.matrix(x - mean_data).T * np.matrix(x - mean_data)) for x in data], axis=0)
        mu_n = (self.lmbda * self.mu + n * mean_data) / (self.lmbda + n)
        lmbda_n = self.lmbda + n
        nu_n = self.nu + n
        psi_n = self.psi + sum_squares + self.lmbda * n / float(self.lmbda + n) * np.array(np.matrix(mean_data - self.mu).T * np.matrix(mean_data - self.mu))
        return NormalInverseWishartDistribution(mu_n, lmbda_n, nu_n, psi_n)

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
    

#http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section5_2-Dirichlet-Processes.ipynb
def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()

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



###↓###発話→場所の認識############################################
def Location_from_speech(filename,THETA,iteration,sample):
  datasetNUM = 0
  datasetname = datasets[int(datasetNUM)]
  
  #教示位置データを読み込み平均値を算出（xx,xy）
  XX = []
  count = 0
  Ct = []
  ##Ctの読み込み
  for line in open('./data/' + filename +'/' + filename + '_Ct_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            Ct = Ct + [int(itemList[i])]
  It = []
  ##Ctの読み込み
  for line in open('./data/' + filename +'/' + filename + '_It_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            It = It + [int(itemList[i])]
  ctc = collections.Counter(Ct)
  itc = collections.Counter(It)

  ItC = []
  #それぞれの場所の中央座標を出す（10カ所）
  s = 0
  #正解データを読み込みIT
  for line in open(datasetfolder + datasetname + correct_It, 'r'):
      itemList = line[:].split(',')
      for i in xrange(len(itemList)):
        if (itemList[i] != '') and (s < 50):
          ItC = ItC + [int(itemList[i])]
        s += 1
        
  ic = collections.Counter(ItC)
  icitems = ic.items()  # [(it番号,カウント数),(),...]
  
  X = [[] for i in range(len(ic))]
  Y = [[] for i in range(len(ic))]
  if (1):
        Xt = []
        for line3 in open('./../sample/' + data_name, 'r'):
          itemList3 = line3[:-1].split(',')
          Xt.append([float(itemList3[0]), float(itemList3[1])])
          count = count + 1
        
        for j in xrange(len(ic)):  #教示場所の種類数
          Xtemp  = []
          for i in xrange(len(ItC)): #要はステップ数（=50）
            if (icitems[j][0] == ItC[i]):
              Xtemp = Xtemp + [Xt[i]]
              X[j] = X[j] + [Xt[i][0]]
              Y[j] = Y[j] + [Xt[i][1]]
          
          #print len(Xtemp),Xtemp,ic[icitems[j][0]]
          XX = XX + [sum(np.array(Xtemp))/float(ic[icitems[j][0]])]
        #Xt = []
        #最終時点step=50での位置座標を読み込み(これだと今のパーティクル番号の最終時刻の位置情報になるが細かいことは気にしない)
        #Xt = np.array( ReadParticleData2(step,particle, trialname) )
  
  
  
  #THETA = [W,W_index,Myu,S,pi,phi_l]
  W = THETA[0]
  W_index = THETA[1]
  Myu = THETA[2]
  S = THETA[3]
  pi = THETA[4]
  phi_l = THETA[5]
  
  
  
  ##自己位置推定用の音声ファイルを読み込み
  # wavファイルを指定
  files = glob.glob(speech_folder_go)   #./../../../Julius/directory/CC3Th2/ (相対パス)
  #genkan,teeburu,teeburu,hondana,sofa,kittin,daidokoro,gomibako,terebimae
  files.sort()
  
  LAR = [] #0.0
  
  ##パーティクルをばらまく（全ての各位置分布に従う点を一点サンプリング）
  Xp = []
  
  for j in range(K):
    #x1,y1 = np.random.multivariate_normal([Myu[j][0][0],Myu[j][1][0]],S[j],1).T
    #位置分布の平均値と位置分布からサンプリングした10点をxtの候補とした
    if (itc[j] != 0):
      for i in range(9):    
        x1,y1 = np.mean(np.array([ np.random.multivariate_normal([Myu[j][0][0],Myu[j][1][0]],S[j],1).T ]),0)
        Xp = Xp + [[x1,y1]]
      Xp = Xp + [[Myu[j][0][0],Myu[j][1][0]]]
  
  WordDictionaryUpdate2(iteration+1, filename, W_index)       ##単語辞書登録
  
  k = 0
  ##学習した単語辞書を用いて音声認識し、BoWを得る
  for f in files:
    St = RecogLattice( f , iteration+1 , filename , N_best_number)
    Otb_B = [0 for i in xrange(len(W_index))]
    for j in range(len(St)):
      for i in range(5):
              St[j] = St[j].replace(" <s> ", "")
              St[j] = St[j].replace("<sp>", "")
              St[j] = St[j].replace(" </s>", "")
              St[j] = St[j].replace("  ", " ") 
              St[j] = St[j].replace("\n", "")
              
      print j,St[j]
      Otb = St[j].split(" ")
      ##データごとにBOW化
      #Otb_B = [ [] for s in xrange(len(files)) ]
      #for n in xrange(len(files)):
      #  Otb_B[n] = [0 for i in xrange(len(W_index))]
      
      
      #for n in xrange(N):
      for j2 in xrange(len(Otb)):
          #print n,j,len(Otb_Samp[sample][n])
          for i in xrange(len(W_index)):
            #print W_index[i].decode('sjis'),Otb[j]
            if (W_index[i].decode('sjis') == Otb[j2] ):
            #####if (W_index[i].decode('utf8') == Otb[j] ):
              Otb_B[i] = Otb_B[i] + 1
              #print W_index[i].decode('sjis'),Otb[j]
    print iteration,Otb_B
    
    
    
    pox = [0.0 for i in xrange(len(Xp))]
    ##パーティクルごとにP(xt|Ot,θ)の確率値を計算、最大の座標を保存
    ##位置データごとに
    for xdata in xrange(len(Xp)):
        
        ###提案手法による尤度計算####################
        #Ot_index = 0
        
        #for otb in xrange(len(W_index)):
        #Otb_B = [0 for j in xrange(len(W_index))]
        #Otb_B[Ot_index] = 1
        temp = [0.0 for c in range(L)]
        #print Otb_B
        for c in xrange(L) :
          if(ctc[c] != 0):
            ##場所の名前、多項分布の計算
            W_temp = Multinomial(W[c])
            temp[c] = W_temp.pmf(Otb_B)
            #temp[c] = W[c][otb]
            ##場所概念の多項分布、piの計算
            temp[c] = temp[c] * pi[c]
            
            ##itでサメーション
            it_sum = 0.0
            for it in xrange(K):
              if(itc[it] != 0):
                if (S[it][0][0] < pow(10,-100)) or (S[it][1][1] < pow(10,-100)) :    ##共分散の値が0だとゼロワリになるので回避
                    if int(Xp[xdata][0]) == int(Myu[it][0]) and int(Xp[xdata][1]) == int(Myu[it][1]) :  ##他の方法の方が良いかも
                        g2 = 1.0
                        print "gauss 1"
                    else : 
                        g2 = 0.0
                        print "gauss 0"
                else : 
                    g2 = gaussian2d(Xp[xdata][0],Xp[xdata][1],Myu[it][0],Myu[it][1],S[it])  #2次元ガウス分布を計算
                it_sum = it_sum + g2 * phi_l[c][it]
                
            temp[c] = temp[c] * it_sum
        
        pox[xdata] = sum(temp)
        
        #print Ot_index,pox[Ot_index]
        #Ot_index = Ot_index + 1
        #POX = POX + [pox.index(max(pox))]
        
        #print pox.index(max(pox))
        #print W_index_p[pox.index(max(pox))]
        
    
    
    Xt_max = [ Xp[pox.index(max(pox))][0], Xp[pox.index(max(pox))][1] ] #[0.0,0.0] ##確率最大の座標候補
    
    
    ##正解をどうするか
    ##正解の区間の座標であれば正解とする
    PXO = 0.0  ##座標が正解(1)か不正解か(0)
    
    #for i in range(K): #発話ごとに正解の場所の領域がわかるはず
    if (1):
      ##正解区間設定(上下左右10のマージン)margin
      #i = k
      print "k=",k
      
      if(k == 3): # ikidomari 2kasyo 
          #X[4].append(4)
          #X[4].append(6)
          #Y[4].append(-1)
          #Y[4].append(-4)
          #x座標の最小値-10
          xmin1 = min(X[4])
          #x座標の最大値+10
          xmax1 = max(X[4])
          #y座標の最小値-10
          ymin1 = min(Y[4])
          #y座標の最大値+10
          ymax1 = max(Y[4])
          
          #X[5].append(-6)
          #X[5].append(-10)
          #Y[5].append(-1)
          #Y[5].append(-4)
          #x座標の最小値-10
          xmin2 = min(X[5])
          #x座標の最大値+10
          xmax2 = max(X[5])
          #y座標の最小値-10
          ymin2 = min(Y[5])
          #y座標の最大値+10
          ymax2 = max(Y[5])
          
          #正解判定
          if( ((xmin1-margin <= Xt_max[0] <= xmax1+margin) and (ymin1-margin <= Xt_max[1] <= ymax1+margin)) or ((xmin2-margin <= Xt_max[0] <= xmax2+margin) and (ymin2-margin <= Xt_max[1] <= ymax2+margin)) ):
            PXO = PXO + 1
            print iteration,sample,k,Xt_max," OK!"
          else:
            print iteration,sample,k,Xt_max," NG!"
      
      elif(k == 1): # kyuukeijyo 2kasyo
          #X[1].append(9)
          #X[1].append(6.5)
          #Y[1].append(-1)
          #Y[1].append(4)
          #x座標の最小値-10
          xmin1 = min(X[1])
          #x座標の最大値+10
          xmax1 = max(X[1])
          #y座標の最小値-10
          ymin1 = min(Y[1])
          #y座標の最大値+10
          ymax1 = max(Y[1])
          
          #X[2].append(-5)
          #X[2].append(-10)
          #Y[2].append(-1)
          #Y[2].append(4)
          #x座標の最小値-10
          xmin2 = min(X[2])
          #x座標の最大値+10
          xmax2 = max(X[2])
          #y座標の最小値-10
          ymin2 = min(Y[2])
          #y座標の最大値+10
          ymax2 = max(Y[2])
          
          
          #正解判定
          if( ((xmin1-margin <= Xt_max[0] <= xmax1+margin) and (ymin1-margin <= Xt_max[1] <= ymax1+margin)) or ((xmin2-margin <= Xt_max[0] <= xmax2+margin) and (ymin2-margin <= Xt_max[1] <= ymax2+margin)) ):
            PXO = PXO + 1
            print iteration,sample,k,Xt_max," OK!"
          else:
            print iteration,sample,k,Xt_max," NG!"
      elif(k == 6 or k == 7): #purintaabeya and daidokoro
          #X[8].append(-4)
          #X[8].append(-6)
          #Y[8].append(-10)
          #Y[8].append(-4)
          #x座標の最小値-10
          xmin1 = min(X[8])
          #x座標の最大値+10
          xmax1 = max(X[8])
          #y座標の最小値-10
          ymin1 = min(Y[8])
          #y座標の最大値+10
          ymax1 = max(Y[8])
          
          #正解判定
          if( ((xmin1-margin <= Xt_max[0] <= xmax1+margin) and (ymin1-margin <= Xt_max[1] <= ymax1+margin)) ):
            PXO = PXO + 1
            print iteration,sample,k,Xt_max," OK!"
          else:
            print iteration,sample,k,Xt_max," NG!"
      
      else:
          if (k == 0):
            i = 0
            #X[i].append(2)
            #X[i].append(6.5)
            #Y[i].append(-1.5)
            #Y[i].append(4)
          elif (k == 2):
            i = 3
            #X[i].append(-0.5)
            #X[i].append(3)
            #Y[i].append(-1)
            #Y[i].append(2)
          elif (k == 4):
            i = 6
            #X[i].append(0.5)
            #X[i].append(4)
            #Y[i].append(-2)
            #Y[i].append(-4)
          elif (k == 5):
            i = 7
            #X[i].append(-4)
            #X[i].append(1)
            #Y[i].append(-4)
            #Y[i].append(-10)
          elif (k == 8):
            i = 9
            #X[i].append(-0.5)
            #X[i].append(-3)
            #Y[i].append(-1)
            #Y[i].append(4)
          #x座標の最小値-10
          xmin = min(X[i]) #min(X[i*10:i*10 + 10])
          #x座標の最大値+10
          xmax = max(X[i])
          #y座標の最小値-10
          ymin = min(Y[i])
          #y座標の最大値+10
          ymax = max(Y[i])
          
          #正解判定
          if( (xmin-margin <= Xt_max[0] <= xmax+margin) and (ymin-margin <= Xt_max[1] <= ymax+margin) ):
            PXO = PXO + 1
            print iteration,sample,k,Xt_max," OK!"
          else:
            print iteration,sample,k,Xt_max," NG!"
    
    LAR = LAR + [PXO]
    k = k + 1
    
  
  #LARの平均値を算出(各発話ごとの正解の割合え)
  LAR_mean = sum(LAR) / float(len(LAR))
  print LAR
  print LAR_mean
  
  return LAR_mean
###↑###発話→場所の認識############################################

###↓###単語辞書読み込み書き込み追加############################################
#MAX_Samp : 重みが最大のパーティクル
def WordDictionaryUpdate2(iteration, filename, W_list):
  LIST = []
  LIST_plus = []
  #i_best = len(W_list[MAX_Samp])    ##相互情報量上位の単語をどれだけ使うか（len(W_list)：すべて）
  i_best = len(W_list)
  #W_list = W_list[MAX_Samp]
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
  if (1):
    ##W_listの単語を順番に処理していく
    for c in xrange(i_best):    # i_best = len(W_list)
          #W_list_sj = unicode(MI_best[c][i], encoding='shift_jis')
          W_list_sj = unicode(W_list[c], encoding='shift_jis')
          if len(W_list_sj) != 1:  ##１文字は除外
            #for moji in xrange(len(W_list_sj)):
            moji = 0
            while (moji < len(W_list_sj)):
              flag_moji = 0
              #print len(W_list_sj),str(W_list_sj),moji,W_list_sj[moji]#,len(unicode(W_list[i], encoding='shift_jis'))
              
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
                  #else:
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_list_sj[moji]):
                      ###print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
            print hatsuon[c]
          else:
            print W_list[c] + " (one name)"
  
  ##各場所の名前の単語ごとに
  meishi = u'名詞'
  meishi = meishi.encode('shift-jis')
  
  ##単語辞書ファイル生成
  fp = open( "./data/" + filename + '/WDonly_' + str(iteration) + '.htkdic', 'w')
  for list in xrange(len(LIST)):
    if (list < 3):
        fp.write(LIST[list])
  #if (UseLM == 1):
  if (1):
    ##新しい単語を追加
    c = 0
    for mi in xrange(i_best):    # i_best = len(W_list)
        if hatsuon[mi] != "":
            if ((W_list[mi] in LIST_plus) == False):  #同一単語を除外
              flag_tango = 0
              for j in xrange(len(TANGO)):
                if(W_list[mi] == TANGO[j][0]):
                  flag_tango = -1
              if flag_tango == 0:
                LIST_plus = LIST_plus + [W_list[mi]]
                
                fp.write(LIST_plus[c] + "+" + meishi +"	[" + LIST_plus[c] + "]	" + hatsuon[mi])
                fp.write('\n')
                c = c+1
  
  fp.close()
  ###↑###単語辞書読み込み書き込み追加############################################


def Evaluation2(filename):
  
  #相互推定の学習結果データを読み込む
  MI_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  #ARI_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  #PARs_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  #PARw_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  LAR_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)] 
  # location accuracy rate from a name of place 
  MAX_Samp = [0 for j in xrange(ITERATION)]
  
  #イテレーションごとに選ばれた学習結果の評価値をすべて保存するファイル
  fp = open('./data/' + filename + '/' + filename + '_A_sougo_EvaluationPRR.csv', 'w')  
  
  #相互推定のイテレーションと単語分割結果の候補のすべてのパターンの評価値を保存
  #fp_ARI = open('./data/' + filename + '/' + filename + '_A_sougo_ARI.csv', 'w')  
  #fp_PARs = open('./data/' + filename + '/' + filename + '_A_sougo_PARs.csv', 'w')  
  #fp_PARw = open('./data/' + filename + '/' + filename + '_A_sougo_PARw.csv', 'w')  
  #fp_MI = open('./data/' + filename + '/' + filename + '_A_sougo_MI.csv', 'w')  
  fp_LAR = open('./data/' + filename + '/' + filename + '_A_sougo_PRR.csv', 'w')  
  #fp.write('MI,ARI,PARs,PARw\n')
  fp.write('LAR\n')
  
  #相互推定のイテレーションごとに
  for iteration in xrange(ITERATION):
    
    #./data/filename/filename_sougo_MI_iteration.csvを読み込み
    for line in open('./data/' + filename + '/' + filename + '_sougo_MI_' + str(iteration+1) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        if (int(itemList[0]) < sample_num):
          MI_List[iteration][int(itemList[0])] = float(itemList[1])
    #    fp_MI.write(str(float(itemList[1])))
    #    fp_MI.write(',')
    MAX_Samp[iteration] = MI_List[iteration].index(max(MI_List[iteration]))  #相互情報量が最大のサンプル番号
    #fp_MI.write(',' + str(MAX_Samp[iteration]))
    #fp_MI.write('\n')

    sample = MAX_Samp[iteration]
    #単語分割結果の候補ごとに
    #for sample in xrange(sample_num):
    #if (sample == MAX_Samp[iteration]):  
    for sample in xrange(sample_num):
      W_index= []
      
      i = 0
      #テキストファイルを読み込み
      for line in open('./data/' + filename +'/' + filename + '_w_index_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):   ##*_samp.100を順番に読み込む
        itemList = line[:-1].split(',')
        
        if(i == 1):
            for j in range(len(itemList)):
              W_index = W_index + [itemList[j]]
            
        i = i + 1
      
      #####パラメータW、μ、Σ、φ、πを入力する#####
      Myu = [ np.array([[ int( random.uniform(1,10) ) ],[ int( random.uniform(1,10) ) ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
      S = [ np.array([ [10000.0, 0.0],[0.0, 10000.0] ]) for i in xrange(K) ]      #位置分布の共分散(2×2次元)[K]
      W = [ [beta0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布：W_index次元)[L]
      pi = [ 0 for c in xrange(L)]     #場所概念のindexの多項分布(L次元)
      phi_l = [ [0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K次元)[L]
      
      Ct = []
      
      i = 0
      ##Myuの読み込み
      for line in open('./data/' + filename +'/' + filename + '_Myu_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        itemList[1] = itemList[1].replace("_"+str(sample), "")
        Myu[i] = np.array([[ float(itemList[0]) ],[ float(itemList[1]) ]])
        
        i = i + 1
      
      i = 0
      ##Sの読み込み
      for line in open('./data/' + filename +'/' + filename + '_S_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        itemList[2] = itemList[2].replace("_"+str(sample), "")
        S[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3]) ]])
        
        i = i + 1
      
      ##phiの読み込み
      c = 0
      #テキストファイルを読み込み
      for line in open('./data/' + filename +'/' + filename + '_phi_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
            if itemList[i] != "":
              phi_l[c][i] = float(itemList[i])
        c = c + 1
      
      
      ##piの読み込み
      for line in open('./data/' + filename +'/' + filename + '_pi_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            pi[i] = float(itemList[i])
        
      ##Ctの読み込み
      for line in open('./data/' + filename +'/' + filename + '_Ct_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            Ct = Ct + [int(itemList[i])]
        
      ##Wの読み込み
      c = 0
      #テキストファイルを読み込み
      for line in open('./data/' + filename +'/' + filename + '_W_' + str(iteration+1) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #print c
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              W[c][i] = float(itemList[i])
              
              #print itemList
        c = c + 1
      
      #############################################################
      
      print iteration,sample
      
      #print "ARI"
      #ARI_List[iteration][sample] = ARI(Ct)
      
      #print "PAR_S"
      #PARs_List[iteration][sample] = PAR_sentence(iteration,sample)
       
      THETA = [W,W_index,Myu,S,pi,phi_l]
      #NOP = []
      #print "PAR_W"
      #PARw_List[iteration][sample] = Name_of_Place(THETA)
      
      LAR_List[iteration][sample] = Location_from_speech(filename,THETA,iteration,sample)
      
      print "OK!"
      #fp_ARI.write(str(ARI_List[iteration][sample]))
      #fp_ARI.write(',')
      #fp_PARs.write(str(PARs_List[iteration][sample]))
      #fp_PARs.write(',')
      #fp_PARw.write(str(PARw_List[iteration][sample]))
      #fp_PARw.write(',')
      fp_LAR.write(str( LAR_List[iteration][sample] ))
      fp_LAR.write(',')
    
    #fp_ARI.write(',')
    #smean = sum(ARI_List[iteration])/sample_num
    #fp_ARI.write(str(smean))
    #fp_ARI.write('\n')
    
    #fp_PARs.write(',')
    #smean = sum(PARs_List[iteration])/sample_num
    #fp_PARs.write(str(smean))
    #fp_PARs.write('\n')
    
    #fp_PARw.write(',')
    #smean = sum(PARw_List[iteration])/sample_num
    #fp_PARw.write(str(smean))
    #fp_PARw.write('\n')
    
    fp_LAR.write(',')
    smean = sum(LAR_List[iteration])/sample_num
    fp_LAR.write(str(smean))
    fp_LAR.write('\n')
    
    #MI,ARI,PARs,PARw,
    
    #fp.write( str(MI_List[iteration][MAX_Samp[iteration]])+','+ str(ARI_List[iteration][MAX_Samp[iteration]])+','+ str(PARs_List[iteration][MAX_Samp[iteration]])+','+str(PARw_List[iteration][MAX_Samp[iteration]]) )
    fp.write( str(LAR_List[iteration][MAX_Samp[iteration]]) )
    fp.write('\n')
    
  print "close."
  
  fp.close()
  #fp_ARI.close()
  #fp_PARs.close()
  #fp_PARw.close()
  #fp_MI.close()
  fp_LAR.close()
  
if __name__ == '__main__':
    #出力ファイル名を要求
    trialname = raw_input("trialname?(**_num) >") #"tamd2_sig_mswp_01" 
    ITERATION = 10
    sample_num = 6
    
    if ("p1" in trialname):
      R = 1
    elif ("p30" in trialname):
      R = 30
    
    if ("nf" in trialname):
      UseFT = 0
    else:
      UseFT = 1
    
    if ("nl" in trialname):
      UseLM = 0
    else:
      UseLM = 1
    
    #おてがるコース（SpCoA）
    #ITERATION = 1  #相互推定のイテレーション回数
    
    for i in range(1,11):
      Evaluation2(trialname + str(i).zfill(3))
