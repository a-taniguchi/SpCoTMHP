#coding:utf-8
#Akira Taniguchi 2017/02/15
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
from initSpCoSLAM import *

#相互推定のイテレーションごとに選ばれた学習結果の評価値（ARI、コーパスPAR、単語PAR）、（事後確率値）を出力
##単語PARはp(O_best|x_t)と正解を比較。x_tは正しいデータの平均値とする。

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


###↓###ARI############################################ok
def ARI(Ct,It):  #Ct = []  推定された場所概念のindex  
  CtC = [] #正解の分類結果
  ItC = [] #正解の分類結果

  datasetNUM = 0
  datasetname = datasets[int(datasetNUM)]
  #正解データを読み込みCT
  for line in open( datasetfolder + datasetname + correct_Ct, 'r'):
    itemList = line[:].split(',')
    
    for i in xrange(len(itemList)):
      if itemList[i] != '':
        CtC = CtC + [int(itemList[i])]
  
  #正解データを読み込み
  for line in open(datasetfolder + datasetname + correct_It, 'r'):
    itemList = line[:].split(',')
    
    for i in xrange(len(itemList)):
      if itemList[i] != '':
        ItC = ItC + [int(itemList[i])]

  #ARIを計算
  ARIc = adjusted_rand_score(CtC, Ct)
  print CtC
  print Ct  
  print str(ARIc)

  #ARIを計算
  ARIi = adjusted_rand_score(ItC, It)
  print ItC
  print It  
  print str(ARIi)
  
  return ARIc,ARIi
###↑###ARI############################################

###↓###NMI############################################ok
def NMI(Ct,It):  #Ct = []  推定された場所概念のindex  
  CtC = [] #正解の分類結果
  ItC = [] #正解の分類結果

  datasetNUM = 0
  datasetname = datasets[int(datasetNUM)]
  #正解データを読み込みCT
  for line in open( datasetfolder + datasetname + correct_Ct, 'r'):
    itemList = line[:].split(',')
    
    for i in xrange(len(itemList)):
      if itemList[i] != '':
        CtC = CtC + [int(itemList[i])]
  
  #正解データを読み込み
  for line in open(datasetfolder + datasetname + correct_It, 'r'):
    itemList = line[:].split(',')
    
    for i in xrange(len(itemList)):
      if itemList[i] != '':
        ItC = ItC + [int(itemList[i])]

  #ARIを計算
  NMI_CT = normalized_mutual_info_score(CtC, Ct)
  print CtC
  print Ct  
  print str(NMI_CT)

  #ARIを計算
  NMI_IT = normalized_mutual_info_score(ItC, It)
  print ItC
  print It  
  print str(NMI_IT)
  
  return NMI_CT, NMI_IT
###↑###NMI############################################

###↓###文章まるごとのPAR############################################
def PAR_sentence(iteration,sample):
  #区切り位置も一文字としてカウントして計算。
  #音素は一文字ではないので、複数文字で表現された音素を別な記号に置き換える
  datasetNUM = 0
  datasetname = datasets[int(datasetNUM)]
  RS = []      #認識音節列
  CS = []      #正解音節列
  RS_p = []      #認識音素列
  CS_p = []      #正解音素列
  
  TSEG = 0     #正解の単語分割数（分割文字+１）
  ESEG = 0     #推定の単語分割数（分割文字+１）
  
  j = 0
  for line in open(datasetfolder + datasetname + correct_data, 'r'):
        itemList = line[:-1].split(',')
        #itemList = itemList.replace(",", "")
        #W_index = W_index + [itemList]
        CS = CS + [[]]
        for i in xrange(len(itemList)):
          if itemList[i] != "":
            CS[j].append(itemList[i])
        j = j + 1
  
  
  LIST = []
  hatsuon = [ "" for i in xrange(len(CS)) ]
  TANGO = []
  ##単語辞書の読み込み
  for line in open("./lang_m/phonemes.htkdic", 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
      
  ##単語辞書の読み込み
  for line in open("./lang_m/web.000.htkdic", 'r'):
      itemList = line[:-1].split('	')
      LIST = LIST + [line]
      for j in xrange(len(itemList)):
          itemList[j] = itemList[j].replace("[", "")
          itemList[j] = itemList[j].replace("]", "")
      
      TANGO = TANGO + [[itemList[1],itemList[2]]]
  
  
  #print TANGO
  
  ##単語を順番に処理していく
  for c in xrange(len(CS)):   
    CS_p = CS_p + [[]]
    for jc in xrange(len(CS[c])):
          W_index_sj = unicode(CS[c][jc], encoding='shift_jis')
          hatsuon[c] = ""
          #print W_index_sj
          #if len(W_index_sj) != 1:  ##１文字は除外
          #for moji in xrange(len(W_index_sj)):
          moji = 0
          while (moji < len(W_index_sj)):
              flag_moji = 0
              #print len(W_index_sj), W_index_sj,moji,W_index_sj[moji]#,str(W_index_sj),moji,W_index_sj[moji]#,len(unicode(W_index[i], encoding='shift_jis'))
              #print moji              
              #print W_index_sj[moji]
              if (W_index_sj[moji] == u' ') or (W_index_sj[moji] == ' '):
                hatsuon[c] = hatsuon[c] + str("|")
                moji = moji + 1
                flag_moji = 1
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-2 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+"_"+W_index_sj[moji+2]) and (W_index_sj[moji+1] == "_"): 
                    #print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+W_index_sj[moji+1]):
                    #print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_index_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_index_sj) > moji) and (flag_moji == 0):
                  #else:
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                      #print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
          #print hatsuon[c]
          #CS_p = CS_p + [[]]
          CS_p[c] = CS_p[c] + [hatsuon[c]]
          #else:
          #  print W_index[c] + " (one name)"
        
  #一文字ではない表記の音素を別の文字へ置き換える	
  for c in xrange(len(CS_p)): 
    for j in xrange(len(CS_p[c])):
      CS_p[c][j] = CS_p[c][j].replace("my", "1")
      CS_p[c][j] = CS_p[c][j].replace("ky", "2")
      CS_p[c][j] = CS_p[c][j].replace("dy", "3")
      CS_p[c][j] = CS_p[c][j].replace("by", "4")
      CS_p[c][j] = CS_p[c][j].replace("gy", "5")
      CS_p[c][j] = CS_p[c][j].replace("ny", "6")
      CS_p[c][j] = CS_p[c][j].replace("hy", "7")
      CS_p[c][j] = CS_p[c][j].replace("ry", "8")
      CS_p[c][j] = CS_p[c][j].replace("py", "9")
      CS_p[c][j] = CS_p[c][j].replace("ts", "0")
      CS_p[c][j] = CS_p[c][j].replace("ch", "c")
      CS_p[c][j] = CS_p[c][j].replace("sh", "x")
      CS_p[c][j] = CS_p[c][j].replace(" ", "")
    #print "CS[%d]:%s" % (c,CS_p[c])
  
  N = 0
  
  if (os.path.exists("./data/" + filename + "/out_gmm_" + str(iteration+1) + "/samp_best.100") == True):
    #テキストファイルを読み込み
    for line in open("./data/" + filename + "/out_gmm_" + str(iteration+1) + "/samp_best.100" , 'r'):
        #######
        itemList = line[:-1]#.split(',')
        #itemList = itemList.replace(",", "")
        #<s>,<sp>,</s>を除く処理：単語に区切られていた場合
        """
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
        """
        #print itemList
        #<s>,<sp>,</s>を除く処理：単語中に存在している場合
        for j in xrange(len(itemList)):
          #itemList = itemList.replace("<s><s> ", "")
          itemList = itemList.replace("<sp>", "")
          itemList = itemList.replace("  ", " ")
          itemList = itemList.replace("  ", " ")
          itemList = itemList.replace("<s> ", "")
          #itemList = itemList.replace("<sp> ", "")
          #itemList = itemList.replace(" <sp>", "")
          itemList = itemList.replace("<s><s>", "")
          itemList = itemList.replace("<s>", "")
          itemList = itemList.replace(" </s>", "")
          itemList = itemList.replace("</s>", "")
          for b in xrange(5):
              itemList = itemList.replace("  ", " ")
        """
        for b in xrange(5):
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        """
        #W_index = W_index + [itemList]
        #for i in xrange(len(itemList)):
        if itemList != "":
          RS.append(itemList)
          #temp = temp + 1
          N = N + 1  #count
  else:
    for line in open("./data/" + filename + "/out_gmm_" + str(iteration+1) + "/" + str(sample) + "_samp.100" , 'r'):
    #######
        itemList = line[:-1]#.split(',')
        #itemList = itemList.replace(",", "")
        #<s>,<sp>,</s>を除く処理：単語に区切られていた場合
        """
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
        """
        #print itemList
        #<s>,<sp>,</s>を除く処理：単語中に存在している場合
        for j in xrange(len(itemList)):
          #itemList = itemList.replace("<s><s> ", "")
          itemList = itemList.replace("<sp>", "")
          itemList = itemList.replace("  ", " ")
          itemList = itemList.replace("  ", " ")
          itemList = itemList.replace("<s> ", "")
          #itemList = itemList.replace("<sp> ", "")
          #itemList = itemList.replace(" <sp>", "")
          itemList = itemList.replace("<s><s>", "")
          itemList = itemList.replace("<s>", "")
          itemList = itemList.replace(" </s>", "")
          itemList = itemList.replace("</s>", "")
          for b in xrange(5):
              itemList = itemList.replace("  ", " ")
        """
        for b in xrange(5):
          if ("" in itemList):
            itemList.pop(itemList.index(""))
        """
        #W_index = W_index + [itemList]
        #for i in xrange(len(itemList)):
        if itemList != "":
          RS.append(itemList)
          #temp = temp + 1
          N = N + 1  #count
        
  
  hatsuon = [ "" for i in xrange(len(RS)) ]
  ##単語を順番に処理していく
  for c in xrange(len(RS)):   
          W_index_sj = unicode(RS[c], encoding='shift_jis')
          #if len(W_index_sj) != 1:  ##１文字は除外
          #for moji in xrange(len(W_index_sj)):
          moji = 0
          while (moji < len(W_index_sj)):
              flag_moji = 0
              #print len(W_index_sj),str(W_index_sj),moji,W_index_sj[moji]#,len(unicode(W_index[i], encoding='shift_jis'))
              if (W_index_sj[moji] == u' ') or (W_index_sj[moji] == ' '):
                hatsuon[c] = hatsuon[c] + str("|")
                moji = moji + 1
                flag_moji = 1
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-2 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+"_"+W_index_sj[moji+2]) and (W_index_sj[moji+1] == "_"): 
                    #print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+W_index_sj[moji+1]):
                    #print moji,j,TANGO[j][0]
                    hatsuon[c] = hatsuon[c] + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_index_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_index_sj) > moji) and (flag_moji == 0):
                  #else:
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                      #print moji,j,TANGO[j][0]
                      hatsuon[c] = hatsuon[c] + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
          #print hatsuon[c]
          
          RS_p = RS_p + [hatsuon[c]]
          #else:
          #  print W_index[c] + " (one name)"
  
  #一文字ではない表記の音素を別の文字へ置き換える	
  for c in xrange(len(RS)): 
    RS_p[c] = RS_p[c].replace("my", "1")
    RS_p[c] = RS_p[c].replace("ky", "2")
    RS_p[c] = RS_p[c].replace("dy", "3")
    RS_p[c] = RS_p[c].replace("by", "4")
    RS_p[c] = RS_p[c].replace("gy", "5")
    RS_p[c] = RS_p[c].replace("ny", "6")
    RS_p[c] = RS_p[c].replace("hy", "7")
    RS_p[c] = RS_p[c].replace("ry", "8")
    RS_p[c] = RS_p[c].replace("py", "9")
    RS_p[c] = RS_p[c].replace("ts", "0")
    RS_p[c] = RS_p[c].replace("ch", "c")
    RS_p[c] = RS_p[c].replace("sh", "x")
    RS_p[c] = RS_p[c].replace(" ", "")
    #print RS_p[c]
    #print "RS[%d]:%s" % (c,RS_p[c])
  
  
  #print N
  #print CS
  #print RS
  
  ###単語分割結果の音素表記をファイル保存
  
  
  #LD = [0 for t in range(N)]
  LD = 0
  
  ##文章全体に対してSERを算出
  #LD_sum = fsum(LD)
  #CSN = 0
  #for t in xrange(N):
  #  CSN = CSN + len(CS_p[t])        #正解音節列の音節数
  #CSN = CSN/2    #文字コードの関係（2byte1文字のbyte換算のため）
  #SER = float(LD_sum)/CSN  #音節誤り率
  
  
  #print "LD:",LD
  #print "LD_sum:",LD_sum
  #print "CSN:",CSN
  #print "PER:"
  #print SER
  
  #一文一文のSERを出してそれを平均
  SER2 = 0.0
  
  for t in xrange(N):
    SER_temp = 1.0
    for j in xrange(len(CS_p[t])):
      ###編集距離
      LD = int( levenshtein_distance(unicode(RS_p[t], sys.stdin.encoding), unicode(CS_p[t][j], sys.stdin.encoding)) )
      print RS_p[t],CS_p[t][j],len(CS_p[t][j]),LD,float(LD)/(len(CS_p[t][j]))
      if ( (1.0 - SER_temp) <= (1.0 - (float(LD)/len(CS_p[t][j]))) ):
        SER_temp = float(LD)/(len(CS_p[t][j]))
        
        TSEGtemp = CS_p[t][j].count("|") + 1
        ESEGtemp = RS_p[t].count("|") + 1
        #print SER_temp
      
    SER2 = SER2 + SER_temp #float(LD)/(len(CS_p[t]))
    #print SER2
    
    TSEG += TSEGtemp
    ESEG += ESEGtemp
  
  SER2 = float(SER2) / N
  print "PAR_s:"
  #print SER2
  PAR_S = 1.0 - SER2
  print PAR_S
  
  return PAR_S,ESEG
###↑###文章まるごとのPAR############################################


###↓###場所の名前のPAR############################################
def Name_of_Place(THETA):
    datasetNUM = 0
    datasetname = datasets[int(datasetNUM)]
    CN = []    #正解の音素列を格納する配列
    #correct_nameを読み込む
    for line in open(datasetfolder + datasetname + correct_name, 'r'):
        itemList = line[:-1].split(',')
        
        #W_index = W_index + [itemList]
        for i in xrange(len(itemList)):
          itemList[i] = itemList[i].replace(",", "")
          itemList[i] = itemList[i].replace("\r", "")
        if (itemList[0] != ""): # and (itemList[0] == itemList[1]):
            CN.append(itemList)
    
    
    #教示位置データを読み込み平均値を算出（xx,xy）
    XX = []
    count = 0
    
    ItC = []
    #それぞれの場所の中央座標を出す（10カ所）
    s = 0
    #正解データを読み込みIT
    for line in open(datasetfolder + datasetname + correct_It, 'r'):
      itemList = line[:].split(',')
      for i in xrange(len(itemList)):
        if (itemList[i] != ''):
          ItC = ItC + [int(itemList[i])]
        s += 1
        
    ic = collections.Counter(ItC)
    icitems = ic.items()  # [(it番号,カウント数),(),...]
    
    
    if (data_name != 'test000'):
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
          
          #print len(Xtemp),Xtemp,ic[icitems[j][0]]
          XX = XX + [sum(np.array(Xtemp))/float(ic[icitems[j][0]])]
        
        #for i in xrange(int(count/5)):
        #  XX = XX + [[0.0,0.0]]
        #  for j in xrange(5):
        #    XX[i][0] = XX[i][0] + Xt[int(i*5 + j)][0]
        #    XX[i][1] = XX[i][1] + Xt[int(i*5 + j)][1]
        #  XX[i][0] = XX[i][0] / 5.0
        #  XX[i][1] = XX[i][1] / 5.0
    """
    else:
        ###SIGVerse###
        HTW = []
        for line2 in open('./../sample/' + data_name +  '_HTW.csv', 'r'):
          itemList2 = line2[:-1].split(',')
          HTW = HTW + [itemList2[0]]
        
        #i = 0
        Xt_temp = []
        Xt = [(0.0,0.0) for n in xrange(len(HTW)) ]
        #TN = []
        for line3 in open('./../sample/' + data_name +  '_X_To.csv', 'r'):
          itemList3 = line3[:-1].split(',')
          Xt_temp = Xt_temp + [(float(itemList3[2]) + 500, float(itemList3[1]) + 250)]
          #TN = TN + [i]
          #print TN
          #i = i + 1
        
        #げんかん(ge)0-9、てえぶるのあたり黒(teb)10-19、白(tew)20-29、ほんだな(hd)30-39、
        #そふぁあまえ(sf)40-49、きっちん(kt)50-59、だいどころ(dd)60-69、ごみばこ(go)70-79、てれびまえ(tv)80-89
        ge = 0
        teb = 10
        tew = 20
        hd = 30
        sf = 40
        kt = 50
        dd = 60
        go = 70
        tv = 80
        
        for i in xrange(len(HTW)):
          htw = HTW[i]
          if (htw == "ge"):
            Xt[ge] = Xt_temp[i]
            ge = ge + 1
           
          if (htw == "teb"):
            Xt[teb] = Xt_temp[i]
            teb = teb + 1
            
          if (htw == "tew"):
            Xt[tew] = Xt_temp[i]
            tew = tew + 1
            
          if (htw == "hd"):
            Xt[hd] = Xt_temp[i]
            hd = hd + 1
            
          if (htw == "sf"):
            Xt[sf] = Xt_temp[i]
            sf = sf + 1
            
          if (htw == "kt"):
            Xt[kt] = Xt_temp[i]
            kt = kt + 1
            
          if (htw == "dd"):
            Xt[dd] = Xt_temp[i]
            dd = dd + 1
            
          if (htw == "tv"):
            Xt[tv] = Xt_temp[i]
            tv = tv + 1
            
          if (htw == "go"):
            Xt[go] = Xt_temp[i]
            go = go + 1
            
        #print kyouji_count
        #EndStep = len(Xt)-1
        for i in xrange(int(kyouji_count/10)):
          XX = XX + [[0.0,0.0]]
          for j in xrange(10):
              #print XX[i][0]
              #print Xt[int(i*10 + j)][0]
              XX[i][0] = XX[i][0] + Xt[int(i*10 + j)][0]
              XX[i][1] = XX[i][1] + Xt[int(i*10 + j)][1]
          XX[i][0] = XX[i][0] / 10.0
          XX[i][1] = XX[i][1] / 10.0

         #以下はSIGVerse実験用なため変更が必要
         #XX = [750,180,180,780,300,700,450,480,100]    #データの平均値
         #XY = [380,120,120,120,300,220,100,400,450]    #データの平均値
         #CC = ["sofaamae","daidokoro","kiqchiN","terebimae","teeburunoatari","teeburunoatari","gomibako","hoNdana","geNkaN"]
        ###SIGVerse###
    """  
    print XX
    
    #THETA = [W,W_index,Myu,S,pi,phi_l]
    W = THETA[0]
    W_index = THETA[1]
    Myu = THETA[2]
    S = THETA[3]
    pi = THETA[4]
    phi_l = THETA[5]
    
    
    POX_PAR = []
    
    ##教示位置データごとに
    for xdata in xrange(len(XX)):
    
      ###提案手法による尤度計算####################
      #Ot_index = 0
      pox = [0.0 for i in xrange(len(W_index))]
      for otb in xrange(len(W_index)):
        #Otb_B = [0 for j in xrange(len(W_index))]
        #Otb_B[Ot_index] = 1
        temp = [0.0 for c in range(L)]
        #print Otb_B
        for c in xrange(L) :
            ##場所の名前、多項分布の計算
            #W_temp = Multinomial(W[c])
            #temp[c] = W_temp.pmf(Otb_B)
            temp[c] = W[c][otb]
            ##場所概念の多項分布、piの計算
            temp[c] = temp[c] * pi[c]

            ##itでサメーション
            it_sum = 0.0
            for it in xrange(K):
                if (S[it][0][0] < pow(10,-10)) or (S[it][1][1] < pow(10,-10)) :    ##共分散の値が0だとゼロワリになるので回避
                    if int(XX[xdata][0]) == int(Myu[it][0]) and int(XX[xdata][1]) == int(Myu[it][1]) :  ##他の方法の方が良いかも
                        g2 = 1.0
                        print "gauss 1"
                    else : 
                        g2 = 0.0
                        print "gauss 0"
                else : 
                    g2 = gaussian2d(XX[xdata][0],XX[xdata][1],Myu[it][0],Myu[it][1],S[it])  #2次元ガウス分布を計算
                it_sum = it_sum + g2 * phi_l[c][it]
                    
            temp[c] = temp[c] * it_sum
                    
        pox[otb] = sum(temp)
        
        #print Ot_index,pox[Ot_index]
        #Ot_index = Ot_index + 1
      #POX = POX + [pox.index(max(pox))]
      
      #print pox.index(max(pox))
      #print W_index_p[pox.index(max(pox))]
      
      
      #RS = [W_index[pox.index(max(pox))].replace(" ", "")]#[]      #認識音節列
      
      LIST = []
      hatsuon = "" #[ "" for i in xrange(len(W_index)) ]
      TANGO = []
      ##単語辞書の読み込み
      for line in open("./lang_m/web.000.htkdic", 'r'):
        itemList = line[:-1].split('	')
        LIST = LIST + [line]
        for j in xrange(len(itemList)):
            itemList[j] = itemList[j].replace("[", "")
            itemList[j] = itemList[j].replace("]", "")
        
        TANGO = TANGO + [[itemList[1],itemList[2]]]
        
      
      #for c in xrange(len(W_index)):    # i_best = len(W_index)
      #W_index_sj = unicode(MI_best[c][i], encoding='shift_jis')
      W_index_sj = unicode(W_index[pox.index(max(pox))], encoding='shift_jis')
      #if len(W_index_sj) != 1:  ##１文字は除外
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
                    #print moji,j,TANGO[j][0]
                    hatsuon = hatsuon + TANGO[j][1]
                    moji = moji + 3
                    flag_moji = 1
                    
              for j in xrange(len(TANGO)):
                if (len(W_index_sj)-1 > moji) and (flag_moji == 0): 
                  #print TANGO[j],j
                  #print moji
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]+W_index_sj[moji+1]):
                    #print moji,j,TANGO[j][0]
                    hatsuon = hatsuon + TANGO[j][1]
                    moji = moji + 2
                    flag_moji = 1
                    
                #print len(W_index_sj),moji
              for j in xrange(len(TANGO)):
                if (len(W_index_sj) > moji) and (flag_moji == 0):
                  if (unicode(TANGO[j][0], encoding='shift_jis') == W_index_sj[moji]):
                      #print moji,j,TANGO[j][0]
                      hatsuon = hatsuon + TANGO[j][1]
                      moji = moji + 1
                      flag_moji = 1
      #print hatsuon #[c]
      RS = [hatsuon]
      #else:
      #print W_index[c] + " (one name)"
      
      
      
      #RS = [W_index_p[pox.index(max(pox))].replace(" ", "")]#[]      #認識音節列
      PAR = 0.0
      for ndata in xrange(len(CN[xdata])):    
        CS = [CN[xdata][ndata]]  #[]      #正解音節列
        LD = 0
        #print CS,RS
        #print CS
        
        #一文字ではない表記の音素を別の文字へ置き換える	
        CS[0] = CS[0].replace("my", "1")
        CS[0] = CS[0].replace("ky", "2")
        CS[0] = CS[0].replace("dy", "3")
        CS[0] = CS[0].replace("by", "4")
        CS[0] = CS[0].replace("gy", "5")
        CS[0] = CS[0].replace("ny", "6")
        CS[0] = CS[0].replace("hy", "7")
        CS[0] = CS[0].replace("ry", "8")
        CS[0] = CS[0].replace("py", "9")
        CS[0] = CS[0].replace("ts", "0")
        CS[0] = CS[0].replace("ch", "c")
        CS[0] = CS[0].replace("sh", "x")
        CS[0] = CS[0].replace(" ", "")
        
        RS[0] = RS[0].replace("my", "1")
        RS[0] = RS[0].replace("ky", "2")
        RS[0] = RS[0].replace("dy", "3")
        RS[0] = RS[0].replace("by", "4")
        RS[0] = RS[0].replace("gy", "5")
        RS[0] = RS[0].replace("ny", "6")
        RS[0] = RS[0].replace("hy", "7")
        RS[0] = RS[0].replace("ry", "8")
        RS[0] = RS[0].replace("py", "9")
        RS[0] = RS[0].replace("ts", "0")
        RS[0] = RS[0].replace("ch", "c")
        RS[0] = RS[0].replace("sh", "x")
        RS[0] = RS[0].replace(" ", "")
        
        print CS,RS
        ###編集距離
        LD = int( levenshtein_distance(unicode(RS[0], sys.stdin.encoding), unicode(CS[0], sys.stdin.encoding)) )
        #print LD[t]
      
        
        CSN = len(CS[0])        #正解音節列の音節数
        #CSN = CSN/2    #文字コードの関係（2byte1文字のbyte換算のため）
        PER = float(LD)/CSN  #音節誤り率
        if (PAR <= (1.0 - PER)):        
            PAR = 1.0 - PER
      
        print "LD:",LD
        #print "LD_sum:",LD_sum
        print "CSN:",CSN
      print "PAR:"
      print PAR
      POX_PAR = POX_PAR + [PAR]
      
    #PARの平均値を算出
    PAR_mean = sum(POX_PAR) / len(XX)
    print PAR_mean
    
    
    return PAR_mean
###↑###場所の名前のPAR############################################


def Evaluation(filename):
  
  #相互推定の学習結果データを読み込む
  MI_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  ARIc_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]  #
  ARIi_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]  #
  NMIc_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]  #NMI
  NMIi_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]  #NMI
  PARs_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  PARw_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  MAX_Samp = [0 for j in xrange(ITERATION)]
  
  L_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  K_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  SEG_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  #SEG_List = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
  
  #イテレーションごとに選ばれた学習結果の評価値をすべて保存するファイル
  fp = open('./data/' + filename + '/' + filename + '_A_sougo_Evaluation.csv', 'w')  
  
  #相互推定のイテレーションと単語分割結果の候補のすべてのパターンの評価値を保存
  fp_ARIc = open('./data/' + filename + '/' + filename + '_A_sougo_ARIc.csv', 'w')  
  fp_ARIi = open('./data/' + filename + '/' + filename + '_A_sougo_ARIi.csv', 'w')  
  fp_NMIc = open('./data/' + filename + '/' + filename + '_A_sougo_NMIc.csv', 'w')  
  fp_NMIi = open('./data/' + filename + '/' + filename + '_A_sougo_NMIi.csv', 'w')  
  fp_PARs = open('./data/' + filename + '/' + filename + '_A_sougo_PARs.csv', 'w')  
  fp_PARw = open('./data/' + filename + '/' + filename + '_A_sougo_PARw.csv', 'w')  
  fp_L = open('./data/' + filename + '/' + filename + '_A_sougo_L.csv', 'w')  
  fp_K = open('./data/' + filename + '/' + filename + '_A_sougo_K.csv', 'w')  
  fp_SEG = open('./data/' + filename + '/' + filename + '_A_sougo_SEG.csv', 'w')  
  fp_MI = open('./data/' + filename + '/' + filename + '_A_sougo_MI.csv', 'w')  
  
  #fp.write('MI,NMIc,NMIi,PARs,PARw\n')
  fp.write('MI,ARIc,ARIi,NMIc,NMIi,PARs,PARw,L,K,ESEG\n')
  
  #相互推定のイテレーションごとに
  for iteration in xrange(ITERATION):
    
    #./data/filename/filename_sougo_MI_iteration.csvを読み込み
    for line in open('./data/' + filename + '/' + filename + '_sougo_MI_' + str(iteration+1) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        if (int(itemList[0]) < sample_num):
          MI_List[iteration][int(itemList[0])] = float(itemList[1])
        fp_MI.write(str(float(itemList[1])))
        fp_MI.write(',')
    MAX_Samp[iteration] = MI_List[iteration].index(max(MI_List[iteration]))  #相互情報量が最大のサンプル番号
    fp_MI.write(',' + str(MAX_Samp[iteration]))
    fp_MI.write('\n')
    
    #単語分割結果の候補ごとに
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
      It = []
      
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
        
      ##itの読み込み
      for line in open('./data/' + filename +'/' + filename + '_It_'+str(iteration+1) + "_" + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            It = It + [int(itemList[i])]
        
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
      
      print "ARI"
      ARIc_List[iteration][sample],ARIi_List[iteration][sample] = ARI(Ct,It)
      
      print "NMI"
      NMIc_List[iteration][sample],NMIi_List[iteration][sample] = NMI(Ct,It)
      if (1):
        print "PAR_S"
        PARs_List[iteration][sample],SEG_List[iteration][sample] = PAR_sentence(iteration,sample)
        
        THETA = [W,W_index,Myu,S,pi,phi_l]
        #NOP = []
        print "PAR_W"
        PARw_List[iteration][sample] = Name_of_Place(THETA)
        
        cc = collections.Counter(Ct) #｛Ct番号：カウント数｝
        L_List[iteration][sample] = len(cc)   #場所概念の数
        
        ic = collections.Counter(It) #｛it番号：カウント数｝
        K_List[iteration][sample] = len(ic)   #位置分布の数
        
      
      print "OK!"
      fp_ARIc.write(str(ARIc_List[iteration][sample]))
      fp_ARIc.write(',')
      fp_ARIi.write(str(ARIi_List[iteration][sample]))
      fp_ARIi.write(',')
      fp_NMIc.write(str(NMIc_List[iteration][sample]))
      fp_NMIc.write(',')
      fp_NMIi.write(str(NMIi_List[iteration][sample]))
      fp_NMIi.write(',')
      fp_PARs.write(str(PARs_List[iteration][sample]))
      fp_PARs.write(',')
      fp_PARw.write(str(PARw_List[iteration][sample]))
      fp_PARw.write(',')
      fp_L.write(str(L_List[iteration][sample]))
      fp_L.write(',')
      fp_K.write(str(K_List[iteration][sample]))
      fp_K.write(',')
      fp_SEG.write(str(SEG_List[iteration][sample]))
      fp_SEG.write(',')
    
    fp_ARIc.write(',')
    smean = sum(ARIc_List[iteration])/sample_num
    fp_ARIc.write(str(smean))
    fp_ARIc.write('\n')

    fp_ARIi.write(',')
    smean = sum(ARIi_List[iteration])/sample_num
    fp_ARIi.write(str(smean))
    fp_ARIi.write('\n')

    fp_NMIc.write(',')
    smean = sum(NMIc_List[iteration])/sample_num
    fp_NMIc.write(str(smean))
    fp_NMIc.write('\n')

    fp_NMIi.write(',')
    smean = sum(NMIi_List[iteration])/sample_num
    fp_NMIi.write(str(smean))
    fp_NMIi.write('\n')

    fp_PARs.write(',')
    smean = sum(PARs_List[iteration])/sample_num
    fp_PARs.write(str(smean))
    fp_PARs.write('\n')
    
    fp_PARw.write(',')
    smean = sum(PARw_List[iteration])/sample_num
    fp_PARw.write(str(smean))
    fp_PARw.write('\n')
    
    fp_L.write(',')
    smean = sum(L_List[iteration])/sample_num
    fp_L.write(str(smean))
    fp_L.write('\n')

    fp_K.write(',')
    smean = sum(K_List[iteration])/sample_num
    fp_K.write(str(smean))
    fp_K.write('\n')
    
    fp_SEG.write(',')
    smean = sum(SEG_List[iteration])/sample_num
    fp_SEG.write(str(smean))
    fp_SEG.write('\n')
    
    #MI,ARI,PARs,PARw,
    
    fp.write( str(MI_List[iteration][MAX_Samp[iteration]])+','+ str(ARIc_List[iteration][MAX_Samp[iteration]])+','+ str(ARIi_List[iteration][MAX_Samp[iteration]])+','+ str(NMIc_List[iteration][MAX_Samp[iteration]])+','+ str(NMIi_List[iteration][MAX_Samp[iteration]])+','+ str(PARs_List[iteration][MAX_Samp[iteration]])+','+str(PARw_List[iteration][MAX_Samp[iteration]])+','+ str(L_List[iteration][MAX_Samp[iteration]])+','+ str(K_List[iteration][MAX_Samp[iteration]])+','+ str(SEG_List[iteration][MAX_Samp[iteration]]) )
    fp.write('\n')
    
  print "close."
  
  fp.close()
  fp_ARIc.close()
  fp_ARIi.close()
  fp_NMIc.close()
  fp_NMIi.close()
  fp_PARs.close()
  fp_PARw.close()
  fp_L.close()
  fp_K.close()
  fp_SEG.close()
  fp_MI.close()
  
  
if __name__ == '__main__':
    #出力ファイル名を要求
    filename = raw_input("trialname?(folder) >") #"tamd2_sig_mswp_01" 
    ITERATION = 10
    sample_num = 6
    
    """
    if ("sig" in filename):
      data_name = 'test000'
      L = 50
      K = 50
      kyouji_count = 90
      correct_Ct = 'Ct_correct.csv'  #データごとの正解のCt番号
      correct_data = 'TAMD1_human.txt'  #データごとの正解の文章（単語列、区切り文字つき）(./data/)
      correct_name = 'name_correct.csv'  #データごとの正解の場所の名前（音素列）
      
    else:
      data_name = 'datah.csv'
      L = 100
      K = 100
      kyouji_count = 100
      correct_Ct = 'Ct_correct_turtle.csv'  #データごとの正解のCt番号
      correct_data = 'TAMD1_human_turtle.csv'  #データごとの正解の文章（単語列、区切り文字つき）(./data/)
      correct_name = 'name_correct_turtle.csv'  #データごとの正解の場所の名前（音素列）
      
    if ("p" in filename):
      lang_init = 'phonemes.htkdic'
    else:
      lang_init = 'web.000.htkdic'
    if ("ms" in filename):
      ITERATION = 10
      sample_num = 6
    if ("m0" in filename):
      ITERATION = 10
      sample_num = 1
    if ("nakamura" in filename):
      ITERATION = 10
      sample_num = 1
    if (("000t" in filename) or ("000b" in filename)):
      ITERATION = 1
      sample_num = 1
    """
    #for i in range(1,11):
    #  Evaluation(filename + str(i).zfill(3))
    Evaluation(filename)
