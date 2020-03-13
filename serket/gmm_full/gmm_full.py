# encoding: utf-8
import numpy as np
import random
import math
import os
import pickle
from sklearn.cluster import KMeans

var = 2.0 #0.1 #共分散行列の行列成分のハイパーパラメータのスケール係数

##Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
#alpha0 = 20.0        #Hyperparameter of CRP in multinomial distribution for index of spatial concept
gamma0 = 0.1         #Hyperparameter of CRP in multinomial distribution for index of position distribution
#beta0 = 0.1          #Hyperparameter in multinomial distribution P(W) for place names 
#chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
k0 = 1e-3            #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*var  #Hyperparameter in Inverse Wishart distribution P(Σ) (prior covariance matrix) 
n0 = 3.0             #Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))

##Initial (hyper)-parameters
#num_iter = 100           #The number of iterations of Gibbs sampling for spatial concept learning
#L = 10                  #The number of spatial concepts #50 #100
#K = 10                  #The number of position distributions #50 #100
#alpha = 1.0                  #Hyperparameter of multinomial distributions for index of position distirubitons phi #1.5 #0.1
#gamma = 1.0                  #Hyperparameter of multinomial distributions for index of spatial concepts pi #8.0 #20.0
#beta0 = 0.1                  #Hyperparameter of multinomial distributions for words (place names) W #0.5 #0.2
#kappa0 = 1e-3                #For μ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (scale: kappa0>0)
#m0 = np.array([[0.0],[0.0]]) #For μ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (mean prior)
#V0 = np.eye(2)*2             #For Σ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (covariance matrix prior)
#nu0 = 3.0 #3.0               #For Σ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (degree of freedom: dimension+1)
sig_init =  1.0 

# 計算された共分散行列のパラメータが正定値性を満たすか簡易チェック
def Check_VN(VN):
  if (VN[0][0] <= 0 or VN[1][1] <= 0 ):
    print "ERROR!!!! Posterior parameter VN is negative."
    print VN
    VN = V0
  return VN

#"""
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
#"""

#http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section5_2-Dirichlet-Processes.ipynb
def stick_breaking(alpha, k):
    betas = np.random.beta(1, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    return p/p.sum()


###########################################################

# 確率を計算するためのクラス
class GaussWishart():
    def __init__(self,dim, mean, var):
        # 事前分布のパラメータ
        self.__dim = dim
        self.__r0 = 1e-3 #1
        self.__nu0 = dim + 2 #1 #2
        self.__m0 = mean.reshape((dim,1))
        self.__S0 = np.eye(dim, dim ) * var

        self.__X = np.zeros( (dim,1) )
        self.__C = np.zeros( (dim, dim) )
        self.__r = self.__r0
        self.__nu = self.__nu0
        self.__N = 0

        self.__update_param()

    def __update_param(self):
        self.__m = (self.__X + self.__r0 * self.__m0)/(self.__r0 + self.__N )
        self.__S = - self.__r * self.__m * self.__m.T + self.__C + self.__S0 + self.__r0 * self.__m0 * self.__m0.T;

    def add_data(self, x):
        x = x.reshape((self.__dim,1))  # 縦ベクトルにする
        self.__X += x
        self.__C += x.dot( x.T )
        self.__r += 1
        self.__nu += 1
        self.__N += 1
        self.__update_param()

    def delete_data(self, x):
        x = x.reshape((self.__dim,1))  # 縦ベクトルにする
        self.__X -= x
        self.__C -= x.dot( x.T )
        self.__r -= 1
        self.__nu -= 1
        self.__N -= 1
        self.__update_param()

    def calc_loglik(self, x):
        def _calc_loglik(self):
            p = - self.__N * self.__dim * 0.5 * math.log( math.pi )
            p+= - self.__dim * 0.5 * math.log( self.__r )
            p+= - self.__nu * 0.5 * math.log( np.linalg.det( self.__S ) );

            for d in range(1,self.__dim+1):
                p += math.lgamma( 0.5*(self.__nu+1-d) )

            return p

        # log(P(X))
        p1 = _calc_loglik( self )

        # log(P(x,X))
        self.add_data(x)
        p2 = _calc_loglik( self )
        self.delete_data(x)

        # log(P(x|X)) = log(P(x,X)) - log(P(X))
        return p2 - p1

    def get_loglik(self):
         p = - self.__N * self.__dim * 0.5 * math.log( math.pi )
         p+= - self.__dim * 0.5 * math.log( self.__r )
         p+= - self.__nu * 0.5 * math.log( np.linalg.det( self.__S ) );

         for d in range(1,self.__dim+1):
             p += math.lgamma( 0.5*(self.__nu+1-d) )

         return p
    
    def get_mean(self):
        return self.__m

    # 事後逆ウィシャート分布から共分散行列の期待値を計算
    def get_variance(self):
        sig = self.__S / (self.__nu - self.__dim - 1.0)
        return sig

    def get_num_data(self):
        return self.__N

    def get_param(self):
        return [self.__X, self.__C, self.__r, self.__nu, self.__N, self.__m0]
    
    def load_params(self, params):
        self.__X = params[0]
        self.__C = params[1]
        self.__r = params[2]
        self.__nu = params[3]
        self.__N = params[4]
        self.__m0 = params[5]

        self.__update_param()


def calc_probability( dist, d ):
    return dist.get_num_data() * math.exp( dist.calc_loglik( d ) )


def sample_class( d, distributions, i, bias_dz ):
    K = len(distributions)
    P =  np.zeros(K) #[ 0.0 ] * K
    
    # 事後確率分布を計算
    for k in range(K):
        P[k] = calc_probability( distributions[k], d ) * bias_dz[i][k]

    P = P / np.sum(P)  #Normalization
    sample_list = np.random.multinomial(1,P)
    sample_index = np.where(sample_list == 1)[0][0] 
    return sample_index


    # 累積確率を計算
    #P[0] = calc_probability( distributions[0], d ) * bias_dz[i][0]
    #for k in range(1,K):
    #    P[k] = P[k-1] + calc_probability( distributions[k], d ) * bias_dz[i][k]

    # サンプリング
    #rnd = P[K-1] * random.random()
    #for k in range(K):
    #    if P[k] >= rnd:
    #        return k


def calc_acc( results, correct ):
    K = np.max(results)+1     # カテゴリ数
    N = len(results)          # データ数
    max_acc = 0               # 精度の最大値
    changed = True            # 変化したかどうか

    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                tmp_result = np.zeros( N )

                # iとjを入れ替える
                for n in range(N):
                    if results[n]==i: tmp_result[n]=j
                    elif results[n]==j: tmp_result[n]=i
                    else: tmp_result[n] = results[n]

                # 精度を計算
                acc = (tmp_result==correct).sum()/float(N)

                # 精度が高くなって入れば保存
                if acc > max_acc:
                    max_acc = acc
                    results = tmp_result
                    changed = True

    return max_acc, results

# モデルの保存
def save_model( Pdz, mu, sig, classes, save_dir, categories, distributions, liks, load_dir ):
    if not os.path.exists( save_dir ):
        os.makedirs( save_dir )
    
    if load_dir is None:
        # モデルパラメータの保存
        K = len(Pdz[0])
        params = []
        for k in range(K):
            params.append(distributions[k].get_param())
        with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
            pickle.dump( params, f )
        # muと尤度の保存
        np.savetxt( os.path.join( save_dir, "mu.txt" ), mu )
        np.savetxt( os.path.join( save_dir, "liklihood.txt" ), liks, fmt="%f" )
        # sigの保存: add by Akira
        np.savetxt( os.path.join( save_dir, "sig.txt" ), sig )

    # 確率の保存
    np.savetxt( os.path.join( save_dir, "Pdz.txt" ), Pdz, fmt="%f" )
    
    # 分類結果・精度の計算と保存
    if categories is not None:
        acc, results = calc_acc( classes, categories )
        np.savetxt( os.path.join( save_dir, "class.txt" ), results, fmt="%d" )
        np.savetxt( os.path.join( save_dir, "acc.txt" ), [acc], fmt="%f" )
        
    else:
        np.savetxt( os.path.join( save_dir, "class.txt" ), classes, fmt="%d" )

# モデルパラメータの読み込み
def load_model( load_dir, distributions ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open( model_path, "rb" ) as f:
        a = pickle.load( f )
    
    K = len(distributions)
    for k in range(K):
        distributions[k].load_params(a[k])

# gmmメイン
def train( data, K, num_itr=100, save_dir="model", bias_dz=None, categories=None, load_dir=None ):
    # データ数
    N = len(data)
    # データの次元
    dim = len(data[0])
    
    # 尤度のリスト
    liks = []

    Pdz = np.zeros((N,K))
    mu  = np.zeros((N,dim))
    sig = np.zeros( (N, 2*dim) ) #

    # 初期値の決定：データをランダムに分類 (-> k-meansで分類)
    #kmeans = KMeans(n_clusters=10, init='k-means++').fit(data)
    classes = np.random.randint( K , size=N ) #kmeans.labels_  #
    #print classes

    # ガウス-ウィシャート分布の生成
    mean = np.mean( data, axis=0 )
    distributions = [ GaussWishart(dim, mean , var) for _ in range(K) ]
    #distributions = [ GaussWishart(dim, mean[k_] , var) for k_ in range(K) ]
        
    # 学習モード時はガウス-ウィシャート分布のパラメータを計算
    if load_dir is None:    
        for i in range(N):
            c = classes[i]
            x = data[i]
            distributions[c].add_data(x)
    # 認識モード時は学習したモデルパラメータを読み込む
    else:
        load_model(load_dir, distributions)

    # 学習
    if load_dir is None:
        for it in range(num_itr):
            # メインの処理
            for i in range(N):
                d = data[i]
                k_old = classes[i]  # 現在のクラス
    
                # データをクラスから除きパラメータを更新
                distributions[k_old].delete_data( d )
                classes[i] = -1
    
                # 新たなクラスをサンプリング
                k_new = sample_class( d, distributions, i, bias_dz )
    
                # サンプリングされたクラスに更新
                classes[i] = k_new
                
                # サンプリングされたクラスのパラメータを更新
                distributions[k_new].add_data( d )
            
            # 尤度の計算
            lik = 0
            for k in range(K):
                lik += distributions[k].get_loglik()
            liks.append( lik )

        for n in range(N):
            for k in range(K):
                Pdz[n][k] = calc_probability( distributions[k], data[n] )
                if classes[n] == k:
                    mu[n] = distributions[k].get_mean().reshape((1,dim))[0]
                    sig[n] = distributions[k].get_variance().reshape((1,dim*2))[0] #
                    #print sig[n]
    
    # 認識
    if load_dir is not None:
        for n in range(N):
            for k in range(K):
                Pdz[n][k] = calc_probability( distributions[k], data[n] )
        classes = np.argmax(Pdz, axis=1)

    # 正規化
    Pdz = (Pdz.T / np.sum(Pdz,1)).T
    
    save_model( Pdz, mu, sig, classes, save_dir, categories, distributions, liks, load_dir )

    return Pdz, mu

