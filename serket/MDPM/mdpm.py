# encoding: utf8
# 2020/03/12 Akira Taniguchi 編集途中(未完成)
#from __future__ import unicode_literals
import numpy as np
import random
import pickle
import os
#from numba import jit

import scipy.stats as ss
import random
import sklearn.cluster
import sklearn.metrics
import time

# ハイパーパラメータ
__alpha = 1.0
__beta = 1.0


def calc_lda_param( docs_mdn, topics_mdn, K, dims ):
    M = len(docs_mdn)
    D = len(docs_mdn[0])

    # 各物体dにおいてトピックzが発生した回数
    n_dz = np.zeros((D,K))

    # 各トピックzにおいて特徴wが発生した回数
    n_mzw = [ np.zeros((K,dims[m])) for m in range(M)]

    # 各トピックが発生した回数
    n_mz = [ np.zeros(K) for m in range(M) ]

    # 数え上げる
    for d in range(D):
        for m in range(M):
            if dims[m]==0:
                continue
            N = len(docs_mdn[m][d])    # 物体に含まれる特徴数
            for n in range(N):
                w = docs_mdn[m][d][n]       # 物体dのn番目の特徴のインデックス
                z = topics_mdn[m][d][n]     # 特徴に割り当てれれているトピック
                n_dz[d][z] += 1
                n_mzw[m][z][w] += 1
                n_mz[m][z] += 1

    return n_dz, n_mzw, n_mz

#@jit #numbaのエラーが出るためコメントアウト:Akira
def sample_topic( d, w, n_dz, n_zw, n_z, K, V, bias_dz ):
    # 累積確率を計算
    P = (n_dz[d,:] + __alpha )*(n_zw[:,w] + __beta) / (n_z[:] + V *__beta) * bias_dz[d]
    for z in range(1,K):
        P[z] = P[z] + P[z-1]
        
    # サンプリング
    rnd = P[K-1] * random.random()
    for z in range(K):
        if P[z] >= rnd:
            return z

    return -1

# 単語を一列に並べたリスト変換
def conv_to_word_list( data ):
    V = len(data)
    doc = []
    for v in range(V):  # v:語彙のインデックス
        for n in range(data[v]): # 語彙の発生した回数文forを回す
            doc.append(v)
    return doc

# 尤度計算
def calc_liklihood( data, n_dz, n_zw, n_z, K, V ):
    lik = 0

    P_wz = (n_zw.T + __beta) / (n_z + V *__beta)
    for d in range(len(data)):
        Pz = (n_dz[d] + __alpha )/( np.sum(n_dz[d]) + K *__alpha )
        Pwz = Pz * P_wz
        Pw = np.sum( Pwz , 1 ) + 0.000001
        lik += np.sum( data[d] * np.log(Pw) )

    return lik

def calc_acc(results, correct):
    K = np.max(results)+1  # カテゴリ数
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
def save_model( save_dir, n_dz, n_mzw, n_mz, M, dims, categories, liks, load_dir ):
    if not os.path.exists( save_dir ):
        os.makedirs( save_dir )
        
    # 尤度の保存
    np.savetxt( os.path.join( save_dir, "liklihood.txt" ), liks, fmt="%f" )
    
    # 確率の計算と保存
    Pdz = n_dz + __alpha
    Pdz = (Pdz.T / Pdz.sum(1)).T
    
    np.savetxt( os.path.join( save_dir, "Pdz.txt" ), Pdz, fmt="%f" )
    
    Pmdw = []
    for m in range(M):
        Pwz = (n_mzw[m].T + __beta) / (n_mz[m] + dims[m] *__beta)
        Pdw = Pdz.dot(Pwz.T)
        Pmdw.append( Pdw )
        np.savetxt( os.path.join( save_dir, "Pmdw[{}].txt".format(m) ) , Pdw )

    if load_dir is None:
        # モデルパラメータの保存
        with open( os.path.join( save_dir, "model.pickle" ), "wb" ) as f:
            pickle.dump( [n_mzw, n_mz], f )
        # 尤度の保存
        np.savetxt( os.path.join( save_dir, "liklihood.txt" ), liks, fmt="%f" )
    
    # 分類結果・精度の計算と保存
    results = np.argmax( Pdz, -1 )
    if categories is not None:
        results = np.argmax( Pdz, -1 )
        acc, results = calc_acc( results, categories )
        np.savetxt( os.path.join( save_dir, "categories.txt" ), results, fmt="%d" )
        np.savetxt( os.path.join( save_dir, "acc.txt" ), [acc], fmt="%f" )
        
    else:
        np.savetxt( os.path.join( save_dir, "categories.txt" ), results, fmt="%d" )
        
    return Pdz, Pmdw

# モデルパラメータの読み込み
def load_model( load_dir ):
    model_path = os.path.join( load_dir, "model.pickle" )
    with open(model_path, "rb" ) as f:
        a,b = pickle.load( f )

    return a,b

# ldaメイン
def train( data, K, num_itr=100, save_dir="model", bias_dz=None, categories=None, load_dir=None ):
    
    # 尤度のリスト
    liks = []

    M = len(data)       # モダリティ数

    dims = []
    for m in range(M):
        if data[m] is not None:
            dims.append( len(data[m][0]) )
            D = len(data[m])    # 物体数
        else:
            dims.append( 0 )

    # data内の単語を一列に並べる（計算しやすくするため）
    docs_mdn = [[ None for i in range(D) ] for m in range(M)]
    topics_mdn = [[ None for i in range(D) ] for m in range(M)]
    for d in range(D):
         for m in range(M):
            if data[m] is not None:
                docs_mdn[m][d] = conv_to_word_list( data[m][d] )
                topics_mdn[m][d] = np.random.randint( 0, K, len(docs_mdn[m][d]) ) # 各単語にランダムでトピックを割り当てる

    # LDAのパラメータを計算
    n_dz, n_mzw, n_mz = calc_lda_param( docs_mdn, topics_mdn, K, dims )

    # 認識モードの時は学習したパラメータを読み込み
    if load_dir is not None:
        n_mzw, n_mz = load_model( load_dir )

    for it in range(num_itr):
        # メインの処理
        for d in range(D):
            for m in range(M):
                if data[m] is None:
                    continue

                N = len(docs_mdn[m][d]) # 物体dのモダリティmに含まれる特徴数
                for n in range(N):
                    w = docs_mdn[m][d][n]       # 特徴のインデックス
                    z = topics_mdn[m][d][n]     # 特徴に割り当てられているカテゴリ


                    # データを取り除きパラメータを更新
                    n_dz[d][z] -= 1

                    if load_dir is None:
                        n_mzw[m][z][w] -= 1
                        n_mz[m][z] -= 1

                    # サンプリング
                    z = sample_topic( d, w, n_dz, n_mzw[m], n_mz[m], K, dims[m], bias_dz )

                    # データをサンプリングされたクラスに追加してパラメータを更新
                    topics_mdn[m][d][n] = z
                    n_dz[d][z] += 1

                    if load_dir is None:
                        n_mzw[m][z][w] += 1
                        n_mz[m][z] += 1

        # 尤度計算
        if load_dir is None:
            lik = 0
            for m in range(M):
                if data[m] is not None:
                    lik += calc_liklihood( data[m], n_dz, n_mzw[m], n_mz[m], K, dims[m] )
            liks.append( lik )
        
    params = save_model( save_dir, n_dz, n_mzw, n_mz, M, dims, categories, liks, load_dir )
    
    return params


class Multimodal_DPM():
	def __init__(self, data, num_class, num_itr=100, save_dir="model", bias_dz=None, categories=None, load_dir=None ): #pose,data1,data2,ganmma,num_class):
		self.num_iter  = num_itr # 100
		self.num_class = num_class # K
        self.M = len(data)       # モダリティ数

        dims = []
        for m in range(M):
            if data[m] is not None:
                dims.append( len(data[m][0]) )
                D  = len(data[m])    # 物体数
         else:
             dims.append( 0 )
		
        #self.pose      = pose
		self.data      = data   #モダリティ数可変のため変更が必要
        #self.data1     = data1
		#self.data2     = data2

		self.ganmma    = np.ones([num_class])*ganmma
		self.pi        = self.stick_breaking(ganmma,self.num_class)
		self.alpha1    = np.ones([data1.shape[1]])*2
		self.alpha2    = np.ones([data2.shape[1]])*2
		self.phi1     = ss.dirichlet.rvs(self.alpha1,num_class)
		self.phi2     = ss.dirichlet.rvs(self.alpha2,num_class)
		self.C_t       = np.random.multinomial(1,self.pi,size=data1.shape[0])
		
		self.V         = np.eye(pose.shape[1])*0.05 #np.cov(data,rowvar=0) /(10)
		self.v0        = 15 #pose.shape[1]+1
		self.m0        = np.mean(pose, axis=0)
		self.k0        = 0.01

		self.mu        =  sklearn.cluster.KMeans(n_clusters=num_class, random_state=random.randint(1,100)).fit(pose).cluster_centers_
		self.sigma     =  [self.V for n in xrange(self.num_class)]



	def stick_breaking(self,ganmma, num_class):
		betas = np.random.beta(1, ganmma, num_class)
		remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
		p = betas * remaining_pieces
		return p/p.sum()


	def fit(self):
		for gibbs in range(self.num_iter):
			print gibbs
			self.gibbs_sampling()

	def gibbs_sampling(self):
		self.phi1   =  self.dirichlet_multinomial1(self.alpha1, self.data1,self.C_t)
		self.phi2   =  self.dirichlet_multinomial1(self.alpha2, self.data2,self.C_t)
		self.pi      =  self.dirichlet_multinomial2(self.ganmma, self.C_t)
		self.C_t     =  self.multinomial_multinomial_gaussian()
		self.sigma, self.mu = self.gaussian_inverse_wishart(self.pose) 

	def dirichlet_multinomial1(self,eta,data,z_i):
		beta = np.ones([self.num_class,data.shape[1]])
		for i in range(self.num_class):
			beta[i]  = ss.dirichlet.rvs((z_i[:,i]*data.T).sum(1)+eta)
		return beta

	def dirichlet_multinomial2(self,alpha,z_i):
		return ss.dirichlet.rvs(alpha+z_i.sum(0))


	def multinomial_multinomial_gaussian(self):
		log_phi1  = np.log(self.phi1)
		log_phi2  = np.log(self.phi2)
		z_i        = np.zeros([self.data1.shape[0],self.num_class])
		log_likely = self.data1.dot(log_phi1.T)+self.data2.dot(log_phi2.T)
		sub_pi     = self.pi * np.exp(log_likely-np.array([log_likely.max(1) for j in range(self.num_class)]).T)
		probab     = sub_pi/np.array([sub_pi.sum(1) for j in range(self.num_class)]).T

		self.new_pi = np.zeros((self.pose.shape[0],self.num_class))
		for i in xrange(self.num_class):
			self.new_pi[:,i] = self.pi[0,i] #ss.multivariate_normal.pdf(self.pose,self.mu[i],self.sigma[i])*self.pi[0,i] 

		self.new_pi = self.new_pi * probab
		for i in range(self.data1.shape[0]):
			if self.new_pi.sum(1)[i] != 0:
				self.pi_sub = (self.new_pi[i].T/self.new_pi.sum(1)[i]).T
				z_i[i]      = np.random.multinomial(1,self.pi_sub,size=1)
		return z_i

    """
	def gaussian_inverse_wishart(self,pose):
		hist           = self.C_t.sum(0)
		self.mean      = np.zeros((self.num_class,pose.shape[1]))

		self.Vn        = [np.linalg.inv(self.V) for i in xrange(self.num_class)]
		self.mn        = [self.m0 * self.k0     for i in xrange(self.num_class)]
		self.kn        = self.k0  + self.C_t.sum(0) 
		self.vn        = self.v0  + self.C_t.sum(0)
		for i in xrange(self.num_class):
			if hist[i] != 0:
				self.sub_pose   =  pose * np.array([self.C_t[:,i] for j in xrange(pose.shape[1])]).T
				self.mean[i]    =  self.sub_pose.sum(0)/float(hist[i])
				self.mn[i]      = (self.mn[i]   + self.sub_pose.sum(0))/self.kn[i]
				self.var        =  np.array([self.mean[i] for t in range(pose.shape[0])]) * np.array([self.C_t[:,i] for j in xrange(pose.shape[1])]).T
				self.Vn[i]     += (self.sub_pose - self.var).T.dot(self.sub_pose - self.var) + \
									self.k0 * hist[i] / self.kn[i] * (self.mean[i] -self.m0)[:, np.newaxis].dot((self.mean[i] -self.m0)[np.newaxis,:])

				#if min(np.linalg.eigvalsh(self.Vn[i]))>0:
				self.sigma[i]   =  ss.invwishart.rvs(self.vn[i],self.Vn[i])
				self.mu[i]      =  ss.multivariate_normal.rvs(self.mn[i], np.linalg.inv(self.Vn[i]) / self.kn[i])
				#else:
				#	self.Vn[i] += self.V/10
			else:
				self.sigma[i]   =  ss.invwishart.rvs(self.v0,self.V)
				self.mu[i]      =  ss.multivariate_normal.rvs(self.m0, np.linalg.inv(self.V) / self.k0)

		return self.sigma,self.mu
    """
    def message():
        self.Pdz, self.Pdw = save_model( self.save_dir, self.n_dz, self.n_mzw, self.n_mz, self.M, self.dims, self.categories, self.liks, self.load_dir )
        return self.Pdz, self.Pdw


		