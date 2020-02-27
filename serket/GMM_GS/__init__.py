#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys
sys.path.append( "../" )
import os
import serket as srk
import numpy as np
import scipy.stats as stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""
class GMM_em(srk.Module):
    def __init__(self, K, D, itr=10, name="gmm" ):
        super(GMM_em, self).__init__(name, True)
        self.__K = K
        self.__D = D
        self.__itr = itr
        #self.__pi = np.ones(self.__K) / self.__K
        #self.__mean = stats.multivariate_normal.rvs(mean=np.zeros(self.__D), cov=np.identity(self.__D), size=self.__K)
        #self.__cov = np.array(list(np.identity(self.__D)) * self.__K).reshape(self.__K, self.__D, self.__D)
        self.__pi = None
        self.__mean = None
        self.__cov = None

    #initialize parameter using K-means
    def initialize_parameter(self):
        K = self.__K
        data = np.concatenate(self.get_observations()).reshape(-1, self.__D)
        result = KMeans(n_clusters=K).fit_predict(data)
        self.__pi = np.array([(result == k).sum().astype(float) for k in range(K)]) / data.shape[0]
        self.__mean = np.array([data[result == k].mean(axis=0) for k in range(K)])
        self.__cov = np.array([np.cov(data[result == k], rowvar=False) for k in range(K)])

    #update by EM algorithm.
    def update(self, itr=-1):
        if itr is -1:
            itr = self.__itr

        if self.__pi is None:
            self.initialize_parameter()
        data = np.concatenate(self.get_observations()).reshape(-1, self.__D)
        N = data.shape[0]

        pi, mean, cov, taxrate = gmm_em(data, self.__pi, self.__mean, self.__cov, taxmessage=self.get_backward_msg(), itr=itr)

        self.__pi = pi
        self.__mean = mean
        self.__cov = cov

        # メッセージの送信
        self.set_forward_msg( taxrate )
        #self.send_backward_msgs( Pdw )

        ##########################
        save_dir = self.get_name()
        try:
            os.mkdir( save_dir )
        except:
            pass

        cov2 = [[cov[k][0][0], cov[k][0][1], cov[k][1][0], cov[k][1][1]] for k in range(self.__K)]
        np.savetxt( os.path.join( save_dir, "pi.txt" ), pi )
        np.savetxt( os.path.join( save_dir, "mu.txt" ), mean )
        np.savetxt( os.path.join( save_dir, "Sig.txt" ), cov2 )
        np.savetxt( os.path.join( save_dir, "taxrate.txt" ), taxrate )
        ##########################

    def getParameterOf(self, name):
        if name == "pi":
            return self.__pi
        elif name == "mean":
            return self.__mean
        elif name == "cov":
            return self.__cov
        else:
            raise ValueError("argument ``%s\" is invalid!!" % name)

def gmm_em(data, pi, mean, cov, taxmessage=None, itr=10):
    K = pi.shape[0]
    N = data.shape[0]
    D = data.shape[1]

    if taxmessage is None:
        taxmessage = np.ones((N, K)) / K
    elif (taxmessage.ndim is 1) and (taxmessage.shape[0] is K):
        taxmessage = np.ones((N, K)) * taxmessage

    taxrate = np.empty((N, K))

    for t in range(itr):
        # E step
        for k in range(K):
            if np.linalg.matrix_rank(cov[k]) != D:
                cov[k] = np.identity(D) * D
            taxrate[:,k] = stats.multivariate_normal.pdf(data, mean=mean[k], cov=cov[k], allow_singular=True)  #########
        taxrate *= taxmessage * pi
        taxrate = (taxrate.T / taxrate.sum(axis=1)).T
        # M step
        Nk = taxrate.sum(axis=0)
        pi = Nk / Nk.sum()

        tmp = taxrate.T.dot(data)
        mean = (tmp.T / Nk).T

        for k in range(K):
            diff = data - mean[k]
            cov[k] = (diff.T * taxrate[:,k]).dot(diff) / Nk[k]

    #calculate message for upper module
    for k in range(K):
        if np.linalg.matrix_rank(cov[k]) != D:
            cov[k] = np.identity(D) * D
        taxrate[:,k] = stats.multivariate_normal.pdf(data, mean=mean[k], cov=cov[k])
    taxrate *= pi
    taxrate = (taxrate.T / taxrate.sum(axis=1)).T
    return pi, mean, cov, taxrate
"""
class GMM_gibbs(srk.Module):
    def __init__(self, K, D, itr=100, name="gmm", category=None, load_dir=None ):
        super(GMM_gibbs, self).__init__(name, True)
        self.__K = K
        self.__D = D
        self.__itr = itr
        self.__category = category
        self.__load_dir = load_dir
        self.__n = 0

        self.__pi = None
        self.__mean = None
        self.__cov = None

        self.__alpha_0 = np.ones(K) * 10.0 #
        self.__mu_0 = np.zeros((K, D))
        self.__Sigma_0 = np.empty((K, D, D))
        for k in range(K):
            self.__Sigma_0[k] = np.identity(D)
        self.__nu_0 = np.ones(K) * D + 5
        self.__Psi_0 = np.empty((K, D, D))
        for k in range(K):
            self.__Psi_0[k] = np.identity(D) #* 0.1 #2.0 #

    #initialize parameter by prior distributions.
    def initialize_parameter(self):
        K = self.__K
        D = self.__D
        self.__pi = stats.dirichlet.rvs(self.__alpha_0)[0]
        mean = self.__mean = np.empty((K, D))
        cov = self.__cov = np.empty((K, D, D))
        for k in range(K):
            mean[k] = stats.multivariate_normal.rvs(mean=self.__mu_0[k], cov=self.__Sigma_0[k])
            cov[k] = stats.invwishart.rvs(df=self.__nu_0[k], scale=self.__Psi_0[k])

    #update by Gibbs sampling.
    def update(self, itr=-1):
        if itr is -1:
            itr = self.__itr
        if self.__pi is None:
            self.initialize_parameter()
        K = self.__K
        D = self.__D

        data = self.get_observations()
        datas = np.concatenate(data).reshape(-1, D)
        N = datas.shape[0]
        k_vec = np.zeros(K)
        D_zeros = np.matrix(np.zeros(D))
        p_yi = np.empty((N, K)) #np.zeros((N,K)) #Pdz
        hidden_state = np.zeros(N, dtype=int)
        hidden_state_count = np.zeros(K, dtype=int)

        taxmessage = self.get_backward_msg()
        if taxmessage is None:
            taxmessage = np.ones((N, K)) / K
        elif taxmessage.ndim == 1 and taxmessage.shape[0] == K:
            taxmessage = np.ones((N, K)) * taxmessage

        if self.__load_dir is None:
            save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
        else:
            save_dir = os.path.join( self.get_name(), "recog" )

        pi = self.__pi
        mu = self.__mean
        Sigma = self.__cov
        #print mu,Sigma

        for t in range(itr):

            for k in range(K):
                p_yi[:,k] = stats.multivariate_normal.pdf(datas, mean=mu[k], cov=Sigma[k])
                #print p_yi[:,k]
            p_yi *= pi
            #print pi,p_yi

            # Resampling labels.
            for n in range(N):
                k_vec[:] = p_yi[n] * taxmessage[n]
                #print k_vec
                k_vec /= k_vec.sum()
                hidden_state[n] = np.random.choice(K, p=k_vec)
            counting(hidden_state, K, out=hidden_state_count)

            # Resampling parameters phase.
            # Resampling pi.
            alpha_hat = self.__alpha_0 + hidden_state_count
            pi[:] = stats.dirichlet.rvs(alpha_hat)

            # Resampling mu_k and Sigma_k
            for k in range(K):
                data_k = np.matrix(datas[hidden_state == k])
                if data_k.shape[0] == 0:
                    data_k = D_zeros
                m_k = hidden_state_count[k]
                sum_data_k = data_k.sum(axis=0)

                # Resampling Sigma_k.
                tmp = np.matrix(data_k - mu[k])
                Psi_hat = tmp.T.dot(tmp) + self.__Psi_0[k]
                nu_hat = m_k + self.__nu_0[k]
                Sigma[k] = stats.invwishart.rvs(df=nu_hat, scale=Psi_hat)

                # Resampling mu_k.
                Sigma_hat = np.array((m_k * np.matrix(Sigma[k]).I + np.matrix(self.__Sigma_0[k]).I).I)
                mu_hat = np.array((sum_data_k.dot(np.matrix(Sigma[k]).I) + self.__mu_0[k].dot(np.matrix(self.__Sigma_0[k]).I)).dot(Sigma_hat))[0]
                mu[k] = stats.multivariate_normal.rvs(mean=mu_hat, cov=Sigma_hat)

        # self.__pi = pi
        # self.__mean = mu
        # self.__cov = Sigma

        for k in range(K):
            p_yi[:,k] = stats.multivariate_normal.pdf(datas, mean=mu[k], cov=Sigma[k], allow_singular=True)  #########
        p_yi *= pi
        taxrate = (p_yi.T / p_yi.sum(axis=1)).T

        self.__n += 1

        # メッセージの送信
        self.set_forward_msg( taxrate )
        self.send_backward_msgs( [mu] )

        ##########################
        #save_dir = self.get_name()
        if not os.path.exists( save_dir ):
            os.makedirs( save_dir )

        cov2 = [[Sigma[k][0][0], Sigma[k][0][1], Sigma[k][1][0], Sigma[k][1][1]] for k in range(self.__K)]
        np.savetxt( os.path.join( save_dir, "pi.txt" ), pi )
        np.savetxt( os.path.join( save_dir, "mu.txt" ), mu )
        np.savetxt( os.path.join( save_dir, "Sig.txt" ), cov2 )
        np.savetxt( os.path.join( save_dir, "taxrate.txt" ), taxrate, fmt="%f" )
        #np.savetxt( os.path.join( save_dir, "class.txt" ), hidden_state, fmt="%d" )
        classes = hidden_state
        categories = self.__category

        # 分類結果・精度の計算と保存
        if categories is not None:
            acc, results = calc_acc( classes, categories )
            np.savetxt( os.path.join( save_dir, "class.txt" ), results, fmt="%d" )
            np.savetxt( os.path.join( save_dir, "acc.txt" ), [acc], fmt="%f" )
            
        else:
            np.savetxt( os.path.join( save_dir, "class.txt" ), classes, fmt="%d" )
        ##########################

    def getParameterOf(self, name):
        if name == "pi":
            return self.__pi
        elif name == "mean":
            return self.__mean
        elif name == "cov":
            return self.__cov
        else:
            raise ValueError("argument ``%s\" is invalid!!" % name)

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

def counting(datas, size, out=None):
    if out is None:
        cnt = np.zeros(size, dtype=int)
    else:
        cnt = out
        cnt[:] = 0
    if datas is None:
        return cnt
    for i in range(size):
        cnt[i] = (datas == i).sum()
    return cnt
