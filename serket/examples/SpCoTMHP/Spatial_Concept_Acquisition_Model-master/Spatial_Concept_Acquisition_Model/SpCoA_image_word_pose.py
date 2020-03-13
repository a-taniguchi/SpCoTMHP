# coding: utf-8
import numpy as np
import scipy.stats as ss
import random
import sklearn.cluster
import sklearn.metrics
import time

class SpCoA():
	def __init__(self,pose,data1,data2,ganmma,num_class):
		self.num_iter  = 100
		self.num_class = num_class

		self.pose      = pose
		self.data1     = data1
		self.data2     = data2

		self.ganmma    = np.ones([num_class])*ganmma
		self.pi        = self.stick_breaking(ganmma,self.num_class)
		self.alpha1    = np.ones([data1.shape[1]])*2
		self.alpha2    = np.ones([data2.shape[1]])*2
		self.phai1     = ss.dirichlet.rvs(self.alpha1,num_class)
		self.phai2     = ss.dirichlet.rvs(self.alpha2,num_class)
		self.C_t       = np.random.multinomial(1,self.pi,size=data1.shape[0])
		
		self.V         = np.eye(pose.shape[1])*0.05#np.cov(data,rowvar=0) /(10)
		self.v0        = 15#pose.shape[1]+1
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
		self.phai1   =  self.dirichlet_multinomial1(self.alpha1, self.data1,self.C_t)
		self.phai2   =  self.dirichlet_multinomial1(self.alpha2, self.data2,self.C_t)
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
		log_phai1  = np.log(self.phai1)
		log_phai2  = np.log(self.phai2)
		z_i        = np.zeros([self.data1.shape[0],self.num_class])
		log_likely = self.data1.dot(log_phai1.T)+self.data2.dot(log_phai2.T)
		sub_pi     = self.pi * np.exp(log_likely-np.array([log_likely.max(1) for j in range(self.num_class)]).T)
		probab     = sub_pi/np.array([sub_pi.sum(1) for j in range(self.num_class)]).T

		self.new_pi = np.zeros((self.pose.shape[0],self.num_class))
		for i in xrange(self.num_class):
			self.new_pi[:,i] = ss.multivariate_normal.pdf(self.pose,self.mu[i],self.sigma[i])*self.pi[0,i] 

		self.new_pi = self.new_pi * probab
		for i in range(self.data1.shape[0]):
			if self.new_pi.sum(1)[i] != 0:
				self.pi_sub = (self.new_pi[i].T/self.new_pi.sum(1)[i]).T
				z_i[i]      = np.random.multinomial(1,self.pi_sub,size=1)
		return z_i

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

		



if __name__ == '__main__':
	pose      = np.loadtxt("pose.csv"    ,delimiter=",")
	data1     = np.loadtxt("word.csv"    ,delimiter=",")
	data2     = np.loadtxt("image.csv"   ,delimiter=",")

	#label     = np.loadtxt("label.csv"   ,delimiter=",")

	start     = time.time()

	spcoa     = SpCoA(pose,data1,data2,10.0,30)
	spcoa.fit()

	elapsed_time = time.time() - start
	print "time :",elapsed_time

	for i in spcoa.C_t.dot(range(spcoa.num_class)):
		print int(i), ",",
	print 





