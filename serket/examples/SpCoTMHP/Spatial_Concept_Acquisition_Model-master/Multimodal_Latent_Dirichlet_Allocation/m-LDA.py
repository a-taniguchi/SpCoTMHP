# coding: utf-8
import numpy as np
import scipy.stats as ss
import random
import math

class M_LDA():
	def __init__(self,data1,data2,num_class):
		self.num_iter  = 100
		self.num_class = num_class
		self.alpha     = np.ones([num_class])*10
		self.eta1      = np.ones([data1.shape[1]])*10
		self.eta2      = np.ones([data2.shape[1]])*10
		self.data1     = data1
		self.data2     = data2
		self.theta     = ss.dirichlet.rvs(self.alpha)
		self.beta1     = ss.dirichlet.rvs(self.eta1,num_class)
		self.beta2     = ss.dirichlet.rvs(self.eta2,num_class)
		self.z_i       = np.random.multinomial(1,self.theta[0],size=data1.shape[0])


	def fit(self):
		for gibbs in range(self.num_iter):
			print gibbs
			self.gibbs_sampling()


	def gibbs_sampling(self):
		self.beta1   =  self.dirichlet_multinomial1(self.eta1, self.data1,self.z_i)
		self.beta2   =  self.dirichlet_multinomial1(self.eta2, self.data2,self.z_i)
		self.theta   =  self.dirichlet_multinomial2(self.alpha,self.z_i)
		self.z_i     =  self.multinomial_multinomial()

	def dirichlet_multinomial1(self,eta,data,z_i):
		beta = np.ones([self.num_class,data.shape[1]])
		for i in range(self.num_class):
			beta[i]  = ss.dirichlet.rvs((z_i[:,i]*data.T).sum(1)+eta)
		return beta

	def dirichlet_multinomial2(self,alpha,z_i):
		return ss.dirichlet.rvs(alpha+z_i.sum(0))

	def multinomial_multinomial(self):
		self.log_beta1  = np.log(self.beta1)
		self.log_beta2  = np.log(self.beta2)
		self.z_i        = np.zeros([self.data1.shape[0],self.num_class])
		self.log_likely = self.data1.dot(self.log_beta1.T)+self.data2.dot(self.log_beta2.T)
		self.sub_theta  = self.theta * np.exp(self.log_likely-np.array([self.log_likely.max(1) for j in range(self.num_class)]).T)
		self.probab     = self.sub_theta/np.array([self.sub_theta.sum(1) for j in range(self.num_class)]).T
		for i in range(self.data1.shape[0]):
			self.z_i[i]    = np.random.multinomial(1,self.probab[i],size=1)
		return self.z_i
		
	def predict(self,new_data1,new_data2):
		self.new_data1 = new_data1
		self.new_data2 = new_data2
		log_beta1  = np.log(self.beta1)
		log_beta2  = np.log(self.beta2)
		z_i        = np.zeros([self.data1.shape[0],self.num_class])
		log_likely = self.new_data1.dot(log_beta1.T)+self.new_data2.dot(log_beta2.T)
		sub_theta  = self.theta * np.exp(log_likely-np.array([log_likely.max(1) for j in range(self.num_class)]).T)
		probab     = sub_theta/np.array([sub_theta.sum(1) for j in range(self.num_class)]).T
		for i in range(self.data1.shape[0]):
			z_i[i]    = np.random.multinomial(1,probab[i],size=1)
		return z_i


if __name__ == '__main__':
	
	data1     = np.loadtxt("op1_mfcc_GMM.csv",delimiter=",")
	data2     = np.loadtxt("op1_color.csv"   ,delimiter=",")
	mlda      = M_LDA(data1,data2,10)
	mlda.fit()



