# encoding: utf8
#from __future__ import unicode_literals
import sys
sys.path.append( "../" )

import serket as srk
from GMM import *
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import pickle
import os
from numba import jit

from matplotlib import gridspec
from matplotlib import patches

def plot_mixingrate(axes, mixing_rate, colors=None):
    size = mixing_rate.shape[0]
    arg = np.argsort(mixing_rate)[::-1]
    cumsum_rev = np.cumsum(mixing_rate[arg])[::-1]
    if colors is not None:
        colors = colors[arg][::-1]
    return axes.barh([0] * size, cumsum_rev, color=colors, height=2.0).get_children()

def plot_mean(axes, mean, color=None, mark="o"):
    return axes.scatter(mean[0], mean[1], marker=mark, c=color, zorder=1)

def plot_covariance(axes, mean, covariance, scale=3.0, color=None):
    la, v = np.linalg.eig(covariance)
    std = np.sqrt(la)
    angle = rad2deg(np.arctan2(v[1,0], v[0,0]))
    e = patches.Ellipse((mean[0], mean[1]), 2*std[0]*scale, 2*std[1]*scale, angle=angle, linewidth=1, fill=False, color=color, zorder=2)
    axes.add_artist(e)
    return e

def plot_datas(axes, datas, color=None, mark="+"):
    return axes.scatter(datas[:,0], datas[:,1], marker=mark, c=color, zorder=1)

def rad2deg(rad):
    return rad * 180.0 / np.pi

T = 100
truth_K = 3
data_N = [3000, 3000, 3000]
N = sum(data_N)
truth_pi = np.array(data_N) / N
truth_mean = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
truth_cov = np.array(list(np.identity(2)) * truth_K).reshape(truth_K, 2, 2)

colors = np.array(["red", "green", "blue"])

observations = stats.multivariate_normal.rvs(mean=truth_mean[0], cov=truth_cov[0], size=data_N[0])
for k in range(truth_K-1):
    observations = np.concatenate( (observations, stats.multivariate_normal.rvs(mean=truth_mean[k+1], cov=truth_cov[k+1], size=data_N[k+1])) )

fig = plt.figure(figsize=(12,8))
ax = plt.gca()

plot_datas(ax, observations)
for k in range(truth_K):
    plot_mean(ax, truth_mean[k], color=colors[k])
    plot_covariance(ax, truth_mean[k], truth_cov[k], color=colors[k])
plt.title("Truth data.")
plt.xlim(-10, 10)
plt.pause(0.5)

obs_mod = srk.Observation( observations )
K = 3
D = 2
gmm = GMM( K, D, itr=1 )

gmm.connect( obs_mod )
#gmm.set_backward_msg(np.array([0.02, 0.44, 0.44]))

for t in range(T):
    gmm.update()
    gmm_forward = gmm.get_forward_msg()
    mean = gmm.getParameterOf("mean")
    cov = gmm.getParameterOf("cov")
    ax.cla()
    plot_datas(ax, observations, color=gmm_forward)
    for k in range(truth_K):
        plot_mean(ax, mean[k], color=colors[k])
        plot_covariance(ax, mean[k], truth_cov[k], color=colors[k])
    plt.title("Iteration at %d" % (t + 1))
    plt.xlim(-10, 10)
    plt.pause(0.01)

"""
pi = np.ones(K) / K
mean = stats.multivariate_normal.rvs(mean=np.zeros(D), cov=np.identity(D) * 3, size=K)
cov = np.array(list(np.identity(D) * 10) * K).reshape(K, D, D)

for t in range(T):
    pi, mean, cov, _ = em(observations, pi, mean, cov, itr=1)
    print cov
"""
