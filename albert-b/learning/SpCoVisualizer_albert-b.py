#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Spatial concept Topometric Map Visualizaer (Python only w/o ROS)
# 場所概念の位置分布（ガウス分布）および、その隣接関係のグラフ（遷移確率）を地図上に描画する
# SpCoA++の要領で複数回施行している場合は、相互情報量最大の候補を選択
# Akira Taniguchi 2020/05/16 - 2020/05/18
# This code is based on em_spcoae_map_srv.py
# [command] $ python3 SpCoVisualizer_albert-b.py <trialname>

import sys
import numpy as np
import scipy.stats as ss
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from matplotlib.patches import Ellipse
from PIL import Image,ImageOps #, ImageDraw
import yaml
from __init__ import *

iteration = 0 # if (ITERATION == 1 )

# 学習済みパラメータフォルダ名 trialname を得る
trialname = sys.argv[1]

# map の origin and resolution を .yaml file から読み込む
with open(datasetfolder + map_file + '.yaml') as file:
    obj = yaml.safe_load(file)
    origin     = obj['origin']
    resolution = obj['resolution'] 

print((origin,resolution))

# For the map on albert-B dataset
x_min = 380 #X_init_index[0] - T_horizon
x_max = 800 #X_init_index[0] + T_horizon
y_min = 180 #X_init_index[1] - T_horizon
y_max = 510 #X_init_index[1] + T_horizon

# map の .pgm file を読み込み
map_file_path = datasetfolder + map_file + '.pgm' #roslib.packages.get_pkg_dir('em_spco_ae') + '/map/' + self.map_file + '/map.pgm'
map_image     = Image.open(map_file_path)
map_image     = ImageOps.flip(map_image)      # 上下反転

# height and width を得る
width, height = map_image.size
print(map_image.size)



# MIが最大のsampleの番号を得る
MI_List   = [[0.0 for i in range(sample_num)] for j in range(ITERATION)]
MAX_Samp  = [0 for j in range(ITERATION)]

#./data/trialname/trialname_sougo_MI_iteration.csvを読み込み
for line in open(outputfolder + trialname + '/' + trialname + '_sougo_MI_' + str(iteration+1) + '.csv', 'r'):
    itemList = line[:-1].split(',')
    if (int(itemList[0]) < sample_num):
        MI_List[iteration][int(itemList[0])] = float(itemList[1])
MAX_Samp[iteration] = MI_List[iteration].index(max(MI_List[iteration]))  #相互情報量が最大のサンプル番号
sample_max = MAX_Samp[iteration]


file_trialname   = outputfolder + trialname +'/' + trialname
iteration_sample = str(iteration+1) + "_" + str(sample_max) 

# Spatial concept の Gaussian distribution parameters (mu and sigma) を読み込む
Mu  = np.loadtxt(file_trialname + '_Mu_' + iteration_sample + '.csv', delimiter=',')
Mu_origin = ( Mu - np.array([origin[0],origin[1]]) ) / resolution
Sig = np.load(file_trialname + '_Sig_'   + iteration_sample + '.npy')

# Spatial concept の Transition probability parameter (psi) を読み込む
psi = np.load(file_trialname + '_psi_'   + iteration_sample + '.npy')

##itの読み込み
It = np.loadtxt( file_trialname + '_It_'+ iteration_sample + '.csv', dtype=int )


# 位置分布を描画する処理 #########################################
#分散共分散行列描画
fig  = plt.figure()
ax   = fig.add_subplot(1,1,1)
el_c = np.sqrt(ss.chi2.ppf(el_prob, 2))
for k in range(K):
    #データ点が割り当てられていない位置分布は描画しない
    if list(It).count(k) == 0:
        continue
    try:
        lmda, vec            = np.linalg.eig(Sig[k]/(resolution**(2)))
        el_width,el_height   = 2 * el_c * np.sqrt(lmda) #* resolution
        el_angle             = np.rad2deg(np.arctan2(vec[1,0],vec[0,0]))
        el                   = Ellipse(xy=Mu_origin[k],width=el_width,height=el_height,angle=el_angle,facecolor=COLOR[k],alpha=0.3,edgecolor=None)  # matplotlib.patches.Ellipse
        ax.add_patch(el)
        
        # 中心点の描画
        plt.scatter(Mu_origin[k][0], Mu_origin[k][1], s=5,color=COLOR[k], zorder=2)
        #el2                  = Ellipse(xy=Mu_origin[k],width=1.0,height=1.0,color=COLOR[k])  # matplotlib.patches.Ellipse
        #ax.add_patch(el2)

        txt = ax.text(Mu_origin[k][0]+4, Mu_origin[k][1]+4, str(k), size = 14, color = "midnightblue", zorder=3)
        txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white', alpha=0.8),
                              path_effects.Normal()])
    except:
        break

# グラフノードと遷移確率の描画
edge_list = []
psi_sym = (psi + psi.T) / 2.0 
#psi_sym = psi_sym / np.sum(psi_sym,0)
#print psi_sym,np.sum(psi_sym[0])
ignore_value = 1.0 / K
for k1 in range(K):
    if list(It).count(k1) == 0:
        continue
    for k2 in range(K):
        if list(It).count(k2) == 0:
          continue
        if (psi_sym[k1][k2] > ignore_value) and (k1 != k2) and (psi_sym[k1][k2] != 1.0):
          if ( (k2,k1) not in edge_list ):
              edge_list += [(k1,k2)]
              print(k1,k2,psi_sym[k1][k2])
print(edge_list)


# 地図描画
plt.imshow(map_image,cmap='gray')
#,extent=(origin[0],origin[0]+height*resolution,origin[1],origin[1]+width*resolution)

for edge in edge_list:
    plt.plot( (Mu_origin[edge[0]][0], Mu_origin[edge[1]][0]), (Mu_origin[edge[0]][1], Mu_origin[edge[1]][1]),
              color='dimgray', linestyle = "-", linewidth=10*psi_sym[edge[0]][edge[1]], zorder=1 ) #, alpha=10*psi_sym[edge[0]][edge[1]]) #psi_sym[edge[0]][edge[1]])
#darkslategrey

##axの場合
#ax.set_axisbelow(True)

##rcParamsの場合
#plt.rcParams['axes.axisbelow'] = True

plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=8)
plt.xlabel('X', fontsize=10)
plt.ylabel('Y', fontsize=10)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.savefig(file_trialname + '_A_SpCoGraph_' + iteration_sample + '.png', dpi=300)
plt.savefig(file_trialname + '_A_SpCoGraph_' + iteration_sample + '.pdf', dpi=300)

#plt.show()

plt.cla()
plt.clf()
plt.close()
