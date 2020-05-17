#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Spatial concept Topometric Map Visualizaer (Python only w/o ROS)
# 場所概念の位置分布（ガウス分布）および、その隣接関係のグラフ（遷移確率）を地図上に描画する
# Akira Taniguchi 2020/05/16- 
# This code is based on em_spcoae_map_srv.py

import sys
import numpy as np
import scipy.stats as ss
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image,ImageOps
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
MI_List   = [[0.0 for i in xrange(sample_num)] for j in xrange(ITERATION)]
MAX_Samp  = [0 for j in xrange(ITERATION)]

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
Mu_origin = Mu/(resolution) - np.array([origin[0],origin[1]])/(resolution)
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
for k in xrange(K):
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
        el2                  = Ellipse(xy=Mu_origin[k],width=1.0,height=1.0,color=COLOR[k])  # matplotlib.patches.Ellipse
        ax.add_patch(el2)
    except:
        break

# グラフノードと遷移確率の描画


# 地図描画
plt.imshow(map_image,cmap='gray')
#,extent=(origin[0],origin[0]+height*resolution,origin[1],origin[1]+width*resolution)

plt.tick_params(axis='x', which='major', labelsize=8)
plt.tick_params(axis='y', which='major', labelsize=8)
plt.xlabel('X', fontsize=10)
plt.ylabel('Y', fontsize=10)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.savefig(file_trialname + '_A_SpCoGraph_' + iteration_sample + '.png', dpi=300)
plt.savefig(file_trialname + '_A_SpCoGraph_' + iteration_sample + '.pdf', dpi=300)

plt.show()

#plt.cla()
#plt.clf()
#plt.close()



################################################################################################
"""
map_callback関数でmapのtopicをsubscribeして探索候補点を算出し、変数で保持。
map_server関数でROSサービスとして返す。


#from __init__ import *
#from em_spco_ae.srv import *

class Map(object):
    def map_callback(self, msg):
        self.msg = msg
        info = msg.info
        data = msg.data

        height       = info.height
        width        = info.width
        resolution   = np.round(info.resolution,decimals=3)
        origin       = info.origin

        #探索候補点の間隔（m）
        interval     = 0.8
        #何m四方内に占有セルがある場合に探索候補点から取り除くか
        square       = 0.5
        around       = int(round((square-resolution)/resolution)) / 2
        #探索候補点のindex格納リスト
        self.free_list    = list()
        width_list   = range(0,width,int(interval/resolution))
        height_list  = range(0,height,int(interval/resolution))
        for y in width_list:
            for x in height_list:
                if data[y * height + x] == 0:
                    free = True
                    #50cm四方内に占有セルがある場合は候補から除去
                    for y_ in range(y-around,y+around+1,1):
                        for x_ in range(x-around,x+around+1,1):
                            if data[y_ * height + x_] == 100:
                                free = False
                    if free:
                        #探索候補点のmapトピック上のindexをリストに格納
                        self.free_list.append(y * height + x)

        #ROSサービスで返すためにMultiArrayに変換
        self.free_array = Int64MultiArray()
        self.free_array.data = self.free_list


        #以下は、真の位置分布を描画する処理#########################################

        #ディレクトリ作成と真のパラメータコピー
        subprocess.Popen('mkdir -p ' + data_path + '/' + self.map_file, shell=True)
        param        = parameter_copy(self.map_file)
        _mu_k        = param[3]
        _sigma_k     = param[4]

        #分散共分散行列描画
        fig  = plt.figure()
        ax   = fig.add_subplot(1,1,1)
        for k in xrange(K):
            try:
                lmda, vec            = np.linalg.eig(_sigma_k[k])
                el_width,el_height   = 2 * el_c * np.sqrt(lmda)
                el_angle             = np.rad2deg(np.arctan2(vec[1,0],vec[0,0]))
                el                   = Ellipse(xy=_mu_k[k],width=el_width,height=el_height,angle=el_angle,color=colorlist[k],alpha=0.3)
                ax.add_patch(el)
            except:
                break

        #探索候補点プロット
        for xy in xrange(len(self.free_list)):
            #探索点のindex→探索点の座標
            point        = index2point(xy,self.free_list,info)
            point_pdf    = np.zeros(K)
            for k in xrange(K):
                try:
                    point_pdf[k] = ss.multivariate_normal(mean=_mu_k[k],cov=_sigma_k[k]).pdf(np.array([point.x,point.y]))
                except:
                    break
            index_i = np.argmax(point_pdf)
            plt.plot(point.x,point.y,color=colorlist[index_i],marker='o',markersize=4)

        #地図描画
        map_file_path    = roslib.packages.get_pkg_dir('em_spco_ae') + '/map/' + self.map_file + '/map.pgm'
        map_image        = Image.open(map_file_path)
        plt.imshow(map_image,extent=(origin.position.x,origin.position.x+height*resolution,origin.position.y,origin.position.y+width*resolution),cmap='gray')

        #未知領域を全て描画する場合
        # self.x_min = origin.position.x
        # self.x_max = origin.position.x+height*resolution
        # self.y_min = origin.position.y
        # self.y_max = origin.position.y+width*resolution
        # plt.xlim(origin.position.x,origin.position.x+height*resolution)
        # plt.ylim(origin.position.y,origin.position.y+width*resolution)

        #未知領域を削って描画する場合##################################
        #地図を描画する際のxyの最大最小値を算出
        map_data     = np.array([data[i:i+width] for i in range(0, len(data), width)])
        map_x        = np.sum(map_data, axis=0)
        map_y        = np.sum(map_data, axis=1)
        for i in range(0,width,1):
            if map_x[i] != width*-1:
                self.x_min_ = i * resolution + origin.position.x
                break
        for i in range(width-1,-1,-1):
            if map_x[i] != width*-1:
                self.x_max_ = i * resolution + origin.position.x
                break
        for i in range(0,height,1):
            if map_y[i] != height*-1:
                self.y_min_ = i * resolution + origin.position.y
                break
        for i in range(height-1,-1,-1):
            if map_y[i] != height*-1:
                self.y_max_ = i * resolution + origin.position.y
                break
        #未知領域を描画する幅（m）
        self.margin = 0.5
        self.x_min = self.x_min_-self.margin
        self.x_max = self.x_max_+self.margin
        self.y_min = self.y_min_-self.margin
        self.y_max = self.y_max_+self.margin
        ##########################################################

        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(self.x_min,self.x_max)
        plt.ylim(self.y_min,self.y_max)
        plt.savefig(data_path + '/' + self.map_file + '/true_' + self.map_file + '.png')
        plt.savefig(data_path + '/' + self.map_file + '/true_' + self.map_file + '.pdf')
        plt.cla()
        plt.clf()
        plt.close()

        self.map_callback_signal = True

    def map_server(self, request):
        #map_callback関数で算出した変数を返す。
        #self.free_array：探索候補点のindexリスト
        #len(self.free_list)：self.free_listの要素数
        #self.msg：mapのtopic
        #self.x_min：地図描画のx軸最小値
        #self.x_max：地図描画のx軸最大値
        #self.y_min：地図描画のy軸最小値
        #self.y_max：地図描画のy軸最大値
        return em_spcoae_mapResponse(self.free_array, len(self.free_list), self.msg, self.x_min, self.x_max, self.y_min, self.y_max)

    def __init__(self):
        self.map_file        = rospy.get_param('~map_file')
        self.map_callback_signal = False
        self.map_server_signal = False

        rospy.Subscriber('/map_' + self.map_file, OccupancyGrid, self.map_callback, queue_size=1)

        #map_callback関数の処理が終わってからmap_server定義
        while self.map_server_signal == False:
            if self.map_callback_signal:
                s_map = rospy.Service('em_spcoae/map_' + self.map_file, em_spcoae_map, self.map_server)
                rospy.loginfo('[Server em_spcoae/map]          Ready')
                self.map_server_signal = True

if __name__ == '__main__':
    rospy.init_node('em_map_server')
    hoge = Map()
    rospy.spin()
"""
