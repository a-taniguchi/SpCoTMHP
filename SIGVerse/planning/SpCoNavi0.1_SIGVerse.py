#coding:utf-8

###########################################################
# SpCoNavi: Spatial Concept-based Path-Planning Program for SIGVerse
# Akira Taniguchi 2019/06/24-2019/07/13
###########################################################

### NEW for SIGVerse ###
## Save path length and log likelihood
## Remove using numba

##############################################
import os
import sys
import glob
import time
import random
import numpy as np
import scipy as sp
#from numpy.random import multinomial #,uniform #,dirichlet
from scipy.stats import multivariate_normal,multinomial #,t,invwishart,rv_discrete
#from math import pi as PI
from math import cos,sin,sqrt,exp,log,degrees,radians,atan2 #,gamma,lgamma,fabs,fsum
from __init__ import *
#from JuliusNbest_dec_SIGVerse import *
from submodules import *
from scipy.io import mmwrite, mmread
from scipy.sparse import lil_matrix, csr_matrix
from itertools import izip
import collections

#Read the map data⇒2-dimension array に格納
def ReadMap(outputfile):
    #outputfolder + trialname + navigation_folder + map.csv
    gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
    print "Read map: " + outputfile + "map.csv"
    return gridmap

#Read the cost map data⇒2-dimension array に格納
def ReadCostMap(outputfile):
    #outputfolder + trialname + navigation_folder + contmap.csv
    costmap = np.loadtxt(outputfile + "costmap.csv", delimiter=",")
    print "Read costmap: " + outputfile + "contmap.csv"
    return costmap

#Read the parameters of learned spatial concepts
def ReadParameters(iteration, sample, filename, trialname):
    #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    #r = iteration
    """
    i = 0
    for line in open(filename + 'index' + str(r) + '.csv', 'r'):   ##読み込む
        itemList = line[:-1].split(',')
        #print itemList
        if (i == 0):
          L = len(itemList) -1
        elif (i == 1):
          K = len(itemList) -1
        i += 1
    print "L:",L,"K:",K
    """

    W_index = []
    i = 0
    #Read the text file
    for line in open(filename + "/" + trialname + '_w_index_' + str(iteration) + '_' + str(sample) + '.csv', 'r'): 
        itemList = line[:-1].split(',')
        if(i == 1):
            for j in xrange(len(itemList)):
              if (itemList[j] != ""):
                W_index = W_index + [itemList[j]]
        i = i + 1
    
    #####パラメータW, μ, Σ, φ, πを入力する#####
    Mu    = [ np.array([ 0.0, 0.0 ]) for i in xrange(K) ]  #[ np.array([[ 0.0 ],[ 0.0 ]]) for i in xrange(K) ]      #位置分布の平均(x,y)[K]
    Sig   = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in xrange(K) ]      #位置分布の共分散(2×2-dimension)[K]
    W     = [ [0.0 for j in xrange(len(W_index))] for c in xrange(L) ]  #場所の名前(多項分布: W_index-dimension)[L]
    #theta = [ [0.0 for j in xrange(DimImg)] for c in xrange(L) ] 
    Pi    = [ 0.0 for c in xrange(L)]     #場所概念のindexの多項分布(L-dimension)
    Phi_l = [ [0.0 for i in xrange(K)] for c in xrange(L) ]  #位置分布のindexの多項分布(K-dimension)[L]
      
    i = 0
    ##Mu is read from the file
    for line in open(filename + "/" + trialname + '_Myu_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #Mu[i] = np.array([ float(itemList[0]) - origin[0] , float(itemList[1]) - origin[1] ]) / resolution
        Mu[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
        i = i + 1
      
    i = 0
    ##Sig is read from the file
    for line in open(filename + "/" + trialname + '_S_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        #Sig[i] = np.array([[ float(itemList[0])/ resolution, float(itemList[1]) ], [ float(itemList[2]), float(itemList[3])/ resolution ]]) #/ resolution
        Sig[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3]) ]]) 
        i = i + 1
      
    ##phi is read from the file
    c = 0
    #Read the text file
    for line in open(filename + "/" + trialname + '_phi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
            if itemList[i] != "":
              Phi_l[c][i] = float(itemList[i])
        c = c + 1
        
    ##Pi is read from the file
    for line in open(filename + "/" + trialname + '_pi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
          if itemList[i] != '':
            Pi[i] = float(itemList[i])
      
    ##W is read from the file
    c = 0
    #Read the text file
    for line in open(filename + "/" + trialname + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              W[c][i] = float(itemList[i])
        c = c + 1

    """
    ##theta is read from the file
    c = 0
    #Read the text file
    for line in open(filename + 'theta' + str(r) + '.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in xrange(len(itemList)):
            if itemList[i] != '':
              #print c,i,itemList[i]
              theta[c][i] = float(itemList[i])
        c = c + 1
    """

    THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    return THETA

"""
#角度を[-π,π]に変換(参考: https://github.com/AtsushiSakai/PythonRobotics)
def pi_2_pi(angle):
    return (angle + PI) % (2 * PI) - PI

#Triangular distribution PDF
def Prob_Triangular_distribution_pdf(a,b):
    prob = max( 0, ( 1 / (sqrt(6)*b) ) - ( abs(a) / (6*(b**2)) ) )
    return prob

#Selection of probabilistic distribution
def Motion_Model_Prob(a,b):
    if (MotionModelDist == "Gauss"):
      p = multivariate_normal.pdf(a, 0, b)
    elif (MotionModelDist == "Triangular"):
      p = Prob_Triangular_distribution_pdf(a, b)
    return p

#Odometry motion model (確率ロボティクスp.122) #現状, 不使用
def Motion_Model_Odometry(xt,ut,xt_1):
    #ut = (xt_1_bar, xt_bar), xt_1_bar = (x_bar, y_bar, theta_bar), xt_bar = (x_dash_bar, y_dash_bar, theta_dash_bar)
    x_dash, y_dash, theta_dash = xt
    x, y, theta = xt_1
    xt_1_bar, xt_bar = ut
    x_dash_bar, y_dash_bar, theta_dash_bar = xt_bar
    x_bar, y_bar, theta_bar = xt_1_bar

    delta_rot1  = atan2(y_dash_bar - y_bar, x_dash_bar - x_bar) - theta_bar
    delta_trans = sqrt( (x_dash_bar - x_bar)**2 + (y_dash_bar - y_bar)**2 )
    delta_rot2  = theta_dash_bar - theta_bar - delta_rot1

    delta_rot1_hat  = atan2(y_dash - y, x_dash - x) - theta
    delta_trans_hat = sqrt( (x_dash - x)**2 + (y_dash - y)**2 )
    delta_rot2_hat  = theta_dash - theta - delta_rot1_hat

    p1 = Motion_Model_Prob(pi_2_pi(delta_rot1 - delta_rot1_hat), odom_alpha1*(delta_rot1_hat**2) + odom_alpha2*(delta_trans_hat**2))
    p2 = Motion_Model_Prob(delta_trans - delta_trans_hat, odom_alpha3*(delta_trans_hat**2) + odom_alpha4*(delta_rot1_hat**2+delta_rot2_hat**2))
    p3 = Motion_Model_Prob(pi_2_pi(delta_rot2 - delta_rot2_hat), odom_alpha1*(delta_rot2_hat**2) + odom_alpha2*(delta_trans_hat**2))

    return p1*p2*p3

#Odometry motion model (簡略版) #角度は考慮せず, 移動量に応じて確率が決まる(ドーナツ型分布)
def Motion_Model_Odometry_No_theta(xt,ut,xt_1):
    #ut = (xt_1_bar, xt_bar), xt_1_bar = (x_bar, y_bar), xt_bar = (x_dash_bar, y_dash_bar)
    #utは相対的な位置関係で良い
    x_dash, y_dash = xt
    x, y = xt_1

    delta_trans = cmd_vel #sqrt( (x_dash_bar - x_bar)**2 + (y_dash_bar - y_bar)**2 )
    delta_trans_hat = sqrt( (x_dash - x)**2 + (y_dash - y)**2 )

    p2 = Motion_Model_Prob( delta_trans - delta_trans_hat, odom_alpha3*(delta_trans_hat**2) )

    return p2  #p1*p2*p3

#Motion model (original) #角度は考慮せず, 移動先位置に応じて確率が決まる(Gaussian distribution)
def Motion_Model_Original(xt,ut,xt_1):
    xt = np.array(xt)
    #ut = np.array(ut)
    xt_1 = np.array(xt_1)
    dist = np.sum((xt-xt_1)**2)
    
    px = Motion_Model_Prob( xt[0] - (xt_1[0]+ut[0]), odom_alpha3*dist )
    py = Motion_Model_Prob( xt[1] - (xt_1[1]+ut[1]), odom_alpha3*dist )
    return px*py
"""

#ROSのmap 座標系をPython内の2-dimension array index 番号に対応付ける
def Map_coordinates_To_Array_index(X):
    X = np.array(X)
    Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
    return Index

#Python内の2-dimension array index 番号からROSのmap 座標系への変換
def Array_index_To_Map_coordinates(Index):
    Index = np.array(Index)
    X = np.array( (Index * resolution) + origin )
    return X

#gridmap and costmap から確率の形のCostMapProbを得ておく
def CostMapProb_jit(gridmap, costmap):
    CostMapProb = (100.0 - costmap) / 100.0     #Change the costmap to the probabilistic costmap
    #gridの数値が0（非占有）のところだけ数値を持つようにマスクする
    GridMapProb = 1*(gridmap == 0)  #gridmap * (gridmap != 100) * (gridmap != -1)  #gridmap[][]が障害物(100)または未探索(-1)であれば確率0にする
    
    return CostMapProb * GridMapProb


def PostProb_ij(Index_temp,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K):
    if (CostMapProb[Index_temp[1]][Index_temp[0]] != 0.0): 
      X_temp = Array_index_To_Map_coordinates(Index_temp)  #map と縦横の座標系の軸が合っているか要確認
      #print X_temp,Mu
      sum_i_GaussMulti = [ np.sum([multivariate_normal.pdf(X_temp, mean=Mu[k], cov=Sig[k]) * Phi_l[c][k] for k in xrange(K)]) for c in xrange(L) ] ##########np.array( ) !!! np.arrayにすると, numbaがエラーを吐く
      PostProb = np.sum( LookupTable_ProbCt * sum_i_GaussMulti ) #sum_c_ProbCtsum_i
    else:
      PostProb = 0.0
    return PostProb

#@jit(parallel=True)  #並列化されていない？1CPUだけ使用される
def PostProbMap_nparray_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K): #,IndexMap):
    PostProbMap = np.array([ [ PostProb_ij([width, length],Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) for width in xrange(map_width) ] for length in xrange(map_length) ])
    
    return CostMapProb * PostProbMap


#Global path estimation by dynamic programming (calculation of SpCoNavi)
def PathPlanner(S_Nbest, X_init, THETA, CostMapProb): #gridmap, costmap):
    print "[RUN] PathPlanner"
    #THETAを展開
    W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA

    #ROSの座標系の現在位置を2-dimension array index にする
    X_init_index = X_init ###TEST  #Map_coordinates_To_Array_index(X_init)
    print "Initial Xt:",X_init_index

    #length and width of the MAP cells
    map_length = len(CostMapProb)     #len(costmap)
    map_width  = len(CostMapProb[0])  #len(costmap[0])
    print "MAP[length][width]:",map_length,map_width

    #Pre-calculation できるものはしておく
    LookupTable_ProbCt = np.array([multinomial.pmf(S_Nbest, sum(S_Nbest), W[c])*Pi[c] for c in xrange(L)])  #Ctごとの確率分布 p(St|W_Ct)×p(Ct|Pi) の確率値
    ###SaveLookupTable(LookupTable_ProbCt, outputfile)
    ###LookupTable_ProbCt = ReadLookupTable(outputfile)  #Read the result from the Pre-calculation file(計算する場合と大差ないかも)


    print "Please wait for PostProbMap"
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
    if (os.path.isfile(output) == False) or (UPDATE_PostProbMap == 1):  #すでにファイルがあれば作成しない
      #PathWeightMap = PostProbMap_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #マルチCPUで高速化できるかも #CostMapProb * PostProbMap #後の処理のために, この時点ではlogにしない
      PathWeightMap = PostProbMap_nparray_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #,IndexMap) 
      
      #[TEST]計算結果を先に保存
      SaveProbMap(PathWeightMap, outputfile)
    else:
      PathWeightMap = ReadProbMap(outputfile)
      #print "already exists:", output
    print "[Done] PathWeightMap."

    PathWeightMap_origin = PathWeightMap


    #[メモリ・処理の軽減]初期位置のセルからT_horizonよりも離れた位置のセルをすべて２-dimension array から消す([(2*T_horizon)+1][(2*T_horizon)+1]の array になる)
    Bug_removal_savior = 0  #座標変換の際にバグを生まないようにするためのフラグ
    x_min = X_init_index[0] - T_horizon
    x_max = X_init_index[0] + T_horizon
    y_min = X_init_index[1] - T_horizon
    y_max = X_init_index[1] + T_horizon
    if (x_min>=0 and x_max<=map_width and y_min>=0 and y_max<=map_length) and (memory_reduction == 1):
      PathWeightMap = PathWeightMap[x_min:x_max+1, y_min:y_max+1] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
      X_init_index = [T_horizon, T_horizon]
      print "Re Initial Xt:", X_init_index
      #再度, length and width of the MAP cells
      map_length = len(PathWeightMap)
      map_width  = len(PathWeightMap[0])
    elif(memory_reduction == 0):
      print "NO memory reduction process."
      Bug_removal_savior = 1  #バグを生まない(1)
    else:
      print "[WARNING] The initial position (or init_pos +/- T_horizon) is outside the map."
      Bug_removal_savior = 1  #バグを生まない(1)
      #print X_init, X_init_index

    #計算量削減のため状態数を減らす(状態空間をone-dimension array にする⇒0の要素を除く)
    #PathWeight = np.ravel(PathWeightMap)
    PathWeight_one_NOzero = PathWeightMap[PathWeightMap!=float(0.0)]
    state_num = len(PathWeight_one_NOzero)
    print "PathWeight_one_NOzero state_num:", state_num

    #map の2-dimension array インデックスとone-dimension array の対応を保持する
    IndexMap = np.array([[(i,j) for j in xrange(map_width)] for i in xrange(map_length)])
    IndexMap_one_NOzero = IndexMap[PathWeightMap!=float(0.0)].tolist() #先にリスト型にしてしまう #実装上, np.arrayではなく2-dimension array リストにしている
    print "IndexMap_one_NOzero",len(IndexMap_one_NOzero)

    #one-dimension array 上の初期位置
    if (X_init_index in IndexMap_one_NOzero):
      X_init_index_one = IndexMap_one_NOzero.index(X_init_index)
    else:
      print "[ERROR] The initial position is not a movable position on the map."
      #print X_init, X_init_index
      X_init_index_one = 0
      exit()
    print "Initial index", X_init_index_one

    #移動先候補 index 座標のリスト(相対座標)
    MoveIndex_list = MovePosition_2D([0,0]) #.tolist()
    #MoveIndex_list = np.round(MovePosition(X_init_index)).astype(int)
    print "MoveIndex_list"

    """
    #状態遷移確率(Motion model)の計算
    print "Please wait for Transition"
    output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse.mtx" # + "_Transition_log.csv"
    if (os.path.isfile(output_transition) == False):  #すでにファイルがあれば作成しない
      #IndexMap_one_NOzero内の2-dimension array 上 index と一致した要素のみ確率1を持つようにする
      #Transition = Transition_log_jit(state_num,IndexMap_one_NOzero,MoveIndex_list)
      Transition = Transition_sparse_jit(state_num,IndexMap_one_NOzero,MoveIndex_list)

      #[TEST]計算結果を先に保存
      #SaveTransition(Transition, outputfile)
      SaveTransition_sparse(Transition, outputfile)
    else:
      Transition = ReadTransition_sparse(state_num, outputfile) #ReadTransition(state_num, outputfile)
      #print "already exists:", output_transition

    Transition_one_NOzero = Transition #[PathWeightMap!=0.0]
    print "[Done] Transition distribution."
    """

    #Viterbi Algorithmを実行
    Path_one = ViterbiPath(X_init_index_one, np.log(PathWeight_one_NOzero), state_num,IndexMap_one_NOzero,MoveIndex_list, outputname, X_init, Bug_removal_savior) #, Transition_one_NOzero)

    #one-dimension array index を2-dimension array index へ⇒ROSの座標系にする
    Path_2D_index = np.array([ IndexMap_one_NOzero[Path_one[i]] for i in xrange(len(Path_one)) ])
    if ( Bug_removal_savior == 0):
      Path_2D_index_original = Path_2D_index + np.array(X_init) - T_horizon
    else:
      Path_2D_index_original = Path_2D_index
    Path_ROS = Array_index_To_Map_coordinates(Path_2D_index_original) #ROSのパスの形式にできればなおよい

    #Path = Path_2D_index_original #Path_ROS #必要な方をPathとして返す
    print "Init:", X_init
    print "Path:\n", Path_2D_index_original
    return Path_2D_index_original, Path_ROS, PathWeightMap_origin, Path_one #, LogLikelihood_step, LogLikelihood_sum


#移動位置の候補: 現在の位置(2-dimension array index )の近傍8セル+現在位置1セル
def MovePosition_2D(Xt): 
  if (NANAME == 1):
    PostPosition_list = np.array([ [-1,-1],[-1,0],[-1,1], [0,-1],[0,0], [0,1], [1,-1],[1,0],[1,1] ])*cmd_vel + np.array(Xt)
  else:
    PostPosition_list = np.array([ [-1,0], [0,-1],[0,0], [0,1], [1,0] ])*cmd_vel + np.array(Xt)
    
    return PostPosition_list


#Viterbi Path計算用関数(参考: https://qiita.com/kkdd/items/6cbd949d03bc56e33e8e)
def update(cost, trans, emiss):
    COST = 0 #COST, INDEX = range(2)  #0,1
    arr = [c[COST]+t for c, t in zip(cost, trans)]
    max_arr = max(arr)
    #print max_arr + emiss, arr.index(max_arr)
    return max_arr + emiss, arr.index(max_arr)


def update_lite(cost, n, emiss, state_num,IndexMap_one_NOzero,MoveIndex_list,Transition):
    #Transition = np.array([approx_log_zero for j in xrange(state_num)]) #emissのindex番号に応じて, これをつくる処理を入れる
    for i in xrange(len(Transition)):
      Transition[i] = float('-inf') #approx_log_zero #-infでも計算結果に変わりはない模様

    #今, 想定している位置1セルと隣接する8セルのみの遷移を考えるようにすればよい
    #Index_2D = IndexMap_one_NOzero[n] #.tolist()
    MoveIndex_list_n = MoveIndex_list + IndexMap_one_NOzero[n] #Index_2D #絶対座標系にする
    MoveIndex_list_n_list = MoveIndex_list_n.tolist()
    #print MoveIndex_list_n_list

    count_t = 0
    for c in xrange(len(MoveIndex_list_n_list)): 
        if (MoveIndex_list_n_list[c] in IndexMap_one_NOzero):
          m = IndexMap_one_NOzero.index(MoveIndex_list_n_list[c])  #cは移動可能な状態(セル)とは限らない
          Transition[m] = 0.0 #1 #Transition probability from state to state (index of this array is not x, y of map)
          count_t += 1
          #print c, MoveIndex_list_n_list[c]
    
    #計算上おかしい場合はエラー表示を出す．
    if (count_t == 0): #遷移確率がすべて0．移動できないということを意味する．
      print "[ERROR] All transition is approx_log_zero."
    elif (count_t == 1): #遷移確率がひとつだけある．移動可能な座標が一択．（このWARNINGが出ても問題ない場合がある？）
      print "[WARNING] One transition can move only."
    #elif (count_t != 5):
    #  print count_t, MoveIndex_list_n_list
    
    #trans = Transition #np.array(Transition)
    arr = cost + Transition #trans
    #max_arr = np.max(arr)
    max_arr_index = np.argmax(arr)
    #return max_arr + emiss, np.where(arr == max_arr)[0][0] #np.argmax(arr)#arr.index(max_arr)
    return arr[max_arr_index] + emiss, max_arr_index

#def transition(m, n):
#    return [[1.0 for i in xrange(m)] for j in xrange(n)]
#def emission(n):
#    return [random.random() for j in xrange(n)]

#ViterbiPathを計算してPath(軌道)を返す
def ViterbiPath(X_init, PathWeight, state_num,IndexMap_one_NOzero,MoveIndex_list, outputname, X_init_original, Bug_removal_savior): #, Transition):
    #Path = [[0,0] for t in xrange(T_horizon)]  #各tにおけるセル番号[x,y]
    print "Start Viterbi Algorithm"

    INDEX = 1 #COST, INDEX = range(2)  #0,1
    INITIAL = (approx_log_zero, X_init)  # (cost, index) #indexに初期値のone-dimension array インデックスを入れる
    #print "Initial:",X_init

    cost = [INITIAL for i in xrange(len(PathWeight))] 
    cost[X_init] = (0.0, X_init) #初期位置は一意に与えられる(確率log(1.0))
    trellis = []

    e = PathWeight #emission(nstates[i])
    m = [i for i in xrange(len(PathWeight))] #Transition #transition(nstates[i-1], nstates[i]) #一つ前から現在への遷移
    
    #Transition = np.array([approx_log_zero for j in xrange(state_num)]) #参照渡しになってしまう
    Transition = np.array([float('-inf') for j in xrange(state_num)]) #参照渡しになってしまう

    temp = 1
    #Forward
    print "Forward"
    for i in xrange(T_horizon):  #len(nstates)): #計画区間まで1セルずつ移動していく+1+1
      #このfor文の中でiを別途インディケータとして使わないこと
      print "T:",i+1
      if (i+1 == T_restart):
        outputname_restart = outputfile + "T"+str(T_restart)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)
        trellis = ReadTrellis(outputname_restart, i+1)
        cost = trellis[-1]
      if (i+1 >= T_restart):
        #cost = [update(cost, t, f) for t, f in zip(m, e)]
        #cost = [update_sparse(cost, Transition[t], f) for t, f in zip(m, e)] #なぜか遅い
        cost_np = np.array([cost[c][0] for c in xrange(len(cost))])
        #Transition = np.array([approx_log_zero for j in xrange(state_num)]) #参照渡しになってしまう

        #cost = [update_lite(cost_np, t, e[t], state_num,IndexMap_one_NOzero,MoveIndex_list) for t in xrange(len(e))]
        cost = [update_lite(cost_np, t, f, state_num,IndexMap_one_NOzero,MoveIndex_list,Transition) for t, f in izip(m, e)] #izipの方がメモリ効率は良いが, zipとしても処理速度は変わらない
        trellis.append(cost)
        if (float('inf') in cost) or (float('-inf') in cost) or (float('nan') in cost):
            print("[ERROR] cost:", str(cost))
        #print "i", i, [(c[COST], c[INDEX]) for c in cost] #前のノードがどこだったか（どこから来たか）を記録している
        if (SAVE_T_temp == temp):
            #Backward temp
            last = [trellis[-1][j][0] for j in xrange(len(trellis[-1]))]
            path_one = [last.index(max(last))] #最終的にいらないが計算上必要⇒最後のノードの最大値インデックスを保持する形でもできるはず
            #print "last",last,"max",path

            for x in reversed(trellis):
              path_one = [x[path_one[0]][INDEX]] + path_one
              #print "x", len(x), x
            path_one = path_one[1:len(path_one)] #初期位置と処理上追加した最後の遷移を除く
            
            SavePathTemp(X_init_original, path_one, i+1, outputname, IndexMap_one_NOzero, Bug_removal_savior)
            
            ##log likelihood 
            #PathWeight (log)とpath_oneからlog likelihoodの値を再計算する
            LogLikelihood_step = np.zeros(i+1)
            LogLikelihood_sum = np.zeros(i+1)
    
            for t in range(i+1):
                LogLikelihood_step[t] = PathWeight[ path_one[t]]
                if (t == 0):
                     LogLikelihood_sum[t] = LogLikelihood_step[t]
                elif (t >= 1):
                     LogLikelihood_sum[t] = LogLikelihood_sum[t-1] + LogLikelihood_step[t]

            SaveLogLikelihood(LogLikelihood_step,0,i+1)
            SaveLogLikelihood(LogLikelihood_sum,1,i+1)

            #The moving distance of the path
            Distance = PathDistance(path_one)
    
            #Save the moving distance of the path
            SavePathDistance_temp(Distance, i+1)

            if (SAVE_Trellis == 1):
              SaveTrellis(trellis, outputname, i+1)
            temp = 0
        temp += 1

    #最後の遷移確率は一様にすればよいはず
    e_last = [0.0]
    m_last = [[0.0 for i in range(len(PathWeight))]]
    cost = [update(cost, t, f) for t, f in zip(m_last, e_last)]
    trellis.append(cost)

    #Backward
    print "Backward"
    #last = [trellis[-1][i][0] for i in xrange(len(trellis[-1]))]
    path = [0]  #[last.index(max(last))] #最終的にいらないが計算上必要⇒最後のノードの最大値インデックスを保持する形でもできるはず
    #print "last",last,"max",path

    for x in reversed(trellis):
        path = [x[path[0]][INDEX]] + path
        #print "x", len(x), x
    path = path[1:len(path)-1] #初期位置と処理上追加した最後の遷移を除く
    print 'Maximum prob path:', path
    return path

#推定されたパスを（トピックかサービスで）送る
#def SendPath(Path):

#Save the path trajectory
def SavePath(X_init, Path, Path_ROS, outputname):
    print "PathSave"
    if (SAVE_X_init == 1):
      # Save the robot initial position to the file (index)
      np.savetxt(outputname + "_X_init.csv", X_init, delimiter=",")
      # Save the robot initial position to the file (ROS)
      np.savetxt(outputname + "_X_init_ROS.csv", Array_index_To_Map_coordinates(X_init), delimiter=",")

    # Save the result to the file (index)
    np.savetxt(outputname + "_Path.csv", Path, delimiter=",")
    # Save the result to the file (ROS)
    np.savetxt(outputname + "_Path_ROS.csv", Path_ROS, delimiter=",")
    print "Save Path: " + outputname + "_Path.csv and _Path_ROS.csv"

#Save the path trajectory
def SavePathTemp(X_init, Path_one, temp, outputname, IndexMap_one_NOzero, Bug_removal_savior):
    print "PathSaveTemp"

    #one-dimension array index を2-dimension array index へ⇒ROSの座標系にする
    Path_2D_index = np.array([ IndexMap_one_NOzero[Path_one[i]] for i in xrange(len(Path_one)) ])
    if ( Bug_removal_savior == 0):
      Path_2D_index_original = Path_2D_index + np.array(X_init) - T_horizon
    else:
      Path_2D_index_original = Path_2D_index
    Path_ROS = Array_index_To_Map_coordinates(Path_2D_index_original) #

    #Path = Path_2D_index_original #Path_ROS #必要な方をPathとして返す
    # Save the result to the file (index)
    np.savetxt(outputname + "_Path" + str(temp) + ".csv", Path_2D_index_original, delimiter=",")
    # Save the result to the file (ROS)
    np.savetxt(outputname + "_Path_ROS" + str(temp) + ".csv", Path_ROS, delimiter=",")
    print "Save Path: " + outputname + "_Path" + str(temp) + ".csv and _Path_ROS" + str(temp) + ".csv"

def SaveTrellis(trellis, outputname, temp):
    print "SaveTrellis"
    # Save the result to the file 
    np.save(outputname + "_trellis" + str(temp) + ".npy", trellis) #, delimiter=",")
    print "Save trellis: " + outputname + "_trellis" + str(temp) + ".npy"

def ReadTrellis(outputname, temp):
    print "ReadTrellis"
    # Save the result to the file 
    trellis = np.load(outputname + "_trellis" + str(temp) + ".npy") #, delimiter=",")
    print "Read trellis: " + outputname + "_trellis" + str(temp) + ".npy"
    return trellis

#パス計算のために使用したLookupTable_ProbCtをファイル保存する
def SaveLookupTable(LookupTable_ProbCt, outputfile):
    # Save the result to the file 
    output = outputfile + "LookupTable_ProbCt.csv"
    np.savetxt( output, LookupTable_ProbCt, delimiter=",")
    print "Save LookupTable_ProbCt: " + output

#パス計算のために使用したLookupTable_ProbCtをファイル読み込みする
def ReadLookupTable(outputfile):
    # Read the result from the file
    output = outputfile + "LookupTable_ProbCt.csv"
    LookupTable_ProbCt = np.loadtxt(output, delimiter=",")
    print "Read LookupTable_ProbCt: " + output
    return LookupTable_ProbCt


#パス計算のために使用した確率値コストマップをファイル保存する
def SaveCostMapProb(CostMapProb, outputfile):
    # Save the result to the file 
    output = outputfile + "CostMapProb.csv"
    np.savetxt( output, CostMapProb, delimiter=",")
    print "Save CostMapProb: " + output

#Load the probability cost map used for path calculation
def ReadCostMapProb(outputfile):
    # Read the result from the file
    output = outputfile + "CostMapProb.csv"
    CostMapProb = np.loadtxt(output, delimiter=",")
    print "Read CostMapProb: " + output
    return CostMapProb

#パス計算のために使用した確率値マップを（トピックかサービスで）送る
#def SendProbMap(PathWeightMap):

#Save the probability value map used for path calculation
def SaveProbMap(PathWeightMap, outputfile):
    # Save the result to the file 
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
    np.savetxt( output, PathWeightMap, delimiter=",")
    print "Save PathWeightMap: " + output

#Load the probability value map used for path calculation
def ReadProbMap(outputfile):
    # Read the result from the file
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
    PathWeightMap = np.loadtxt(output, delimiter=",")
    print "Read PathWeightMap: " + output
    return PathWeightMap

def SaveTransition(Transition, outputfile):
    # Save the result to the file 
    output_transition = outputfile + "T"+str(T_horizon) + "_Transition_log.csv"
    #np.savetxt(outputfile + "_Transition_log.csv", Transition, delimiter=",")
    f = open( output_transition , "w")
    for i in xrange(len(Transition)):
      for j in xrange(len(Transition[i])):
        f.write(str(Transition[i][j]) + ",")
      f.write('\n')
    f.close()
    print "Save Transition: " + output_transition

def ReadTransition(state_num, outputfile):
    Transition = [[approx_log_zero for j in xrange(state_num)] for i in xrange(state_num)] 
    # Read the result from the file
    output_transition = outputfile + "T"+str(T_horizon) + "_Transition_log.csv"
    #Transition = np.loadtxt(outputfile + "_Transition_log.csv", delimiter=",")
    i = 0
    #Read the text file
    for line in open(output_transition, 'r'):
        itemList = line[:-1].split(',')
        for j in xrange(len(itemList)):
            if itemList[j] != '':
              Transition[i][j] = float(itemList[j])
        i = i + 1

    print "Read Transition: " + output_transition
    return Transition

def SaveTransition_sparse(Transition, outputfile):
    # Save the result to the file (.mtx形式)
    output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse"
    mmwrite(output_transition, Transition)

    print "Save Transition: " + output_transition

def ReadTransition_sparse(state_num, outputfile):
    #Transition = [[0 for j in xrange(state_num)] for i in xrange(state_num)] 
    # Read the result from the file
    output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse.mtx"
    Transition = mmread(output_transition).tocsr()  #.todense()

    print "Read Transition: " + output_transition
    return Transition

#Save the log likelihood for each time-step
def SaveLogLikelihood(LogLikelihood,flag,flag2):
    # Save the result to the file 
    if (flag2 == 0):
      if   (flag == 0):
        output_likelihood = outputname + "_Log_likelihood_step.csv"
      elif (flag == 1):
        output_likelihood = outputname + "_Log_likelihood_sum.csv"
    else:
      if   (flag == 0):
        output_likelihood = outputname + "_Log_likelihood_step" + str(flag2) + ".csv"
      elif (flag == 1):
        output_likelihood = outputname + "_Log_likelihood_sum" + str(flag2) + ".csv"

    np.savetxt( output_likelihood, LogLikelihood, delimiter=",")
    print "Save LogLikekihood: " + output_likelihood

#The moving distance of the pathを計算する
def PathDistance(Path):
    Distance = len(collections.Counter(Path))
    print "Path Distance is ", Distance
    return Distance

#Save the moving distance of the path
def SavePathDistance(Distance):
    # Save the result to the file 
    output = outputname + "_Distance.csv"
    np.savetxt( output, np.array([Distance]), delimiter=",")
    print "Save Distance: " + output

#Save the moving distance of the path
def SavePathDistance_temp(Distance,temp):
    # Save the result to the file 
    output = outputname + "_Distance"+str(temp)+".csv"
    np.savetxt( output, np.array([Distance]), delimiter=",")
    print "Save Distance: " + output


########################################
if __name__ == '__main__': 
    print "[START] SpCoNavi."
    #Request a folder name for learned parameters.
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")

    #Request iteration value
    iteration = sys.argv[2] #1

    #Request sample value
    sample = sys.argv[3] #0

    #Request the index number of the robot initial position
    init_position_num = sys.argv[4] #0

    #Request the file number of the speech instruction   
    speech_num = sys.argv[5] #0


    if (SAVE_time == 1):
      #Substitution of start time
      start_time = time.time()

    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" 
    print filename, iteration, sample
    outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
    outputname = outputfile + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)

    #Makedir( outputfolder + trialname )
    Makedir( outputfile )
    #Makedir( outputname )

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA = ReadParameters(iteration, sample, filename, trialname)
    W_index = THETA[1]


    if (os.path.isfile(outputfile + "CostMapProb.csv") == False):  #すでにファイルがあれば計算しない
      print "If you do not have map.csv, please run commands for cost map acquisition procedure in advance."
      ##Read the map file
      gridmap = ReadMap(outputfile)
      ##Read the cost map file
      costmap = ReadCostMap(outputfile)

      #Change the costmap to the probabilistic costmap
      CostMapProb = CostMapProb_jit(gridmap, costmap)
      #Write the probabilistic cost map file
      SaveCostMapProb(CostMapProb, outputfile)
    else:
      #Read the probabilistic cost map file
      CostMapProb = ReadCostMapProb(outputfile)

    ##Read the speech file
    #speech_file = ReadSpeech(int(speech_num))
    BoW = [Goal_Word[int(speech_num)]]
    if ( "AND" in BoW ):
      BoW = Example_AND
    elif ( "OR" in BoW ):
      BoW = Example_OR

    Otb_B = [int(W_index[i] in BoW) * N_best for i in xrange(len(W_index))]
    print "BoW:",  Otb_B

    while (sum(Otb_B) == 0):
      print("[ERROR] BoW is all zero.", W_index)
      word_temp = raw_input("Please word?>")
      Otb_B = [int(W_index[i] == word_temp) * N_best for i in xrange(len(W_index))]
      print("BoW (NEW):",  Otb_B)

    #Path-Planning
    Path, Path_ROS, PathWeightMap, Path_one = PathPlanner(Otb_B, Start_Position[int(init_position_num)], THETA, CostMapProb) #gridmap, costmap)


    if (SAVE_time == 1):
      #PP終了時刻を保持
      end_pp_time = time.time()
      time_pp = end_pp_time - start_time #end_recog_time
      fp = open( outputname + "_time_pp.txt", 'w')
      fp.write(str(time_pp)+"\n")
      fp.close()

    #The moving distance of the path
    Distance = PathDistance(Path_one)
    
    #Save the moving distance of the path
    SavePathDistance(Distance)

    #Send the path
    #SendPath(Path)
    #Save the path
    SavePath(Start_Position[int(init_position_num)], Path, Path_ROS, outputname)

    #Send the PathWeightMap
    #SendProbMap(PathWeightMap)

    #Save the PathWeightMap(PathPlanner内部で実行)
    #####SaveProbMap(PathWeightMap, outputname)
    
    #PathWeightMapとPathからlog likelihoodの値を再計算する
    LogLikelihood_step = np.zeros(T_horizon)
    LogLikelihood_sum = np.zeros(T_horizon)
    
    for t in range(T_horizon):
         #print PathWeightMap.shape, Path[t][0], Path[t][1]
         LogLikelihood_step[t] = np.log(PathWeightMap[ Path[t][0] ][ Path[t][1] ])
         if (t == 0):
             LogLikelihood_sum[t] = LogLikelihood_step[t]
         elif (t >= 1):
             LogLikelihood_sum[t] = LogLikelihood_sum[t-1] + LogLikelihood_step[t]
    
    
    #すべてのステップにおけるlog likelihoodの値を保存
    SaveLogLikelihood(LogLikelihood_step,0,0)
    
    #すべてのステップにおける累積報酬（sum log likelihood）の値を保存
    SaveLogLikelihood(LogLikelihood_sum,1,0)
    
    
    print "[END] SpCoNavi."

########################################

