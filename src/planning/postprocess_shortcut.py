#!/usr/bin/env python
#coding:utf-8

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import spconavi_read_data
import spconavi_save_data
from __init__ import *
from submodules import *


tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()


#################################################
if __name__ == '__main__': 
    print("[START] Path Smoothing (Post Process).")
    
    #Request iteration value
    iteration = sys.argv[2] #1

    #Request sample value
    sample = sys.argv[3] #0

    #Request the index number of the robot initial position
    init_position_num = int(sys.argv[4]) #位置分布インデックス(-1で座標直接指定)

    #Request the file number of the speech instruction   
    speech_num = sys.argv[5] #0
    
    #初期値を指定する場合はTHETA読み込み以降に記載(int(sys.argv[6]) , int(sys.argv[7]))

    #中間地点の単語番号を指定 (未実装：複数指定の場合、コンマ区切りする)
    waypoint_word = sys.argv[8] # -1:中間なし


    WP_list    = waypoint_word[:].split(',')
    print("WP:", WP_list)

    if (int(WP_list[0]) == -1):
        tyukan = 0
    else:
        tyukan = 1


    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" 
    print(filename, iteration, sample)
    outputfile      = filename + navigation_folder #outputfolder + trialname + navigation_folder
    #outputsubfolder = outputfile + "astar_node_gauss/"
    outputname_d    = outputfile + "dijkstra_result_wd/"

    #Makedir( outputname_d )

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    #THETA   = read_data.ReadParameters(iteration, sample, filename, trialname)
    #W, W_index, Mu, Sig, pi, phi_l, K, L = THETA
    #psi   = read_data.ReadPsi(iteration, sample, filename, trialname)
    
    ### マップを読み込む
    gridmap = read_data.ReadMap(outputfile)
    map_length, map_width = gridmap.shape
    
    #CostMapProb = read_data.ReadCostMapProb(outputfile)
    #CostMap = read_data.ReadCostMap(outputfile)
    #CostMapProb = (100.0 - CostMap)/100


    #描画
    plt.imshow(gridmap + (40+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100, interpolation='none') #, vmin = 0.0, vmax = 1.0)
    plt.xticks(rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.tick_params(axis='y', which='major', labelsize=8)
    #plt.xlim([380,800])             #x軸の範囲
    #plt.ylim([180,510])             #y軸の範囲
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)
    #plt.xticks(np.arange(width), np.arange(width))
    #plt.yticks(np.arange(height), np.arange(height))
    plt.gca().set_aspect('equal')

 
    

    # 初期位置の設定
    if (init_position_num == -1): #初期値を指定する場合
        start = [int(sys.argv[6]) , int(sys.argv[7])]
        start_inv = [start[1], start[0]]
        print("Start:", start)
    else: #初期値は__init__.pyのリストから選択する場合
        start_inv = Start_Position[init_position_num]
        start = [start_inv[1], start_inv[0]]
    

    outputname      = outputname_d + "Dijkstra_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)

    
    ### パスを読み込む



    if (SAVE_time == 1):
        #開始時刻を保持
        start_time = time.time()

    ### ある区間（パス上の2地点）をとって、その間を（A*or）平均化してスムージング（中継点の前後？）
    ### 中継点の前後[-5,5]であれば効果的⇒パス座標列から中継点座標を探す必要がある（保留）

    ## 何度か繰り返す


        ## パスから（ランダムに？）2点取り出す



        ## 2点を直線で結ぶような部分経路を生成



        ## 部分経路がマップの障害物に被れば不採用、被らなければ採用




    if (SAVE_time == 1):
        #PP終了時刻を保持
        end_pp_time = time.time()
        time_pp = end_pp_time - start_time #end_recog_time
        fp = open( outputname + "_time_pp.txt", 'w')
        fp.write(str(time_pp)+"\n")
        fp.close()


    ### スムージングしたパスとその距離を保存
    
    ## フォルダ名だけ変えて、ファイル内容や形式は元と同じにする


    print("[END] Path Smoothing (Post Process).")