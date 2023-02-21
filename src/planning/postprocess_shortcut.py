#!/usr/bin/env python
#coding:utf-8

##Command: 
#python postprocess_shortcut.py trialname alpha beta windowSize
#python postprocess_shortcut.py 3LDK_01 0.5 0.2 5

import sys
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from __init__ import *
from submodules import *
import spconavi_read_data
import spconavi_save_data

from scipy.signal import savgol_filter

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()

## パス平滑化用Python3コード(Window size = 1)
# new  = (1-alpha)*current + alpha*origin 
# next = new - beta * (2*new - mae - ato) = (1 - 2beta)*new + beta*mae + beta*ato
def PathSmoothing(path,alpha,beta,windowSize):
    optPath = path.copy() #元のパスをコピー

    #windowSize = 5
    torelance = 0.00001 #パスの変化量の閾値(変化量がこの値以下の時平滑化を終了)
    change = torelance
    while change >= torelance:
        change = 0 #初期化
        for ip in range(1, len(path[:, 0]) - 1): #始点と終点は固定
            prePath = optPath[ip, :] #変化量計測用
            optPath[ip, :] = optPath[ip, :] - alpha * (optPath[ip, :] - path[ip, :]) 
            for w in range(1,windowSize): #ip - windowSize, ip + windowSize):
                if ( (ip - w >= 0) and (ip + w < len(path[:, 0])) ):
                    optPath[ip, :] = optPath[ip, :] - beta * (2 * optPath[ip, :] - optPath[ip-w, :] - optPath[ip+w, :])
            change += np.linalg.norm(optPath[ip, :] - prePath)
    
    #整数に直す
    #for ip in range(1, len(path[:, 0]) - 1): #始点と終点は固定
    #    optPath
    optPath = np.array(optPath,dtype=int)
    
    return optPath

# ## パスが一直線になってしまう
# def movingAverageSmoothing2d(trajectory, windowSize):
#     # Get the size of the trajectory
#     sizeX = trajectory.shape[0]
#     sizeY = trajectory.shape[1]
#     windowSize = int(windowSize)
    
#     # Create a new array to store the smoothed trajectory
#     smoothedTrajectory = np.zeros((sizeX, sizeY))
    
#     # Loop through each point in the trajectory
#     for x in range(0, sizeX):
#         for y in range(0, sizeY):
#             # Calculate the average of the points in the window
#             sum = np.array([0,0])
#             count = 0
#             for i in range(x - windowSize, x + windowSize):
#                 for j in range(y - windowSize, y + windowSize):
#                     if (i >= 0 and i < sizeX and j >= 0 and j < sizeY):
#                         sum += trajectory[i, j]
#                         count+=1
#             avg = sum / count
#             smoothedTrajectory[x, y] = avg
    
#     return smoothedTrajectory


# ## 上手く動作しない
# def moving_average_filter(data, window_size):
#     window_size = int(window_size)
#     """Apply a simple moving average filter to the input data."""
#     window = np.ones(window_size) / window_size
#     smoothed_data = np.convolve(data, window, mode='same')
#     return smoothed_data

# ## パスの連続性が失われるためボツ
# def Savitzky_Golay_filter(path,window_size):
#     window_size = int(window_size)

#     # Extract x and y coordinates of the path
#     x = path[:, 0]
#     y = path[:, 1]

#     # Apply Savitzky-Golay filter to x and y coordinates separately
#     x_smooth = savgol_filter(x, window_length=7, polyorder=2)
#     y_smooth = savgol_filter(y, window_length=7, polyorder=2)

#     # Combine smoothed x and y coordinates into a new path
#     smooth_path = np.column_stack((x_smooth, y_smooth))
    
#     optPath = np.array(smooth_path,dtype=int)

#     return optPath

#################################################
if __name__ == '__main__': 
    print("[START] Path Smoothing (Post Process).")
    

    
    #Request a folder name for learned parameters.
    trialname = sys.argv[1]
    
    #method_id = sys.argv[2]
    
    #平準化パラメータ
    alpha = float(sys.argv[2])
    beta  = float(sys.argv[3])
    windowSize = int(sys.argv[4])
    # #Request iteration value
    # iteration = sys.argv[2] #1

    # #Request sample value
    # sample = sys.argv[3] #0

    # #Request the index number of the robot initial position
    # init_position_num = int(sys.argv[4]) #位置分布インデックス(-1で座標直接指定)

    # #Request the file number of the speech instruction   
    # speech_num = sys.argv[5] #0
    
    # #初期値を指定する場合はTHETA読み込み以降に記載(int(sys.argv[6]) , int(sys.argv[7]))

    # #中間地点の単語番号を指定 (未実装：複数指定の場合、コンマ区切りする)
    # waypoint_word = sys.argv[8] # -1:中間なし


    # WP_list    = waypoint_word[:].split(',')
    # print("WP:", WP_list)

    # if (int(WP_list[0]) == -1):
    #     tyukan = 0
    # else:
    #     tyukan = 1


    ##FullPath of folder
    filename        = outputfolder_SIG + trialname #+ "/" 
    outputfile      = filename + navigation_folder 
    
    outputname_d    = outputfile + "dijkstra_result_wd/"     # read folder
    outputname_ps   = outputfile + "dijkstra_result_wd_ps_" + str(windowSize) + "_" + str(alpha) + "_" + str(beta) + "/"  # save folder

    if not os.path.exists(outputname_ps) :
        Makedir( outputname_ps )
    
    ## 元のフォルダ内のファイルコピー
    # if not os.path.exists(outputname_ps) :
    #     shutil.rmtree(outputname_ps)
    #     shutil.copytree(outputname_d,outputname_ps)

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA = read_data.ReadParameters(1, 0, filename, trialname)
    W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA
    # psi   = read_data.ReadPsi(iteration, sample, filename, trialname)
    
    #Ito# 遷移確率の低いエッジは計算しないようにするために擬似的にpsi_setting.csvを読み込む
    CoonectMatricx     = [ [0.0 for atem in range(K)] for aky in range(K) ]
    c=0
    for line in open(filename + "/" + trialname + '_psi_'  + 'setting.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
                CoonectMatricx[c][i] = float(itemList[i])
        c = c + 1 
    
    ### マップを読み込む
    gridmap = read_data.ReadMap(outputfile)
    map_length, map_width = gridmap.shape
    
    #CostMapProb = read_data.ReadCostMapProb(outputfile)
    #CostMap = read_data.ReadCostMap(outputfile)
    #CostMapProb = (100.0 - CostMap)/100
    
    # # 初期位置の設定
    # if (init_position_num == -1): #初期値を指定する場合
    #     start = [int(sys.argv[6]) , int(sys.argv[7])]
    #     start_inv = [start[1], start[0]]
    #     print("Start:", start)
    # else: #初期値は__init__.pyのリストから選択する場合
    #     start_inv = Start_Position[init_position_num]
    #     start = [start_inv[1], start_inv[0]]
    
    
    #存在しない（計算されていない）エッジはnanになる
    # Path        = [ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ]

    for i in reversed(range(K)):
      for j in reversed(range(K)):
        #print(i,j)
        if (int(CoonectMatricx[i][j]) == 0): #直接の接続関係がないとき


            for tyukan in range(K):
                if (tyukan != i) and (tyukan != j): # 中間がスタートとゴールと同じにならないようにする
        
                    # Mu[j]が対応するWordを推測 p(word | Mu[j]) = sum_c p(word | W[c]) p( j | Phi_l[c][j])
                    EstimatedWord = W_index[ np.argmax( [ np.sum( W[c][word] * Phi_l[c][j] * Pi[c] for c in range(L) ) for word in range(len(W_index)) ] ) ] 

                    #Mu[i]をスタートとする
                    #print(i,Mu,len(Mu))
                    start = tools.Map_coordinates_To_Array_index(Mu[i]).tolist()

                    true_ie = j
                    #Mu[j]が推定した単語をゴールとする
                    speech_num = Goal_Word.index(EstimatedWord)
                    print(i,j, start, str(EstimatedWord), speech_num)
                
                
                    # Mu[tyukan]が対応するWordを推測 p(word | Mu[tyukan]) = sum_c p(word | W[c]) p( tyukan | Phi_l[c][tyukan])
                    EstimatedWord_tyukan = W_index[ np.argmax( [ np.sum( W[c][word] * Phi_l[c][tyukan] * Pi[c] for c in range(L) ) for word in range(len(W_index)) ] ) ] 
                    tyukan_num = Goal_Word.index(EstimatedWord_tyukan)                
                    print("tyukan:", str(EstimatedWord_tyukan), tyukan_num)
                    waypoint_word = tyukan_num
                    
        
                    if (speech_num != tyukan_num): #中間の単語インデックスとゴールの単語インデックスが同じにならないようにする       
                                    
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

                    

                        outputname      = outputname_d + "Dijkstra_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)
                        #print(outputname)
                        
                        ### パスを読み込む
                        Path_ROS = read_data.ReadPathROS(outputname + "_fin")
                        #print(Path_ROS)

                        #The moving distance of the path
                        Distance = read_data.ReadPathDistance(outputname + "_fin") #tools.PathDistance(list(Path_ROS))
                        print("Original Path distance is "+ str(Distance))


                        if (SAVE_time == 1):
                            #開始時刻を保持
                            start_time = time.time()
                            
                        ### Path smoothing (移動平均平滑化)
                        optPath_ROS = PathSmoothing(Path_ROS,alpha,beta,windowSize) 
                        #movingAverageSmoothing2d(Path_ROS, alpha) #Savitzky_Golay_filter(Path_ROS, alpha) #0.5,0.2
                        #print(optPath_ROS)


                        ### ある区間（パス上の2地点）をとって、その間を（A*or）平均化してスムージング（中継点の前後？）
                        ### 中継点の前後[-5,5]であれば効果的⇒パス座標列から中継点座標を探す必要がある（保留）

                        ## 何度か繰り返す
                            ## パスから（ランダムに？）2点取り出す
                            ## 2点を直線で結ぶような部分経路を生成
                            ## 部分経路がマップの障害物に被れば不採用、被らなければ採用


                        ## フォルダ名だけ変えて、ファイル内容や形式は元と同じにする
                        outputfile_ps      = outputname_ps + "Dijkstra_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)
                        print(outputfile_ps)

                        if (SAVE_time == 1):
                            #PP終了時刻を保持
                            end_pp_time = time.time()
                            time_pp = end_pp_time - start_time #end_recog_time
                            fp = open( outputfile_ps + "_time_pathsmooting.txt", 'w')
                            fp.write(str(time_pp)+"\n")
                            fp.close()



                        ### スムージングしたパスとその距離を保存    
                        disp = [ tools.PathDistance([tuple(e) for e in optPath_ROS]) ] # tupple in list  #len(Path_ROS)
                        print("Smoothed Path distance is "+ str(disp))
                        
                        if(disp > Distance):
                            optPath_ROS = Path_ROS
                            disp        = [Distance]
                        
                        for ip in range(len(optPath_ROS)):
                            plt.plot(optPath_ROS[ip][1], optPath_ROS[ip][0], "s", color="tab:red", markersize=1)
                        plt.savefig(outputfile_ps + '_Path.pdf', dpi=300)#, transparent=True
                        plt.savefig(outputfile_ps + '_Path.png', dpi=300)#, transparent=True
                        
                        np.savetxt(outputfile_ps + "_fin_Path_ROS.csv", optPath_ROS, delimiter=",")
                        np.savetxt(outputfile_ps + "_fin_Distance.csv", disp, delimiter=",")
                        plt.clf()
                        
    print("[END] Path Smoothing (Post Process).")
    