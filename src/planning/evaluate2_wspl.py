#coding:utf-8
#評価用プログラム（範囲はデータセットから決まる版）　環境ごとに評価値を算出
#Evaluation metrics: W-SPL (正しい経由点かつ目的地に到達できたときのパスの長さで重み付けた成功率) <- using PL and W-SR, based on SPL
#Akira Taniguchi (2023/02/16)

##Command: 
#python3 evaluate2_wspl.py trialname method_id
#python3 evaluate2_wspl.py 3LDK_01 1

import sys
import numpy as np
#import itertools
from __init__ import *
from submodules import *
import spconavi_read_data
import spconavi_save_data

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()

def ReadDatasetWord(DataSetFolder,DATA_NUM):
      N = 0
      Otb = []
      #Read text file
      for word_data_num in range(DATA_NUM):
        f = open(DataSetFolder + word_folder + str(word_data_num) + ".txt", "r")
        line = f.read()
        itemList = line[:].split(' ')
        
        #remove <s>,<sp>,</s> and "\r", "": if its were segmented to words.
        #itemList = Ignore_SP_Tags(itemList)
        
        #Otb[sample] = Otb[sample] + [itemList]
        if (itemList != "" and itemList != "\n" and itemList != "\r" and itemList != "," and itemList != " "):
          Otb = Otb + [itemList]
          N = N + 1  #count
        
        #for j in range(len(itemList)):
        #    print "%s " % (str(itemList[j])),
        #print ""  #改行用
      
      ##For index of multinominal distribution of place names
      W_index = []
      for n in range(N):
        for j in range(len(Otb[n])):
          if ( (Otb[n][j] in W_index) == False ):
            W_index.append(Otb[n][j])
            #print str(W_index),len(W_index)
      
      #print "[",
      for i in range(len(W_index)):
        print("\""+ str(i) + ":" + str(W_index[i]) + "\",")
      #print "]"
      
      ##Vectorize: Bag-of-Words for each time-step n (=t)
      Otb_B = [ [0 for i in range(len(W_index))] for n in range(N) ]
      
      for n in range(N):
        for j in range(len(Otb[n])):
          for i in range(len(W_index)):
            if ( W_index[i] == Otb[n][j] ):
              Otb_B[n][i] += word_increment
      #print Otb_B
      
      return Otb,Otb_B
      
def ReadDatasetPosition(directory,DATA_NUM):
    all_position=[] 
    hosei = 1  # 04 is *2, 06 is -1, 10 is *1.5.
    
    ##### 座標の補正 #####
    if ("04" in directory):
      hosei *= 2
      #print "hosei",hosei
    elif ("06" in directory):
      hosei *= -1
      #print "hosei",hosei
    elif ("10" in directory):
      hosei *= 1.5
      #print "hosei",hosei
    ######################

    for i in range(DATA_NUM):
      #if  (i in test_num)==False:
      f=directory+"/position/"+repr(i)+".txt"
      position=[] #(x,y,sin,cos)
      itigyoume = 1
      for line in open(f, 'r').readlines():
          if (itigyoume == 1):
            data=line[:-1].split('	')
            #print data
            position +=[float(data[0])*(-1) + float(origin[0]*resolution)*hosei]
            position +=[float(data[1])]
            itigyoume = 0
      all_position.append(position)
    
    return np.array(all_position)


#データセットから正解の範囲を決める
def RectangularArea(filename, trialname, true_ie, mu,DATA_NUM):
    area = np.zeros((4,2)) # [[x_min, y_min], [x_max, y_max]]
    
    DataSetFolder = inputfolder_SIG + trialname
    
    
    #Otb, dataset_word = ReadDatasetWord(DataSetFolder,DATA_NUM)
    dataset_position = ReadDatasetPosition(DataSetFolder,DATA_NUM)
    
    ItC = []
    s = 0
    #正解データを読み込みIT
    for line in open(filename + "/" + trialname + "_It_1_0.csv", 'r'):
        itemList = line[:].split(',')
        for i in range(len(itemList)):
            if (itemList[i] != ''):
                ItC = ItC + [int(itemList[i])]
                s += 1
    
    X = []
    Y = []
    for i in range(DATA_NUM):
        if ( int(ItC[i]) == int(true_ie) ):
            position_ai = tools.Map_coordinates_To_Array_index(dataset_position[i])
            X.append(position_ai[0])
            Y.append(position_ai[1])
            
    #元々の学習データからガウスの平均値をずらしたものがあるので補正する
    meanX = np.mean(X) # mean of dataset positions
    meanY = np.mean(Y) # mean of dataset positions
    mu_ai = tools.Map_coordinates_To_Array_index(mu)
    disX = mu_ai[0] - meanX
    disY = mu_ai[1] - meanY
    print("mean(data,mu):",[meanX,meanY], mu_ai)
    
    minX = min(X)-margin+disX
    minY = min(Y)-margin+disY
    maxX = max(X)+margin+disX
    maxY = max(Y)+margin+disY
    minmaxX = [float(minX), float(maxX)]
    minmaxY = [float(minY), float(maxY)]

    # true_ie における 範囲+-marginを作成
    #area[0] = [minX,minY]
    #area[1] = [maxX,minY]
    #area[2] = [minX,maxY]
    #area[3] = [maxX,maxY]
    
    return minmaxX,minmaxY

#パスの中に正解の範囲に入っている座標があるかどうか
def SuccessChecker_WP(filename,trialname, true_ie, path, minmaxX_list,minmaxY_list):
    check = 0 # 0 or 1
    minmaxX,minmaxY = minmaxX,minmaxY =  minmaxX_list[true_ie],minmaxY_list[true_ie] #RectangularArea(filename,trialname, true_ie, mu,DATA_NUM) #.tolist()
    #check = int(in_rect(area,x))
    for x_list in path:
        x_list = [x_list[0],x_list[1]]
        if (minmaxX[0] <= x_list[1] <= minmaxX[1]):
            if (minmaxY[0] <= x_list[0] <= minmaxY[1]):
                check = 1
            
        
    print("X:",minmaxX, "Y:",minmaxY)    
    print("check",check)
    
    return check

#xが正解の範囲内かどうかをチェックする
def SuccessChecker(filename, trialname, true_ie, x_list, minmaxX_list,minmaxY_list):
    check = -1 # 0 or 1
    minmaxX,minmaxY =  minmaxX_list[true_ie],minmaxY_list[true_ie] #RectangularArea(filename,trialname, true_ie, mu) #.tolist()
    #check = int(in_rect(area,x))
    
    if (minmaxX[0] <= x_list[0] <= minmaxX[1]):
        if (minmaxY[0] <= x_list[1] <= minmaxY[1]):
            check = 1
        else:
            check = 0
    else:
        check = 0
        
        
    print("X:",minmaxX, "Y:",minmaxY)    
    print("check",check, x_list)
    
    return check

"""
def in_rect(rect,target):
    a = (rect[0][0], rect[0][1])
    b = (rect[1][0], rect[1][1])
    c = (rect[2][0], rect[2][1])
    d = (rect[3][0], rect[3][1])
    e = (target[0], target[1])

    # 原点から点へのベクトルを求める
    vector_a = np.array(a)
    vector_b = np.array(b)
    vector_c = np.array(c)
    vector_d = np.array(d)
    vector_e = np.array(e)

    # 点から点へのベクトルを求める
    vector_ab = vector_b - vector_a
    vector_ae = vector_e - vector_a
    vector_bc = vector_c - vector_b
    vector_be = vector_e - vector_b
    vector_cd = vector_d - vector_c
    vector_ce = vector_e - vector_c
    vector_da = vector_a - vector_d
    vector_de = vector_e - vector_d

    # 外積を求める
    vector_cross_ab_ae = np.cross(vector_ab, vector_ae)
    vector_cross_bc_be = np.cross(vector_bc, vector_be)
    vector_cross_cd_ce = np.cross(vector_cd, vector_ce)
    vector_cross_da_de = np.cross(vector_da, vector_de)

    return vector_cross_ab_ae < 0 and vector_cross_bc_be < 0 and vector_cross_cd_ce < 0 and vector_cross_da_de < 0
"""

#################################################
#Request a folder name for learned parameters.
trialname = sys.argv[1]

method_id = sys.argv[2]

##FullPath of folder
filename    = outputfolder_SIG + trialname #+ "/" 
outputfile  = filename + navigation_folder

truthfolder = outputfile + "truth_astar/"
DataSetFolder = inputfolder_SIG + trialname

Makedir( outputfile + "/evaluate2/" )

##S## ##### Ishibushi's code #####
env_para = np.genfromtxt(DataSetFolder+"/Environment_parameter.txt",dtype= None,delimiter =" ")
DATA_initial_index = int(env_para[5][1]) #Initial data num
DATA_last_index    = int(env_para[6][1]) #Last data num
DATA_NUM = DATA_last_index - DATA_initial_index + 1
##E## ##### Ishibushi's code ######

#Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
THETA = read_data.ReadParameters(1, 0, filename, trialname)
W, W_index, Mu, Sig, Pi, Phi_l, K, L = THETA

print("Evaluation metrics: W-SPL")

### 比較手法 , "dijkstra_result"
Methods = [ "spconavi_astar_min_J1", "spconavi_astar_min", "astar_result_d_gauss", "astar_result_d2_gauss", "dijkstra_result_wd" ] 
#, "viterbi_result2", "viterbi_result"
if (int(method_id) != -1) and (method_id != ""):
    Methods = [ Methods[int(method_id)] ]
print("Methods:", Methods)


#Ito# 遷移確率の低いエッジは計算しないようにするために擬似的にpsi_setting.csvを読み込む
CoonectMatricx     = [ [0.0 for atem in range(K)] for aky in range(K) ]
c=0
for line in open(filename + "/" + trialname + '_psi_'  + 'setting.csv', 'r'):
    itemList = line[:-1].split(',')
    for i in range(len(itemList)):
        if itemList[i] != "":
          CoonectMatricx[c][i] = float(itemList[i])
    c = c + 1 
    
minmaxX_list = [0.0 for i in range(K)]
minmaxY_list = [0.0 for i in range(K)]
for i in range(K):
    true_ie = i
    mu = Mu[i]
    minmaxX_list[i],minmaxY_list[i] = RectangularArea(filename,trialname, true_ie, mu,DATA_NUM) #.tolist()
    
#spconavi_error = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])

for method in range(len(Methods)):
    print("[start]", Methods[method])
    outputsubfolder = outputfile + Methods[method] + "/"
    outputname_d    = outputsubfolder

    #存在しない（計算されていない）エッジはnanになる
    Path        = [ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ]
    SR          = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    Near_SR     = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    WP_SR       = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    PL          = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    #Arrival_PL  = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    N_SPL       = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    W_SPL       = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    NW_SPL      = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    #Time        = np.array([ [ [np.nan for i in range(K)] for c in range(K) ] for d in range(K) ])
    
    
    

    #for i,j in itertools.product(range(K), range(K)):
    for i in reversed(range(K)):
      for j in reversed(range(K)):
        #print(i,j)
        if (int(CoonectMatricx[i][j]) == 0): #直接の接続関係がないとき


            for tyukan in range(K):
                if (tyukan != i) and (tyukan != j): # 中間がスタートとゴールと同じにならないようにする
        
                    # Mu[j]が対応するWordを推測 p(word | Mu[j]) = sum_c p(word | W[c]) p( j | phi_l[c][j])
                    EstimatedWord = W_index[ np.argmax( [ np.sum( W[c][word] * Phi_l[c][j] * Pi[c] for c in range(L) ) for word in range(len(W_index)) ] ) ] 

                    #Mu[i]をスタートとする
                    start = tools.Map_coordinates_To_Array_index(Mu[i]).tolist()

                    true_ie = j
                    #Mu[j]が推定した単語をゴールとする
                    speech_num = Goal_Word.index(EstimatedWord)
                    print(i,j, start, str(EstimatedWord), speech_num)
                
                
                    # Mu[tyukan]が対応するWordを推測 p(word | Mu[tyukan]) = sum_c p(word | W[c]) p( tyukan | phi_l[c][tyukan])
                    EstimatedWord_tyukan = W_index[ np.argmax( [ np.sum( W[c][word] * Phi_l[c][tyukan] * Pi[c] for c in range(L) ) for word in range(len(W_index)) ] ) ] 
                    tyukan_num = Goal_Word.index(EstimatedWord_tyukan)                
                    print("tyukan:", str(EstimatedWord_tyukan), tyukan_num)
                    waypoint_word = tyukan_num
                    
                    
                    if (speech_num != tyukan_num): #中間の単語インデックスとゴールの単語インデックスが同じにならないようにする
                
                        # output = outputfile + "evaluate2/" + Methods[method] + "_" + str(i) + "_" + str(j) + "_" #+ str(tyukan) + "_"
                        # SR[i][j]        = np.loadtxt(output+"all_SR.csv", delimiter=",")
                        # Near_SR[i][j]   = np.loadtxt(output+"all_Near_SR.csv", delimiter=",")
                        # WP_SR[i][j]     = np.loadtxt(output+"all_WP_SR.csv", delimiter=",")
                        # PL[i][j]        = np.loadtxt(output+"all_PL.csv", delimiter=",")

                        # [ "spconavi_viterbi", "spconavi_astar_min_J1", "spconavi_astar_min", "astar_result_d_gauss", "astar_result_d2_gauss", "dijkstra_result_wd", "dijkstra_result", "viterbi_result" ]
                        if (Methods[method] == "spconavi_viterbi"):
                            outputname = outputsubfolder + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)+"/"
                        elif (Methods[method] == "spconavi_astar_min_J1"):
                            Sampling_J = 1
                            outputname = outputsubfolder + "J"+str(Sampling_J)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(tuple(start))+"H"+str(waypoint_word)+"G"+str(speech_num)+"/"
                        elif (Methods[method] == "spconavi_astar_min"):
                            Sampling_J = 10
                            outputname = outputsubfolder + "J"+str(Sampling_J)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(tuple(start))+"H"+str(waypoint_word)+"G"+str(speech_num)+"/"
                        elif (Methods[method] == "astar_result_d_gauss"):
                            waypoint_word = -1
                            outputname = outputname_d + "Astar_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(tuple(start))+"H"+str(waypoint_word)+"G"+str(speech_num)
                        elif (Methods[method] == "astar_result_d2_gauss"):
                            waypoint_word = -1
                            outputname = outputname_d + "Astar_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(tuple(start))+"H"+str(waypoint_word)+"G"+str(speech_num)
                        elif (Methods[method] == "dijkstra_result_wd"):
                            outputname = outputname_d + "Dijkstra_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)
                        elif (Methods[method] == "dijkstra_result"):
                            outputname = outputname_d + "Dijkstra_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)
                        elif (Methods[method] == "viterbi_result_wd"):
                            outputname = outputname_d + "Viterbi_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)
                        elif (Methods[method] == "viterbi_result"):
                            outputname = outputname_d + "Viterbi_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)
                        
                        outputname_fin = outputname
                        if ( (Methods[method] != "spconavi_viterbi") 
                        and (Methods[method] != "spconavi_astar_min_J1")
                        and (Methods[method] != "spconavi_astar_min") ):
                            outputname_fin = outputname + "_fin"
                            Path[i][j][tyukan] = read_data.ReadPathROS(outputname_fin) #Path_ROS.csv
                            # Pathが無い場合は、スタート位置とする
                            if (len(Path[i][j][tyukan]) == 0):
                                start_inv = [start[1],start[0]]
                                Path[i][j][tyukan] = [ start_inv ]
                        else:
                            Path[i][j][tyukan] = read_data.ReadPath(outputname) #Path.csv
                            
                        #if (Methods[method] == "spconavi_viterbi"):
                        #    start_inv = [start[1],start[0]]
                        #    print(start_inv, "=?", Path[i][j][tyukan][0])
                        #    ## SpCoNavi Viterbiでエラーだったもの（Pathの初めが初期値と異なっている場合）を除く
                        #    ## 初期値から１マスずれは許容する
                        #    if (start_inv[0]-2 <= Path[i][j][tyukan][0][0] <= start_inv[0]+2) and (start_inv[1]-2 <= Path[i][j][tyukan][0][1] <= start_inv[1]+2):
                        #        spconavi_error[i][j][tyukan] = 1
                
                
                        ### 計算済みの評価値を読み込む(PL, Time)
                        PL[i][j][tyukan]   = read_data.ReadPathDistance(outputname_fin)
                        #Time[i][j][tyukan] = read_data.ReadTime(outputname)


                        ### 評価値を計算する(SR, Near-SR, WP-SR, Arrival-PL, SPL)
                        ## SR (寝室はどこいってもOK, 3LDK_07のみ２つの廊下もOK)
                        if (len(Path[i][j][tyukan]) == 0):
                                start_inv = [start[1],start[0]]
                                Path[i][j][tyukan] = [ start_inv ]
                                print("No path",Path[i][j][tyukan])
                        x = Path[i][j][tyukan][-1] #Pathの最終地点
                        #if (len(np.array(x).tolist()) == 1):
                        #    x_inv = [Path[i][j][tyukan][1],Path[i][j][tyukan][0]]
                        #else:
                        print("x",x)
                        x_inv = [x[1],x[0]]
                        
                        near = []
                        
                        # Read PL_truth
                        Mu_i_tuple = tuple(tools.Map_coordinates_To_Array_index(Mu[i])) #[0],Mu[i][1])
                        outputtruth = truthfolder + "S"+str(Mu_i_tuple)+"G"+str(j)+"/"
                        PL_truth = read_data.ReadPathDistance(outputtruth)

                        Mu_tyukan_tuple = tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])) #[0],Mu[i][1])
                        outputtruth_tyukan = truthfolder + "S"+str(Mu_tyukan_tuple)+"G"+str(j)+"/"
                        #outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(j)+"/"
                        PL_truth = PL_truth + read_data.ReadPathDistance(outputtruth_tyukan)

                        ## WP-SR (寝室のみを例外的に処理) （実験１では不要）            
                        WP_SR[i][j][tyukan] = SuccessChecker_WP(filename,trialname, tyukan, Path[i][j][tyukan], minmaxX_list,minmaxY_list) # true_ie = j
                        
                        if (trialname == "3LDK_01"):
                            if (tyukan == 7) or (tyukan == 8) or (tyukan == 9):
                                wp7 = SuccessChecker_WP(filename,trialname, 7, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp8 = SuccessChecker_WP(filename,trialname, 8, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp9 = SuccessChecker_WP(filename,trialname, 9, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                WP_SR[i][j][tyukan] = max([wp7,wp8,wp9])

                                # Read PL_truth (tyukan)
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(7)+"/")
                                PL_truth8 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(8)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(9)+"/")
                                PL_truth = min([PL_truth7,PL_truth8,PL_truth9])
                                tyukan_kouho = [7,8,9]
                                tyukan_id = tyukan_kouho[np.argmin([PL_truth7,PL_truth8,PL_truth9])]
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan_id])))+"G"+str(j)+"/"
                                PL_truth = PL_truth + read_data.ReadPathDistance(outputtruth_tyukan)

                            if (true_ie == 7) or (true_ie == 8) or (true_ie == 9):
                                sr7 = SuccessChecker(filename,trialname, 7, x_inv, minmaxX_list,minmaxY_list)
                                sr8 = SuccessChecker(filename,trialname, 8, x_inv, minmaxX_list,minmaxY_list)
                                sr9 = SuccessChecker(filename,trialname, 9, x_inv, minmaxX_list,minmaxY_list)
                                SR[i][j][tyukan] = max([sr7,sr8,sr9])


                                # Read PL_truth (goal)
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(tyukan)+"/"
                                PL_truth = read_data.ReadPathDistance(outputtruth_tyukan)
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(7)+"/")
                                PL_truth8 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(8)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(9)+"/")
                                PL_truth = PL_truth + min([PL_truth7,PL_truth8,PL_truth9])
                                
                                # start からのNear goalを指定
                                if   (i == 0): near = [sr9]
                                elif (i == 1): near = [sr9]
                                elif (i == 2): near = [sr9,sr8]
                                elif (i == 3): near = [sr9,sr8]
                                elif (i == 4): near = [sr7]
                                elif (i == 5): near = [sr7,sr8]
                                elif (i == 6): near = [sr7]
                                elif (i == 7): near = [sr7]
                                elif (i == 8): near = [sr8]
                                elif (i == 9): near = [sr9]
                                elif (i ==10): near = [sr9,sr8]
                            else:
                                SR[i][j][tyukan] = SuccessChecker(filename,trialname, true_ie, x_inv, minmaxX_list,minmaxY_list)
                        elif (trialname == "3LDK_05"):
                            if (tyukan == 7) or (tyukan == 8) or (tyukan == 9):
                                wp7 = SuccessChecker_WP(filename,trialname, 7, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp8 = SuccessChecker_WP(filename,trialname, 8, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp9 = SuccessChecker_WP(filename,trialname, 9, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                WP_SR[i][j][tyukan] = max([wp7,wp8,wp9])
                            
                                # Read PL_truth (tyukan)
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(7)+"/")
                                PL_truth8 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(8)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(9)+"/")
                                PL_truth = min([PL_truth7,PL_truth8,PL_truth9])
                                tyukan_kouho = [7,8,9]
                                tyukan_id = tyukan_kouho[np.argmin([PL_truth7,PL_truth8,PL_truth9])]
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan_id])))+"G"+str(j)+"/"
                                #outputtruth_tyukan = truthfolder + "S"+str(tyukan_kouho[np.argmin([PL_truth7,PL_truth8,PL_truth9])])+"G"+str(j)+"/"
                                PL_truth = PL_truth + read_data.ReadPathDistance(outputtruth_tyukan)

                            if (true_ie == 7) or (true_ie == 8) or (true_ie == 9):
                                sr7 = SuccessChecker(filename,trialname, 7, x_inv, minmaxX_list,minmaxY_list)
                                sr8 = SuccessChecker(filename,trialname, 8, x_inv, minmaxX_list,minmaxY_list)
                                sr9 = SuccessChecker(filename,trialname, 9, x_inv, minmaxX_list,minmaxY_list)
                                SR[i][j][tyukan] = max([sr7,sr8,sr9])
                                
                                # Read PL_truth (goal)
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(tyukan)+"/"
                                PL_truth = read_data.ReadPathDistance(outputtruth_tyukan)
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(7)+"/")
                                PL_truth8 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(8)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(9)+"/")
                                PL_truth = PL_truth + min([PL_truth7,PL_truth8,PL_truth9])
                                
                                # start からのNear goalを指定
                                if   (i == 0): near = [sr9,sr8]
                                elif (i == 1): near = [sr9]
                                elif (i == 2): near = [sr9,sr8]
                                elif (i == 3): near = [sr8]
                                elif (i == 4): near = [sr8,sr9]
                                elif (i == 5): near = [sr7]
                                elif (i == 6): near = [sr7]
                                elif (i == 7): near = [sr7]
                                elif (i == 8): near = [sr8]
                                elif (i == 9): near = [sr9]
                                elif (i ==10): near = [sr7]
                            else:
                                SR[i][j][tyukan] = SuccessChecker(filename,trialname, true_ie, x_inv, minmaxX_list,minmaxY_list)
                        elif (trialname == "3LDK_06"):
                            if (tyukan == 3) or (tyukan == 5) or (tyukan == 7):
                                wp3 = SuccessChecker_WP(filename,trialname, 3, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp5 = SuccessChecker_WP(filename,trialname, 5, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp7 = SuccessChecker_WP(filename,trialname, 7, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                WP_SR[i][j][tyukan] = max([wp3,wp5,wp7])
                            
                                # Read PL_truth (tyukan)
                                PL_truth3 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(3)+"/")
                                PL_truth5 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(5)+"/")
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(7)+"/")
                                PL_truth = min([PL_truth3,PL_truth5,PL_truth7])
                                tyukan_kouho = [3,5,7]
                                tyukan_id = tyukan_kouho[np.argmin([PL_truth3,PL_truth5,PL_truth7])]
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan_id])))+"G"+str(j)+"/"
                                
                                #outputtruth_tyukan = truthfolder + "S"+str(tyukan_kouho[np.argmin([PL_truth3,PL_truth5,PL_truth7])])+"G"+str(j)+"/"
                                PL_truth = PL_truth + read_data.ReadPathDistance(outputtruth_tyukan)

                            if (true_ie == 3) or (true_ie == 5) or (true_ie == 7):
                                sr3 = SuccessChecker(filename,trialname, 3, x_inv, minmaxX_list,minmaxY_list)
                                sr5 = SuccessChecker(filename,trialname, 5, x_inv, minmaxX_list,minmaxY_list)
                                sr7 = SuccessChecker(filename,trialname, 7, x_inv, minmaxX_list,minmaxY_list)
                                SR[i][j][tyukan] = max([sr3,sr5,sr7])
                                

                                # Read PL_truth (goal)
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(tyukan)+"/"
                                PL_truth = read_data.ReadPathDistance(outputtruth_tyukan)
                                PL_truth3 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(3)+"/")
                                PL_truth5 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(5)+"/")
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(7)+"/")
                                PL_truth = PL_truth + min([PL_truth3,PL_truth5,PL_truth7])

                                # start からのNear goalを指定
                                if   (i == 0): near = [sr7]
                                elif (i == 1): near = [sr7]
                                elif (i == 2): near = [sr3,sr5]
                                elif (i == 3): near = [sr3]
                                elif (i == 4): near = [sr3]
                                elif (i == 5): near = [sr5]
                                elif (i == 6): near = [sr7,sr5]
                                elif (i == 7): near = [sr7]
                                elif (i == 8): near = [sr7]
                                elif (i == 9): near = [sr7]
                                elif (i ==10): near = [sr7]
                            else:
                                SR[i][j][tyukan] = SuccessChecker(filename,trialname, true_ie, x_inv, minmaxX_list,minmaxY_list)
                        elif (trialname == "3LDK_07"):
                            if (tyukan == 6) or (tyukan == 7) or (tyukan == 9):
                                wp6 = SuccessChecker_WP(filename,trialname, 6, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp7 = SuccessChecker_WP(filename,trialname, 7, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp9 = SuccessChecker_WP(filename,trialname, 9, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                WP_SR[i][j][tyukan] = max([wp6,wp7,wp9])
                             
                                # Read PL_truth (tyukan)
                                PL_truth6 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(6)+"/")
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(7)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(9)+"/")
                                PL_truth = min([PL_truth6,PL_truth7,PL_truth9])
                                tyukan_kouho = [6,7,9]
                                tyukan_id = tyukan_kouho[np.argmin([PL_truth6,PL_truth7,PL_truth9])]
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan_id])))+"G"+str(j)+"/"
                                
                                #outputtruth_tyukan = truthfolder + "S"+str(tyukan_kouho[np.argmin([PL_truth6,PL_truth7,PL_truth9])])+"G"+str(j)+"/"
                                PL_truth = PL_truth + read_data.ReadPathDistance(outputtruth_tyukan)
                            elif (tyukan == 2) or (tyukan == 10):
                                wp2 = SuccessChecker_WP(filename,trialname, 2, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp10 = SuccessChecker_WP(filename,trialname, 10, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                WP_SR[i][j][tyukan] = max([wp2,wp10])

                                # Read PL_truth (tyukan)
                                PL_truth2 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(2)+"/")
                                PL_truth10 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(10)+"/")
                                PL_truth = min([PL_truth2,PL_truth10])
                                tyukan_kouho = [2,10]
                                tyukan_id = tyukan_kouho[np.argmin([PL_truth2,PL_truth10])]
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan_id])))+"G"+str(j)+"/"
                                
                                #outputtruth_tyukan = truthfolder + "S"+str(tyukan_kouho[np.argmin([PL_truth2,PL_truth10])])+"G"+str(j)+"/"
                                PL_truth = PL_truth + read_data.ReadPathDistance(outputtruth_tyukan)

                            if (true_ie == 6) or (true_ie == 7) or (true_ie == 9):
                                sr6 = SuccessChecker(filename,trialname, 6, x_inv, minmaxX_list,minmaxY_list)
                                sr7 = SuccessChecker(filename,trialname, 7, x_inv, minmaxX_list,minmaxY_list)
                                sr9 = SuccessChecker(filename,trialname, 9, x_inv, minmaxX_list,minmaxY_list)
                                SR[i][j][tyukan] = max([sr6,sr7,sr9])
                               

                                # Read PL_truth (goal)
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(tyukan)+"/"
                                PL_truth = read_data.ReadPathDistance(outputtruth_tyukan)
                                PL_truth6 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(6)+"/")
                                PL_truth7 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(7)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(9)+"/")
                                PL_truth = PL_truth + min([PL_truth6,PL_truth7,PL_truth9])
                                
                                # start からのNear goalを指定
                                if   (i == 0): near = [sr6]
                                elif (i == 1): near = [sr6]
                                elif (i == 2): near = [sr6,sr7]
                                elif (i == 3): near = [sr6]
                                elif (i == 4): near = [sr6]
                                elif (i == 5): near = [sr6]
                                elif (i == 6): near = [sr6]
                                elif (i == 7): near = [sr7]
                                elif (i == 8): near = [sr7]
                                elif (i == 9): near = [sr9]
                                elif (i ==10): near = [sr9]
                            elif ((true_ie == 2) or (true_ie == 10)) and (speech_num == 9): #単語が"廊下"のとき
                                sr2 = SuccessChecker(filename,trialname, 2, x_inv, minmaxX_list,minmaxY_list)
                                sr10 = SuccessChecker(filename,trialname, 10, x_inv, minmaxX_list,minmaxY_list)
                                SR[i][j][tyukan] = max([sr2,sr10])
                                
                               

                                # Read PL_truth (goal)
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(tyukan)+"/"
                                PL_truth = read_data.ReadPathDistance(outputtruth_tyukan)
                                PL_truth2 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(2)+"/")
                                PL_truth10 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(10)+"/")
                                PL_truth = PL_truth + min([PL_truth2,PL_truth10])
                                
                                # start からのNear goalを指定
                                if   (i == 0): near = [sr2,sr10]
                                elif (i == 1): near = [sr2]
                                elif (i == 2): near = [sr2]
                                elif (i == 3): near = [sr2]
                                elif (i == 4): near = [sr2]
                                elif (i == 5): near = [sr2]
                                elif (i == 6): near = [sr2]
                                elif (i == 7): near = [sr2]
                                elif (i == 8): near = [sr10]
                                elif (i == 9): near = [sr10]
                                elif (i ==10): near = [sr10]
                            else:
                                SR[i][j][tyukan] = SuccessChecker(filename,trialname, true_ie, x_inv, minmaxX_list,minmaxY_list)
                        elif (trialname == "3LDK_09"):
                            if (tyukan == 1) or (tyukan == 3) or (tyukan == 9):
                                wp1 = SuccessChecker_WP(filename,trialname, 1, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp3 = SuccessChecker_WP(filename,trialname, 3, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                wp9 = SuccessChecker_WP(filename,trialname, 9, Path[i][j][tyukan], minmaxX_list,minmaxY_list)
                                WP_SR[i][j][tyukan] = max([wp1,wp3,wp9])

                                # Read PL_truth (tyukan)
                                PL_truth1 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(1)+"/")
                                PL_truth3 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(3)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(9)+"/")
                                PL_truth = min([PL_truth1,PL_truth3,PL_truth9])
                                tyukan_kouho = [1,3,9]
                                tyukan_id = tyukan_kouho[np.argmin([PL_truth1,PL_truth3,PL_truth9])]
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan_id])))+"G"+str(j)+"/"
                                
                                #outputtruth_tyukan = truthfolder + "S"+str(tyukan_kouho[np.argmin([PL_truth1,PL_truth3,PL_truth9])])+"G"+str(j)+"/"
                                PL_truth = PL_truth + read_data.ReadPathDistance(outputtruth_tyukan)

                            if (true_ie == 1) or (true_ie == 3) or (true_ie == 9):
                                sr1 = SuccessChecker(filename,trialname, 1, x_inv, minmaxX_list,minmaxY_list)
                                sr3 = SuccessChecker(filename,trialname, 3, x_inv, minmaxX_list,minmaxY_list)
                                sr9 = SuccessChecker(filename,trialname, 9, x_inv, minmaxX_list,minmaxY_list)
                                SR[i][j][tyukan] = max([sr1,sr3,sr9])
                                
                                # Read PL_truth (goal)
                                outputtruth_tyukan = truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[i])))+"G"+str(tyukan)+"/"
                                PL_truth = read_data.ReadPathDistance(outputtruth_tyukan)
                                PL_truth1 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(1)+"/")
                                PL_truth3 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(3)+"/")
                                PL_truth9 = read_data.ReadPathDistance(truthfolder + "S"+str(tuple(tools.Map_coordinates_To_Array_index(Mu[tyukan])))+"G"+str(9)+"/")
                                PL_truth = PL_truth + min([PL_truth1,PL_truth3,PL_truth9])

                                # start からのNear goalを指定
                                if   (i == 0): near = [sr9,sr1]
                                elif (i == 1): near = [sr1]
                                elif (i == 2): near = [sr9,sr1]
                                elif (i == 3): near = [sr3]
                                elif (i == 4): near = [sr3]
                                elif (i == 5): near = [sr1,sr9,sr3]
                                elif (i == 6): near = [sr9]
                                elif (i == 7): near = [sr9]
                                elif (i == 8): near = [sr1,sr9,sr3]
                                elif (i == 9): near = [sr9]
                                elif (i ==10): near = [sr3]
                            else:
                                SR[i][j][tyukan] = SuccessChecker(filename,trialname, true_ie, x_inv, minmaxX_list,minmaxY_list)
                                

                        
                        ## Near-SR (寝室のみを例外的に処理)
                        if (near == []):
                            Near_SR[i][j][tyukan] = SR[i][j][tyukan]
                        else:
                            Near_SR[i][j][tyukan] = max(near)
                        
                        
                        ## Arrival-PL
                        # Arrival_PL[i][j][tyukan] = SR[i][j][tyukan] * PL[i][j][tyukan]
                        # if (int(SR[i][j][tyukan]) == 0):
                        #     Arrival_PL[i][j][tyukan] = np.nan
                        
                        
                        
                        ## SPL: SR* (PL_truth/ max(PL,PL_truth)): PL_truthは正解ゴール指定のA*
                        PL_rate =  PL_truth / float(max( PL[i][j][tyukan], PL_truth ) )
                        W_SPL[i][j][tyukan] = WP_SR[i][j][tyukan] * PL_rate
                        N_SPL[i][j][tyukan] = Near_SR[i][j][tyukan] * PL_rate
                        NW_SPL[i][j][tyukan] = WP_SR[i][j][tyukan] * Near_SR[i][j][tyukan] * PL_rate
                        
            
            output = outputfile + "/evaluate2/" + Methods[method] + "_" + str(i) + "_" + str(j) + "_" #+ str(tyukan) + "_"
            print("Save each evaluation values")
            ## 環境一つ分の全評価値をまとめて保存（SpCoNavi Viterbiでエラーだったものを含む）
            # #np.savetxt(output+"all_Path.csv", Path, delimiter=",")
            # np.savetxt(output+"all_SR.csv", SR[i][j], delimiter=",", fmt='%s')
            # np.savetxt(output+"all_Near_SR.csv", Near_SR[i][j], delimiter=",", fmt='%s')
            # np.savetxt(output+"all_WP_SR.csv", WP_SR[i][j], delimiter=",", fmt='%s')
            # np.savetxt(output+"all_PL.csv", PL[i][j], delimiter=",", fmt='%s')
            #np.savetxt(output+"all_Arrival_PL.csv", Arrival_PL[i][j], delimiter=",", fmt='%s')
            np.savetxt(output+"all_W_SPL.csv", W_SPL[i][j], delimiter=",", fmt='%s')
            np.savetxt(output+"all_N_SPL.csv", N_SPL[i][j], delimiter=",", fmt='%s')
            np.savetxt(output+"all_NW_SPL.csv", NW_SPL[i][j], delimiter=",", fmt='%s')
            #np.savetxt(output+"all_Time.csv", Time[i][j], delimiter=",", fmt='%s')
    
    #if (Methods[method] == "spconavi_viterbi"):
    #    np.savetxt(output+"spconavi_error.csv", spconavi_error, delimiter=",", fmt='%s')

    
    ### 評価値の平均を取る（SpCoNavi Viterbiでエラーだったものを含む場合の平均）
    # mean_SR = np.nanmean(SR) #nanを除いた平均
    # std_SR  = np.nanstd(SR)  #nanを除いた標準偏差
    
    # mean_Near_SR = np.nanmean(Near_SR) #nanを除いた平均
    # std_Near_SR  = np.nanstd(Near_SR)  #nanを除いた標準偏差
    
    # mean_WP_SR = np.nanmean(WP_SR) #nanを除いた平均
    # std_WP_SR  = np.nanstd(WP_SR)  #nanを除いた標準偏差
    
    # mean_PL = np.nanmean(PL) #nanを除いた平均
    # std_PL  = np.nanstd(PL)  #nanを除いた標準偏差
    
    # mean_Arrival_PL = np.nanmean(Arrival_PL) #nanを除いた平均
    # std_Arrival_PL  = np.nanstd(Arrival_PL)  #nanを除いた標準偏差
    
    mean_W_SPL = np.nanmean(W_SPL) #nanを除いた平均
    std_W_SPL  = np.nanstd(W_SPL)  #nanを除いた標準偏差

    mean_N_SPL = np.nanmean(N_SPL) #nanを除いた平均
    std_N_SPL  = np.nanstd(N_SPL)  #nanを除いた標準偏差

    mean_NW_SPL = np.nanmean(NW_SPL) #nanを除いた平均
    std_NW_SPL  = np.nanstd(NW_SPL)  #nanを除いた標準偏差

    # mean_Time = np.nanmean(Time) #nanを除いた平均
    # std_Time  = np.nanstd(Time)  #nanを除いた標準偏差
    
    Evaluate_all = np.array([
        ["W_SPL", "N_SPL", "NW_SPL"],
        [mean_W_SPL, mean_N_SPL, mean_NW_SPL],
        [std_W_SPL, std_N_SPL, std_NW_SPL] 
                   ])
    
    ### 評価値の平均を保存する
    output2 = outputfile + "/evaluate2/" + Methods[method] + "_"
    np.savetxt(output2+"all_Evaluate_new.csv", Evaluate_all, delimiter=",", fmt='%s')
    print(Evaluate_all)
    #print("[end]", Methods[method])
    
    
    #count_nan = np.count_nonzero(np.isnan(SR))
    #print(count_nan)
    

    print("[end]", Methods[method])
