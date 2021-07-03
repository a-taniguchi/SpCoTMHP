#!/usr/bin/env python
#coding:utf-8
import os
import collections
import spconavi_read_data
import spconavi_save_data
from scipy.stats import multinomial
from __init__ import *
from submodules import *
from itertools import izip

read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()

class PathPlanner:

    #Global path estimation by dynamic programming (calculation of SpCoNavi)
    def PathPlanner(self, S_Nbest, X_init, THETA, CostMapProb, outputfile, speech_num, outputname): #gridmap, costmap):
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
            PathWeightMap = read_data.PostProbMap_nparray_jit(CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K) #,IndexMap) 
        
            #[TEST]計算結果を先に保存
            save_data.SaveProbMap(PathWeightMap, outputfile, speech_num)
        else:
            PathWeightMap = read_data.ReadProbMap(outputfile)
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
        MoveIndex_list = self.MovePosition_2D([0,0]) #.tolist()
        #MoveIndex_list = np.round(MovePosition(X_init_index)).astype(int)
        print "MoveIndex_list"

        #Viterbi Algorithmを実行
        Path_one = self.ViterbiPath(X_init_index_one, np.log(PathWeight_one_NOzero), state_num,IndexMap_one_NOzero,MoveIndex_list, outputname, X_init, Bug_removal_savior) #, Transition_one_NOzero)

        #one-dimension array index を2-dimension array index へ⇒ROSの座標系にする
        Path_2D_index = np.array([ IndexMap_one_NOzero[Path_one[i]] for i in xrange(len(Path_one)) ])
        if ( Bug_removal_savior == 0):
            Path_2D_index_original = Path_2D_index + np.array(X_init) - T_horizon
        else:
            Path_2D_index_original = Path_2D_index
        Path_ROS = read_data.Array_index_To_Map_coordinates(Path_2D_index_original) #ROSのパスの形式にできればなおよい

        #Path = Path_2D_index_original #Path_ROS #必要な方をPathとして返す
        print "Init:", X_init
        print "Path:\n", Path_2D_index_original
        return Path_2D_index_original, Path_ROS, PathWeightMap_origin, Path_one #, LogLikelihood_step, LogLikelihood_sum

    
    #移動位置の候補: 現在の位置(2-dimension array index )の近傍8セル+現在位置1セル
    def MovePosition_2D(self, Xt): 
        if (NANAME == 1):
            PostPosition_list = np.array([ [-1,-1],[-1,0],[-1,1], [0,-1],[0,0], [0,1], [1,-1],[1,0],[1,1] ])*cmd_vel + np.array(Xt)
        else:
            PostPosition_list = np.array([ [-1,0], [0,-1],[0,0], [0,1], [1,0] ])*cmd_vel + np.array(Xt)
        
            return PostPosition_list
    

    #Viterbi Path計算用関数(参考: https://qiita.com/kkdd/items/6cbd949d03bc56e33e8e)
    def update(self, cost, trans, emiss):
        COST = 0 #COST, INDEX = range(2)  #0,1
        arr = [c[COST]+t for c, t in zip(cost, trans)]
        max_arr = max(arr)
        #print max_arr + emiss, arr.index(max_arr)
        return max_arr + emiss, arr.index(max_arr)


    def update_lite(self, cost, n, emiss, state_num,IndexMap_one_NOzero,MoveIndex_list,Transition):
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
        #    print count_t, MoveIndex_list_n_list
        
        #trans = Transition #np.array(Transition)
        arr = cost + Transition #trans
        #max_arr = np.max(arr)
        max_arr_index = np.argmax(arr)
        #return max_arr + emiss, np.where(arr == max_arr)[0][0] #np.argmax(arr)#arr.index(max_arr)
        return arr[max_arr_index] + emiss, max_arr_index

    #def transition(self, m, n):
    #    return [[1.0 for i in xrange(m)] for j in xrange(n)]
    #def emission(self, n):
    #    return [random.random() for j in xrange(n)]

    #ViterbiPathを計算してPath(軌道)を返す
    def ViterbiPath(self, X_init, PathWeight, state_num,IndexMap_one_NOzero,MoveIndex_list, outputname, X_init_original, Bug_removal_savior): #, Transition):
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
                cost = [self.update_lite(cost_np, t, f, state_num,IndexMap_one_NOzero,MoveIndex_list,Transition) for t, f in izip(m, e)] #izipの方がメモリ効率は良いが, zipとしても処理速度は変わらない
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
                
                    save_data.SavePathTemp(X_init_original, path_one, i+1, outputname, IndexMap_one_NOzero, Bug_removal_savior)
                
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

                    save_data.SaveLogLikelihood(LogLikelihood_step,0,i+1, outputname)
                    save_data.SaveLogLikelihood(LogLikelihood_sum,1,i+1, outputname)

                    #The moving distance of the path
                    Distance = self.PathDistance(path_one)
        
                    #Save the moving distance of the path
                    save_data.SavePathDistance_temp(Distance, i+1, outputname)

                    if (SAVE_Trellis == 1):
                        save_data.SaveTrellis(trellis, outputname, i+1)
                    temp = 0
                temp += 1

        #最後の遷移確率は一様にすればよいはず
        e_last = [0.0]
        m_last = [[0.0 for i in range(len(PathWeight))]]
        cost = [self.update(cost, t, f) for t, f in zip(m_last, e_last)]
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
    #def SendPath(self, Path):

    #The moving distance of the pathを計算する
    def PathDistance(self, Path):
        Distance = len(collections.Counter(Path))
        print "Path Distance is ", Distance
        return Distance

