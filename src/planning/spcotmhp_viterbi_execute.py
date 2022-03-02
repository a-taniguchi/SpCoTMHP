#!/usr/bin/env python
#coding:utf-8

import sys
import time
#from math import log
from scipy.stats import multivariate_normal,multinomial
import matplotlib.pyplot as plt
#import collections
#import spconavi_path_calculate
import spconavi_read_data
import spconavi_save_data
from __init__ import *
from submodules import *

##Command: 
#python3 spcotmhp_viterbi_execute.py trialname iteration sample init_position_num speech_num initial_position_x initial_position_y waypoint_word
#python3 spcotmhp_viterbi_execute.py 3LDK_01 1 0 -1 7 100 100 -1 

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
#path_calculate = spconavi_path_calculate.PathPlanner()

#Definition of action (functions in spconavi_read_data)
action_functions = [tools.right, tools.left, tools.up, tools.down, tools.stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]


#GaussMap make (same to the function in spcotmhp_viterbi_path_calculate) #Ito
def PostProbMap_Gauss(CostMapProb,Mu,Sig,map_length,map_width,it): #,IndexMap):
        x,y = np.meshgrid(np.linspace(-10.0,9.92,map_width),np.linspace(-10.0,9.92,map_length))
        pos = np.dstack((x,y))    
        #PostProbMap = np.array([ [ PostProb_ij([width, length],Mu,Sig,map_length,map_width, CostMapProb,it) for width in xrange(map_width) ] for length in xrange(map_length) ])
        PostProb=multivariate_normal(Mu[it],Sig[it]).pdf(pos)

        return CostMapProb * PostProb


# The weight of the topoligical-level likelihood in word Se
def TopoWeight(i,j,Connect,w,THETA,psi):
    #THETAを展開
    W, W_index, Myu, S, pi, phi, K, L = THETA

    if (Connect == 1):
        # i_e: i, i_e-1: j, S_e:w
        sum_c  = np.sum([phi[c][i]*(W[c][w]**N_best)*pi[c] for c in range(L)])
        bunbo2 = np.sum([phi[c][i] for c in range(L)])
        weight = psi[i][j] * sum_c / bunbo2
    else:
      weight = 1.0

    return weight




### A star algorithm (by Ryo Ozaki) ############################################
def a_star(start, goal, maze, action_functions, cost_of_actions, PathWeightMap):
    if (maze[goal[0]][goal[1]] != 0):
        print("[ERROR] goal",maze[goal[0]][goal[1]],"is not 0.")

    ###START A*
    open_list = []
    open_list_cost = []
    open_list_key = []
    closed_list = []
    closed_list_cost = []
    closed_list_key = []
    open_list.append(start)
    open_list_cost.append(0)
    open_list_key.append(0 + tools.Manhattan_distance(start, goal))
    OYA = {}
    ko = (0), (0)
    Path = []

    while open_list:
        sorted_idx = np.argsort(open_list_key, kind="stable")
        pop_idx = sorted_idx[0]
        p = open_list.pop(pop_idx)
        p_cost = open_list_cost.pop(pop_idx)
        p_key = open_list_key.pop(pop_idx)
        closed_list.append(p)
        closed_list_cost.append(p_cost)
        closed_list_key.append(p_key)
        if p == goal:
            break
        for act_func, act_cost in zip(action_functions, cost_of_actions):
            q = act_func(p)
            if (int(maze[q]) != 0):
                continue
            q_cost = p_cost - act_cost - np.log(PathWeightMap[q[0]][q[1]])  #current sum cost and action cost 
            q_pev = tools.Manhattan_distance(q, goal) * np.log(float(len(action_functions))) #heuristic function
            q_key = q_cost - q_pev

            if q in open_list:
                idx = open_list.index(q)
                key = open_list_key[idx]
                if key > q_key:
                    open_list_key[idx] = q_key
                    open_list_cost[idx] = q_cost
            elif q in closed_list:
                idx = closed_list.index(q)
                key = closed_list_key[idx]
                if key > q_key:
                    closed_list.pop(idx)
                    closed_list_cost.pop(idx)
                    closed_list_key.pop(idx)
                    open_list.append(q)
                    open_list_cost.append(q_cost)
                    open_list_key.append(q_key)
                    #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                    OYA[(q[1], q[0])] = (p[1], p[0])
                    ko = (q[1]), (q[0])
                    #print(ko)
            else:
                open_list.append(q)
                open_list_cost.append(q_cost)
                open_list_key.append(q_key)
                #plt.quiver(p[1], p[0], (q[1]-p[1]), (q[0]-p[0]), angles='xy', scale_units='xy', scale=1, color="tab:red")
                OYA[(q[1], q[0])] = (p[1], p[0])
                ko = (q[1]), (q[0])
                #print(ko)

    #最適経路の決定: ゴールから親ノード（どこから来たか）を順次たどっていく
    #i = len(OYA)
    #for oyako in reversed(OYA):
    ko_origin = ko
    ko = (goal[1], goal[0])
    print(ko,goal)
    #for i in range(p_cost):
    while(ko != (start[1],start[0])):
        #print(OYA[ko])
        try:
            Path = Path + [OYA[ko]]
        except KeyError:
            ko = ko_origin
            Path = Path + [OYA[ko]]
            print("NOT END GOAL.")
        
        ko = OYA[ko]
        #i = len(Path)
        #print(i, ko)
        #i -= 1

    print(goal,": Total cost using A* algorithm is "+ str(p_cost))
    return Path, p_cost
### A star algorithm (by Ryo Ozaki) ############################################



    
def ReadDistanceMtrx(outputfile):
    dis  = [ [0.0 for atem in range(K)] for aky in range(K) ]
    c=0
    for line in open(outputfile + 'Astar_SpCoTMHP_distance.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              dis[c][i] = float(itemList[i])
        c = c + 1
    return dis


def ReadCostMtrx(outputfile):
    cost  = [ [0.0 for atem in range(K)] for aky in range(K) ]
    c=0
    for line in open(outputfile + 'Astar_SpCoTMHP_cost.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              cost[c][i] = float(itemList[i])
        c = c + 1
    return cost
        

def ReadPathDistance(outputsubfolder,st,gl):
    for line in open(outputsubfolder + "Astar_SpCoTMHP_S" + str(st) + "_G" + str(gl) + "_Distance.csv", 'r'):
        distance = float(line[:-1])
        #itemList = line[:-1].split(',')
        #for i in range(len(itemList)):
        #    if itemList[i] != "":
        #      cost[c][i] = float(itemList[i])
        #c = c + 1
    return int(distance)

def ReadLogLikelihood(outputsubfolder,st,gl):
    LogLikelihood  = [ 0.0 ]
    c=0
    for line in open(outputsubfolder + "Astar_SpCoTMHP_S" + str(st) + "_G" + str(gl) + "_Log_likelihood_sum.csv", 'r'):
        #itemList = line[:-1].split(',')
        #for i in range(len(itemList)):
        #    if itemList[i] != "":
        LogLikelihood.append(float(line[:-1]))
        c = c + 1
    dist = int(ReadPathDistance(outputsubfolder,st,gl))

    return LogLikelihood[dist]

def ReadLogLkhdMtrx(outputsubfolder,CoonectMatricx):
    LogLkhdMtrx  = np.array([ [ 0.0 for gl in range(K)] for st in range(K) ])
    for st in range(K):
        for gl in range(K):
            if (CoonectMatricx[st][gl] == 1) and (st != gl):
                LogLkhdMtrx[st][gl] = ReadLogLikelihood(outputsubfolder,st,gl)
    return LogLkhdMtrx

    
# おそらく最適探索（Dijkstra）が実装されている
def SearchTopoDijkstra(glf, st, cost): # goal index, start index, cost
    ev   = [ 0.0 for aky in range(K) ]
    node = [ [0,-2] for aky in range(K) ]
    node[st]=[1,-1]

    fin=0
    while fin != 1:
      for i in range(K):
        if node[i][0]==2:
            for k in range(K):  
                if cost[i][k]>0:
                    if node[k][0]==0:
                        node[k][0]=1
                        node[k][1]=i
                        ev[k]=ev[i]+cost[i][k]
                    elif node[k][0]==1:
                        if ev[k]>ev[i]+cost[i][k]:
                            ev[k]=ev[i]+cost[i][k]
                            node[k][1]=i
            node[i][0]=3
            break
           
      minm=100000000000000000  
      minm_i=-1   
      for i in range(K):
        if node[i][0]==1:
          if minm > ev[i]:
            minm=ev[i]  
            minm_i=i
      if minm_i == glf:          
         fin=1
         #print("fin")
      else:
       node[minm_i][0]=2
    
    i=glf
    root=[]    
    while i!= -1 :
       for k in range(K):
         if i==k:
           root.insert(0,k)
           i=node[k][1]
    return root


# おそらく動的計画法（Viterbi）が実装されている
def SearchTopoViterbi(s_i, weight_W_index, S_list, Step): # st, TopoWeight_W_index, lk, S_list, T_topo
    lkt=[ [0.0 for k in range(K)] for t in range(Step)]#sum likelihood
    rt=[ [-1 for k in range(K)] for t in range(Step)]#oya node save
    node =[0 for k in range(K) ] #
    node[s_i]=1
    ndc=[0 for k in range(K) ]#
    root=[0 for t in range(Step) ]
    mxs=-1#
    mxg=-1
    mx=0
    #print(lk)
    #print(node)
    #np_lk = np.array(lk)


    for t in range(Step):
      #w = argmax([ np.array(S_list[t]) for _ in range(len(W_index)) ])
      #weight_inv = -1.0 * (TopoWeight_W_index[S_list[t]] + lk + np.diag(same_lk))
      weight = weight_W_index[S_list[t]] #np.max(weight_inv) - weight_inv
      #print(weight[6])
      #print(t,S_list[t])
      for i in range(K):
        if t==0:
          if node[i]==1:
            for j in range(K):
              if ((weight[i][j]>0) and (lk[i][j]!=0)) or (i==j):
                 lkt[t][j]=weight[i][j]
                 rt[t][j]=i
                 ndc[j]=1
        else:
          if node[i]==1:
            for j in range(K):
              if ((weight[i][j]>0) and (lk[i][j]!=0)) or (i==j):
                if lkt[t][j]<lkt[t-1][i]+weight[i][j]:
                  lkt[t][j]=lkt[t-1][i]+weight[i][j]
                  rt[t][j]=i
                  ndc[j]=1
      
      for f in range(K):
         if ndc[f]==1 and node[f]==0:
           node[f]=1
           
    #print(rt)        
    for i in range(K):
        if mx<lkt[Step-1][i]:
          mx=lkt[Step-1][i]
          mxg=i
    #print(mxg)    

    # Backword path              
    for i in range(Step):   
        root[Step-i-1]=rt[Step-i-1][mxg]
        mxg=rt[Step-i-1][mxg]

    return root


## パスを読み込み、つなげる (For A*)
def PathConnect(outputsubfolder, dis, Path_ROS, root):
    for v in range(len(root)-1):
        path=[ (0,0) for k in range(int(dis[root[v]][root[v+1]]))]
        #print(dis[root[v]][root[v+1]])
        i=0
        if ( root[v] != root[v+1] ):
          for line in open(outputsubfolder + "Astar_SpCoTMHP_S" + str(root[v]) + '_G' + str(root[v+1]) +  '_Path_ROS.csv', 'r'):
            itemList = line[:-1].split(',')
            path[i] = ( float(itemList[0]) , float(itemList[1]) )
            #path[i][0]=int(po[0])
            #path[i][1]=int(po[1])
            i = i + 1
            #print(i)
          Path_ROS=Path_ROS+path
    return Path_ROS


#################################################
if __name__ == '__main__': 
    print("[START] SpCoTMHP (Viterbi).")

    #Request a folder name for learned parameters.
    trialname = sys.argv[1]

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
    outputsubfolder = outputfile + "astar_node_gauss/"
    outputname_d    = outputfile + "viterbi_result/"
    #Viterbi_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(Hf)+"G"+str(gl)
    
    Makedir( outputname_d )

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA   = read_data.ReadParameters(iteration, sample, filename, trialname)
    W, W_index, Mu, Sig, pi, phi_l, K, L = THETA
    psi   = read_data.ReadPsi(iteration, sample, filename, trialname)
    

    gridmap = read_data.ReadMap(outputfile)
    map_length, map_width = gridmap.shape
    
    CostMapProb = read_data.ReadCostMapProb(outputfile)
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
    

    outputname      = outputname_d + "Viterbi_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(waypoint_word)+"G"+str(speech_num)
    
    dis  = ReadDistanceMtrx(outputfile)    
    cost = ReadCostMtrx(outputfile)   
    #lk   = -1.0 * np.array(cost)  # cost is -log likelihood  


    #Read CoonectMatricx (psi_setting.csv) file
    CoonectMatricx=[ [0.0 for i in range(K)] for c in range(K) ]
    c = 0
    for line in open(filename + "/" + trialname + '_psi_setting.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              CoonectMatricx[c][i] = float(itemList[i])
        c = c + 1

    lk = ReadLogLkhdMtrx(outputsubfolder,CoonectMatricx) # log likelihood

    ### 事前計算として、W_indexごとにトポロジカル重みを計算
    TopoWeight_W_index = [ np.log(np.array([ [TopoWeight(i,j,CoonectMatricx[i][j],w,THETA,psi) for i in range(K)] for j in range(K) ])) for w in range(len(W_index)) ]


    same_lk = np.array([multivariate_normal(Mu[it],Sig[it]).pdf(Mu[it])*0.5 for it in range(K)])
    #same_lk = np.array([CostMapProb[tools.Map_coordinates_To_Array_index(Mu[it])[0]][tools.Map_coordinates_To_Array_index(Mu[it])[1]] for it in range(K)])
    same_lk = np.diag(np.log(same_lk))
    #print("same_lk",same_lk)

    duration_term_lk = np.array([ [-1.0 * dis[i][j] * np.log(T_topo) for j in range(K)] for i in range(K) ]) # log duration prob


    whole_log_lk_inv_W_index =  -1.0 * np.array([TopoWeight_W_index[w] + lk + same_lk + duration_term_lk for w in range(len(W_index))])

    #weight_inv = -1.0 * (TopoWeight_W_index[S_list[t]] + lk + np.diag(same_lk))
    weight_W_index = whole_log_lk_inv_W_index #[ whole_log_lk_inv_W_index[w] - np.min(whole_log_lk_inv_W_index[w]) for w in range(len(W_index))]
    #[np.max(whole_log_lk_inv_W_index[w]) - whole_log_lk_inv_W_index[w] for w in range(len(W_index))]


    if (SAVE_time == 1):
      #Substitution of start time
      start_time = time.time()

    ### 初期位置から一番近くの位置分布を推定（ガウスの重みが一番高い分布のインデックス）
    near_node = np.argmax([multivariate_normal(Mu[k],Sig[k]).pdf(tools.Array_index_To_Map_coordinates(start)) for k in range(K)])
      


    #ss=Map_coordinates_To_Array_index(Mu[S_i])
    #start=(150,130)

    ###v###近くの位置分布のindexのガウスの平均をゴールとしたA*を実行###v###
    PathWeightMap = PostProbMap_Gauss(CostMapProb,Mu,Sig,map_length,map_width,near_node)
    
    gg=tools.Map_coordinates_To_Array_index(Mu[near_node])
    #print(gg[0])
    goal_candidate=[[gg[1],gg[0]]]
    #goal_candidate=[[55,135],[89,103]]
    #J = len(goal_candidate)
    J=1
    #if(J != THETA[6]):
    # print("[WARNING] J is not K",J,K)
    p_cost_candidate = [0.0 for j in range(J)]
    Path_candidate   = [[0.0] for j in range(J)]
    #print(goal_candidate)

    ###goal候補ごとにA*を実行
    for gc_index in range(J):
        goal = goal_candidate[gc_index]
        Path, p_cost = a_star(start_inv, goal, gridmap, action_functions, cost_of_actions, PathWeightMap)    
        
        #first_cost = p_cost / float(len(Path))
        p_cost_candidate[gc_index] = p_cost / float(len(Path)+1)
        Path_candidate[gc_index] = Path    


    ### select the goal of expected cost
    expect_gc_index = np.argmin(p_cost_candidate)
    Path = Path_candidate[expect_gc_index]
    goal = goal_candidate[expect_gc_index]
    print("Goal:", goal)


    #The moving distance of the path
    Distance = tools.PathDistance(Path)
    print("Path distance using A* algorithm is "+ str(Distance))

    #計算上パスのx,yが逆になっているので直す
    Path_inv = [ (Path[t][1], Path[t][0]) for t in range(len(Path)) ]
    Path_inv.reverse()
    Path_ROS = Path_inv #使わないので暫定的な措置
    #パスを保存
    #SavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)
    ###^###近くの位置分布のindexのガウスの平均をゴールとしたA*を実行###^###





    #####Estimate the goal point by spatial concept
    Otb_B      = [int(W_index[i] == Goal_Word[int(speech_num)]) * N_best for i in range(len(W_index))]   
    print("BoW:", Otb_B)
    
    Otb_B_list = [ [int(W_index[i] == Goal_Word[int(WP_list[w])]) * N_best for i in range(len(W_index))] for w in range(len(WP_list))]
    print("WP:BoW_list:", Otb_B_list)
    len_Otb_B_list = len(Otb_B_list)
          
    st = near_node
    #gl = 0
    S_list = [] # W_indexの番号の配列
    i = 0
    ### 中間地点が指定されている場合、中間地点（がなくなる）までパスプランニング、中間地点がなければゴールへのプランニング ###
    while (tyukan == 1):
        print("tyukan",i)
        ## Otb_B_listの中身を前半、Otb_Bを後半にしたS_{1:E}を作る
        for half in range( int(T_topo / (len_Otb_B_list+1)) ):
          S_list.append( np.argmax(Otb_B_list[i]) )


        i = i + 1

        ## 中間地点がなければゴールへのプランニングへ
        if (i >= len(Otb_B_list)):
            tyukan = 0
    

    ## Otb_Bを後半にしたS_{1:E}を作る
    while (len(S_list) <= T_topo):
      S_list.append( np.argmax(Otb_B) )

    ## 最終ゴールまでのトポロジカルプランニング
    root = SearchTopoViterbi(st, weight_W_index, S_list, T_topo) #10
    print("root_g:",root)

    ## パスを読み込み、つなげる
    Path_ROS = PathConnect(outputsubfolder, dis, Path_ROS, root) 

   



    if (SAVE_time == 1):
        #PP終了時刻を保持
        end_pp_time = time.time()
        time_pp = end_pp_time - start_time #end_recog_time
        fp = open( outputname + "_time_pp.txt", 'w')
        fp.write(str(time_pp)+"\n")
        fp.close()    
        
    print("[END] SpCoTMHP (Viterbi).")
    # print(Path_ROS)

    disp = [tools.PathDistance(Path_ROS)] # tupple in list  #len(Path_ROS)

    for i in range(len(Path_ROS)):
      plt.plot(Path_ROS[i][1], Path_ROS[i][0], "s", color="tab:red", markersize=1)
    plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
    plt.savefig(outputname + '_Path.png', dpi=300)#, transparent=True
    
    np.savetxt(outputname + "_fin_Path_ROS.csv", Path_ROS, delimiter=",")
    np.savetxt(outputname + "_fin_Distance.csv", disp, delimiter=",")
    plt.clf()
    print("DONE.",outputname)
