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

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()
save_data = spconavi_save_data.SavingData()
#path_calculate = spconavi_path_calculate.PathPlanner()

#Definition of action (functions in spconavi_read_data)
action_functions = [tools.right, tools.left, tools.up, tools.down, tools.stay] #, migiue, hidariue, migisita, hidarisita]
cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]



def ReadLikelihood(i,j,outputfile):
    fname=outputfile +"/SpCoTMHP_S"+str(i)+"G"+str(j)+"_Log_likelihood_sum.csv"
    for line in open(fname,'r'):
      itemList = line[:-1].split(',')
    
    return float(itemList[0])
 

def ReadDistance(i,j,outputfile):
    fname=outputfile +"/SpCoTMHP_S"+str(root[i])+"G"+str(root[i+1])+"_Distance.csv"
    for line in open(fname,'r'):
      itemList = line[:-1].split(',')
    
    return float(itemList[0])   




#GaussMap make (same to the function in spcotmhp_astar_path_calculate) #Ito
def PostProbMap_nparray_jit_Ito(CostMapProb,Mu,Sig,map_length,map_width,it): #,IndexMap):
        x,y = np.meshgrid(np.linspace(-10.0,9.92,map_width),np.linspace(-10.0,9.92,map_length))
        pos = np.dstack((x,y))    
        #PostProbMap = np.array([ [ PostProb_ij([width, length],Mu,Sig,map_length,map_width, CostMapProb,it) for width in xrange(map_width) ] for length in xrange(map_length) ])
        PostProb=multivariate_normal(Mu[it],Sig[it]).pdf(pos)
        '''
        mmm=CostMapProb * PostProb
        fig = plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        ax.contourf(x,y,mmm)
        ax.set_xlim(-10,9.92)
        ax.set_ylim(-10,9.92)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        '''
        return CostMapProb * PostProb
    


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
    for line in open(outputfile +    'Viterbi_SpCoTMHP_distance.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              dis[c][i] = float(itemList[i])
        c = c + 1
    return dis


def ReadLikeliMtrx(outputfile):
    cost  = [ [0.0 for atem in range(K)] for aky in range(K) ]
    c=0
    for line in open(outputfile + 'Astar_SpCoTMHP_likelihood.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              cost[c][i] = float(itemList[i])
        c = c + 1
    return cost
        

# おそらく最適探索（Dijkstra）が実装されている
def Search_ITO(glf, st, cost): # goal index, start index, cost
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
         print("fin")
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
def Search_ITO_2(s_i, lk, Step): # start index, likelihoods, T_topo
    lkt=[ [0.0 for i in range(K)] for k in range(Step)]#sum likewood
    rt=[ [-1 for i in range(K)] for k in range(Step)]#oya node save
    node =[0 for i in range(K) ] #
    node[s_i]=1
    ndc=[0 for i in range(K) ]#
    root=[0 for i in range(Step) ]
    mxs=-1#
    mxg=-1
    mx=0
    #print(lk)
    #print(node)

    for t in range(Step):
      for i in range(K):
        if t==0:
          if node[i]==1:
            for j in range(K):
              if lk[i][j]>0:
                 lkt[t][j]=lk[i][j]
                 rt[t][j]=i
                 ndc[j]=1
        else:
          if node[i]==1:
            for j in range(K):
              if lk[i][j]>0:
                if lkt[t][j]<lkt[t-1][i]+lk[i][j]:
                  lkt[t][j]=lkt[t-1][i]+lk[i][j]
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
    print(mxg)                  
    for i in range(Step):   
        root[Step-i-1]=rt[Step-i-1][mxg]
        mxg=rt[Step-i-1][mxg]
    return root



if __name__ == '__main__': 
    print("[START] SpCoTMHP (Viterbi).")
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
    
    ### hard cording ###
    tyukan=0
    t_i=0 #中間地点の位置分布index
    #x_s=
    #y_s=
    s_i=1 #スタート位置近くの位置分布index
    gl=3  #ゴールの位置分布index
    start=(120,50)
    ### hard cording ###


    if tyukan==1:
        glf=t_i #暫定ゴールを中間地点に設定
        Hf=t_i  #中間
    else :
        glf=gl  #ゴールを指定した最終ゴールに設定
        Hf="N"  #中間なしの場合
         
         
    if (SAVE_time == 1):
      #Substitution of start time
      start_time = time.time()

    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" 
    print(filename, iteration, sample)
    outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
    outputsubfolder = outputfile + "viterbi_node/"
    outputname = outputfile + "viterbi_result/Viterbi_SpCoTMHP_"+"T"+str(T_topo)+"S"+str(start)+"H"+str(Hf)+"G"+str(gl)
    
    #Makedir( outputfolder + trialname )
    #Makedir( outputfile )
    Makedir( outputfile + "viterbi_result/" )

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA   = read_data.ReadParameters(iteration, sample, filename, trialname)
    gridmap = read_data.ReadMap(outputfile)
    map_length, map_width = gridmap.shape
    
    #gridmap = maze
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
    CostMap = read_data.ReadCostMap(outputfile)
    CostMapProb = (100.0 - CostMap)/100


        
    W, W_index, Mu, Sig, pi, phi_l, K, L = THETA
    
    dis  = ReadDistanceMtrx(outputfile)      
    lk   = ReadLikeliMtrx(outputfile)    


    #Read CoonectMatricx (psi_setting.csv) file
    CoonectMatricx=[ [0.0 for i in range(K)] for c in range(K) ]
    c = 0
    for line in open(filename + "/" + trialname + '_psi_setting.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              CoonectMatricx[c][i] = float(itemList[i])
        c = c + 1

    #lk=[ [0.0 for i in range(K)] for c in range(K) ]
    #ndl=[0.454,0.314,0.218,0.372,0.706,0.700,0.748,0.333,0.348,0.591,0.583,]
    #for i in range(K):
    #  for j in range(K):
    #    if i==j:
    #     lk[i][j]=float(ndl[i])*100.0*float(phi_l[j][gl])
    #    elif CoonectMatricx[i][j]==1:
    #     #fname=outputfile +"SpCoTMHP_S"+str(i)+"G"+str(j)+"_Log_likelihood_sum.csv"
    #     lk[i][j]= float(ReadLikelihood(i,j,outputfile))*phi_l[j][gl]
    #    else:
    #     lk[i][j]=0
        
        
    #ss=Map_coordinates_To_Array_index(Mu[S_i])
    #start=(150,130)

    ###v###近くの位置分布のindexのガウスの平均をゴールとしたA*を実行###v###
    PathWeightMap = PostProbMap_nparray_jit_Ito(CostMapProb,Mu,Sig,map_length,map_width,s_i)
    
    gg=tools.Map_coordinates_To_Array_index(Mu[s_i])
    #print(gg[0])
    goal_candidate=[[gg[1],gg[0]]]
    #goal_candidate=[[55,135],[89,103]]
    #J = len(goal_candidate)
    J=1
    #if(J != THETA[6]):
    # print("[WARNING] J is not K",J,K)
    p_cost_candidate = [0.0 for j in range(J)]
    Path_candidate   = [[0.0] for j in range(J)]
    print(goal_candidate)

    ###goal候補ごとにA*を実行
    for gc_index in range(J):
        goal = goal_candidate[gc_index]
        Path, p_cost = a_star(start, goal, gridmap, action_functions, cost_of_actions, PathWeightMap)    
        
        first_cost = p_cost / float(len(Path))
        p_cost_candidate[gc_index] = p_cost / float(len(Path))
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
    Path_inv = [[Path[t][1], Path[t][0]] for t in range(len(Path))]
    Path_inv.reverse()
    Path_ROS = Path_inv #使わないので暫定的な措置
    #パスを保存
    #SavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)
    ###^###近くの位置分布のindexのガウスの平均をゴールとしたA*を実行###^###

          
    root = Search_ITO_2(s_i, lk, T_topo) #10
    print(root)


    
    
    """
    PathDist=[0.0 for i in range(Step)]
    dissum=0
    for i in range(Step-1):
      if root[i]!=root[i+1]:
        #fname=outputfile +"/viterbi_node/SpCoTMHP_S"+str(root[i])+"G"+str(root[i+1])+"_Distance.csv"
        PathDist[i]=ReadDistance(root[i],root[j],outputfile)
        dissum+=PathDist[i]  
    #print(dissum)
    print(PathDist)
    dissum+=Distance
    disp=[dissum]    
    #po=[0,0]
    #print(root)
    """

    ## パスを読み込み、つなげる
    for v in range(len(root)-1):
        path=[[0,0]for k in range(int(T_horizon))] #100
        i=0
        if root[v]!=root[v+1]:
         for line in open(outputsubfolder + "SpCoTMHP_S" + str(root[v]) + 'G' + str(root[v+1]) +  '_Path.csv', 'r'):
           
           itemList = line[:-1].split(',')
           path[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
           #path[i][0]=int(po[0])
           #path[i][1]=int(po[1])
           i = i + 1
           #print(i)
         Path_ROS=Path_ROS+path 



    if (SAVE_time == 1):
        #PP終了時刻を保持
        end_pp_time = time.time()
        time_pp = end_pp_time - start_time #end_recog_time
        fp = open( outputname + "_time_pp.txt", 'w')
        fp.write(str(time_pp)+"\n")
        fp.close()    
        
    disp=[tools.PathDistance(Path_ROS)] #len(Path_ROS)

    for i in range(len(Path_ROS)):
      plt.plot(Path_ROS[i][1], Path_ROS[i][0], "s", color="tab:red", markersize=1)
    plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
    np.savetxt(outputname + "_fin_Path_ROS.csv", Path_ROS, delimiter=",")
    np.savetxt(outputname + "_fin_Distance.csv", disp, delimiter=",")
    plt.clf()
     