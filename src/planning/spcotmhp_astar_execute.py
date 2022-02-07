#!/usr/bin/env python
#coding:utf-8
#import os
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

#GaussMap make (same to the function in spcotmhp_astar_path_calculate) #Ito
def PostProbMap_nparray_jit(CostMapProb,Mu,Sig,map_length,map_width,it): #,IndexMap):
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

    
if __name__ == '__main__': 
    print("[START] SpCoTMHP. (A star)")
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
    
    
    tyukan=0
    t_i=0
    #x_s=
    #y_s=
    s_i=1
    gl=3
    start=(120,50)
    if tyukan==1:
       glf=t_i
    else :
       glf=gl
       
    if (SAVE_time == 1):
      #Substitution of start time
      start_time = time.time()

    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" 
    print(filename, iteration, sample)
    outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
    #outputname = outputfile + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(init_position_num)+"G"+str(speech_num)

    #Makedir( outputfolder + trialname )
    Makedir( outputfile )
    #Makedir( outputname )

    #Read the files of learned parameters  #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
    THETA = read_data.ReadParameters(iteration, sample, filename, trialname)
    maze  = read_data.ReadMap(outputfile)
    map_length, map_width = maze.shape
    
    gridmap = maze
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
    #St=st_i
    Gl=s_i
    if tyukan==1:
        Hf=t_i
    else:
        Hf="N"

    
    outputname = outputfile + "astar_result/" + "Astar_SpCoTMHP_"+"S"+str(start)+"H"+str(Hf)+"_G"+str(gl)
    
    #action_functions = [right, left, up, down, stay] #, migiue, hidariue, migisita, hidarisita]
    #cost_of_actions  = np.log( np.ones(len(action_functions)) / float(len(action_functions)) ) #[    1/5,    1/5,  1/5,    1/5,    1/5]) #, ,    1,        1,        1,          1]
    W, W_index, Mu, Sig, pi, phi_l, K, L = THETA
    PathWeightMap = PostProbMap_nparray_jit(CostMapProb,Mu,Sig,map_length,map_width,Gl)
    cos     = [ [0.0 for atem in range(K)] for aky in range(K) ]
    dis     = [ [0.0 for atem in range(K)] for aky in range(K) ]
    ev     = [ 0.0 for aky in range(K) ]
    node = [ [0,-2] for aky in range(K) ]
    node[s_i]=[1,-1]
    c=0
    #i=0
    for line in open(outputfile + "/" +'Astar_SpCoTMHP_distance.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              dis[c][i] = float(itemList[i])
        c = c + 1      
    c=0
    #i=0
    for line in open(outputfile + "/" +'Astar_SpCoTMHP_cost.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
              cos[c][i] = float(itemList[i])
              cos[c][i]=cos[c][i]/(phi_l[c][glf]*100.0)
              if dis[c][i] != 0:
                 cos[c][i]=cos[c][i]/float(dis[c][i])
        c = c + 1
        
        
        
        
    #ss=Map_coordinates_To_Array_index(Mu[S_i])
    #start=(150,130)

    gg=tools.Map_coordinates_To_Array_index(Mu[s_i])
    #print(gg[0])
    goal_candidate=[[gg[1],gg[0]]]
    #goal_candidate=[[55,135],[89,103]]
    #J = len(goal_candidate)
    J=1
    #if(J != THETA[6]):
    # print("[WARNING] J is not K",J,K)
    p_cost_candidate = [0.0 for j in range(J)]
    Path_candidate = [[0.0] for j in range(J)]
    print(goal_candidate)

    ###goal候補ごとにA*を実行
    for gc_index in range(J):
        goal = goal_candidate[gc_index]
        Path, p_cost = a_star(start, goal, maze, action_functions, cost_of_actions, PathWeightMap)    

        first_cost = p_cost / float(len(Path))
        p_cost_candidate[gc_index] = p_cost / float(len(Path))
        Path_candidate[gc_index] = Path    

        """
        #PathWeightMapとPathからlog likelihoodの値を再計算する
        LogLikelihood_step = np.zeros(T_horizon)
        LogLikelihood_sum = np.zeros(T_horizon)

        for i in range(T_horizon):
        if (i < len(Path)):
            t = i
        else:
            t = len(Path) -1
        #print PathWeightMap.shape, Path[t][0], Path[t][1]
        LogLikelihood_step[i] = np.log(PathWeightMap[ Path_inv[t][0] ][ Path_inv[t][1] ])
        if (t == 0):
            LogLikelihood_sum[i] = LogLikelihood_step[i]
        elif (t >= 1):
            LogLikelihood_sum[i] = LogLikelihood_sum[i-1] + LogLikelihood_step[i]
        
        LogLikelihood_step_candidate[gc_index] = LogLikelihood_step
        LogLikelihood_sum_candidate[gc_index] = LogLikelihood_sum
        """

    ### select the goal of expected cost
    expect_gc_index = np.argmin(p_cost_candidate)
    Path = Path_candidate[expect_gc_index]
    goal = goal_candidate[expect_gc_index]
    print("Goal:", goal)
    #LogLikelihood_step
    #LogLikelihood_sum

    if (SAVE_time == 1):
        #PP終了時刻を保持
        end_pp_time = time.time()
        time_pp = end_pp_time - start_time #end_recog_time
        fp = open( outputname + "_time_pp.txt", 'w')
        fp.write(str(time_pp)+"\n")
        fp.close()
        
    #for i in range(len(Path)):
    #  plt.plot(Path[i][0], Path[i][1], "s", color="tab:red", markersize=1)


    #The moving distance of the path
    Distance = tools.PathDistance(Path)
    #Distance_save[st_i][gl_i]=Distance
    #Save the moving distance of the path
    #SavePathDistance(Distance)

    print("Path distance using A* algorithm is "+ str(Distance))

    #計算上パスのx,yが逆になっているので直す
    Path_inv = [[Path[t][1], Path[t][0]] for t in range(len(Path))]
    Path_inv.reverse()
    Path_ROS = Path_inv #使わないので暫定的な措置
    #パスを保存
    #SavePath(start, [goal[1], goal[0]], Path_inv, Path_ROS, outputname)
    
      
    fin=0
    while fin != 1:
     
      for i in range(K):
        if node[i][0]==2:
            for k in range(K):  
                if cos[i][k]>0:
                    if node[k][0]==0:
                        node[k][0]=1
                        node[k][1]=i
                        ev[k]=ev[i]+cos[i][k]
                    elif node[k][0]==1:
                        if ev[k]>ev[i]+cos[i][k]:
                            ev[k]=ev[i]+cos[i][k]
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
      
    
    #print "[END] SpCoNavi."
    print(root)
    for i in range(K):
      for k in range(K):  
        cos[i][k]=phi_l[i][glf]*cos[i][k]/phi_l[i][gl]
    #Path_ROS=[]
    #path=[]
    po=[0,0]
    for v in range(len(root)-1):
        path=[[0,0]for k in range(int(dis[root[v]][root[v+1]]))]
        print(dis[root[v]][root[v+1]])
        i=0
        for line in open(outputfile + "astar_result/" + "/Astar_SpCoTMHP_S" + str(root[v]) + '_G' + str(root[v+1]) +  '_Path_ROS.csv', 'r'):
           itemList = line[:-1].split(',')
           path[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
           #path[i][0]=int(po[0])
           #path[i][1]=int(po[1])
           i = i + 1
           #print(i)
        Path_ROS=path+Path_ROS 
         


    
    ev     = [ 0.0 for aky in range(K) ]
    node = [ [0,-2] for aky in range(K) ]
    node[glf]=[1,-1]
    glf=gl
    fin=0
    
    if tyukan==1 :
        while fin != 1:
         
          for i in range(K):
            if node[i][0]==2:
               for k in range(K):  
                 if cos[i][k]>0:
                   if node[k][0]==0:
                    node[k][0]=1
                    node[k][1]=i
                    ev[k]=ev[i]+cos[i][k]
                   elif node[k][0]==1:
                     if ev[k]>ev[i]+cos[i][k]:
                        ev[k]=ev[i]+cos[i][k]
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
        while i != -1 :
           for k in range(K):
              if i==k:
                root.insert(0,k)
                i=node[k][1]

    
        print("[END] SpCoNavi.")
        print(root)
    #print(Path_ROS)

    """
    for v in range(len(root)-1):
        path=[[0,0]for k in range(int(dis[root[v]][root[v+1]]))]
        print(dis[root[v]][root[v+1]])
        i=0
        for line in open(filename + "/navi/astar_node/Astar_SpCoTMHP_S" + str(root[v]) + '_G' + str(root[v+1]) +  '_Path_ROS.csv', 'r'):
           itemList = line[:-1].split(',')
           path[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
           #path[i][0]=int(po[0])
           #path[i][1]=int(po[1])
           i = i + 1
           #print(i)
        Path_ROS=path+Path_ROS 
    """

    for i in range(len(Path_ROS)):
      plt.plot(Path_ROS[i][1], Path_ROS[i][0], "s", color="tab:red", markersize=1)
    np.savetxt(outputname + "_fin_Path_ROS.csv", Path_ROS, delimiter=",")
    plt.savefig(outputname + '_Path.pdf', dpi=300)#, transparent=True
    dism=len(Path_ROS)
    sd=[dism]
    np.savetxt(outputname + "_fin_Distance.csv", sd, delimiter=",")
    plt.clf()
     









