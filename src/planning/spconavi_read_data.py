#!/usr/bin/env python
#coding:utf-8
from scipy.io import mmread
from scipy.stats import multivariate_normal
from __init__ import *
import collections


class Tools:

    #ROSのmap 座標系をPython内の2-dimension array index 番号に対応付ける
    def Map_coordinates_To_Array_index(self, X):
        X = np.array(X)
        Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
        return Index


    #Python内の2-dimension array index 番号からROSのmap 座標系への変換
    def Array_index_To_Map_coordinates(self, Index):
        Index = np.array(Index)
        X = np.array( (Index * resolution) + origin )
        return X


    # Action types of the robot
    def right(self, pos):
        return (pos[0], pos[1] + 1)

    def left(self,pos):
        return (pos[0], pos[1] - 1)

    def up(self,pos):
        return (pos[0] - 1, pos[1])

    def down(self, pos):
        return (pos[0] + 1, pos[1])

    def stay(self, pos):
        return (pos[0], pos[1])


    def Manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


    #The moving distance of the pathを計算する
    def PathDistance(self, Path):
        Distance = len(collections.Counter(Path))
        print("Path Distance is ", Distance)
        return Distance


class ReadingData:

    #Read the path
    def ReadPath(self, outputname):
        # 結果をファイル読み込み
        output = outputname + "_Path.csv"
        Path = np.loadtxt(output, delimiter=",")
        print("Read Path: " + output)
        return Path

    #Read the path per step
    def ReadPath_step(self, outputname,step):
        # Read the result file
        output = outputname + "_Path" + str(step) + ".csv" # ビタビアルゴリズム用
        Path = np.loadtxt(output, delimiter=",")
        print("Read Path: " + output)
        return Path

    #Read the map data⇒2-dimension array に格納
    def ReadMap(self, outputfile):
        #outputfolder + trialname + navigation_folder + map.csv
        gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
        print("Read map: " + outputfile + "map.csv")
        return gridmap


    #Read the cost map data⇒2-dimension array に格納
    def ReadCostMap(self, outputfile):
        #outputfolder + trialname + navigation_folder + contmap.csv
        costmap = np.loadtxt(outputfile + "costmap.csv", delimiter=",")
        print("Read costmap: " + outputfile + "contmap.csv")
        return costmap


    #Read the psi parameters of learned spatial concepts (for SpCoTMHP)
    def ReadPsi(self, iteration, sample, filename, trialname):
        #psi     = [ [0.0 for atem in range(K)] for aky in range(K) ]
        psi = np.load(filename + "/" + trialname + '_psi_' + str(iteration) + '_' + str(sample) + '.npy')
        return psi

    #Read the parameters of learned spatial concepts
    def ReadParameters(self, iteration, sample, filename, trialname):
        #THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
        #r = iteration

        W_index = []
        i = 0
        #Read the text file
        for line in open(filename + "/" + trialname + '_w_index_' + str(iteration) + '_' + str(sample) + '.csv', 'r'): 
            itemList = line[:-1].split(',')
            if(i == 1):
                for j in range(len(itemList)):
                    if (itemList[j] != ""):
                        W_index = W_index + [itemList[j]]
            i = i + 1
            
        #####Parameters W, μ, Σ, φ, π#####
        Mu    = [ np.array([ 0.0, 0.0 ]) for i in range(K) ]               #the position distribution (mean of Gaussian) (x,y)[K]
        Sig   = [ np.array([ [0.0, 0.0],[0.0, 0.0] ]) for i in range(K) ]  #the position distribution (co-variance of Gaussian) (2×2-dimension)[K]
        W     = [ [0.0 for j in range(len(W_index))] for c in range(L) ]  #the name of place (multinomial distribution: W_index-dimension)[L]
        #theta = [ [0.0 for j in range(DimImg)] for c in range(L) ] 
        Pi    = [ 0.0 for c in range(L)]                                   #index of spatial concept of multinomial distribution (L-dimension)
        Phi_l = [ [0.0 for i in range(K)] for c in range(L) ]             #index of position distribution of multinomial distribution (K-dimension)[L]
        
        i = 0
        ##Mu is read from the file
        for line in open(filename + "/" + trialname + '_Mu_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            #Mu[i] = np.array([ float(itemList[0]) - origin[0] , float(itemList[1]) - origin[1] ]) / resolution
            Mu[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
            i = i + 1
        
        #i = 0
        ##Sig is read from the file
        Sig = np.load(filename + "/" + trialname + '_Sig_' + str(iteration) + '_' + str(sample) + '.npy')
        '''
        for line in open(filename + "/" + trialname + '_Sig_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            #Sig[i] = np.array([[ float(itemList[0])/ resolution, float(itemList[1]) ], [ float(itemList[2]), float(itemList[3])/ resolution ]]) #/ resolution
            Sig[i] = np.array([[ float(itemList[0]), float(itemList[1]) ], [ float(itemList[2]), float(itemList[3])]]) 
            i = i + 1
         '''
        ##phi is read from the file
        c = 0
        #Read the text file
        for line in open(filename + "/" + trialname + '_phi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            for i in range(len(itemList)):
                if itemList[i] != "":
                    Phi_l[c][i] = float(itemList[i])
            c = c + 1
            
        ##Pi is read from the file
        for line in open(filename + "/" + trialname + '_pi_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            for i in range(len(itemList)):
                if itemList[i] != '':
                    Pi[i] = float(itemList[i])
        
        ##W is read from the file
        c = 0
        #Read the text file
        for line in open(filename + "/" + trialname + '_W_' + str(iteration) + '_' + str(sample) + '.csv', 'r'):
            itemList = line[:-1].split(',')
            for i in range(len(itemList)):
                if itemList[i] != '':
                    #print c,i,itemList[i]
                    W[c][i] = float(itemList[i])
            c = c + 1

        THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
        return THETA


    def ReadTrellis(self, outputname, temp):
        print ("ReadTrellis")
        # Save the result to the file 
        trellis = np.load(outputname + "_trellis" + str(temp) + ".npy") #, delimiter=",")
        print("Read trellis: " + outputname + "_trellis" + str(temp) + ".npy")
        return trellis


    #パス計算のために使用したLookupTable_ProbCtをファイル読み込みする
    def ReadLookupTable(self, outputfile):
        # Read the result from the file
        output = outputfile + "LookupTable_ProbCt.csv"
        LookupTable_ProbCt = np.loadtxt(output, delimiter=",")
        print("Read LookupTable_ProbCt: " + output)
        return LookupTable_ProbCt    


    #Load the probability cost map used for path calculation
    def ReadCostMapProb(self, outputfile):
        # Read the result from the file
        output = outputfile + "CostMapProb.csv"
        CostMapProb = np.loadtxt(output, delimiter=",")
        print("Read CostMapProb: " + output)
        return CostMapProb  


    #Load the probability value map used for path calculation
    def ReadProbMap(self, outputfile, speech_num):
        # Read the result from the file
        output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
        PathWeightMap = np.loadtxt(output, delimiter=",")
        print("Read PathWeightMap: " + output)
        return PathWeightMap


    def ReadTransition(self, state_num, outputfile):
        Transition = [[approx_log_zero for j in range(state_num)] for i in range(state_num)] 
        # Read the result from the file
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_log.csv"
        #Transition = np.loadtxt(outputfile + "_Transition_log.csv", delimiter=",")
        i = 0
        #Read the text file
        for line in open(output_transition, 'r'):
            itemList = line[:-1].split(',')
            for j in range(len(itemList)):
                if itemList[j] != '':
                    Transition[i][j] = float(itemList[j])
            i = i + 1
        print("Read Transition: " + output_transition)
        return Transition


    def ReadTransition_sparse(self, state_num, outputfile):
        #Transition = [[0 for j in range(state_num)] for i in range(state_num)] 
        # Read the result from the file
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse.mtx"
        Transition = mmread(output_transition).tocsr()  #.todense()
        print("Read Transition: " + output_transition)
        return Transition

    """
    def path_topic_from_csv_data(self, outputfile, outputname):
        pose_list = []
        path_pose = np.loadtxt(outputfile + outputname + "_Path_ROS.csv", delimiter=",")
        print("Read Path Topic: " + outputfile + outputname + "_Path_ROS.csv")
        
        for indx in range(len(path_pose)):
            temp_pose = PoseStamped()
            temp_pose.pose.position.x = self.csv_path_data["x"][indx]
            temp_pose.pose.position.y = self.csv_path_data["y"][indx]
        return pose_list
    """
