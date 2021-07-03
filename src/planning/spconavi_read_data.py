#!/usr/bin/env python
#coding:utf-8
from scipy.io import mmread
from __init__ import *
from scipy.stats import multivariate_normal


class ReadingData:

    #Read the map data⇒2-dimension array に格納
    def ReadMap(self, outputfile):
        #outputfolder + trialname + navigation_folder + map.csv
        gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
        print "Read map: " + outputfile + "map.csv"
        return gridmap


    #Read the cost map data⇒2-dimension array に格納
    def ReadCostMap(self, outputfile):
        #outputfolder + trialname + navigation_folder + contmap.csv
        costmap = np.loadtxt(outputfile + "costmap.csv", delimiter=",")
        print "Read costmap: " + outputfile + "contmap.csv"
        return costmap


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

        THETA = [W,W_index,Mu,Sig,Pi,Phi_l,K,L]
        return THETA


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


    #gridmap and costmap から確率の形のCostMapProbを得ておく
    def CostMapProb_jit(self, gridmap, costmap):
        CostMapProb = (100.0 - costmap) / 100.0     #Change the costmap to the probabilistic costmap
        #gridの数値が0（非占有）のところだけ数値を持つようにマスクする
        GridMapProb = 1*(gridmap == 0)  #gridmap * (gridmap != 100) * (gridmap != -1)  #gridmap[][]が障害物(100)または未探索(-1)であれば確率0にする
        return CostMapProb * GridMapProb

    def PostProb_ij(self, Index_temp,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K, CostMapProb):
        if (CostMapProb[Index_temp[1]][Index_temp[0]] != 0.0): 
            X_temp = self.Array_index_To_Map_coordinates(Index_temp)  #map と縦横の座標系の軸が合っているか要確認
            #print X_temp,Mu
            sum_i_GaussMulti = [ np.sum([multivariate_normal.pdf(X_temp, mean=Mu[k], cov=Sig[k]) * Phi_l[c][k] for k in xrange(K)]) for c in xrange(L) ] ##########np.array( ) !!! np.arrayにすると, numbaがエラーを吐く
            PostProb = np.sum( LookupTable_ProbCt * sum_i_GaussMulti ) #sum_c_ProbCtsum_i
        else:
            PostProb = 0.0
        return PostProb


    #@jit(parallel=True)  #並列化されていない？1CPUだけ使用される
    def PostProbMap_nparray_jit(self, CostMapProb,Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K): #,IndexMap):
        PostProbMap = np.array([ [ self.PostProb_ij([width, length],Mu,Sig,Phi_l,LookupTable_ProbCt,map_length,map_width,L,K, CostMapProb) for width in xrange(map_width) ] for length in xrange(map_length) ])
        return CostMapProb * PostProbMap


    def ReadTrellis(self, outputname, temp):
        print "ReadTrellis"
        # Save the result to the file 
        trellis = np.load(outputname + "_trellis" + str(temp) + ".npy") #, delimiter=",")
        print "Read trellis: " + outputname + "_trellis" + str(temp) + ".npy"
        return trellis


    #パス計算のために使用したLookupTable_ProbCtをファイル読み込みする
    def ReadLookupTable(self, outputfile):
        # Read the result from the file
        output = outputfile + "LookupTable_ProbCt.csv"
        LookupTable_ProbCt = np.loadtxt(output, delimiter=",")
        print "Read LookupTable_ProbCt: " + output
        return LookupTable_ProbCt    


    #Load the probability cost map used for path calculation
    def ReadCostMapProb(self, outputfile):
        # Read the result from the file
        output = outputfile + "CostMapProb.csv"
        CostMapProb = np.loadtxt(output, delimiter=",")
        print "Read CostMapProb: " + output
        return CostMapProb  


    #Load the probability value map used for path calculation
    def ReadProbMap(self, outputfile):
        # Read the result from the file
        output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
        PathWeightMap = np.loadtxt(output, delimiter=",")
        print "Read PathWeightMap: " + output
        return PathWeightMap


    def ReadTransition(self, state_num, outputfile):
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


    def ReadTransition_sparse(self, state_num, outputfile):
        #Transition = [[0 for j in xrange(state_num)] for i in xrange(state_num)] 
        # Read the result from the file
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse.mtx"
        Transition = mmread(output_transition).tocsr()  #.todense()
        print "Read Transition: " + output_transition
        return Transition


    def path_topic_from_csv_data(self, outputfile, outputname):
        pose_list = []
        path_pose = np.loadtxt(outputfile + outputname + "_Path_ROS.csv", delimiter=",")
        print "Read Path Topic: " + outputfile + outputname + "_Path_ROS.csv"
        
        for indx in range(len(path_pose)):
            temp_pose = PoseStamped()
            temp_pose.pose.position.x = self.csv_path_data["x"][indx]
            temp_pose.pose.position.y = self.csv_path_data["y"][indx]
        return pose_list

