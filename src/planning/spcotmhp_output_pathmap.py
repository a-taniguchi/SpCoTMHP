#coding:utf-8
#Akira Taniguchi (Ito Shuya) 2022/02/05
#For Visualization of Path and Posterior emission probability (PathWeightMap) on the grid map
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from __init__ import *
from submodules import *
import spconavi_read_data

tools     = spconavi_read_data.Tools()
read_data = spconavi_read_data.ReadingData()

##Command: 
##python ./spcotmhp_output_pathmap.py 3LDK_01


########################################
if __name__ == '__main__': 
    #Request a folder name for learned parameters.
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")

    #Request the file number of the speech instruction      
    #speech_num = sys.argv[2] #0
    
    #Request the file name of path
    #path_file = sys.argv[3]
  
    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" #+ str(step) +"/"
    #print filename #, particle_num
    outputfile = filename + navigation_folder 
    outputsubfolder = outputfile + "viterbi_node/"

    #Read the map file
    gridmap = read_data.ReadMap(outputfile)



    """
	    y_min = 380 #X_init_index[0] - T_horizon
	    y_max = 800 #X_init_index[0] + T_horizon
	    x_min = 180 #X_init_index[1] - T_horizon
	    x_max = 510 #X_init_index[1] + T_horizon
	    #if (x_min>=0 and x_max<=map_width and y_min>=0 and y_max<=map_length):
	    PathWeightMap = PathWeightMap[x_min:x_max, y_min:y_max] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
	    gridmap = gridmap[x_min:x_max, y_min:y_max]
    """

	#length and width of the MAP cells
    map_length = len(gridmap)  #len(costmap)
    map_width  = len(gridmap[0])  #len(costmap[0])
    print("MAP[length][width]:",map_length,map_width)

    #Ito# 遷移確率の低いエッジは計算しないようにするために擬似的にpsi_setting.csvを読み込む
    #Ito# psiそのものの確率値ではないことに注意
    psi     = [ [0.0 for atem in range(K)] for aky in range(K) ]
    c=0
    for line in open(filename + "/" + trialname + '_psi_'  + 'setting.csv', 'r'):
        itemList = line[:-1].split(',')
        for i in range(len(itemList)):
            if itemList[i] != "":
                psi[c][i] = float(itemList[i])
        c = c + 1



    for st_i in range(K):
        for gl_i in range(K):
            if (psi[st_i][gl_i] == 1) and (st_i != gl_i):
                output = outputsubfolder + "SpCoTMHP_S"+str(st_i)+"_G"+str(gl_i) #"navi4"

                #Read the PathWeightMap file
                PathWeightMap = read_data.ReadProbMap_TMHP(outputfile, st_i, gl_i)

                ###v### Add by Ito ###v###
                Path    = [ np.array([ 0.0, 0.0 ]) for i in range(T_horizon) ]
                PathR   = [ np.array([ 0.0, 0.0 ]) for i in range(T_horizon) ]
                i = 0
                ##Mu is read from the file
                for line in open(output+'_Path.csv', 'r'): # T100N6A1S1G4_Path100
                    itemList = line[:-1].split(',')
                    #Mu[i] = np.array([ float(itemList[0]) - origin[0] , float(itemList[1]) - origin[1] ]) / resolution
                    Path[i] = np.array([ float(itemList[0]) , float(itemList[1]) ])
                    #PathR[i]=Map_coordinates_To_Array_index(Path[i])
                    i = i + 1
                    
                PathMap = np.array([[np.inf for j in xrange(map_width)] for i in xrange(map_length)])
                
                for t in xrange(len(Path)):
                    for i in xrange(map_length):
                        for j in xrange(map_width):
                            if (Path[t][0] == i) and (Path[t][1] == j):
                              PathMap[i][j] = 1.0

                #PathMap[142][124]=1.0    
                #print(str(PathWeightMap[142][124]))
                #print(str(PathWeightMap[137][119]))      
                #PathMap[PathR[T_horizon][1]][PathR[T_horizon][0]] = 1.0
                PathWeightMap = PathWeightMap[0:map_width,0:map_length] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
                PathMap = PathMap[0:map_width,0:map_length] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
                gridmap = gridmap[0:map_width,0:map_length]  
                ###^### Add by Ito ###^###
                
                #Add the weights on the map (heatmap)
                plt.imshow(gridmap + (40+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100, interpolation='none') #, vmin = 0.0, vmax = 1.0)
                plt.imshow(PathWeightMap,norm=LogNorm(), origin='lower', cmap='viridis', interpolation='none') #, vmin=wmin, vmax=wmax) #gnuplot, inferno,magma,plasma  #
                #extent=[0, PathWeightMap.shape[1], PathWeightMap.shape[0],0 ] ) #, vmin = 0.0, vmax = 1.0) + np.log((np.exp(wmin)+np.exp(wmax))/2.0)
                #Path=np.array[0.0,0.0 for c in range(L)]
                    
                pp=plt.colorbar (orientation="vertical",shrink=0.8) # Color barの表示 
                pp.set_label("Probability (log scale)", fontname="Arial", fontsize=10) #Color barのラベル
                pp.ax.tick_params(labelsize=8)
                plt.tick_params(axis='x', which='major', labelsize=8)
                plt.tick_params(axis='y', which='major', labelsize=8)
                #plt.xlim([380,800])             #x軸の範囲
                #plt.ylim([180,510])             #y軸の範囲
                plt.xlabel('X', fontsize=10)
                plt.ylabel('Y', fontsize=10)

                plt.savefig(output + '_Weight.pdf', dpi=300, transparent=True)
                plt.savefig(output + '_Weight.png', dpi=300, transparent=True)

                plt.imshow(PathMap, origin='lower', cmap='autumn', interpolation='none')

                #Save the emission probability in the map as a color image 
                #output = outputfile + "S"+str(N_best)+"G"+str(speech_num) #"navi4"

                plt.savefig(output + '_PathWeight.pdf', dpi=300, transparent=True)
                plt.savefig(output + '_PathWeight.png', dpi=300, transparent=True)
                plt.clf()

