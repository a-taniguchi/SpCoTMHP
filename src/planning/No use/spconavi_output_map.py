#coding:utf-8
#Akira Taniguchi 2019/01/22-2019/02/05-
#For Visualization of Posterior emission probability (PathWeightMap) on the grid map
import sys
#from math import pi as PI
#from math import cos,sin,sqrt,exp,log,fabs,fsum,degrees,radians,atan2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from __init__ import *
from submodules import *
##Command: 
##python ./weight_visualizer.py alg2wicWSLAG10lln008 8


#Read the map data⇒2-dimension array
def ReadMap(outputfile):
    #outputfolder + trialname + navigation_folder + map.csv
    gridmap = np.loadtxt(outputfile + "map.csv", delimiter=",")
    print "Read map: " + outputfile + "map.csv"
    return gridmap

#Load the probability value map used for path calculation
def ReadProbMap(outputfile):
    # Read the result file
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
    PathWeightMap = np.loadtxt(output, delimiter=",")
    print "Read PathWeightMap: " + output
    return PathWeightMap

#ROSの地図座標系をPython内の2次元配列のインデックス番号に対応付ける
def Map_coordinates_To_Array_index(X):
    X = np.array(X)
    Index = np.round( (X - origin) / resolution ).astype(int) #四捨五入してint型にする
    return Index

#Python内の2次元配列のインデックス番号からROSの地図座標系への変換
def Array_index_To_Map_coordinates(Index):
    Index = np.array(Index)
    X = np.array( (Index * resolution) + origin )
    return X


########################################
if __name__ == '__main__': 
    #Request a folder name for learned parameters.
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")

    #Request the file number of the speech instruction      
    speech_num = sys.argv[2] #0
  
    ##FullPath of folder
    #filename = datafolder + trialname + "/" + str(step) +"/"
    #print filename #, particle_num
    outputfile = outputfolder_SIG + trialname + navigation_folder

    #Read the map file
    gridmap = ReadMap(outputfile)

    #Read the PathWeightMap file
    PathWeightMap = ReadProbMap(outputfile)

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
    map_length = len(PathWeightMap)  #len(costmap)
    map_width  = len(PathWeightMap[0])  #len(costmap[0])
    print "MAP[length][width]:",map_length,map_width

    #Add the weights on the map (heatmap)
    plt.imshow(gridmap + (40+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100, interpolation='none') #, vmin = 0.0, vmax = 1.0)
    plt.imshow(PathWeightMap,norm=LogNorm(), origin='lower', cmap='viridis', interpolation='none') #, vmin=wmin, vmax=wmax) #gnuplot, inferno,magma,plasma  #
    #extent=[0, PathWeightMap.shape[1], PathWeightMap.shape[0],0 ] ) #, vmin = 0.0, vmax = 1.0) + np.log((np.exp(wmin)+np.exp(wmax))/2.0)
    

    pp=plt.colorbar (orientation="vertical",shrink=0.8) # Color barの表示 
    pp.set_label("Probability (log scale)", fontname="Arial", fontsize=10) #Color barのラベル
    pp.ax.tick_params(labelsize=8)
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.tick_params(axis='y', which='major', labelsize=8)
    #plt.xlim([380,800])             #x軸の範囲
    #plt.ylim([180,510])             #y軸の範囲
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)

    #Save the emission probability in the map as a color image
    output = outputfile + "N"+str(N_best)+"G"+str(speech_num)
    #plt.savefig(output + '_Weight.eps', dpi=300)#, transparent=True
    plt.savefig(output + '_Weight.png', dpi=300)#, transparent=True
    plt.savefig(output + '_Weight.pdf', dpi=300, transparent=True)#, transparent=True
    plt.clf()

    #plt.show()
    