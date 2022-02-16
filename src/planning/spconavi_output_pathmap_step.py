#coding:utf-8
#Akira Taniguchi 2022/02/07
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
#python ./spconavi_output_pathmap_step.py trialname init_position_num speech_num initial_position_x initial_position_y
#Example: python ./spconavi_output_pathmap_step.py 3LDK_01 0 7 100 100

#output = "/root/HSR/catkin_ws/src/spconavi_ros/src/data/3LDK_01/navi/Astar_Approx_expect_N6A1SG7" + "_Path" + str(temp) + ".csv" # A*用


########################################
if __name__ == '__main__': 
    #Request a folder name for learned parameters.
    trialname = sys.argv[1]
    #print trialname
    #trialname = raw_input("trialname?(folder) >")

    #Request the index number of the robot initial position
    init_position_num = sys.argv[2] #0

    #Request the file number of the speech instruction   
    speech_num = sys.argv[3] #0
  
    start = [int(sys.argv[4]), int(sys.argv[5])] #Start_Position[int(init_position_num)]
    #start[0] = int(sys.argv[4]) #0
    #start[1] = int(sys.argv[5]) #0
    #start = [start_list[0], start_list[1]]
    print("Start:", start)


    #init_position_num = 0
    X_init = start #Start_Position[int(init_position_num)]
    print(X_init)

    ##FullPath of folder
    filename = outputfolder_SIG + trialname #+ "/" 
    #print(filename, iteration, sample)
    outputfile = filename + navigation_folder #outputfolder + trialname + navigation_folder
    outputsubfolder = outputfile + "spconavi_viterbi/"
    #outputname = outputsubfolder + "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)+"/"  

    conditions = "T"+str(T_horizon)+"N"+str(N_best)+"A"+str(Approx)+"S"+str(start)+"G"+str(speech_num)+"/"  
    outputname = outputsubfolder + conditions
    
    Makedir(outputfile + "step")

    temp = T_horizon #400
    for temp in range(SAVE_T_temp,T_horizon+SAVE_T_temp,SAVE_T_temp):
      #Read the map file
      gridmap = read_data.ReadMap(outputfile)

      #Read the PathWeightMap file
      PathWeightMap = read_data.ReadProbMap(outputfile,speech_num)

      #Read the Path file
      Path = read_data.ReadPath_step(outputname,temp)
      #Makedir( outputfile + "step" )
      #print(Path)
    
      #length and width of the MAP cells
      map_length = len(gridmap)  #len(costmap)
      map_width  = len(gridmap[0])  #len(costmap[0])

      #パスの２-dimension array を作成
      PathMap = np.array([[np.inf for j in xrange(map_width)] for i in xrange(map_length)])
      
      for i in xrange(map_length):
        for j in xrange(map_width):
            if (X_init[1] == i) and (X_init[0] == j):
              PathMap[i][j] = 1.0
            for t in xrange(len(Path)):
              if ( Path[t][0] == i ) and ( Path[t][1] == j ): ################バグがないならこっちを使う
                #if ( int(Path[t][0] -X_init[0]+T_horizon) == i) and ( int(Path[t][1] -X_init[1]+T_horizon) == j): ################バグに対処療法した
                PathMap[i][j] = 1.0
      
      """
      y_min = 380 #X_init_index[0] - T_horizon
      y_max = 800 #X_init_index[0] + T_horizon
      x_min = 180 #X_init_index[1] - T_horizon
      x_max = 510 #X_init_index[1] + T_horizon
      #if (x_min>=0 and x_max<=map_width and y_min>=0 and y_max<=map_length):
      PathWeightMap = PathWeightMap[x_min:x_max, y_min:y_max] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
      PathMap = PathMap[x_min:x_max, y_min:y_max] # X[-T+I[0]:T+I[0],-T+I[1]:T+I[1]]
      gridmap = gridmap[x_min:x_max, y_min:y_max]
      """

      #length and width of the MAP cells
      map_length = len(gridmap)  #len(costmap)
      map_width  = len(gridmap[0])  #len(costmap[0])
      #print("MAP[length][width]:",map_length,map_width)

      #Add the weights on the map (heatmap)
      plt.imshow(gridmap + (40+1)*(gridmap == -1), origin='lower', cmap='binary', vmin = 0, vmax = 100, interpolation='none') #, vmin = 0.0, vmax = 1.0)
      plt.imshow(PathWeightMap,norm=LogNorm(), origin='lower', cmap='viridis', interpolation='none') #, vmin=wmin, vmax=wmax) #gnuplot, inferno,magma,plasma  #
    

      pp=plt.colorbar (orientation="vertical",shrink=0.8) # Color barの表示 
      pp.set_label("Probability (log scale)", fontname="Arial", fontsize=10) #Color barのラベル
      pp.ax.tick_params(labelsize=8)
      plt.tick_params(axis='x', which='major', labelsize=8)
      plt.tick_params(axis='y', which='major', labelsize=8)
      #plt.xlim([380,800])             #x軸の範囲
      #plt.ylim([180,510])             #y軸の範囲
      plt.xlabel('X', fontsize=10)
      plt.ylabel('Y', fontsize=10)

      plt.savefig(outputname + '_Weight' +  str(temp).zfill(3) + '.png', dpi=300)#, transparent=True
      plt.savefig(outputname + '_Weight' +  str(temp).zfill(3) + '.pdf', dpi=300, transparent=True)#, transparent=True

      plt.imshow(PathMap, origin='lower', cmap='autumn', interpolation='none') #, vmin=wmin, vmax=wmax) #gnuplot, inferno,magma,plasma  #


      #Save path trajectory and the emission probability in the map as a color image
      plt.savefig(outputname + '_Path_Weight' +  str(temp).zfill(3) + '.png', dpi=300)#, transparent=True
      plt.savefig(outputname + '_Path_Weight' +  str(temp).zfill(3) + '.pdf', dpi=300, transparent=True)#, transparent=True
      plt.clf()

    #plt.show()
    
