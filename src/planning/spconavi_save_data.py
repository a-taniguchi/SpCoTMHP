#!/usr/bin/env python
#coding:utf-8
from __init__ import *
from submodules import *
from scipy.io import mmwrite
import collections
import spconavi_read_data

read_data = spconavi_read_data.ReadingData()


class SavingData:

    #Save the path trajectory
    def SavePath(self, X_init, Path, Path_ROS, outputname):
        print "PathSave"
        if (SAVE_X_init == 1):
            # Save the robot initial position to the file (index)
            np.savetxt(outputname + "_X_init.csv", X_init, delimiter=",")
            # Save the robot initial position to the file (ROS)
            np.savetxt(outputname + "_X_init_ROS.csv", read_data.Array_index_To_Map_coordinates(X_init), delimiter=",")

        # Save the result to the file (index)
        np.savetxt(outputname + "_Path.csv", Path, delimiter=",")
        # Save the result to the file (ROS)
        np.savetxt(outputname + "_Path_ROS.csv", Path_ROS, delimiter=",")
        print "Save Path: " + outputname + "_Path.csv and _Path_ROS.csv"

    #Save the path trajectory
    def SavePathTemp(self, X_init, Path_one, temp, outputname, IndexMap_one_NOzero, Bug_removal_savior):
        print "PathSaveTemp"

        #one-dimension array index を2-dimension array index へ⇒ROSの座標系にする
        Path_2D_index = np.array([ IndexMap_one_NOzero[Path_one[i]] for i in xrange(len(Path_one)) ])
        if ( Bug_removal_savior == 0):
            Path_2D_index_original = Path_2D_index + np.array(X_init) - T_horizon
        else:
            Path_2D_index_original = Path_2D_index
        Path_ROS = read_data.Array_index_To_Map_coordinates(Path_2D_index_original) #

        #Path = Path_2D_index_original #Path_ROS #必要な方をPathとして返す
        # Save the result to the file (index)
        np.savetxt(outputname + "_Path" + str(temp) + ".csv", Path_2D_index_original, delimiter=",")
        # Save the result to the file (ROS)
        np.savetxt(outputname + "_Path_ROS" + str(temp) + ".csv", Path_ROS, delimiter=",")
        print "Save Path: " + outputname + "_Path" + str(temp) + ".csv and _Path_ROS" + str(temp) + ".csv"

    def SaveTrellis(self, trellis, outputname, temp):
        print "SaveTrellis"
        # Save the result to the file 
        np.save(outputname + "_trellis" + str(temp) + ".npy", trellis) #, delimiter=",")
        print "Save trellis: " + outputname + "_trellis" + str(temp) + ".npy"

    #パス計算のために使用したLookupTable_ProbCtをファイル保存する
    def SaveLookupTable(self, LookupTable_ProbCt, outputfile):
        # Save the result to the file 
        output = outputfile + "LookupTable_ProbCt.csv"
        np.savetxt( output, LookupTable_ProbCt, delimiter=",")
        print "Save LookupTable_ProbCt: " + output

    #パス計算のために使用した確率値コストマップをファイル保存する
    def SaveCostMapProb(self, CostMapProb, outputfile):
        # Save the result to the file 
        output = outputfile + "CostMapProb.csv"
        np.savetxt( output, CostMapProb, delimiter=",")
        print "Save CostMapProb: " + output

    #パス計算のために使用した確率値マップを（トピックかサービスで）送る
    #def SendProbMap(self, PathWeightMap):

    #Save the probability value map used for path calculation
    def SaveProbMap(self, PathWeightMap, outputfile, speech_num):
        # Save the result to the file 
        output = outputfile + "N"+str(N_best)+"G"+str(speech_num) + "_PathWeightMap.csv"
        np.savetxt( output, PathWeightMap, delimiter=",")
        print "Save PathWeightMap: " + output

    def SaveTransition(self, Transition, outputfile):
        # Save the result to the file 
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_log.csv"
        #np.savetxt(outputfile + "_Transition_log.csv", Transition, delimiter=",")
        f = open( output_transition , "w")
        for i in xrange(len(Transition)):
            for j in xrange(len(Transition[i])):
                f.write(str(Transition[i][j]) + ",")
            f.write('\n')
        f.close()
        print "Save Transition: " + output_transition

    def SaveTransition_sparse(self, Transition, outputfile):
        # Save the result to the file (.mtx形式)
        output_transition = outputfile + "T"+str(T_horizon) + "_Transition_sparse"
        mmwrite(output_transition, Transition)

        print "Save Transition: " + output_transition

    #Save the log likelihood for each time-step
    def SaveLogLikelihood(self, LogLikelihood,flag,flag2, outputname):
        # Save the result to the file 
        if (flag2 == 0):
            if   (flag == 0):
                output_likelihood = outputname + "_Log_likelihood_step.csv"
            elif (flag == 1):
                output_likelihood = outputname + "_Log_likelihood_sum.csv"
        else:
            if   (flag == 0):
                output_likelihood = outputname + "_Log_likelihood_step" + str(flag2) + ".csv"
            elif (flag == 1):
                output_likelihood = outputname + "_Log_likelihood_sum" + str(flag2) + ".csv"

        np.savetxt( output_likelihood, LogLikelihood, delimiter=",")
        print "Save LogLikekihood: " + output_likelihood

    #Save the moving distance of the path
    def SavePathDistance(self, Distance, outputname):
        # Save the result to the file 
        output = outputname + "_Distance.csv"
        np.savetxt( output, np.array([Distance]), delimiter=",")
        print "Save Distance: " + output

    #Save the moving distance of the path
    def SavePathDistance_temp(self, Distance,temp, outputname):
        # Save the result to the file 
        output = outputname + "_Distance"+str(temp)+".csv"
        np.savetxt( output, np.array([Distance]), delimiter=",")
        print "Save Distance: " + output
