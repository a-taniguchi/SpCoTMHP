#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#Akira Taniguchi (SpCoSLAM, batch learning) 開発・編集中
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import serket as srk
import mlda
import CNN
import gmm
import GMM_GS
import numpy as np

dataset_path = "./../../../albert-b/dataset/serket/"

def main():
    #print [dataset_path + "image/%03d.jpg"%i for i in range(50)]
    #obs1 = CNN.CNNFeatureExtractor( [dataset_path + "image/%03d.jpg"%i for i in range(50)] )
    obs1 = srk.Observation( np.loadtxt(dataset_path + "image_place.txt") ) # 画像特徴
    obs2 = srk.Observation( np.loadtxt(dataset_path + "histogram_w_place.txt") ) # 単語
    obs3 = srk.Observation( np.loadtxt(dataset_path + "position_place.txt") ) # 位置
    
    #obs1 = srk.Observation( np.loadtxt("feature_place.txt") )       # 視覚  
    #obs2 = srk.Observation( np.loadtxt("mfcc.txt") )        # 聴覚
    #obs3 = srk.Observation( np.loadtxt("tactile.txt") )     # 触覚
    #obs4 = srk.Observation( np.loadtxt("angle.txt") )       # 関節角
    
    #object_category = np.loadtxt( "object_category.txt" )
    #motion_category = np.loadtxt( "motion_category.txt" )
    place_category = np.loadtxt( dataset_path + "place_category.txt" )
    position_category = np.loadtxt( dataset_path + "position_category.txt" )
    
    #gmm1 = gmm.GMM( K=10, category=position_category )
    gmm1 = GMM_GS.GMM_gibbs( K=10, D=2, itr=10, category=position_category ) #true category number K=10
    mlda1 = mlda.MLDA(K=10, weights=[200,200,200], itr=10, category=place_category) #true category number K=9
    #mlda2 = mlda.MLDA(10, [200], category=motion_category)
    #mlda3 = mlda.MLDA(10, [100,100])
    #gmm = sklearnGMM.sklearnGMM(10)
    #gmm = GMM.gmm(10)
    
    gmm1.connect( obs3 )
    mlda1.connect( gmm1, obs1, obs2 )
    #mlda2.connect( obs4 )
    #mlda3.connect( mlda1, mlda2 )
    
    for itr in range(50):
        print( "itr:", itr+1 )
        gmm1.update()
        mlda1.update() 
        #gmm1.update() #GMMは精度が悪いので先にMLDAをやっておいた方がいい。

    
if __name__=="__main__":
    main()
    
    
