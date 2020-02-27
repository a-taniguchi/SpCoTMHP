#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import sys
sys.path.append("../../")

import serket as srk
import gmm
import mm
import mlda
import numpy as np

def main():    
    obs = srk.Observation( np.loadtxt("./data.txt") )
    #obs1 = srk.Observation( np.loadtxt("./data/data1.txt") )
    #obs2 = srk.Observation( np.loadtxt("./data/data2.txt") )
    #data_category = np.loadtxt( "./data/category.txt" )
    correct_classes = np.loadtxt( "correct.txt" )
    
    # SpCoSLAM (Serketizetion)
    gmm1 = gmm.GMM( K=4, category=correct_classes )
    #mm1 = mm.MarkovModel()
    mlda1 = mlda.MLDA( K=4, weights=[200,200], category=correct_classes )

    gmm1.connect( obs )
    #mm1.connect( gmm1 )
    mlda1.connect( obs, gmm1 )
    
    for itr in range(5):
        print( "itr:", itr+1 )
        gmm1.update()
        #mm1.update()
        mlda1.update()

if __name__=="__main__":
    main()
    
    