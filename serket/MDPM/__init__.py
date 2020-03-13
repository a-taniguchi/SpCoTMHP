#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# 2020/03/12 Akira Taniguchi 編集途中(未完成)
import sys
sys.path.append( "../" )

from . import mdpm
import serket as srk
import numpy as np

import os


class MDPM(srk.Module):
    def __init__( self, K, weights=None, itr=100, name="mdpm", category=None, load_dir=None ):
        super(MDPM, self).__init__(name, True)
        self.__K = K
        self.__weights = weights
        self.__itr = itr
        self.__category = category
        self.__load_dir = load_dir
        self.__n = 0
        
    def update(self):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)
        #print "0", data[0]
        #print "1", data[1]
        #print "2", data[2]
        M = len( data )     # モダリティ数
        N = len( data[0] )  # データ数
        
        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones( (N, self.__K) ) / self.__K
    
        # データの正規化処理
        for m in range(M):     
            data[m][ data[m]<0 ] = 0
            
        if self.__weights is not None:
            for m in range(M):
                data[m] = (data[m].T / data[m].sum(1)).T * self.__weights[m]
        
        for m in range(M):
            data[m] = np.array( data[m], dtype=np.int32 )
        
        if self.__load_dir is None:
            save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
        else:
            save_dir = os.path.join( self.get_name(), "recog" )
        
        # MDPM学習
        #Pdz, Pdw = mdpm.train( data, self.__K, self.__itr, save_dir, Pdz, self.__category, self.__load_dir )
        multimodal_dpm     = Multimodal_DPM( data, self.__K, self.__itr, save_dir, Pdz, self.__category, self.__load_dir ) # , 10.0, 30)
        multimodal_dpm.fit() # learning by Gibbs Sampling
        Pdz, Pdw = multimodal_dpm.message() # メッセージの確率を計算
        
        self.__n += 1
        
        # メッセージの送信
        self.set_forward_msg( Pdz )
        self.send_backward_msgs( Pdw )