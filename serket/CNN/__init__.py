#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#Akira Taniguchi (Ver. Places CNN)
import sys
sys.path.append( "../" )

import serket as srk
import numpy as np
import cv2

dataset_path = "./../../../albert-b/dataset/serket/" #

class CNNFeatureExtractor(srk.Module):
    def __init__(self, filenames, name="CNNFeatureExtracter" ):
        super(CNNFeatureExtractor, self).__init__(name, False)
        self.is_ninitilized = False
        self.filenames = filenames
        
        CNN_folder = "./../../../../PlaceCNN/places365resnet/"
        proto_file = CNN_folder + "deploy_resnet152_places365.prototxt" #"bvlc_googlenet.prototxt"
        caffemodel_file = CNN_folder+ "resnet152_places365.caffemodel" #"bvlc_googlenet.caffemodel"
        net = cv2.dnn.readNetFromCaffe( proto_file, caffemodel_file)
        
        features = []

        for fname in self.filenames:        
            # 必要なファイルを読み込む
            image = cv2.imread( fname )
            
            # 認識処理
            blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
            net.setInput(blob)
            preds = net.forward('prob') #"pool5/7x7_s1"
            #print preds.shape
            f = preds[0, :] #, 0, 0]
            #print f
            features.append(f)
        #print features, len(features)

        #CNN featureをファイル保存
        np.savetxt( dataset_path + "image_place.txt", features, fmt="%f" ) #dataset_path の指定が必要

        self.is_ninitilized = True
    
        self.set_forward_msg( features )

