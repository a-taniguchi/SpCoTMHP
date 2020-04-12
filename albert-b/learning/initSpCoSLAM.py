#coding:utf-8
#The file for setting parameters [SpCoSLAMとの比較用(learn4_3.py対応) for albert-b]
#Akira Taniguchi 2020/04/11-
import numpy as np

####################Parameters####################
kyouji_count = 50 #100  # The number of training data
M = 2000                # The number of particles (Same value as the condition in learning: 300)
#LAG = 100 + 1          # The number of elements of array (lag value for smoothing + 1)

num_iter = 100          # The number of iterations of Gibbs sampling for spatial concept learning
dimx = 2                # The number of dimensions of xt (x,y)

#limit of map size
#WallX = 1600
#WallY = 1152
WallXmin = -10
WallXmax = 10
WallYmin = 10
WallYmax = -10

#Motion model parameters (TABLE 5.6 in Probabilistic Robotics)
#para1 = 0.01  #0.50
#para2 = 0.01  #0.05
#para3 = 0.2   #0.8
#para4 = 0.5   #20.0
#para_s = [0,para1,para2,para3,para4] #最初の0は配列番号とpara番号を合わせるためのもの

#Sensor model parameters
#sig_hit2 = 2.0  #Note the parameter value. (default: 3)

##Initial (hyper)-parameters
##Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
L = 50               #The number of spatial concepts #50 #100
K = 50               #The number of position distributions #50 #100
alpha0 = 20.0        #Hyperparameter of multinomial distribution for index of spatial concept
gamma0 = 0.1         #Hyperparameter of multinomial distribution for index of position distribution
beta0 = 0.1          #Hyperparameter in multinomial distribution P(W) for place names 
chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
k0 = 1e-3            #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*2  #Hyperparameter in Inverse Wishart distribution P(Σ) (prior covariance matrix) 
n0 = 3.0             #Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))

"""
alpha = 10.0 #0.1 #10.0 #5.0#1.5#10.0               #位置分布のindexの多項分布のハイパーパラメータ1.5#
gamma = 10.0 #20.0 #15.0#8.0#20.0               #場所概念のindexの多項分布のハイパーパラメータ8.0#
beta0 = 0.1 #0.4#0.2               #場所の名前Wのハイパーパラメータ0.5#
kappa0 = 1e-3                #μのパラメータ、旧モデルのbeta0であることに注意
m0 = np.array([[0.0],[0.0]])   #μのパラメータ
V0 = np.eye(2)*2 #*1000              #Σのパラメータ
nu0 = 3.0 #3.0                    #Σのパラメータ、旧モデルでは1としていた(自由度の問題で2の方が良い?)、事後分布の計算のときに1加算していた
"""

sig_init =  10.0 

##latticelm parameters
knownn       = [2,3,4] #[3] #The n-gram length of the language model (3)
unkn         = [3,4] #[3]   #The n-gram length of the spelling model (3)
annealsteps  = [3,5,10]     #The number of annealing steps to perform (3)
anneallength = [5,10,15]    #The length of each annealing step in iterations (5)


##Parameters for mutual estimation in SpCoA++ 
sample_num = len(knownn)*len(unkn)  #The number of samples (candidates for word segmentation results)  #len(knownn)*len(unkn)  
ITERATION = 10                      #The number of iterations for mutual estimation

##単語の選択の閾値
threshold = 0.01

#Plot = 2000#1000  #位置分布ごとの描画の点プロット数
#N_best_number = 10 #n-bestのnをどこまでとるか（n<=10）

#Julius parameters
#Juliusフォルダのsyllable.jconf参照
JuliusVer      = "v4.4"   #"v.4.3.1"
HMMtype        = "DNN"    #"GMM"
lattice_weight = "AMavg"  #"exp" #acoustic likelihood (log likelihood: "AMavg", likelihood: "exp")
wight_scale    = -1.0

if (JuliusVer ==  "v4.4"):
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.4/"
else:
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.3.1-linux/"

if (HMMtype == "DNN"):
  lang_init = 'syllableDNN.htkdic' 
else:
  lang_init = 'web.000.htkdic' # 'trueword_syllable.htkdic' #'phonemes.htkdic' # 初期の単語辞書（./lang_mフォルダ内）
#lang_init_DNN = 'syllableDNN.htkdic' #なごり

#################### Folder PATH ####################
speech_folder = "/home/*/Dropbox/Julius/directory/SpCoSLAM/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
data_name = 'SpCoSLAM.csv'      # 'test000' #位置推定の教示データ(./../sampleフォルダ内)
lmfolder = "/home/akira/Dropbox/SpCoSLAM/learning/lang_m/"
#lang_init = 'web.000.htkdic' #'phonemes.htkdic' #  初期の単語辞書（./lang_mフォルダ内）

#Folder of training data set (rosbag file)
datasetfolder = "/home/akira/Dropbox/SpCoSLAM/rosbag/"   #training data set folder
dataset1      = "albert-b-laser-vision/albert-B-laser-vision-dataset/"
bag1          = "albertBimg.bag"  #Name of rosbag file
datasets      = [dataset1] #[dataset1,dataset2]
bags          = [bag1] #run_rosbag.pyにて使用
scantopic     = ["scan"] #, "base_scan _odom_frame:=odom_combined"]

#dataset2      = "MIT_Stata_Center_Data_Set/"   ##用意できてない
#datasets      = {"albert":dataset1,"MIT":dataset2}
#CNNfolder     = "/home/*/CNN/CNN_Places365/"                        #Folder of CNN model files

#True data files for evaluation (評価用正解データファイル)
correct_Ct = 'Ct_correct.csv'        #データごとの正解のCt番号
correct_It = 'It_correct.csv'        #データごとの正解のIt番号
correct_data = 'SpCoSLAM_human.csv'  #データごとの正解の文章（単語列、区切り文字つき）(./data/)
correct_name = 'name_correct.csv'    #データごとの正解の場所の名前（音素列）

N_best_number = 10  # The number of N of N-best for PRR evaluation (PRR評価用のN-bestのN) (N<=10)
margin = 10*0.05    # margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)
