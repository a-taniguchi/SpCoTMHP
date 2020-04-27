#coding:utf-8
#The file for setting parameters [SpCoA++, SpCoTMHP (learnSpCoTMHP_SIGVerse.py) for SIGVerse]
#Akira Taniguchi 2020/04/11-2020/04/26-
import numpy as np

################### Parameters ###################
num_iter = 100          # The number of iterations of Gibbs sampling for spatial concept learning
DATA_NUM = 60           # The number of training data #100
word_increment = 1.0    # The increment number of word observation data (BoWs)
dimx = 2                # The number of dimensions of xt (x,y)

terminal_output_prams = 0  # Terminalにサンプリングされたパラメータを表示する (ON:1,OFF:0)
SIGVerse = 1

##### SpCoA++ (word segmentation) #####
## latticelm parameters
knownn       = [3] #[2,3,4]  # The n-gram length of the language model (3)
unkn         = [3] #[3,4]    # The n-gram length of the spelling model (3)
#annealsteps  = [3,5,10]     # The number of annealing steps to perform (3) for SpCoSLAM
#anneallength = [5,10,15]    # The length of each annealing step in iterations (5) for SpCoSLAM

## Parameters for mutual estimation in SpCoA++ 
sample_num = len(knownn)*len(unkn)  # The number of samples (candidates for word segmentation results)  
ITERATION  = 1                      # The number of iterations for mutual estimation
threshold  = 0.01                   # 単語の選択の閾値 in SpCoA++ 
#######################################

################### Change models ###################
nonpara = 1     # Nonparametric Bayes method (ON:1,OFF:0)
UseFT   = 1     # 画像特徴を使う場合(1), 使わない場合(0) 
UseLM   = 0     # 言語モデルを更新する場合(1), しない場合(0) (音声認識・単語分割を含む)

##### Add SpCoTMHP #####
IT_mode         = "HMM"  # "HMM" or "GMM"
transition_type = "sym"  # "sym": (事後ハイパーパラメータの)対象行列化, "left2right": そのまま 
                         # "reverse_replay": 逆順データも入力して学習 (未実装)
sampling_method = "DA"   # "DA": Direct Assignment, "BGS":Blocked Gibbs Sampling (未実装)
########################

################### Initial hyper-parameters ################### 
## Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
if (nonpara == 1):
  L = 20             # The number of spatial concepts (weak-limit number) #50 #100
  K = 20             # The number of position distributions (weak-limit number) #50 #100
  alpha0 = 10.00 / float(L)  # Hyperparameter of Dir(π) for index of spatial concept
  gamma0 = 10.00 / float(K)  # Hyperparameter of Dir(φ) for index of position 
                             #  (GMM mixtured component; spatial concept dependent)
  omega0 = 10.00 / float(K)  # Hyperparameter of Dir(ψ) for index of position distribution
                             #  (HMM transition distribution)
else:
  L = 10             # The number of spatial concepts (Setting the true number)
  K = 10             # The number of position distributions (Setting the true number)
  alpha0 = 1.00      # Hyperparameter of Dir(π) for index of spatial concept
  gamma0 = 0.10      # Hyperparameter of Dir(φ) for index of position distribution
                     #  (GMM mixtured component; spatial concept dependent)
  omega0 = 0.10      # Hyperparameter of Dir(ψ) for index of position distribution
                     #  (HMM transition distribution)

beta0 = 0.1          # Hyperparameter in Dir(W) for place names 
chi0  = 1.0          # Hyperparameter in Dir(θ) for image feature
k0 = 1e-3            # Hyperparameter in Gauss(μ) (Influence degree of prior distribution)
m0 = np.zeros(dimx)  # Hyperparameter in Gauss(μ) (prior mean vector)
V0 = np.eye(dimx)*2  # Hyperparameter in Inverse-Wishart(Σ) (prior covariance matrix) 
n0 = 3.0             # Hyperparameter in Inverse-Wishart(Σ) [n0 > the number of dimenssions] 
                     #  (Influence degree of prior distribution)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))

#################### Option setting ####################
approx_zero = 10.0**(-200)   # approximated value of log(0)

## The number of samples for robust sampling 
Robust_W     = 1#000
Robust_Sig   = 1#00
Robust_Mu    = 1
Robust_pi    = 1#000
Robust_phi   = 1#000
Robust_theta = 1#000
Robust_psi   = 1#000 

## Image feature parameter setting
CNNmode = 1            # Select image feature descriptor
Feture_times = 100.0   # 画像特徴量を何倍するか
Feture_sum_1 = 0       # 画像特徴量を足して１になるようにする(1)
Feture_noize = 0.0     # 画像特徴量に微小ノイズを足す(Feture_noize/DimImg) #approx_zero #10.0**(-5)

if (CNNmode == 1):
  Descriptor = "googlenet_prob_AURO" #"CNN_softmax"
  DimImg     = 1000 # Dimension of image feature
  Feture_times = float(Feture_times)/100.0  # googlenet_probのデータはすでに100倍されている
elif (CNNmode == 2):
  Descriptor = "CNN_fc6"
  DimImg     = 4096 # Dimension of image feature
elif (CNNmode == 3):
  Descriptor = "CNN_Place205"
  DimImg     = 205  # Dimension of image feature
elif (CNNmode == 4):
  Descriptor = "hybridCNN"
  DimImg     = 1183 # Dimension of image feature
elif (CNNmode == 5):
  Descriptor = "CNN_Place365"
  DimImg     = 365  # Dimension of image feature

## For initialization of parameters in Gaussian distribution 
sig_init    = 1.0         # initial scale of covariance matrix S
WallXmin, WallXmax, WallYmin, WallYmax = -10, 10, 10, -10  
                          # limit of map size (for mean vector Mu)
#WallX, WallY = 1600, 1152

## map parameters (origin and resolution are same values to map yaml file.)
origin     = np.array([-10.0, -10.0]) #np.array([x,y]) #np.array([-30.000000, -20.000000])
resolution = 0.1              # m/grid (0.050000)
margin     = 10.0*resolution  # margin value for place area in gird map 
                              #  (margin[m] = margin[grid]*resolution)

## Julius parameters (See syllable.jconf in Julius folder)
JuliusVer      = "v4.4"   # "v.4.3.1"
HMMtype        = "DNN"    # "GMM"
lattice_weight = "AMavg"  # Acoustic likelihood (log-likelihood:"AMavg", likelihood:"exp")
wight_scale    = -1.0     # Scale value of lattice_weight
N_best_number  = 10       # The number of N-best for PRR evaluation (N<=10)

################### Folder PATH ###################
if (JuliusVer == "v4.4"):
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.4/"
else:
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.3.1-linux/"
if (HMMtype == "DNN"):
  lang_init = 'syllableDNN.htkdic' # 初期の単語辞書（./lang_m/フォルダ内）
else:
  lang_init = 'web.000.htkdic'     # 'trueword_syllable.htkdic' #'phonemes.htkdic' 
lmfolder = "/home/akira/Dropbox/SpCoSLAM/learning/lang_m/"

##### NEW #####
inputfolder  = "/mnt/hgfs/D/Dropbox/SpCoSLAM/SpCoTMHP/SIGVerse/dataset/similar/3LDK_small/"
outputfolder = "/mnt/hgfs/D/Dropbox/SpCoSLAM/SpCoTMHP/albert-b/data/"  
# "/home/akira/Dropbox/SpCoNavi/data/"
# akira/Dropbox/SpCoNavi/CoRL/dataset/similar/3LDK_small/3LDK_01/

## Folder of training data set
datasetfolder = inputfolder   # training data set folder
#"/home/akira/Dropbox/SpCoSLAM/rosbag/"
datasets      = ["00","01","02","03","04","05","06","07","08","09","10"] 
#["00","01","04","05","06","09","02","03","07","08","10"] #[dataset1,dataset2]
#data_step_num = 60

## 教示の音声データフォルダ(Ubuntu full path) #*.wav"
speech_folder = inputfolder + "speech/*.wav" 
#"/home/akira/Dropbox/Julius/directory/SpCoSLAM/*.wav"  

## 命令の音声データフォルダ(Ubuntu full path) #*.wav" (SpCoNavi for SIGVerseでは未使用)
#speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav" 

## 位置推定の教示データ(旧 ./../sample/フォルダ内)
PositionDataFile = '/position/position_AURO.csv' #'SpCoSLAM.csv'      # 'test000' 

## Word data folder path
word_folder = "SpCoSLAM_human.csv"
"/name/per_100/word" # "/name/" + example_folder + "word"

## Navigation folder (Other output files are also in same folder.)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv

###### example (for SpCoNavi experiments) #######
example = 0 #2 #1
example_folder = ""
if (example == 1):
  example_folder = "example1/"
  word_folder    = "/name/" + example_folder + "word" # "/name/per_100/word"
elif (example == 2):
  example_folder = "example2/"
  word_folder    = "/name/" + example_folder + "word" # "/name/per_100/word"
#################################################

## True data files for evaluation (評価用正解データファイル)
correct_Ct = 'Ct_correct.csv'        # データごとの正解のCt番号
correct_It = 'It_correct.csv'        # データごとの正解のIt番号
correct_data = 'SpCoSLAM_human.csv'  # データごとの正解の文章（単語列、区切り文字つき）(./data/)
correct_name = 'name_correct.csv'    # データごとの正解の場所の名前（音素列）


#################################################
#M = 2000               # The number of particles (Same value as the condition in learning: 300)
#LAG = 100 + 1          # The number of elements of array (lag value for smoothing + 1)

# Motion model parameters (TABLE 5.6 in Probabilistic Robotics)
#para1 = 0.01  #0.50
#para2 = 0.01  #0.05
#para3 = 0.2   #0.8
#para4 = 0.5   #20.0
#para_s = [0,para1,para2,para3,para4] #最初の0は配列番号とpara番号を合わせるためのもの

# Sensor model parameters
#sig_hit2 = 2.0  #Note the parameter value. (default: 3)

#if (CNNmode == 0):
#  Descriptor = "SIFT_BoF"
#  DimImg     = 100  #Dimension of image feature
#el