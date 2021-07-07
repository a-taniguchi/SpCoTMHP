#coding:utf-8
#The file for setting parameters (learning for SpCoNavi,SpCoTMHP on SIGVerse; for learnSpCoTMHP.py)
#Akira Taniguchi 2020/04/11-2021/07/06
import numpy as np

################### Parameters ###################
LEARN_MODE = "GT"
if (LEARN_MODE == "GT"):
  num_iter = 1
else:
  num_iter = 100          # The number of iterations of Gibbs sampling for spatial concept learning
SIGVerse = 1

word_increment = 1.0    # The increment number of word observation data (BoWs)
dimx = 2                # The number of dimensions of xt (x,y)

terminal_output_prams = 0  # Terminalにサンプリングされたパラメータを表示する (ON:1,OFF:0)


##### SpCoA++ (word segmentation) #####
## latticelm parameters
#knownn       = [2,3,4] #[3] #The n-gram length of the language model (3)
#unkn         = [3,4] #[3]   #The n-gram length of the spelling model (3)
#annealsteps  = [3,5,10]     #The number of annealing steps to perform (3)
#anneallength = [5,10,15]    #The length of each annealing step in iterations (5)

## Parameters for mutual estimation in SpCoA++ (Cannot change in this code.)
sample_num = 1  #The number of samples (candidates for word segmentation results)  #len(knownn)*len(unkn)  
ITERATION  = 1  #The number of iterations for mutual estimation
threshold  = 0.01                   # 単語の選択の閾値 in SpCoA++ 
#######################################

################### Change models ###################
nonpara = 0     # Nonparametric Bayes method (ON:1,OFF:0)
UseFT   = 0     # 画像特徴を使う場合(1), 使わない場合(0) 
UseLM   = 0     # 言語モデルを更新する場合(1), しない場合(0) (音声認識・単語分割を含む．音声データが必要)

##### Add SpCoTMHP #####
IT_mode         = "GMM"  # "HMM" or "GMM"
transition_type = "left2right"  # "sym": (事後ハイパーパラメータの)対象行列化, "left2right": そのまま 
                         # "reverse_replay": 逆順データも入力して学習 (未実装)
sampling_method = "DA"   # "DA": Direct Assignment, "BGS":Blocked Gibbs Sampling (未実装)
########################

################### Initial hyper-parameters ################### 
## Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
if (nonpara == 1):
  L = 50             # The number of spatial concepts #50 #100
  K = 50             # The number of position distributions #50 #100
  alpha0 = 20.00 / float(L)  # Hyperparameter of multinomial distribution for index of spatial concept
  gamma0 = 10.00 / float(K)  # Hyperparameter of multinomial distribution for index of position 
                             #  (GMM mixtured component; spatial concept dependent)
  omega0 = 10.00 / float(K)  # Hyperparameter of Dir(ψ) for index of position distribution
                             #  (HMM transition distribution)
else:
  L = 10             # The number of spatial concepts #50 #100
  K = 10             # The number of position distributions #50 #100
  alpha0 = 1.00      # Hyperparameter of multinomial distribution for index of spatial concept
  gamma0 = 0.10      # Hyperparameter of multinomial distribution for index of position distribution
                     #  (GMM mixtured component; spatial concept dependent)
  omega0 = 0.10      # Hyperparameter of Dir(ψ) for index of position distribution
                     #  (HMM transition distribution)

beta0 = 0.1          # Hyperparameter in multinomial distribution P(W) for place names 
chi0  = 0.1          # Hyperparameter in multinomial distribution P(φ) for image feature
k0 = 1e-3            # Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
m0 = np.zeros(dimx)  # Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*2  # Hyperparameter in Inverse Wishart distribution P(Σ) (prior covariance matrix) 
n0 = 3.0             # Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
                     #  (Influence degree of prior distribution)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))

#################### Option setting ####################
approx_zero = 10.0**(-200)   # approximated value of log(0)

## The number of samples for robust sampling 
Robust_W     = 1000
Robust_Sig   = 100
Robust_Mu    = 100 #1
Robust_pi    = 1000
Robust_phi   = 1000
Robust_theta = 1000
Robust_psi   = 1000 

## Image feature parameter setting
if (SIGVerse == 1):
  CNNmode = 1            # Select image feature descriptor
else:
  CNNmode = 5            # Select image feature descriptor
Feture_times = 10.0    # 画像特徴量を何倍するか(SIGVerse dataの場合100を推奨)
Feture_sum_1 = 0       # 画像特徴量を足して１になるようにする(1)
Feture_noize = 1.0     # 画像特徴量に微小ノイズを足す  #approx_zero #10.0**(-5)
Feture_int   = 1       # 画像特徴量を整数化（ヒストグラムのカウント）

if (CNNmode == 1):
  Descriptor = "googlenet_prob_AURO" #"CNN_softmax" (for SIGverse dataset)
  DimImg     = 1000 # Dimension of image feature
  Feture_times = float(Feture_times)/100.0  # googlenet_probのデータはすでに100倍されている
elif (CNNmode == 2):
  Descriptor = "CNN_fc6"
  DimImg     = 4096 # Dimension of image feature
elif (CNNmode == 3):
  Descriptor = "places205"
  DimImg     = 205  # Dimension of image feature
elif (CNNmode == 5):
  Descriptor = "places365"
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

## For SpCoVisualizer.py ############################################
# インデックス毎のカラーリスト
#colorlist = ['red','green','blue','cyan','magenta','darkblue','orange','purple','yellowgreen','yellow','darkred']
COLOR = ['red','green','blue','cyan','magenta','darkblue','darkorange','purple','yellowgreen','goldenrod','darkred',
  #[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],
  [0.5,0.5,0.0],[0.5,0.0,0.5], #4
  [0.0,0.5,0.5],#[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.6,0.2,0.2],#9
  #[0.2,0.6,0.2],[0.2,0.2,0.6],[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4], #14
  [0.7,0.2,0.1],[0.7,0.1,0.2],[0.2,0.7,0.1],[0.1,0.7,0.2],[0.2,0.1,0.7],#19
  [0.1,0.2,0.7],[0.5,0.2,0.3],[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.5,0.3],#24
  [0.3,0.2,0.5],[0.2,0.3,0.5],[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7],#29
  [0.6,0.3,0.1],[0.6,0.1,0.3],[0.1,0.6,0.3],[0.3,0.6,0.1],[0.3,0.1,0.6],#34
  [0.1,0.3,0.6],[0.8,0.2,0.0],[0.8,0.0,0.2],[0.2,0.8,0.0],[0.0,0.8,0.2],#39
  [0.2,0.0,0.8],[0.0,0.2,0.8],[0.7,0.3,0.0],[0.7,0.0,0.3],[0.3,0.7,0.0],#44
  [0.3,0.0,0.7],[0.0,0.7,0.3],[0.0,0.3,0.7],[0.25,0.25,0.5],[0.25,0.5,0.25], #49
  [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.5,0.5,0.0],[0.5,0.0,0.5], #54
  [0.0,0.5,0.5],[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.6,0.2,0.2],#59
  [0.2,0.6,0.2],[0.2,0.2,0.6],[0.4,0.3,0.3],[0.3,0.4,0.3],[0.3,0.3,0.4], #64
  [0,7,0.2,0.1],[0.7,0.1,0.2],[0.2,0.7,0.1],[0.1,0.7,0.2],[0.2,0.1,0.7],#69
  [0.1,0.2,0.7],[0.5,0.2,0.3],[0.5,0.3,0.2],[0.3,0.5,0.2],[0.2,0.5,0.3],#74
  [0.3,0.2,0.5],[0.2,0.3,0.5],[0.7,0.15,0.15],[0.15,0.7,0.15],[0.15,0.15,0.7],#79
  [0.6,0.3,0.1],[0.6,0.1,0.3],[0.1,0.6,0.3],[0.3,0.6,0.1],[0.3,0.1,0.6],#84
  [0.1,0.3,0.6],[0.8,0.2,0.0],[0.8,0.0,0.2],[0.2,0.8,0.0],[0.0,0.8,0.2],#89
  [0.2,0.0,0.8],[0.0,0.2,0.8],[0.7,0.3,0.0],[0.7,0.0,0.3],[0.3,0.7,0.0],#94
  [0.3,0.0,0.7],[0.0,0.7,0.3],[0.0,0.3,0.7],[0.25,0.25,0.5],[0.25,0.5,0.25] #99
]

# 楕円の確率値
el_prob  = 0.68 # 1σ:0.68, 2σ:0.95
#el_c     = np.sqrt(ss.chi2.ppf(el_prob, 2))

#####################################################################

## Julius parameters (See syllable.jconf in Julius folder)
JuliusVer      = "v4.4"   # "v.4.3.1"
HMMtype        = "DNN"    # "GMM"
lattice_weight = "AMavg"  # Acoustic likelihood (log-likelihood:"AMavg", likelihood:"exp")
wight_scale    = -1.0     # Scale value of lattice_weight
N_best_number  = 10       # The number of N-best for PRR evaluation (N<=10)

################### Folder PATH ###################
if (JuliusVer ==  "v4.4"):
  Juliusfolder = "/mnt/hgfs/Dropbox/Julius/dictation-kit-v4.4/"
else:
  Juliusfolder = "/mnt/hgfs/Dropbox/Julius/dictation-kit-v4.3.1-linux/"

if (HMMtype == "DNN"):
  lang_init = 'syllableDNN.htkdic' # 初期の単語辞書（./lang_m/フォルダ内）
else:
  lang_init = 'web.000.htkdic'     # 'trueword_syllable.htkdic' #'phonemes.htkdic' 
lmfolder = "/home/akira/Dropbox/SpCoSLAM/learning/lang_m/"

## Folder of training data set
inputfolder  = "/mnt/hgfs/D/akira/Dropbox/SpCoSLAM/SpCoTMHP/SIGVerse/dataset/similar/3LDK/"  #"/home/akira/Dropbox/SpCoNavi/data/"
## Output folder
outputfolder = "/mnt/hgfs/D/akira/Dropbox/SpCoSLAM/SpCoTMHP/SIGVerse/data/" #+ example_folder  #"/home/akira/Dropbox/SpCoNavi/data/"
## map fileのフォルダファイル名 (**.pgm and **.yaml)
map_file = "/map/s"

# Word data folder path
word_folder = "/name/per_100/word"
#word_folder = "/name/" + example_folder + "word" # "/name/per_100/word"

## Image data folder path
ImageFolder = ""

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








#################### Parameters ####################
#kyouji_count = 50 #100 # The number of training data
#M = 2000               # The number of particles (Same value as the condition in learning: 300)
#LAG = 100 + 1          # The number of elements of array (lag value for smoothing + 1)



#limit of map size
#WallX = 1600
#WallY = 1152
WallXmin = -10
WallXmax = 10
WallYmin = 10
WallYmax = -10


#margin = 10*0.05   # margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)

#Motion model parameters (TABLE 5.6 in Probabilistic Robotics)
#para1 = 0.01  #0.50
#para2 = 0.01  #0.05
#para3 = 0.2   #0.8
#para4 = 0.5   #20.0
#para_s = [0,para1,para2,para3,para4] #最初の0は配列番号とpara番号を合わせるためのもの

#Sensor model parameters
#sig_hit2 = 2.0  #Note the parameter value. (default: 3)






"""
#################### Folder PATH ####################
speech_folder = "/home/*/Dropbox/Julius/directory/SpCoSLAM/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
data_name = 'SpCoSLAM.csv'      # 'test000' #位置推定の教示データ(./../sampleフォルダ内)
lmfolder = "/home/akira/Dropbox/SpCoSLAM/learning/lang_m/"
#lang_init = 'web.000.htkdic' #'phonemes.htkdic' #  初期の単語辞書 (./lang_mフォルダ内) 

datasetfolder = "/home/akira/Dropbox/SpCoSLAM/rosbag/"
dataset1 = "albert-b-laser-vision/albert-B-laser-vision-dataset/"
bag1 = "albertBimg.bag"
dataset2 = "MIT_Stata_Center_Data_Set/"   ##用意できてない
#datasets = {"albert":dataset1,"MIT":dataset2}
datasets = [dataset1,dataset2]
bags = [bag1]
scantopic = ["scan", "base_scan _odom_frame:=odom_combined"]
#map_data : ./jygame/__inti__.py 

## True data files for evaluation (評価用正解データファイル)
correct_Ct = 'Ct_correct.csv'  #データごとの正解のCt番号
correct_It = 'It_correct.csv'  #データごとの正解のIt番号
correct_data = 'SpCoSLAM_human.csv'  #データごとの正解の文章 (単語列, 区切り文字つき) (./data/)
correct_name = 'name_correct.csv'  #データごとの正解の場所の名前 (音素列) 
"""
