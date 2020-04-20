#coding:utf-8
#The file for setting parameters [SpCoSLAMとの比較用(learn4_3.py対応) for SIGVerse]
#Akira Taniguchi 2020/04/11-
import numpy as np

##### Add SpCoTMHP #####
IT_mode = "HMM"  # "HMM" or "GMM"

nonpara    = 1     #Nonparametric Bayes method (ON:1,OFF:0)
Robust_W   = 1000
Robust_Sig = 100
Robust_Mu  = 1
Robust_pi  = 1000
Robust_phi = 1000
Robust_theta = 1000
Robust_psi = 1000  #予約(未使用)

#Navigation folder (Other output files are also in same folder.)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv

#Word data folder path
#word_folder = "/name/" + example_folder + "word" # "/name/per_100/word"

#Same value to map yaml file
resolution = 0.1   #0.050000
origin     = np.array([-10.000000, -10.000000]) #np.array([x,y]) #np.array([-30.000000, -20.000000])

approx_zero = 10.0**(-200)   #approximated value of log(0)

word_increment = 1.0     #Increment number of word observation data (BoWs)

CNNmode = 1             # Select image feature descriptor
Feture_times = 1        # 画像特徴量を何倍するか
Feture_sum_1 = 1        # 画像特徴量を足して１になるようにする(1)
Feture_noize = 10**(-5) # 画像特徴量に微小ノイズを足す(Feture_noize/DimImg)

if (CNNmode == 0):
  Descriptor = "SIFT_BoF"
  DimImg     = 100  #Dimension of image feature
elif (CNNmode == 1):
  Descriptor = "googlenet_prob_AURO" #"CNN_softmax"
  DimImg     = 1000 #Dimension of image feature
  Feture_times = float(Feture_times)/100.0 #googlenet_probのデータはすでに１００倍されている
elif (CNNmode == 2):
  Descriptor = "CNN_fc6"
  DimImg     = 4096 #Dimension of image feature
elif (CNNmode == 3):
  Descriptor = "CNN_Place205"
  DimImg     = 205  #Dimension of image feature
elif (CNNmode == 4):
  Descriptor = "hybridCNN"
  DimImg     = 1183  #Dimension of image feature
elif (CNNmode == 5):
  Descriptor = "CNN_Place365"
  DimImg     = 365  #Dimension of image feature

####################Parameters####################
num_iter = 100          # The number of iterations of Gibbs sampling for spatial concept learnin
DATA_NUM = 60 #100      # The number of training data
#M = 2000               # The number of particles (Same value as the condition in learning: 300)
#LAG = 100 + 1          # The number of elements of array (lag value for smoothing + 1)
dimx = 2                # The number of dimensions of xt (x,y)

#limit of map size
#WallX = 1600
#WallY = 1152
WallXmin = -10
WallXmax = 10
WallYmin = 10
WallYmax = -10

# initial scale of Gaussian distribution 
sig_init =  1.0 

margin = 10*0.05    # margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)

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
if (nonpara == 1):
  L = 20             #The number of spatial concepts #50 #100
  K = 20             #The number of position distributions #50 #100
  alpha0 = 20.0 / float(L)      #Hyperparameter of multinomial distribution for index of spatial concept
  gamma0 = 0.10 / float(K)      #Hyperparameter of multinomial distribution for index of position (GMM mixtured component; spatial concept dependent)
  omega0 = 0.10 / float(K)      #Hyperparameter of multinomial distribution for index of position distribution (HMM transition distribution)
else:
  L = 10             #The number of spatial concepts #50 #100
  K = 10             #The number of position distributions #50 #100
  alpha0 = 1.00      #Hyperparameter of multinomial distribution for index of spatial concept
  gamma0 = 0.10      #Hyperparameter of multinomial distribution for index of position distribution (GMM mixtured component; spatial concept dependent)
  omega0 = 0.10      #Hyperparameter of multinomial distribution for index of position distribution (HMM transition distribution)

beta0 = 0.1          #Hyperparameter in multinomial distribution P(W) for place names 
chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
k0 = 1e-3            #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*2  #Hyperparameter in Inverse Wishart distribution P(Σ) (prior covariance matrix) 
n0 = 3.0             #Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))


##latticelm parameters
knownn       = [3] #[2,3,4] #The n-gram length of the language model (3)
unkn         = [3,4] #[3]   #The n-gram length of the spelling model (3)
annealsteps  = [3,5,10]     #The number of annealing steps to perform (3)
anneallength = [5,10,15]    #The length of each annealing step in iterations (5)


##Parameters for mutual estimation in SpCoA++ 
sample_num = len(knownn)*len(unkn)  #The number of samples (candidates for word segmentation results)  #len(knownn)*len(unkn)  
ITERATION = 10                      #The number of iterations for mutual estimation

##単語の選択の閾値
threshold = 0.01

# The number of N of N-best for PRR evaluation (PRR評価用のN-bestのN) (N<=10)
N_best_number = 10  

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


#################### Folder PATH ####################

##### NEW #####
inputfolder  = "/mnt/hgfs/D/Dropbox/SpCoSLAM/SpCoTMHP/SIGVerse/dataset/similar/3LDK_small/"  #"/home/akira/Dropbox/SpCoNavi/data/"
outputfolder = "/mnt/hgfs/D/Dropbox/SpCoSLAM/SpCoTMHP/albert-b/data/"  #"/home/akira/Dropbox/SpCoNavi/data/"
# akira/Dropbox/SpCoNavi/CoRL/dataset/similar/3LDK_small/3LDK_01/

speech_folder = inputfolder + "speech/*.wav" #"/home/akira/Dropbox/Julius/directory/SpCoSLAM/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntu full path)
#speech_folder = "/home/*/Dropbox/Julius/directory/SpCoSLAM/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
#speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav" #*.wav" #音声の教示データフォルダ(Ubuntuフルパス)
PositionDataFile = '/position/position_AURO.csv' #'SpCoSLAM.csv'      # 'test000' #位置推定の教示データ(./../sampleフォルダ内)
lmfolder = "/home/akira/Dropbox/SpCoSLAM/learning/lang_m/"

#Folder of training data set
datasetfolder = inputfolder #"/home/akira/Dropbox/SpCoSLAM/rosbag/"   #training data set folder
datasets      = ["00","01","02","03","04","05","06","07","08","09","10"] #["00","01","04","05","06","09","02","03","07","08","10"] #[dataset1,dataset2]
data_step_num = 60

#True data files for evaluation (評価用正解データファイル)
correct_Ct = 'Ct_correct.csv'        #データごとの正解のCt番号
correct_It = 'It_correct.csv'        #データごとの正解のIt番号
correct_data = 'SpCoSLAM_human.csv'  #データごとの正解の文章（単語列、区切り文字つき）(./data/)
correct_name = 'name_correct.csv'    #データごとの正解の場所の名前（音素列）
