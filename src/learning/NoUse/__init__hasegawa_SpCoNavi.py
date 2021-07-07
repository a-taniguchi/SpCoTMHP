#coding:utf-8
#The file for setting parameters (learning for SpCoNavi on SIGVerse; for learn4_3SpCoA_GT.py)
#Akira Taniguchi -2019/07/25
import numpy as np

##### example #####
example = 2 #1
example_folder = ""
#Word data folder path
word_folder = "/name/per_100/word"
if (example == 1):
  example_folder = "example1/"
  word_folder = "/name/" + example_folder + "word" # "/name/per_100/word"
elif (example == 2):
  example_folder = "example2/"
  word_folder = "/name/" + example_folder + "word" # "/name/per_100/word"

##### NEW #####
inputfolder_SIG  = "/root/HSR/catkin_ws/src/em_spconavi/src/data/"
outputfolder_SIG = "/root/HSR/catkin_ws/src/em_spconavi/src/data/"

#inputfolder_SIG  = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/dataset/similar/3LDK/"  #"/home/akira/Dropbox/SpCoNavi/data/"
#outputfolder_SIG = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/data/" + example_folder  #"/home/akira/Dropbox/SpCoNavi/data/"

#Navigation folder (Other output files are also in same folder.)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv

#Word data folder path
#word_folder = "/name/" + example_folder + "word" # "/name/per_100/word"

#Same value to map yaml file
resolution = 0.1   #0.050000
origin     = np.array([-10.000000, -10.000000]) #np.array([x,y]) #np.array([-30.000000, -20.000000])

word_increment = 10     #Increment number of word observation data (BoWs)

#################### Parameters ####################
#kyouji_count = 50 #100 #the number of training data
#M = 2000   #the number of particles (Same value as the condition in learning: 300)
#LAG = 100 + 1  ##the number of elements of array (lag value for smoothing + 1)

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
num_iter = 1           #The number of iterations of Gibbs sampling for spatial concept learning
L = 10                  #The number of spatial concepts #50 #100
K = 10                  #The number of position distributions #50 #100
alpha = 1.0                  #Hyperparameter of multinomial distributions for index of position distirubitons phi #1.5 #0.1
gamma = 1.0                  #Hyperparameter of multinomial distributions for index of spatial concepts pi #8.0 #20.0
beta0 = 0.1                  #Hyperparameter of multinomial distributions for words (place names) W #0.5 #0.2
kappa0 = 1e-3                #For μ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (scale: kappa0>0)
m0 = np.array([[0.0],[0.0]]) #For μ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (mean prior)
V0 = np.eye(2)*2             #For Σ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (covariance matrix prior)
nu0 = 3.0 #3.0               #For Σ, Hyperparameters of Gaussian–inverse–Wishart prior distribution (degree of freedom: dimension+1)

sig_init =  1.0 

##latticelm parameters
#knownn       = [2,3,4] #[3] #The n-gram length of the language model (3)
#unkn         = [3,4] #[3]   #The n-gram length of the spelling model (3)
#annealsteps  = [3,5,10]     #The number of annealing steps to perform (3)
#anneallength = [5,10,15]    #The length of each annealing step in iterations (5)

##Parameters for mutual estimation in SpCoA++ (Cannot change in this code.)
sample_num = 1  #The number of samples (candidates for word segmentation results)  #len(knownn)*len(unkn)  
ITERATION  = 1  #The number of iterations for mutual estimation

##単語の選択の閾値
#threshold = 0.01

#Plot = 2000#1000  #位置分布ごとの描画の点プロット数
#N_best_number = 10 #n-bestのnをどこまでとるか (n<=10) 

#Julius parameters
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
  lang_init = 'web.000.htkdic'   # 'trueword_syllable.htkdic' #'phonemes.htkdic' # Initial word dictionary (in ./lang_m/ folder)
#lang_init_DNN = 'syllableDNN.htkdic' #なごり

N_best_number = 10  #N of N-best (N<=10)
#margin = 10*0.05   #margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)

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

correct_Ct = 'Ct_correct.csv'  #データごとの正解のCt番号
correct_It = 'It_correct.csv'  #データごとの正解のIt番号
correct_data = 'SpCoSLAM_human.csv'  #データごとの正解の文章 (単語列, 区切り文字つき) (./data/)
correct_name = 'name_correct.csv'  #データごとの正解の場所の名前 (音素列) 
"""
