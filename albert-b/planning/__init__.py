#coding:utf-8
#The file for setting parameters
#Akira Taniguchi 2018/12/13-2019/03/10-2019/07/25
import numpy as np

##Command
#python ./SpCoNavi0.1s.py trialname particle_num init_position_num speech_num
#python ./SpCoNavi0.1s.py alg2wicWSLAG10lln008 0 0 0

#################### Folder PATH ####################
#Setting of PATH for a folder of learned spatial concept parameters
datafolder    = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/" #"/home/akira/Dropbox/SpCoSLAM/data/" 
#Setting of PATH for output folder
outputfolder  = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/"  #"/home/akira/Dropbox/SpCoNavi/data/"

#File folder of speech data
speech_folder    = "/home/akira/Dropbox/Julius/directory/SpCoSLAM/*.wav"    #Teaching speech data folder
speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav"  #Evaluation speech data folder
lmfolder         = "/mnt/hgfs/D/Dropbox/SpCoSLAM/learning/lang_m/"  #Language model (word dictionary)

#Navigation folder (Other output files are also same folder)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv
# follow folder format of learning result in spatial concept (SpCoSLAM)
#"/navi_s/"は, StのN-bestを別々に計算する版
#"/navi_s2/"は, StのN-bestを別々に計算する版+URの分母の割り算省略版

#Cost map folder
costmap_folder = navigation_folder



#################### Parameters ####################
T_horizon  = 400     #Planning horizon #may be over 150~200. depends on memory and computational limits
N_best     = 10      #N of N-best (N<=10)
step       = 50      #The end number of time-step in SpCoSLAM (the number of training data)

#Initial position (position candidates)
X_candidates = [[340, 590]]  #Index coordinates on 2 dimension list (VR340)
##0:kyouyuseki,1:kyukeijyo,2:roboqtookiba,3:ikidomari,4:miithingusupeesu,5:kyouinkennkyushitsu,6:purintaabeya,7:daidokoro,8:siroitana

#When starting from the mid-flow (value of t to read trellis, from the beginning: 0)
T_restart = 0         #If T_horizon changes, it can not be used at present because the number of states changes in the dimension reduction process. If you do not save trellis, you run from the beginning.

SAVE_time    = 1      #Save computational time (Save:1, Not save:0)
SAVE_X_init  = 0      #Save initial value (Save:1, Not save:0) 
SAVE_T_temp  = 10     #Step interval to save the path temporarily (each SAVE_T_temp value on the way)
SAVE_Trellis = 0      #Save trellis for Viterbi Path estimation (Save:1, Not save:0) 

UPDATE_PostProbMap = 1 #If the file exists already, calculate PostProbMap: (1) 

#Select approximated methods (Proposed method (ver. JSAI2019):0, sampling_Ct_it:1 (Unimplemented), dimension reduction of state x, and so on...(Unimplemented), no approximation:-1 (Unimplemented))
Approx = 0  
#Separated N-best approximation
St_separate = 0   #N-best BoWs: All:0, separate:1 

#Dynamics of state transition (motion model): (Deterministic:0, Probabilistic:1, Approximation:2(Unimplemented))
#Dynamics = 0

cmd_vel = 1  #Movement amount of robot (ROS: cmd_vel [m/s], [rad/s]) [default:1 (int)]
MotionModelDist = "Gauss"  #"Gauss": Gaussian distribution, "Triangular": Triangular distribution

#Odometry motion model parameters (Same values to AMCL or gmapping): unused
odom_alpha1 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's rotation estimate from the rotational component of the robot's motion.  #stt = 0.2 #(float, default: 0.2) #オドメトリの誤差．回転移動に起因する回転移動の誤差．
odom_alpha2 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's rotation estimate from translational component of the robot's motion.  #str = 0.1 #(float, default: 0.1) #オドメトリの誤差．平行移動に起因する回転移動の誤差．
odom_alpha3 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's translation estimate from the translational component of the robot's motion.  #srr = 0.1 #(float, default: 0.1) #オドメトリの誤差．平行移動に起因する平行移動の誤差．
odom_alpha4 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's translation estimate from the rotational component of the robot's motion.  #srt = 0.2 #(float, default: 0.2) #オドメトリの誤差．回転移動に起因する平行移動の誤差．


#ROS topic name
MAP_TOPIC     = "/map"
COSTMAP_TOPIC = "/move_base/global_costmap/costmap"
#PATH_TOPIC = "/spconavi/path" #Unimplemented

#Same value to map yaml file
resolution = 0.050000
origin =  np.array([-30.000000, -20.000000]) #, 0.000000] #np.array([x,y])

#map size (length and width)
#map_length = 0
#map_width  = 0

#Julius parameters
JuliusVer      = "v4.4"   #"v.4.3.1"
HMMtype        = "DNN"    #"GMM"
lattice_weight = "AMavg"  #"exp" #acoustic likelihood (log likelihood: "AMavg", likelihood: "exp")
wight_scale    = -1.0
#WDs = "0"   #DNN版の単語辞書の音素を*_Sだけにする("S"), BIE or Sにする("S"以外)
##In other parameters, please see "main.jconf" in Julius folder

if (JuliusVer ==  "v4.4"):
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.4/"
else:
  Juliusfolder = "/home/akira/Dropbox/Julius/dictation-kit-v4.3.1-linux/"

if (HMMtype == "DNN"):
  lang_init = 'syllableDNN.htkdic' 
else:
  lang_init = 'syllableGMM.htkdic' 
  # 'trueword_syllable.htkdic' #'phonemes.htkdic' # Initial word dictionary (in ./lang_m/ folder)

#dimx = 2           #The number of dimensions of xt (x,y)
#margin = 10*0.05   #margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)
approx_log_zero = np.log(10.0**(-300))   #approximated value of log(0)


####################Particle Class (structure)####################
class Particle:
  def __init__(self,id,x,y,theta,weight,pid):
    self.id = id
    self.x = x
    self.y = y
    self.theta = theta
    self.weight = weight
    self.pid = pid
    #self.Ct = -1
    #self.it = -1

"""
####################Option setting (NOT USE)####################
wic = 1         #1:wic重みつき(理論的にはこちらがより正しい), 0:wic重みなし(Orignal paper of SpCoSLAM)
UseFT = 1       #画像特徴を使う場合(１), 使わない場合(０)
UseLM = 1       #言語モデルを更新する場合(１), しない場合(０)[Without update language modelのため無関係]

#NbestNum = N_best      #N of N-best (N<=10)
#N_best_number = N_best #N of N-best (N<=10) for PRR

##Initial (hyper) parameters
##Posterior (∝likelihood×prior): https://en.wikipedia.org/wiki/Conjugate_prior
alpha0 = 10.0        #Hyperparameter of CRP in multinomial distribution for index of spatial concept
gamma0 = 1.0         #Hyperparameter of CRP in multinomial distribution for index of position distribution
beta0 = 0.1          #Hyperparameter in multinomial distribution P(W) for place names 
chi0  = 0.1          #Hyperparameter in multinomial distribution P(φ) for image feature
k0 = 1e-3            #Hyperparameter in Gaussina distribution P(μ) (Influence degree of prior distribution of μ)
m0 = np.zeros(dimx)  #Hyperparameter in Gaussina distribution P(μ) (prior mean vector)
V0 = np.eye(dimx)*2  #Hyperparameter in Inverse Wishart distribution P(Σ)(prior covariance matrix)
n0 = 3.0             #Hyperparameter in Inverse Wishart distribution P(Σ) {>the number of dimenssions] (Influence degree of prior distribution of Σ)
k0m0m0 = k0*np.dot(np.array([m0]).T,np.array([m0]))
"""
