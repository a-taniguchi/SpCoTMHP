#coding:utf-8
#The file for setting parameters
#Akira Taniguchi 2022/02/05-
import numpy as np

##Command
#python ./spcotmhp_viterbi_planning.py trialname iteration(1) sample(0) init_position_num speech_num
#python ./spcotmhp_astar_execute.py trialname iteration(1) sample(0) init_position_num speech_num
#python ./spcotmhp_viterbi_planning.py 3LDK_01 1 0 0 0

##### example (for SpCoNavi exp) #####
example = 0 #1
example_folder = ""
if (example == 1):
  example_folder = "example1/"
elif (example == 2):
  example_folder = "example2/"

##### NEW #####
inputfolder_SIG  = "../../SIGVerse/dataset/similar/3LDK/"
outputfolder_SIG = "../../SIGVerse/data/"

inputfolder_albert  = "../../albert-b/dataset/TMHP/"
outputfolder_albert = "../../albert-b/data/"

# Note: Don't be tupple! Only list! [*,*]  #(y,x). not (x,y). (Same as coordinates in Astar_*.py) 
Start_Position = [[100,100],[150,130],[120,60],[60,90],[90,120],[75,75],[90,50],[90,60],[110,80],[130,95]] 
Goal_Word      = ["玄関","リビング","ダイニング","キッチン","風呂","洗面所","トイレ","寝室", "物置き", "廊下", "東", "南", "広場"]
#0:玄関, 1:リビング, 2:ダイニング, 3:キッチン, 4:風呂, 5:洗面所, 6:トイレ, 7:寝室, 8:物置き, 9:廊下

Goal_Word_albert = ["きょうゆうせき","きゅうけいじょ","ろぼっとおきば","ろうか","いきどまり","みいてぃんぐすぺいす","きょういんけんきゅうしつ","ぷりんたあべや","だいどころ","がくせいさぎょうば","しろいたな"]

#Same values as /learning/__init.py__
L = 11                   #The number of spatial concepts
K = 11                   #The number of position distributions

memory_reduction = 1 #0    #Memory reduction process (ON:1, OFF:0)
NANAME = 0                 #Action pattern: up, down, left and right (0), and add diagonal (oblique) movements (１)
word_increment = 6 #10     #Increment number of word observation data (BoWs)

#################### Folder PATH ####################
#Setting of PATH for a folder of learned spatial concept parameters
#datafolder    = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/" 
#Setting of PATH for output folder
#outputfolder  = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/"  

# Word data folder path
word_folder = "/name/per_100/word"

#File folder of speech data
#speech_folder    = "/home/akira/Dropbox/Julius/directory/SpCoSLAM/*.wav"    #Teaching speech data folder
#speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav"  #Evaluation speech data folder
#lmfolder         = "/mnt/hgfs/D/Dropbox/SpCoSLAM/learning/lang_m/"  #Language model (word dictionary)

#Navigation folder (Other output files are also in same folder.)
navigation_folder = "/tmhp/"  #outputfolder + trialname + / + navigation_folder + contmap.csv
# follow folder format of learning result in spatial concept

#Cost map folder
costmap_folder = navigation_folder  #"/costmap/" 

## [albert-b] map fileのフォルダファイル名 (**.pgm and **.yaml)
map_file = "map/map312"

#################### Parameters ####################
T_horizon  = 200             #Planning horizon #may be over 150~200. depends on memory and computational limits

T_topo     = 10              #Planning horizon for SpCoTMHP
D_metric   = T_horizon       #Planning horizon for SpCoTMHP

N_best     = word_increment  #N of N-best (N<=10)
step       = 70              # for albert-b: The end number of time-step in SpCoSLAM (the number of training data)

# SpCoNavi_Astar_approx.py: The number of goal position candidates
Sampling_J = 1
Sampling_G = 10 #for SpCoTMHP (albert)
N_ie       = 1 #0 # for SpCoTMHP

#Initial position (position candidates)
X_candidates = Start_Position  #Index coordinates on 2 dimension list

#When starting from the mid-flow (value of t to read trellis, from the beginning: 0)
# SpCoTMHPでは未実装
T_restart = 0         #If T_horizon changes, it can not be used at present because the number of states changes in the dimension reduction process. If you do not save trellis, you run from the beginning.

SAVE_time    = 1      #Save computational time (Save:1, Not save:0)
SAVE_X_init  = 1      #Save initial value (Save:1, Not save:0) 
SAVE_T_temp  = 50     #Step interval to save the path temporarily (each SAVE_T_temp value on the way)
SAVE_Trellis = 0      #Save trellis for Viterbi Path estimation (Save:1, Not save:0) 

UPDATE_PostProbMap = 1 #0 #If the file exists already, calculate PostProbMap: (1) 

#Select approximated methods (Proposed method (ver. SIGVerse):0) -> run SpCoNavi_Astar_approx.py
Approx = 0  
if (NANAME != 1):
  Approx = 1
#Separated N-best approximation version is another program (SpCoNavi0.1s.py)
St_separate = 0   #N-best BoWs: All:0, separate:1


#Dynamics of state transition (motion model): (Deterministic:0, Probabilistic:1, Approximation:2(Unimplemented))
#Dynamics = 0


cmd_vel = 1  #Movement amount of robot (ROS: cmd_vel [m/s], [rad/s]) [default:1 (int)]
#MotionModelDist = "Gauss"  #"Gauss": Gaussian distribution, "Triangular": Triangular distribution

#Odometry motion model parameters (Same values to AMCL or gmapping): unused
#odom_alpha1 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's rotation estimate from the rotational component of the robot's motion. 
#odom_alpha2 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's rotation estimate from translational component of the robot's motion. 
#odom_alpha3 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's translation estimate from the translational component of the robot's motion. 
#odom_alpha4 = 0.2  #(double, default: 0.2) Specifies the expected noise in odometry's translation estimate from the rotational component of the robot's motion. 


#ROS topic name
MAP_TOPIC     = "/map"
COSTMAP_TOPIC = "/move_base/global_costmap/costmap"
#PATH_TOPIC = "/spconavi/path" #Unimplemented

#Same value to map yaml file
resolution = 0.1   #0.050000
origin     = np.array([-10.000000, -10.000000]) #np.array([x,y]) #np.array([-30.000000, -20.000000])

resolution_albert = 0.050000
origin_albert =  np.array([-30.000000, -20.000000]) #, 0.000000] #np.array([x,y])

#map size (length and width)
#map_length = 0
#map_width  = 0

#dimx = 2           #The number of dimensions of xt (x,y)
margin = 10*0.1 #0.05   #margin value for place area in gird map (0.05m/grid)*margin(grid)=0.05*margin(m)
approx_log_zero = np.log(10.0**(-200)) #  float('-inf') #approximated value of log(0)
just_zero = 0.0