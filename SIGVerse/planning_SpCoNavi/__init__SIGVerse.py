#coding:utf-8
#The file for setting parameters
#Akira Taniguchi 2018/12/13-2019/03/10-2019/06/27-
import numpy as np

##実行コマンド
#python ./SpCoNavi0.1_SIGVerse.py trialname iteration sample init_position_num speech_num
#python ./SpCoNavi0.1_SIGVerse.py 3LDK_01 1 0 0 0

##### NEW #####
inputfolder_SIG  = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/dataset/similar/3LDK/"  #"/home/akira/Dropbox/SpCoNavi/data/"
outputfolder_SIG = "/mnt/hgfs/Dropbox/SpCoNavi/CoRL/data/"  #"/home/akira/Dropbox/SpCoNavi/data/"

Start_Position = [[100,100]]
Goal_Word = ["玄関","リビング","ダイニング","キッチン","風呂","洗面所","トイレ","寝室"]
#玄関,リビング,ダイニング,キッチン,風呂,洗面所,トイレ,寝室,

#Same values as /learning/__init.py__
L = 10 #100                  #場所概念の数50#
K = 10 #100                  #位置分布の数50#

memory_reduction = 1 #00 #
NANAME = 0 #斜め座標の遷移も動作に含む（１）

#################### Folder PATH ####################
#Setting of PATH for a folder of learned spatial concept parameters
datafolder    = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/" #"/home/akira/Dropbox/SpCoSLAM/data/"  #
#Setting of PATH for output folder
outputfolder  = "/mnt/hgfs/D/Dropbox/SpCoSLAM/data/"  #"/home/akira/Dropbox/SpCoNavi/data/"

#音声ファイルフォルダ
speech_folder    = "/home/akira/Dropbox/Julius/directory/SpCoSLAM/*.wav"    #音声の教示データフォルダ
speech_folder_go = "/home/akira/Dropbox/Julius/directory/SpCoSLAMgo/*.wav"  #評価用の音声データフォルダ
lmfolder = "/mnt/hgfs/D/Dropbox/SpCoSLAM/learning/lang_m/"  #Language model (word dictionary)

#Navigation folder (他の出力ファイルも同フォルダ)
navigation_folder = "/navi/"  #outputfolder + trialname + / + navigation_folder + contmap.csv
#SpCoSLAMのフォルダ形式に従うようにしている
#"/navi_s/"は、StのN-bestを別々に計算する版
#"/navi_s2/"は、StのN-bestを別々に計算する版+URの分母の割り算省略版

#Cost map folder
costmap_folder = navigation_folder  #"/costmap/" #



#################### Parameters ####################
T_horizon  = 200     #計画区間(予測ホライズン) #150~200以上はほしいがメモリ容量or計算量次第 #値が大きすぎる(400)と，数値計算(おそらく遷移確立)の問題でパス生成がバグることがあるので注意
N_best     = 10      #N of N-best (N<=10)
step       = 50      #使用するSpCoSLAMの学習時のタイムステップ(教示回数)

#自己位置の初期値(候補：目的地以外の理想的な位置分布のデータ平均)
X_candidates = [[340, 590]] ###TEST #2次元配列のインデックス(VR340)
##0:kyouyuseki,1:kyukeijyo,2:roboqtookiba,3:ikidomari,4:miithingusupeesu,5:kyouinkennkyushitsu,6:purintaabeya,7:daidokoro,8:siroitana

#途中から始める場合(trellisを読み込むためのtの値, 最初から:0)
T_restart = 0 #T_horizonが変わると次元削減の処理で計算上の状態数が変わってしまうため現状使えない。trellisを保存していないとそもそも途中から再開できない。

SAVE_time    = 1      #計算時間を保存するかどうか(保存する:1、保存しない:0)
SAVE_X_init  = 0      #初期値をファイル保存するか（このファイルで指定する場合は事前にわかっているので不要）
SAVE_T_temp  = 10     #途中のパスを一時ファイル保存する(途中のTの値ごと)
SAVE_Trellis = 0      #Viterbi Path推定時のトレリスを保存するか(保存する:1、保存しない:0)

UPDATE_PostProbMap = 0 #1 #ファイルが既にあっても、PostProbMapの計算を行う(1) 

#近似手法の選択(Proposed(JSAI2019版):0, samplingCtit:1(未実装), xの次元削減とか...(未実装), 近似せずに厳格に計算:-1)
Approx = 0  
if (NANAME != 1):
  Approx = 1
#現状、N-best近似しない版は別のプログラム（SpCoNavi0.1s.py）

#状態遷移のダイナミクス(動作モデル)の仮定(確定的:0, 確率的:1, 近似:2(未実装))
#Dynamics = 0

cmd_vel = 1  #ロボットの移動量(ROSではcmd_vel [m/s], [rad/s])[基本的に1(整数値)]
MotionModelDist = "Gauss"  #"Gauss"：ガウス分布、"Triangular":三角分布

#オドメトリ動作モデルパラメータ(AMCL or gmappingと同じ値にする)：未使用
odom_alpha1 = 0.2  #(ダブル、デフォルト：0.2) ロボットの動きの回転移動からオドメトリの回転移動のノイズ
odom_alpha2 = 0.2  #(ダブル、デフォルト：0.2) ロボットの動きの平行移動からオドメトリの回転移動のノイズ
odom_alpha3 = 0.2  #(ダブル、デフォルト：0.2) ロボットの動きの平行移動からオドメトリの平行移動のノイズ
odom_alpha4 = 0.2  #(ダブル、デフォルト：0.2) ロボットの動きの回転移動からオドメトリの平行移動のノイズ
#srr = 0.1 #(float, default: 0.1) #オドメトリの誤差．平行移動に起因する平行移動の誤差．
#srt = 0.2 #(float, default: 0.2) #オドメトリの誤差．回転移動に起因する平行移動の誤差．
#str = 0.1 #(float, default: 0.1) #オドメトリの誤差．平行移動に起因する回転移動の誤差．
#stt = 0.2 #(float, default: 0.2) #オドメトリの誤差．回転移動に起因する回転移動の誤差．

#ROSのトピック名
MAP_TOPIC     = "/map"
COSTMAP_TOPIC = "/move_base/global_costmap/costmap"
#PATH_TOPIC = "/spconavi/path" #未実装

#地図のyamlファイルと同じ値にする
resolution = 0.1 #0.050000
origin =  np.array([-10.000000, -10.000000]) #, 0.000000] #np.array([-30.000000, -20.000000]) #, 0.000000]

#地図のサイズの縦横(length and width)があらかじめ分かる場合はこちらに記載しても良いかも
#map_length = 0
#map_width  = 0

#Julius parameters
JuliusVer      = "v4.4"   #"v.4.3.1"
HMMtype        = "DNN"    #"GMM"
lattice_weight = "AMavg"  #"exp" #音響尤度(対数尤度："AMavg"、尤度："exp")
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
  # 'trueword_syllable.htkdic' #'phonemes.htkdic' # 初期の単語辞書(./lang_mフォルダ内)

dimx = 2           #The number of dimensions of xt (x,y)
margin = 10*0.05   #地図のグリッドと位置の値の関係が不明のため(0.05m/grid)*margin(grid)=0.05*margin(m)
approx_log_zero = np.log(10.0**(-300))   #ほぼlog(0)の微小値


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
wic = 1         #1:wic重みつき(理論的にはこちらがより正しい)、0:wic重みなし(Orignal paper of SpCoSLAM)
UseFT = 1       #画像特徴を使う場合(１)、使わない場合(０)
UseLM = 1       #言語モデルを更新する場合(１)、しない場合(０)[Without update language modelのため無関係]

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
