# encoding: utf-8
#!/usr/bin/env python
#グラフ描画プログラム (log-lilelihoodの推移をグラフ化する)
#Akira Taniguchi (2019/09/23-2019/11/11)
#保存ファイル名（プログラム最後の方）と読み込みファイル名の指定に注意

# いらない部分を消す
# フォルダのパス指定
# 手法ごとにlog-likelihoodの値の読み込み
# グラフ化（逐次の値と累積値の両方を出力）

import sys
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from __init__ import *
#import math

sns.set(style="darkgrid")
sns.set_style("whitegrid", {'grid.linestyle': '--'})
#current_palette = sns.color_palette()
#sns.set_palette("muted")
current_palette = sns.color_palette("muted")
sns.color_palette(current_palette)

### FILE PATH
#outputfolder_SIG + 3LDK_01/navi/

### (A) d == 0
#T200N10A1S0G7_Log_likelihood_step.csv
#T200N10A1S0G7_Log_likelihood_sum.csv

### (B) d == 1
#Astar_Approx_N10A1S(ｘ座標, ｙ座礁)G7_Log_likelihood_step.csv
#Astar_Approx_N10A1S(ｘ座標, ｙ座礁)G7_Log_likelihood_sum.csv

### (C) d == 2
#Astar_costmap_SpCo_N10A1S(100, 100)G7_Log_likelihood_step.csv
#Astar_costmap_SpCo_N10A1S(100, 100)G7_Log_likelihood_sum.csv

### (D) d == 3
#Astar_costmap_Database_N10A1S(100, 100)G7_Log_likelihood_step.csv
#Astar_costmap_Database_N10A1S(100, 100)G7_Log_likelihood_sum.csv

### (E) d == 4
#Astar_costmap_Random_N10A1S(100, 100)G7_Log_likelihood_step.csv
#Astar_costmap_Random_N10A1S(100, 100)G7_Log_likelihood_sum.csv

### 環境(trial)番号 と 座標番号 と　座標値の対応
#Start_Position = [0:[100,100],1:[100,110],2:[120,60],3:[60,90],4:[90,120],5:[75,75]] #(y,x). not (x,y). (Same as coordinates in Astar_*.py) 
HOME_ID = [i+1 for i in range(10)] + [i+1 for i in range(10)]
ZAHYO_ID = [0,0,0,0,0,0,0,0,0,1,2,5,2,4,4,2,2,4,2,4]
Start_Position2 = [(100,100),(100,110),(120,60),(60,90),(90,120),(75,75),(90,50),(90,60),(110,80),(130,95)] #(y,x). not (x,y). (Same as coordinates in Astar_*.py) 

#step = 200   ###事前に設定・要確認
DATA = ['(A) SpCoNavi','(B) SpCoNavi (Approx.)','(C) Baseline (Spatial concept)','(D) Baseline (Database)','(E) Baseline (Random)']
HYOUKA = ['step','sum','step','sum']
learningdata = ["T200N10A1S","Astar_Approx_N10A1S","Astar_costmap_SpCo_N10A1S","Astar_costmap_Database_N10A1S","Astar_costmap_Random_N10A1S"]

#trialname = raw_input("data_name?(**_m???_NUM) > ")
hs = input("Graph Plot? [step(0),sum(1),exp_step(2),exp_sum(3)]> ")
h = int(hs)
data_num1 = '01' #raw_input("data_start_num?(DATA_**) > ")
data_num2 = '10' #raw_input("data_end_num?(DATA_**) > ")
N = ( int(data_num2) - int(data_num1) + 1 ) * 2  #10*2 = 20
S = int(data_num1)

LogL_M = [0.0 for c in range(len(DATA)*T_horizon*N)]
hyouka = HYOUKA[int(h)]

i = 0
for d in range(len(DATA)):
  trialname = learningdata[d]
  for s in range(N):
    n = int(s % 10)
    READ_FODLER = outputfolder_SIG + "3LDK_" + str(n+1).zfill(2) + "/navi/"
    
    if (d == 0):
      READ_FILE = READ_FODLER + trialname + str(ZAHYO_ID[s]) + "G7_Log_likelihood_" + hyouka + ".csv" #(A)
    else:
      READ_FILE = READ_FODLER + trialname + str(Start_Position2[ZAHYO_ID[s]]) + "G7_Log_likelihood_" + hyouka + ".csv" 
    print(READ_FILE)
    i = 0
    for line in open(READ_FILE, 'r'):
      itemList = line[:-1].split(',')
      if (itemList[0] != '') and (i < T_horizon):
        LogL_M[d*N + (i)*len(DATA)*N + s] = float(itemList[0])
      i = i + 1
    
if (h == 2 or h == 3):
  LogL_M = list(np.exp(np.array(LogL_M))) 
  hyouka = "exp_" + hyouka

iteration = []
for i in range(T_horizon):
  iteration = iteration + [i+1 for j in range(N*len(DATA))]
method = []  #[DATA[i] for k in range(N) for i in range(len(DATA)) for j in range(step)]
for p in range(T_horizon):
  for d in range(len(DATA)):
    method = method + [DATA[d] for i in range(N)]

subject = [int(i/(len(DATA)*N))+1 for i in range(T_horizon*len(DATA)*N)] #[i for i in range(N)]*len(DATA)*T_horizon
data = {'step':iteration, 'Method':method, 'subject':subject, hyouka:LogL_M}

df2 = pd.DataFrame(data)
print(df2)

# Plot the response with standard error
#markers = ['^','v','o','D','s','H','>','<','d','X']
#AAA = sns.tsplot(data=df2, time="step", unit="subject",condition="method", value=hyouka)  
AAA = sns.lineplot(x="subject", y=hyouka, hue="Method", 
             style="Method", markers=False, dashes=True, data=df2)
plt.subplots_adjust(left=0.15, bottom=0.15, top=0.90, wspace=None, hspace=None) #, right=0.85, top=0.9800

#for i in range(len(DATA)):
#  AAA.lines[i].set_marker(markers[i])
AAA.legend(ncol=1, fontsize=10, labelspacing = 0.5) #loc='lower left',) #prop={'size':10})  #title='method',

plt.xlim([0,200])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
if (h == 2 or h == 3):
  plt.ylabel("Likelihood",fontsize=14)
else:
  plt.ylabel("Log-likelihood",fontsize=14)
plt.xlabel("step",fontsize=14)


######type 1 font#####
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.unicode_minus'] = False

plt.savefig(outputfolder_SIG + hyouka + '.pdf', dpi=300)#, transparent=True
plt.savefig(outputfolder_SIG + hyouka + '.png', dpi=300)#, transparent=True
#plt.savefig(outputfolder_SIG + hyouka + '.eps', dpi=300)#, transparent=True 半透明にできない

#print(df2)
#df2.to_csv("./text"+ hyouka+".csv")
#fig = AAA.get_figure()
#plt.show()
print("close.")

#step.pdfトリミング：10 5 8 12
#sum.pdfトリミング： 10 5 5 12
