# coding:utf-8
# Akira Taniguchi 2017/02/25-2017/07/10-2018/12/13-
# 実行すれば、自動的に指定フォルダ内にある音声ファイルを読み込み、Juliusで通常の音声認識した結果をN-bestで出力してくれる。(Julius v4.4 GMM/DNN対応済み)
# 注意点：指定フォルダ名があっているか確認すること。
# 2018/12/13：エンコーディングをShift-jis -> utf-8に変更（何らかの影響が出るか要確認）-> sjisに戻している
# 2018/12/13：コードの無駄を減らした
import glob
import codecs
import os
#import re
import sys
#from math import exp,log
from __init__ import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass


# julisuに入力するためのwavファイルのリストファイルを作成
def MakeTmpWavListFile( wavfile , trialname):
    Makedir( datafolder + trialname + "/tmp/" )
    Makedir( datafolder + trialname + "/tmp/" + trialname )
    fList = codecs.open( datafolder + trialname + "/tmp/" + trialname + "/list.txt" , "w" , "sjis" ) #sjis
    fList.write( wavfile )
    fList.close()


# N-best認識
def RecogNbest( wavfile, step, trialname ):
    MakeTmpWavListFile( wavfile , trialname )
    if (JuliusVer == "v4.4"):
      binfolder = "bin/linux/julius"
    else:
      binfolder = "bin/julius"
      
    if (step == 0):  #最初は日本語音節のみの単語辞書を使用(step==1が最初)####使用されないはず
      if (JuliusVer == "v4.4" and HMMtype == "DNN"):
        JuliusCMD = Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v " + lmfolder + lang_init + " -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(N_best+1) + " -dnnconf " + Juliusfolder + "julius.dnnconf $*"
        #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS 
      else:
        JuliusCMD = Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + lmfolder + lang_init + " -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(N_best+1)  
    else: #更新した単語辞書を使用
      if (JuliusVer == "v4.4" and HMMtype == "DNN"):
        JuliusCMD = Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v " + datafolder + trialname + "/" + str(step) + "/WDnavi.htkdic -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(N_best+1) + " -dnnconf " + Juliusfolder + "julius.dnnconf $*"
      else:
        JuliusCMD = Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + datafolder + trialname + "/" + str(step) + "/WDnavi.htkdic -demo -filelist "+ datafolder + trialname + "/tmp/" + trialname + "/list.txt -output " + str(N_best+1)

    p = os.popen( JuliusCMD )
    print "Julius", JuliusVer, HMMtype, "Read dic: " + str(step), "N: " + str(N_best)

    startWordGraphData = False
    wordGraphData = []
    wordData = []
    index = 0 ###単語IDを1から始める
    line = p.readline()  #一行ごとに読む？
    #print line
    while line:
        if line.find( "sentence" + str(N_best+1) + ":" ) != -1:
            startWordGraphData = False

        if startWordGraphData==True:
            items = line#.split()  #空白で区切る
            wordData = []
            ##print items
            for n in range(1,N_best+1):
              if ( 'sentence' + str(n) + ":" ) in items : 
                name = items.replace('sentence','') 
                index, wordData = name.split(":")
                #print n,index

                #wordData = wordData.decode("sjis")
                wordData = wordData.decode("sjis")
                #####print n,wordData
                
                if ( (n == 1) and (len(wordGraphData) == 0) ):
                  wordGraphData.append(wordData)
                else:
                  wordGraphData.append(wordData)
                  #wordGraphData[-1] = wordData

        if line.find("Stat: adin_file: input speechfile:") != -1:
            startWordGraphData = True
        line = p.readline()
    p.close()
    
    #print wordGraphData
    return wordGraphData
  
"""
def Julius(iteration , filename):
    iteration = int(iteration)
    Makedir( "data/" + filename + "/fst_gmm_" + str(iteration+1) )
    Makedir( "data/" + filename + "/out_gmm_" + str(iteration+1) )

    # wavファイルを指定
    files = glob.glob(speech_folder)   #./../../../Julius/directory/CC3Th2/ (相対パス)
    #print files
    files.sort()
    
    #Nbest認識結果をnごとに処理、保存
    for nbest in range(1,N_best+1):
      wordDic = []
      #num = 0
      #n=10  #n-bestのnをどこまでとるか（n<=10）
      
      # 1つづつ認識してFSTフォーマットで保存
      for f in files:
        txtfstfile = "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/" + str(nbest) + str(num).zfill(3) + ".txt" #% num
        print "count...", f , num

        # N-best音声認識&保存
        graph = RecogNbest( f, iteration, filename, nbest )


        # Julius独自の記号を削除
        for word in graph:
            for i in range(5):
              word = word.replace(" <s> ", "")
              word = word.replace("<sp> ", "")
              word = word.replace(" </s>", "")
              #word = word.replace(" ", "") 
              word = word.replace("\n", "")           
            
            wordDic.append( word )
            #print wordDic
            print word
        #num += 1
        
      # 認識結果をファイル保存
      f = open( "data/" + filename + "/out_gmm_" + str(iteration+1) + "/0_samp.100" , "w")# , "sjis" )
      #wordDic = list(wordDic)
      for i in range(len(wordDic)):
        #f.write(wordDic[i].encode('sjis'))
        f.write(wordDic[i].encode('utf8'))
        f.write('\n')
      f.close()
"""
      

"""
if __name__ == '__main__':
    #param = sys.argv
    #print param
    param = [0, "test001"]
    Julius_lattice(param[0],param[1])
"""
