# encoding: shift_jis
# 実行すれば、自動的に指定フォルダ内にある音声ファイルを読み込み、Juliusでラティス認識した結果を出力してくれる。
# 注意点：指定フォルダ名があっているか確認すること。
import glob
import codecs
import os
import re
import sys
from initSpCoSLAMSIGVerse import *

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass


# julisuに入力するためのwavファイルのリストファイルを作成
def MakeTmpWavListFile( wavfile , filename):
    Makedir( "tmp" )
    Makedir( "tmp/" + filename )
    fList = codecs.open( "tmp/" + filename + "/list.txt" , "w" , "sjis" )
    fList.write( wavfile )
    fList.close()

# Lattice認識
def RecogLattice( wavfile , iteration , filename ):
    MakeTmpWavListFile( wavfile , filename )
    if (JuliusVer == "v4.4"):
      binfolder = "bin/linux/julius"
    else:
      binfolder = "bin/julius"
    
    if (JuliusVer == "v4.4" and HMMtype == "DNN"):
      if (iteration == 0):  #最初は日本語音節のみの単語辞書を使用
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v " + lmfolder + lang_init + " -demo -filelist tmp/"+ filename + "/list.txt -lattice -dnnconf " + Juliusfolder + "julius.dnnconf $*"  ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS -confnet 
        print "Julius",JuliusVer,HMMtype,"Read dic:" ,lang_init , iteration
      else:  #更新した単語辞書を使用
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v ./data/" + filename + "/web.000s_" + str(iteration) + ".htkdic -demo -filelist tmp/" + filename + "/list.txt -lattice -dnnconf " + Juliusfolder + "julius.dnnconf $*" ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS  -confnet
        print "Julius",JuliusVer,HMMtype,"Read dic:web.000s_" + str(iteration) + ".htkdic" , iteration
    else:
      if (iteration == 0):  #最初は日本語音節のみの単語辞書を使用
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + lmfolder + lang_init + " -demo -filelist tmp/"+ filename + "/list.txt -lattice" ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS -confnet 
        print "Julius",JuliusVer,HMMtype,"Read dic:" ,lang_init , iteration
      else:  #更新した単語辞書を使用
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v ./data/" + filename + "/web.000s_" + str(iteration) + ".htkdic -demo -filelist tmp/" + filename + "/list.txt -lattice" ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS  -confnet
        print "Julius",JuliusVer,HMMtype,"Read dic:web.000s_" + str(iteration) + ".htkdic" , iteration
      
      """
      if (iteration == 0):  #最初は日本語音節のみの単語辞書を使用
        p = os.popen( "~/Dropbox/Julius/dictation-kit-v4.3.1-linux/bin/julius -C ~/Dropbox/Julius/dictation-kit-v4.3.1-linux/syllable.jconf -C ~/Dropbox/Julius/dictation-kit-v4.3.1-linux/am-gmm.jconf -v lang_m/" + lang_init + " -demo -filelist tmp/" + filename + "/list.txt -confnet -lattice" ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS 
        print "Read dic:" ,lang_init , iteration
      else:  #更新した単語辞書を使用
        p = os.popen( "~/Dropbox/Julius/dictation-kit-v4.3.1-linux/bin/julius -C ~/Dropbox/Julius/dictation-kit-v4.3.1-linux/syllable.jconf -C ~/Dropbox/Julius/dictation-kit-v4.3.1-linux/am-gmm.jconf -v data/" + filename + "/web.000s_" + str(iteration) + ".htkdic -demo -filelist tmp/" + filename + "/list.txt -confnet -lattice" ) #元設定-n 5 # -gram type -n 5-charconv UTF-8 SJIS 
        print "Read dic:web.000s_" + str(iteration) + ".htkdic" , iteration
      """


    startWordGraphData = False
    wordGraphData = []
    wordData = {}
    index = 1 ###単語IDを1から始める
    line = p.readline()  #一行ごとに読む？
    while line:
        if line.find("end wordgraph data") != -1:
            startWordGraphData = False

        if startWordGraphData==True:
            items = line.split()  #空白で区切る
            wordData = {}
            wordData["range"] = items[1][1:-1].split("..")
            wordData["index"] = str(index)
            index += 1
            for item in items[2:]:
                name,value = item.replace('"','').split("=")   #各name=valueをイコールで区切り格納
                if name in ( "right" , "right_lscore" , "left" ):
                    value = value.split(",")

                wordData[name] = value

            wordGraphData.append(wordData)

        if line.find("begin wordgraph data") != -1:
            startWordGraphData = True
        line = p.readline()
    p.close()


    return wordGraphData

# 認識したlatticeをopenFST形式で保存
def SaveLattice( wordGraphData , filename ):
    f = codecs.open( filename , "w" , "sjis" )
    for wordData in wordGraphData:
        flag = 0
        for r in wordData.get("right" ,[str(len(wordGraphData)), ]):
            l = wordData["index"].decode("sjis")
            w = wordData["name"].decode("sjis")
            
            
            if int(r) < len(wordGraphData):     #len(wordGraphData)は終端の数字を表す
                s = wordGraphData[int(r)]["AMavg"] #graphcmで良いのか？音響尤度ならばAMavgでは？
                s = str(float(s) *wight_scale)              #AMavgを使用時のみ(HDecodeの場合と同様の処理？)
                if (lattice_weight == "exp"):
                  s_exp = exp(float(s))
                  s = str(s_exp)

                r = str(int(r) + 1)  ###右に繋がっているノードの番号を＋１する                
                #print l,s,w
                #print wordData.get("left","None")
                if ("None" == wordData.get("left","None")) and (flag == 0):
                    l2 = str(0)
                    r2 = l
                    w2 = "<s>"
                    s2 = -1.0
                    f.write(  "%s %s %s %s %s\n" % (l2,r2,w2,w2,s2))
                    flag = 1
                    #l = str(0)
                    #print l
                f.write(  "%s %s %s %s %s\n" % (l,r,w,w,s) )
            else:
                r = str(int(r) + 1)  ###右に繋がっているノードの番号を＋１する 
                f.write(  "%s %s %s %s 1.0\n" % (l,r,w,w) )
    f.write( "%d 0" % int(len(wordGraphData)+1) )
    f.close()

# テキスト形式をバイナリ形式へコンパイル
def FSTCompile( txtfst , syms , outBaseName , filename ):
    Makedir( "tmp" )
    Makedir( "tmp/" + filename )
    os.system( "fstcompile --isymbols=%s --osymbols=%s %s %s.fst" % ( syms , syms , txtfst , outBaseName ) )
    os.system( "fstdraw  --isymbols=%s --osymbols=%s %s.fst > tmp/%s/fst.dot" % ( syms , syms , outBaseName , filename ) )

    # sjisをutf8に変換して，日本語フォントを指定
    #codecs.open( "tmp/" + filename + "/fst_utf.dot" , "w" , "utf-8" ).write( codecs.open( "tmp/" + filename + "/fst.dot" , "r" , "sjis" ).read().replace( 'label' , u'fontname="MS UI Gothic" label' ) )

    # psとして出力
    #os.system( "dot -Tps:cairo tmp/%s/fst_utf.dot > %s.ps" % (filename , outBaseName) )
    # pdf convert
    #os.system( "ps2pdf %s.ps %s.pdf" % (outBaseName, outBaseName) )


def Julius_lattice(iteration , filename):
    iteration = int(iteration)
    Makedir( "data/" + filename + "/fst_gmm_" + str(iteration+1) )
    Makedir( "data/" + filename + "/out_gmm_" + str(iteration+1) )
    #Makedir( "data/" + filename + "/out_gmm_" + str(iteration) )

    # wavファイルを指定
    files = glob.glob(speech_folder)   #./../../../Julius/directory/CC3Th2/ (相対パス)
    #print files
    files.sort()

    wordDic = set()
    num = 0

    # 1つづつ認識してFSTフォーマットで保存
    for f in files:
        txtfstfile = "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/%03d.txt" % num
        print "count...", f , num

        # Lattice認識&保存
        graph = RecogLattice( f , iteration ,filename )
        SaveLattice( graph , txtfstfile )

        # 単語辞書に追加
        for word in graph:
            wordDic.add( word["name"] )

        num += 1
        
    
    # 単語辞書を作成
    f = codecs.open( "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/isyms.txt" , "w" , "sjis" )
    wordDic = list(wordDic)
    f.write( "<eps>	0\n" )  # latticelmでこの2つは必要らしい
    f.write( "<phi>	1\n" )
    for i in range(len(wordDic)):
        f.write(  "%s %d\n" % (wordDic[i].decode("sjis"),i+2) )
    f.close()
    
    # バイナリ形式へコンパイル
    fList = open( "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/fstlist.txt" , "wb" )  # 改行コードがLFでないとダメなのでバイナリ出力で保存
    for i in range(num):
        print "now compile..." , "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/%03d.txt" % i
        
        # FSTコンパイル
        FSTCompile( "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/%03d.txt" % i , "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/isyms.txt" , "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/%03d" % i  ,filename)
        
        # lattice lm用のリストファイルを作成
        fList.write( "data/" + filename + "/fst_gmm_" + str(iteration+1) + "/%03d.fst" % i )
        fList.write( "\n" )
    fList.close()
    #print "fstへの変換は、Ubuntuで行ってください"

"""
if __name__ == '__main__':
    #param = sys.argv
    #print param
    param = [0, "test001"]
    Julius_lattice(param[0],param[1])
"""
