# encoding: shift_jis
# Akira Taniguchi 2020/04/15
# ���s����΁A�����I�Ɏw��t�H���_���ɂ��鉹���t�@�C����ǂݍ��݁AJulius�Ń��e�B�X�F���������ʂ��o�͂��Ă����B
# ���ӓ_�F�w��t�H���_���������Ă��邩�m�F���邱�ƁB
import glob
import codecs
import os
import re
import sys
from initSpCoSLAMSIGVerse import *  # JuliusLattice_dec.py�Ƃ͂����������Ȃ�

def Makedir(dir):
    try:
        os.mkdir( dir )
    except:
        pass


# julisu�ɓ��͂��邽�߂�wav�t�@�C���̃��X�g�t�@�C�����쐬
def MakeTmpWavListFile( wavfile , trialname):
    Makedir( outputfolder + "tmp" )
    Makedir( outputfolder + "tmp/" + trialname )
    fList = codecs.open( outputfolder + "tmp/" + trialname + "/list.txt" , "w" , "sjis" )
    fList.write( wavfile )
    fList.close()

# Lattice�F��
def RecogLattice( wavfile , iteration , trialname ):
    filename = outputfolder + trialname
    MakeTmpWavListFile( wavfile , trialname )
    if (JuliusVer == "v4.4"):
      binfolder = "bin/linux/julius"
    else:
      binfolder = "bin/julius"
    
    if (JuliusVer == "v4.4" and HMMtype == "DNN"):
      if (iteration == 0):  #�ŏ��͓��{�ꉹ�߂݂̂̒P�ꎫ�����g�p
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v " + lmfolder + lang_init + " -demo -filelist " + outputfolder + "tmp/" + trialname+ "/list.txt -lattice -dnnconf " + Juliusfolder + "julius.dnnconf $*"  ) #���ݒ�-n 5 # -gram type -n 5-charconv UTF-8 SJIS -confnet 
        print "Julius",JuliusVer,HMMtype,"Read dic:" ,lang_init , iteration
      else:  #�X�V�����P�ꎫ�����g�p
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-dnn.jconf -v " + filename + "/WD_" + str(iteration) + ".htkdic -demo -filelist " + outputfolder + "tmp/" + trialname + "/list.txt -lattice -dnnconf " + Juliusfolder + "julius.dnnconf $*" ) #���ݒ�-n 5 # -gram type -n 5-charconv UTF-8 SJIS  -confnet
        print "Julius",JuliusVer,HMMtype,"Read dic:WD_" + str(iteration) + ".htkdic" , iteration
    else:
      if (iteration == 0):  #�ŏ��͓��{�ꉹ�߂݂̂̒P�ꎫ�����g�p
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + lmfolder + lang_init + " -demo -filelist " + outputfolder + "tmp/" + trialname+ "/list.txt -lattice" ) #���ݒ�-n 5 # -gram type -n 5-charconv UTF-8 SJIS -confnet 
        print "Julius",JuliusVer,HMMtype,"Read dic:" ,lang_init , iteration
      else:  #�X�V�����P�ꎫ�����g�p
        p = os.popen( Juliusfolder + binfolder + " -C " + Juliusfolder + "syllable.jconf -C " + Juliusfolder + "am-gmm.jconf -v " + filename + "/WD_" + str(iteration) + ".htkdic -demo -filelist " + outputfolder + "tmp/" + trialname + "/list.txt -lattice" ) #���ݒ�-n 5 # -gram type -n 5-charconv UTF-8 SJIS  -confnet
        print "Julius",JuliusVer,HMMtype,"Read dic:WD_" + str(iteration) + ".htkdic" , iteration

    startWordGraphData = False
    wordGraphData = []
    wordData = {}
    index = 1 ###�P��ID��1����n�߂�
    line = p.readline()  #��s���ƂɓǂށH
    while line:
        if line.find("end wordgraph data") != -1:
            startWordGraphData = False

        if startWordGraphData==True:
            items = line.split()  #�󔒂ŋ�؂�
            wordData = {}
            wordData["range"] = items[1][1:-1].split("..")
            wordData["index"] = str(index)
            index += 1
            for item in items[2:]:
                name,value = item.replace('"','').split("=")   #�ename=value���C�R�[���ŋ�؂�i�[
                if name in ( "right" , "right_lscore" , "left" ):
                    value = value.split(",")

                wordData[name] = value

            wordGraphData.append(wordData)

        if line.find("begin wordgraph data") != -1:
            startWordGraphData = True
        line = p.readline()
    p.close()
    return wordGraphData

# �F������lattice��openFST�`���ŕۑ�
def SaveLattice( wordGraphData , filename ):
    f = codecs.open( filename , "w" , "sjis" )
    for wordData in wordGraphData:
        flag = 0
        for r in wordData.get("right" ,[str(len(wordGraphData)), ]):
            l = wordData["index"].decode("sjis")
            w = wordData["name"].decode("sjis")
            
            if int(r) < len(wordGraphData):     #len(wordGraphData)�͏I�[�̐�����\��
                s = wordGraphData[int(r)]["AMavg"] #graphcm�ŗǂ��̂��H�����ޓx�Ȃ��AMavg�ł́H
                s = str(float(s) *wight_scale)              #AMavg���g�p���̂�(HDecode�̏ꍇ�Ɠ��l�̏����H)
                if (lattice_weight == "exp"):
                  s_exp = exp(float(s))
                  s = str(s_exp)
                  
                r = str(int(r) + 1)  ###�E�Ɍq�����Ă���m�[�h�̔ԍ����{�P����                
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
                r = str(int(r) + 1)  ###�E�Ɍq�����Ă���m�[�h�̔ԍ����{�P���� 
                f.write(  "%s %s %s %s 1.0\n" % (l,r,w,w) )
    f.write( "%d 0" % int(len(wordGraphData)+1) )
    f.close()

# �e�L�X�g�`�����o�C�i���`���փR���p�C��
def FSTCompile( txtfst , syms , outBaseName , filename ):
    #Makedir( "tmp" )
    #Makedir( "tmp/" + filename )
    os.system( "fstcompile --isymbols=%s --osymbols=%s %s %s.fst" % ( syms , syms , txtfst , outBaseName ) )
    os.system( "fstdraw  --isymbols=%s --osymbols=%s %s.fst > %s/fst.dot" % ( syms , syms , outBaseName , filename ) )

    # sjis��utf8�ɕϊ����āC���{��t�H���g���w��
    #codecs.open( "tmp/" + filename + "/fst_utf.dot" , "w" , "utf-8" ).write( codecs.open( "tmp/" + filename + "/fst.dot" , "r" , "sjis" ).read().replace( 'label' , u'fontname="MS UI Gothic" label' ) )

    # ps�Ƃ��ďo��
    #os.system( "dot -Tps:cairo tmp/%s/fst_utf.dot > %s.ps" % (filename , outBaseName) )
    # pdf convert
    #os.system( "ps2pdf %s.ps %s.pdf" % (outBaseName, outBaseName) )


def Julius_lattice(iteration , trialname):
    filename = outputfolder + trialname
    iteration = int(iteration)
    Makedir( filename + "/fst_gmm_" + str(iteration+1) )
    Makedir( filename + "/out_gmm_" + str(iteration+1) )
    #Makedir( filename + "/out_gmm_" + str(iteration) )

    # wav�t�@�C�����w��
    files = glob.glob(speech_folder)
    #print files
    files.sort()

    wordDic = set()
    num = 0

    # 1�ÂF������FST�t�H�[�}�b�g�ŕۑ�
    for f in files:
        txtfstfile = filename + "/fst_gmm_" + str(iteration+1) + "/%03d.txt" % num
        print "count...", f , num

        # Lattice�F��&�ۑ�
        graph = RecogLattice( f, iteration, trialname )
        SaveLattice( graph , txtfstfile )

        # �P�ꎫ���ɒǉ�
        for word in graph:
            wordDic.add( word["name"] )

        num += 1
        
    
    # �P�ꎫ�����쐬
    f = codecs.open( filename + "/fst_gmm_" + str(iteration+1) + "/isyms.txt" , "w" , "sjis" )
    wordDic = list(wordDic)
    f.write( "<eps>	0\n" )  # latticelm�ł���2�͕K�v�炵��
    f.write( "<phi>	1\n" )
    for i in range(len(wordDic)):
        f.write(  "%s %d\n" % (wordDic[i].decode("sjis"),i+2) )
    f.close()
    
    # �o�C�i���`���փR���p�C��
    fList = open( filename + "/fst_gmm_" + str(iteration+1) + "/fstlist.txt" , "wb" )  # ���s�R�[�h��LF�łȂ��ƃ_���Ȃ̂Ńo�C�i���o�͂ŕۑ�
    for i in range(num):
        print "now compile..." , filename + "/fst_gmm_" + str(iteration+1) + "/%03d.txt" % i
        
        # FST�R���p�C��
        FSTCompile( filename + "/fst_gmm_" + str(iteration+1) + "/%03d.txt" % i , filename + "/fst_gmm_" + str(iteration+1) + "/isyms.txt" , filename + "/fst_gmm_" + str(iteration+1) + "/%03d" % i  ,filename)
        
        # latticelm�p�̃��X�g�t�@�C�����쐬
        fList.write( filename + "/fst_gmm_" + str(iteration+1) + "/%03d.fst" % i )
        fList.write( "\n" )
    fList.close()
