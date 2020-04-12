#! /bin/sh

PATH=$PATH:../../latticelm
echo -n "trialname?(output_folder; 3LDK_05) >"
read bun
python ./learn4_3SpCoA_GT.py $bun #| tee -a data/log_$bun.txt
