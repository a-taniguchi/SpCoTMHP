#! /bin/sh

PATH=$PATH:../../latticelm
echo -n "trialname?(output_folder) >"
read bun
python ./learn4_3SpCoSLAM_SIGVerse.py $bun #| tee -a data/log_$bun.txt
