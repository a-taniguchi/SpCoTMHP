# Learning codes of SpCoA, SpCoSLAM, SpCoTMHP (Gibbs sampling) 
Spatial concept formation (learning) of the batch learning without image features and lexical acquisition for SIGVerse and albert-b dataset

## 【Files】  
 - `README.md`: Read me file (This file)
 - `__init__.py`: Code for initial setting (PATH and parameters)
 - `evaluateSpCoTMHP.py`: For evaluation of learning result by SpCoTMHP (albert-b)
 - `learnSpCoTMHP_SIGVerse.py`: 
    - Spatial concept formation model (SpCoA without lexical acquisition)
    - For SpCoNavi -> TMHP (on SIGVerse; /3LDK/ dateset) 
    - [Note] Still SpCoSLAM (Gibbs). Can not learn psi.
 - `learnSpCoTMHP.py`: Learning of spatial concepts by SpCoTMHP (albert-b)
  - `SpCoVisualizer_albert-b.py`: Visualization of spatial concepts and the map by SpCoTMHP (albert-b)
 - `submodules.py`: Sub-program for functions


 ## NoUse
 - `learn4_3SpCoA_GT.py`: Main code for learning
 - `learn4_3SpCoA_GT.sh`: Main shell script for `learn4_3SpCoA_GT.py`
 - `new_place_draw.py`: Visualization code for position distributions of spatial concepts
 - `saveSpCoMAP_SIGVerse.rviz`: rviz file for Visualization