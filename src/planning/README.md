# Planning codes of SpCoNavi (SIGVerse)

【Files】  
 - `README.md`: Read me file (This file)

 - `spconavi_astar_path_calculate.py`: Execute code for SpCoNavi (A* Algorithm)

 - `spconavi_execute.py`:  Execute code for SpCoNavi (Viterbi Algorithm)

 - `spconavi_read_data.py`: Module of reading pre-trained spatial concept model parameter, etc.

 - `spconavi_path_caliculate.py`: Module of caliculating path by viterbi algorithm

 - `spconavi_save_data.py`: Module of saving result of path, etc

 - `spconavi_output_map.py`: Program for visualization of path trajectory and emission probability (log scale) for each step in SIGVerse

 - `spconavi_path_visualizer_rviz.py`: Visualize estimated path on rviz

  - `__init__.py`: Code for initial setting (PATH and parameters)

 - `submodules.py`: Sub-program for functions



# SpCoNavi -> SpCoTMHP
Original SpCoNavi code is here： [https://github.com/a-taniguchi/SpCoNavi](https://github.com/a-taniguchi/SpCoNavi)

現状、長谷川くんのspconavi_rosのREADME.mdのコピー

## Content

*   [Execution Environment](#execution-environment)
*   [Execution Procedure](#execution-procedure)
*   [Folder](#folder)
*   [Reference](#reference)
*   [Other Repositories](#other-repositories)
*   [Acknowledgements](#acknowledgements)


## Execution environment  
- Ubuntu：18.04LTS
- Python：2.7.17 (numpy：1.16.6, scipy：1.2.2, matplotlib：2.1.1)
- ROS：Melodic


## Execution procedure

`trialname` is the data folder name of the learning result in SpCoSLAM.  
For example, trialname is `3LDK_01` in `data` folder.  

### Command for learning of spatial concepts  
In the home environment, you need to have a training data set (robot positions, words, and images).  
~~~
cd ~/*/SpCoNavi/SIGVerse/learning/
python ./learn4_3SpCoA_GT.py 3LDK_01
~~~

### Visulalization of the learning result  
~~~
roscore
rosrun map_server map_server ~/*/SpCoNavi/SIGVerse/data/3LDK_01/navi/3LDK_01.yaml
python ./new_place_drawy 3LDK_01 1 0
rviz -d ./saveSpCoMAP_online_SIGVere.rviz 
~~~


### Command for test execution of SpCoNavi (Viterbi algorithm)
Setting parameters and PATH in `__init__.py`  
~~~
cd spconavi_ros/src/planning
python spconavi_execute.py trialname iteration sample init_position_num speech_num
~~~
Example: 
`python spconavi_execute.py 3LDK_01 1 0 0 7`  

### Command for test execution of SpCoNavi (A* algorithm)
Setting parameters and PATH in `__init__.py`  
~~~
cd spconavi_ros/src/planning
python spconavi_astar_path_calculate.py trialname mapname iteration sample init_position_num speech_num initial_position_x initial_position_y
~~~
Example: 
`python spconavi_astar_path_calculate.py 3LDK_01 s1DK_01 1 0 0 7 100 100 `  


### Command for visualization of a path trajectory and the emission probability on the map
~~~
cd spconavi_ros/src/planning
python spconavi_out_map.py trialname init_position_num speech_num  
~~~
Example: 
`python spconavi_output_map.py 3LDK_01 0 7`  



## Folder  
 - `/Supplement/HSR/`: Supplemental files for virtual HSR robot
 - `/data/`: Data folder including sample data
 - `/learning/`: Codes for learning
 - `/planning/`: Codes for planning
 
---
## Reference
[1]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "[Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping](https://ieeexplore.ieee.org/document/8202243)", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.  
[2]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "[Improved and scalable online learning of spatial concepts and language models with mapping](https://link.springer.com/article/10.1007/s10514-020-09905-0)", Autonomous Robots, Vol.44, pp927-pp946, 2020.
[3]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "Path Planning by Spatial Concept-Based Probabilistic Inference from Human Speech Instructions", the 33rd Annual Conference of the Japanese Society for Artificial Intelligence, 2019. (In Japanese; 谷口彰，萩原良信，谷口忠大，稲邑哲也. [場所概念に基づく確率推論による音声命令からのパスプランニング](https://www.jstage.jst.go.jp/article/pjsai/JSAI2019/0/JSAI2019_1L3J1103/_article/-char/ja/). 人工知能学会全国大会 (JSAI). 2019.)   
[4]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "[Spatial concept-based navigation with human speech instructions via probabilistic inference on Bayesian generative model](https://www.tandfonline.com/doi/full/10.1080/01691864.2020.1817777)", Advanced Robotics, pp1213-pp1228, 2020.


## Other repositories  
 - [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots  
 - [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 - [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
 - [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  
 - [SpCoNavi](https://github.com/a-taniguchi/SpCoNavi): Spatial Concept-Based Navigation from Human Speech Instructions by Probabilistic Inference on Bayesian Generative Model



