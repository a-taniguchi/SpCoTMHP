# SpCoNavi for SIGVerse

This README is a copy of old version (original SpCoNavi).  

SpCoNavi: Spatial Concept-Based Navigation from Human Speech Instructions by Probabilistic Inference on Bayesian Generative Model  
<img src="https://github.com/a-taniguchi/SpCoNavi/blob/master/img/outline.png" width="480px">


## Execution environment  
- Ubuntu 16.04 LTS (Virtual Machine by VMware on Windows 10)  
    - Python 2.7.12 and 3.5.2  
        -  Python2: numpy 1.16.4, scipy 1.2.2, maplotlib 1.5.1   
        - (Python3: numpy 1.16.4, scipy 1.3.0, maplotlib 3.0.3)  
    - ROS kinetic  
- Windows 10  
    - Unity 2018.4.0f1  

## Preparation for execution 
### Windows  
Set up Unity and SIGVerseProject according to the wiki page of SIGVerse.  
Launch `HsrTeleop`, which is the sample project of HSR.  
(in `/sigverse_unity_project/SIGVerseProject/Assets/SIGVerse/SampleScenes/HSR/`)  

SIGverse wiki: http://www.sigverse.org/wiki/  
SIGVerse github: https://github.com/SIGVerse  

【Use / change of room environment】  
Load an unity project of your room environment. 

If you use a virtual HSR robot model;  
Incorporate the HSR model into the room environment from `HsrTeleop`.  

You can use the room environments in `SweetHome3D_rooms`.   
SweetHome3D\_rooms: https://github.com/EmergentSystemLabStudent/SweetHome3D_rooms   


### Ubuntu  
Set up SIGVerse and ROS according to the wiki page of SIGVerse.  
Launch Examples in Tutorial.  

If you observe an error in Mongo C ++ driver installation, please execute the following command;  
~~~
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. -DLIBMONGOC_DIR=/usr/local -DLIBBSON_DIR=/usr/local ..
~~~

Be careful to remember the following when running the examples;  
~~~
cd ~/catkin_ws/
source devel/setup.bash
~~~


【Command for preparation】  
~~~
sudo apt install python-pip python3-pip
sudo apt-get install python3-tk
sudo pip install numpy scipy matplotlib
sudo pip3 install numpy scipy matplotlib
~~~

(Option: For version update)  
~~~
sudo pip install numpy --upgrade
sudo pip install scipy --upgrade
sudo pip2 install scipy --ignore-installed
~~~


## Execution procedure
【Command list for cost map acquisition】  
~~~
(Terminal: the movement of HSR by keyboard operation and running rviz)
roslaunch sigverse_hsr_teleop_key teleop_key_with_rviz.launch
---
(Windows)
<< Unity start >>
---
(Another terminal: run gmapping. Including the parameters setting to command, i.e., resolution and map size.)
rosrun gmapping slam_gmapping scan:=/hsrb/base_scan _xmin:=-10.0 _ymin:=-10.0 _xmax:=10.0 _ymax:=10.0 _delta:=0.1
---
(Another terminal: It needs catkin_make in the /costmap_global/ folder before running the following commands.)
source ~/*/SpCoNavi/costmap_global/devel/setup.bash
roslaunch fourth_robot_2dnav global_costmap_SIGVerse.launch

（If you specify a map yaml file.）
roslaunch fourth_robot_2dnav global_costmap_SIGVerse.launch map_file:=my_map.yaml
---
(Another terminal: Save the map and the cost map)
cd ~/*/SpCoNavi/planning
python costmap_SIGVerse.py trialname
rosrun map_server map_saver -f ../SIGVerse/data/trialname/navi/trialname
~~~
`trialname` is the data folder name of the learning result in SpCoSLAM.  
For example, trialname is `3LDK_01` in `data` folder.  

【Command for learning of spatial concepts】  
In the home environment, you need to have a training data set (robot positions, words, and images).  
~~~
cd ~/*/SpCoNavi/SIGVerse/learning/
python ./learn4_3SpCoA_GT.py 3LDK_01
~~~

【Visulalization of the learning result】  
~~~
roscore
rosrun map_server map_server ~/*/SpCoNavi/SIGVerse/data/3LDK_01/navi/3LDK_01.yaml
python ./new_place_drawy 3LDK_01 1 0
rviz -d ./saveSpCoMAP_online_SIGVere.rviz 
~~~


【Command for test execution of SpCoNavi】  
Setting parameters and PATH in `__init__.py`  
~~~
cd ./learning/
python ./SpCoNavi0.1_SIGVerse.py trialname iteration sample init_position_num speech_num
~~~
Example: 
`python ./SpCoNavi0.1_SIGVerse.py 3LDK_01 1 0 0 7`  

【Command for visualization of a path trajectory and the emission probability on the map】
~~~
python ./path_weight_visualizer_step_SIGVerse.py trialname init_position_num speech_num  
~~~
Example: 
`python ./path_weight_visualizer_step_SIGVerse.py 3LDK_01 0 7`  

【Option: Command for A star algorithms】  
This code only works with Python 3.  
~~~
python3 ./Astar_SpCo.py 3LDK_01 s3LDK_01 1 0 0 7 100 100
python3 ./Astar_Database.py 3LDK_01 s3LDK_01 1 0 0 7 100 100
~~~


## Folder  
 - `/Supplement/HSR/`: Supplemental files for virtual HSR robot
 - `/data/`: Data folder including sample data
 - `/learning/`: Codes for learning
 - `/planning/`: Codes for planning
 
---
## Reference
[1]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.  
[2]: Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "Path Planning by Spatial Concept-Based Probabilistic Inference from Human Speech Instructions", the 33rd Annual Conference of the Japanese Society for Artificial Intelligence, 2019. (In Japanese; 谷口彰，萩原良信，谷口忠大，稲邑哲也. 場所概念に基づく確率推論による音声命令からのパスプランニング. 人工知能学会全国大会 (JSAI). 2019.)    


## Other repositories  
 - [SpCoSLAM_Lets](https://github.com/EmergentSystemLabStudent/SpCoSLAM_Lets): Wrapper of SpCoSLAM for mobile robots (Recommended)  
 - [SpCoSLAM](https://github.com/a-taniguchi/SpCoSLAM): Implementation of SpCoSLAM (Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping)   
 - [SpCoSLAM 2.0](https://github.com/a-taniguchi/SpCoSLAM2): An Improved and Scalable Online Learning of Spatial Concepts and Language Models with Mapping (New version of online learning algorithm)   
 - [SpCoSLAM_evaluation](https://github.com/a-taniguchi/SpCoSLAM_evaluation): The codes for the evaluation or the visualization in our paper  

2019/06/25  Akira Taniguchi  
2019/07/12  Akira Taniguchi (Update)  
