# SpCoTMHP for SIGVerse


## Folder  
 - `/SIGVerse/`: Dataset and learning results of home environments in SIGVerse simulator
   - `/data/`: Learning results of spatial concepts
   - `/dataset/similar/`: Dataset for spatial concept learning (for SpCoTMHP partially)
   - `/learning/`: Source codes of learning and visualization for 3LDK dataset

## Execution environment  
- Ubuntu 16.04 LTS / 18.04 LTS (Virtual Machine by VMware on Windows 10)
    - Python 
      - 2.7.12 (numpy: 1.16.4, scipy: 1.2.2, matplotlib: 1.5.1)
      - 3.5.2 (numpy: 1.16.4, scipy: 1.3.0, matplotlib: 3.0.3)
    - ROS kinetic  
- Windows 10  
    - Unity 2018.4.0f1  

## Preparation for execution 
### ***Windows***
Set up Unity and SIGVerseProject according to the wiki page of SIGVerse.  
Launch `HsrTeleop`, which is the sample project of HSR.  
(in `/sigverse_unity_project/SIGVerseProject/Assets/SIGVerse/SampleScenes/HSR/`)  

 - SIGverse wiki: http://www.sigverse.org/wiki/  
 - SIGVerse github: https://github.com/SIGVerse  

#### **Use / change of room environment**
Load an unity project of your room environment. 

If you use a virtual HSR robot model;  
Incorporate the HSR model into the room environment from `HsrTeleop`.  

You can use the room environments in `SweetHome3D_rooms`.   
SweetHome3D\_rooms: https://github.com/EmergentSystemLabStudent/SweetHome3D_rooms   


### ***Ubuntu***
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


#### **Command for preparation**
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
#### **Command list for cost map acquisition**
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
python costmap_SIGVerse.py <trialname>
rosrun map_server map_saver -f ../SIGVerse/data/<trialname>/navi/<trialname>
~~~
`<trialname>` is the data folder name of the learning result in SpCoSLAM.  
For example, `<trialname>` is `3LDK_01` in `data` folder.  


Please see SpCoNavi github project for `costmap_SIGVerse.py`:  
Original SpCoNavi code is here:  [https://github.com/a-taniguchi/SpCoNavi](https://github.com/a-taniguchi/SpCoNavi)

---

#### **Command for learning of spatial concepts**  
In the home environment, you need to have a training data set (robot positions, words, and images).  
~~~
cd ./SpCoTMHP/SIGVerse/learning/
python ./learnSpCoTMHP.py <trialname>
~~~


#### **Visulalization of the learning result**  
~~~
python SpCoVisualizer_SIGVerse.py <trialname>
~~~

