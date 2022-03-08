# Planning codes of SpCoTMHP (for SIGVerse and albert-b dataset)

## Content
*   [Files](#files)
*   [Execution Environment](#execution-environment)
*   [Execution Procedure](#execution-procedure)



## Files
 - `README.md`: Read me file (This file)
 - `__init__.py`: Code for initial setting (PATH and parameters)

 - `albert_b_spconavi_astar_execute.py`: 
 - `albert_b_spconavi_viterbi_path_calculate.py`: 
 - `albert_b_spcotmhp_astar_metric.py`: 
 - `albert_b_spcotmhp_dijkstra_execute.py`: 

 - `evaluate.py`: Evaluation code for exp1
 - `evaluate2.py`: Evaluation code for exp2
 - `evaluate_all_3LDK.py`: Unimplemented
 - `postprocess_shortcut.py`: Unimplemented

 - `run_exp1_***.py`: Run the experiment 1 (basic task) in our paper
 - `run_exp2_***.py`: Run the experiment 2 (advanced task) in our paper
 - `spconavi_astar_execute.py`: Execute code for SpCoNavi (A* Algorithm)
 - `spconavi_output_pathmap_step.py`: Visualization of the estimated path
 - `spconavi_read_data.py`: Module of reading pre-trained spatial concept model parameter, etc.
 - `spconavi_save_data.py`: Module of saving result of path, etc
 - `spconavi_viterbi_execute.py`: Execute code for SpCoNavi (Viterbi Algorithm)
 - `spconavi_viterbi_path_calculate.py`: Module of caliculating path by viterbi algorithm
 - `spcotmhp_astar_execute_d.py`: Heuristic hierarchical path plan-ning using spatial concepts (the cumulative cost of a partial path in A*)
 - `spcotmhp_astar_execute_d2.py`: Heuristic hierarchical path plan-ning using spatial concepts (the partial path distance)
 - `spcotmhp_astar_metric.py`: Execute code for the partial metric path planning by A*
 - `spcotmhp_astar_multicand_metric.py`: Unimplemented
 - `spcotmhp_dijkstra_execute.py`: Execute code for topogical path-planning using Dijkstra (main program)
 - `spcotmhp_dijkstra_multicand_execute.py`: Unimplemented
 - `spcotmhp_output_pathmap.py`: Visualization of the estimated path
 - `spcotmhp_viterbi_execute.py`: Execute code for topogical path-planning using Viterbi (Unimplemented)
 - `submodules.py`: Sub-program for functions 
 - `truth_astar_execute.py`: Execute code for the plannning of the truth path by A* using truth goal (for SPL evaluation)


## Execution environment  
- Ubuntu: 16.04 LTS / 18.04 LTS
- Python: 
    - 2.7.12 (numpy: 1.16.4, scipy: 1.2.2, matplotlib: 1.5.1)
    - 3.5.2 (numpy: 1.16.4, scipy: 1.3.0, matplotlib: 3.0.3)


## Execution procedure

`trialname` is the data folder name of the learning result in SpCoSLAM.  
For example, trialname is `3LDK_01` in `data` folder.  




### **Command for test execution of SpCoTMHP (for metric planning by A\* algorithm)**
Setting parameters and PATH in `__init__.py`  
~~~
python3 spcotmhp_astar_metric.py trialname mapname iteration sample type_gauss
~~~

Example:   

    python3 spcotmhp_astar_metric.py 3LDK_01 s3LDK_01 1 0 g


### **Command for test execution of SpCoTMHP (for topological planning by Dijkstra algorithm)**
Setting parameters and PATH in `__init__.py`  
~~~
python3 spcotmhp_dijkstra_execute.py trialname iteration sample init_position_num speech_num initial_position_x initial_position_y waypoint_word
~~~

Example: 

    python3 spcotmhp_dijkstra_execute.py 3LDK_01 1 0 -1 7 100 100 -1 


Execution commands for other programs are described in each source file. 
