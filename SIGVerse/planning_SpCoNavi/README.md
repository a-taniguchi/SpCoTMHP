# Planning codes of SpCoNavi for SIGVerse

【Files】  
 - `Astar_Database.py`: A star algorithm (goal position selection from database for training data in spatial concept learning)
 - `Astar_SpCo.py`: A star algorithm (goal position selection from learned spatial concepts by conventional method)
 - `README.md`: Read me file (This file)
 - `SpCoNavi0.1_SIGVerse.py`: Main path-planning code of SpCoNavi in SIGVerse
 - `__init__.py`: Code for initial setting (PATH and parameters)
 - `__init__SIGVerse.py`: Initial setting code (backup for our experiment in SIGVerse)
 - `costmap_SIGVerse.py`: Program to get costmap for SIGVerse
 - `path_weight_visualizer_step_SIGVerse.py`: Program for visualization of path trajectory and emission probability (log scale) for each step in SIGVerse
 - `submodules.py`: Sub-program for functions

【Add Files】(in preparation)  
 - `Astar_SpCo_costmap.py` and `Astar_Database_costmap.py`: A star algorithm with a costmap
 - `Astar_SpCo_weight.py` and `Astar_Database_weight.py`: A star algorithm with a costmap and the emission probability weights
 - `SpCoNavi_Astar_approx.py`: SpCoNavi of an approximation version by A star algorithm
