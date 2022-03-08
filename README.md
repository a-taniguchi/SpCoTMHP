# SpCoTMHP
Spatial Concept-based Topometric Semantic Mapping for Hierarchical Path-planning from Speech instructions

<img src="https://github.com/a-taniguchi/SpCoTMHP/blob/master/overview.png" width="480px">

## Folders
 - `/SIGVerse/`: Dataset and learning results of home environments in SIGVerse simulator
   - `/data/`: Learning results of spatial concepts
   - `/dataset/similar/`: Dataset for spatial concept learning (for SpCoTMHP partially)
   - `/learning/`: Source codes of learning and visualization for 3LDK dataset

 - `/albert-b/`: Dataset and learning results in real world environment (based on albert-b dataset)
   - `/data/`: Learning results of spatial concepts
   - `/dataset/`: Dataset for spatial concept learning for SpCoTMHP
   - `/learning/`: Source codes of learning and visualization for albert-b dataset

 - `/src/planning/`: source code folder for path-planning



## Abstract 
    Navigating to destinations using human speech instructions is an important task for autonomous mobile robots that operate in the real world.
    Spatial representations include a semantic level that represents an abstracted location category, a topological level that represents their connectivity, and a metric level that depends on the structure of the environment.
    The purpose of this study is to realize a hierarchical spatial representation using a topometric semantic map and planning efficient paths through human-robot interactions.
    We propose a novel probabilistic generative model, SpCoTMHP, that forms a topometric semantic map that adapts to the environment and leads to hierarchical path planning.
    We also developed approximate inference methods for path planning, where the levels of the hierarchy can influence each other.
    The proposed path planning method is theoretically supported by deriving a formulation based on control as probabilistic inference.
    The navigation experiment using human speech instruction shows that the proposed spatial concept-based hierarchical path planning improves the performance and reduces the calculation cost compared with conventional methods.
    Hierarchical spatial representation provides a mutually understandable form for humans and robots to render language-based navigation tasks feasible. 


## Reference
 -   Akira Taniguchi, Shuya Ito, Tadahiro Taniguchi, "Spatial Concept-based Topometric Semantic Mapping for Hierarchical Path-planning from Speech Instructions", Submitted to IROS 2022.



---
## Related references
1. Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "[Online Spatial Concept and Lexical Acquisition with Simultaneous Localization and Mapping](https://ieeexplore.ieee.org/document/8202243)", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017.  
1. Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, and Tetsunari Inamura, "[Improved and scalable online learning of spatial concepts and language models with mapping](https://link.springer.com/article/10.1007/s10514-020-09905-0)", Autonomous Robots, Vol.44, pp927-pp946, 2020.
1. Akira Taniguchi, Yoshinobu Hagiwara, Tadahiro Taniguchi, Tetsunari Inamura, "[Spatial concept-based navigation with human speech instructions via probabilistic inference on Bayesian generative model](https://www.tandfonline.com/doi/full/10.1080/01691864.2020.1817777)", Advanced Robotics, pp1213-pp1228, 2020.
    -  Original SpCoNavi code is here:  [https://github.com/a-taniguchi/SpCoNavi](https://github.com/a-taniguchi/SpCoNavi)




