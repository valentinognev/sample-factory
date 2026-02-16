# Reward function
- make sure to scale the different reward function parameters so that they will have roughly the same scale (mean & variance)
- after scaling, you can alter weights according to importance
- (neet to be tested) penalizing actions might yield more economical/optimal use of them, especially if some actions are more "expensive" than others

# Observation & Action Spaces
- since the network will only learn to ineract with the numbers it was trained under (maybe a bit more, but not much) - these steps are needed to ensure "good" generalization capability:
1. use relative frames-of-reference - this way, the network does not depend on absolute numbers, but rather on a small range of relativistic numbers  (i.e use body frame, that is, range from me and angle from me, rather than global positions)
2. even when using relativistic frames, clip "dangerous" parameters so the network will not see off-the-scale values 
(i.e. if your network was trained for maximum distance from me D, then clip the input of that parameter in the inference observation space to D - if the object is further than that you don't care about the actual distance anyway)

# Env-to-Real 
- some key points where it is *vital* for the environment to simulate the real-space:
1. action & observation spaces (not necessarily in scale, but at least in essence)
2. frames of reference (axis directions, angle directions, transformation frames, etc...)
3. dynamic behavior (the level of accuracy needed is still up for debate)

# Testing good-practices
1. overfitting is a feature, not a bug - overfit to isolate the core issue, then when the basic problem is solved - expand the scope and teach the system to generalize

# uncategorized
- converging training - when training for waypoint - started by training for large radius for most of training (~80%), then trained for very small radius for the remaining. this lets the agent experience both long-range following (larger velocities, steadier state...), and also focus on the end zone (slow down).
