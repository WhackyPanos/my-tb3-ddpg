# my-tb3-ddpg
Turtlebot3 path planning using DDPG algorithm

Dept. of PEM, Technical University of Crete. Thesis by Panos Skoulaxinos (in Greek), https://doi.org/10.26233/heallink.tuc.94117

Implemented in Python with TensorFlow/Keras and runs as a ROS node on TB3.

## Some info

### Intro
The system is set up to run on ROS for the Turtlebot 3 Burger robotic platform. The agent is first trained on a simulation environment using Gazebo. It learns to navigate in environments with obstacles, where after a good training it learns to go around them, avoiding collisions. The system was tested in both known and unknown environments with different levels of complexity, where it proved that it is capable of fuctioning properly, navigating the robot to its goal without colliding. 

### About DDPG
The Deep Deterministic Policy Gradient algorithm decides the policy of the agent in a continuous field, rather than picking a predetermined policy. It utillises stochastic behaviour policy for more effective exploration of the given environment (Ornstein-Uhlenbeck in this case). It uses 2 pairs of neural networks, the "regular" Actor and the Critic as well as the "target" Actor and Critic.

#### Actor
The actor NN is responsible for outputting an action (a) after taking the state (s) of the agent as an input. 

#### Critic
The critic NN takes as an input both the state and the action and evaluates them, outputting a quality (Q) value.

#### Target pair of NNs
Characteristic of the DDPG algorithm is this copy of the "regular" NNs. They are a "lagged" version of the regular NN pair, where their weights are updated less frequently. This helps in stability during the training process, as it prevents the agent from developing unwanted behaviours. 

#### Structure
![NN structure](/networks.png?raw=true "NN structure")

## Instructions 
### Training
By editing `rl.py` one can set up the training hyperparameters. By running this script, `actor.h5` and `critic.h5` (included) will be imported as weights for the NNs and starts the training process in the current environment. By editing the `goals` variable one can setup predetermined goals that the robot will attempt to navigate to sequentially. It will advance to the next goal when it successfully reaches the previous one. When the training process ends, ie. the set number of training episodes is reached, statistics are printed, a plot with the average episodic reward per episode is exported as well as a log with the MDP, for diagnostic purposes.

### Using the path planner
By importing the `framework.py` file in a new script, one can command the robot to navigate in the current environment. The movement commands are given like this: 
```python
import framework
f= framework.Mover()

f.set_goal([x, y]) # where x, y the coordinates the robot should navigate to
f.move()
```

## References
* https://github.com/rcampbell95/turtlebot3_ddpg
* https://keras.io/examples/rl/ddpg_pendulum/
