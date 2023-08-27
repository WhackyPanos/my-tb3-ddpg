#!/usr/bin/env python3

import rospy
import navigation
import simulation
import logger

from geometry_msgs.msg import Twist, Pose, PoseArray, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import euler_from_quaternion

import numpy as np
import matplotlib.pyplot as plt
import sys, random
import math
from datetime import datetime
from time import sleep
from statistics import mean
import shutil
from random import randrange

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam

num_states = 14
num_actions = 2
critic_lr = 0.0001
actor_lr = 0.001
total_episodes = 10000
max_steps = 700
gamma = 0.90
tau = 0.01
buffer_size = 200000
batch_size = 512
arena_dims = [4, 4]

rospy.init_node('drl_ddpg_node')
nav = navigation.Navigation()
class Buffer:
    def __init__(self, buffer_capacity=200000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# class OUActionNoise:
#     def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
#         self.theta = theta
#         self.mean = mean
#         self.std_dev = std_deviation
#         self.dt = dt
#         self.x_initial = x_initial
#         self.reset()
#
#     def __call__(self):
#         # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
#         x = (
#             self.x_prev
#             + self.theta * (self.mean - self.x_prev) * self.dt
#             + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
#         )
#         # Store x into x_prev
#         # Makes next noise dependent on current one
#         self.x_prev = x
#         return x
#
#     def reset(self):
#         if self.x_initial is not None:
#             self.x_prev = self.x_initial
#         else:
#             self.x_prev = np.zeros_like(self.mean)
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period= 600000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def __call__(self, t=0):
        ou_state = self.evolve_state()
        # print('noise' + str(ou_state))
        decaying = float(float(t)/ self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(512, activation="relu")(inputs)
    out = layers.Dense(512, activation="relu")(out)
    out = layers.Dense(512, activation="relu")(out)
    out1 = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    out2 = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)(out)
    concat = layers.Concatenate()([out1, out2])

    outputs = concat
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(512, activation="relu")(state_input)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(512, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(512, activation="relu")(concat)
    # out = layers.Dense(512, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object, t):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object(t)
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + 0.1*noise

    # We make sure action is within bounds
    legal_action = list()
    legal_action.append(np.clip(sampled_actions[0], -1, 1))
    legal_action.append( np.clip(sampled_actions[1], 0, 1))
    #print(np.squeeze(legal_action))
    return [np.squeeze(legal_action)]

def get_policy(state):
    sampled_actions = tf.squeeze(actor_model(state))
    # noise = noise_object(t)
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() #+ noise

    # We make sure action is within bounds
    legal_action = list()
    legal_action.append(np.clip(sampled_actions[0], -1, 1))
    legal_action.append( np.clip(sampled_actions[1], 0, 1))
    #print(np.squeeze(legal_action))
    return [np.squeeze(legal_action)]

def get_state(vel):
    laser_reward = 0
    collisionFlag = False
    x, y, _ = nav.returnOdometry()

    distance = (goal_pos[0] - x)**2 + (goal_pos[1] - y)**2
    distance = math.sqrt(distance)

    inc_y = goal_pos[1] - y
    inc_x = goal_pos[0] - x
    angle_to_goal = math.atan2(inc_y, inc_x)
    angle = angle_to_goal - nav.theta
    cur_angle = nav.theta

    if cur_angle <0:
        cur_angle = cur_angle +2*math.pi

    if angle_to_goal <0:
        angle_to_goal = angle_to_goal +2*math.pi

    if angle < -math.pi:
        angle = angle +2*math.pi
    if angle > math.pi:
        angle = angle -2*math.pi
    laser_reward = 0
    for i in nav.laser:
        if i<0.1:
            collisionFlag = True
        elif i < 2*0.2:
            laser_reward = -80

    normalized_laser = [(x)/3.5 for x in (nav.laser)]
    state=list(normalized_laser)
    laser_dist_reward = sum(normalized_laser)-24

    if vel[0][0] <0.2:
        linear_punish = -2

    if vel[0][1] >0.8 or vel[0][1]<-0.8:
        angular_punish = -1


    state.append(angle)
    state.append(distance)
    state.append(vel[0][0])
    state.append(vel[0][1])



    # preprosessing & convert to tensor
    state = np.array(state, dtype=object).astype('float32')
    state[state==np.inf] = 3.5
    state[state==np.nan] = 0
    #print(state)
    state = tf.convert_to_tensor(state)
    #state = tf.reshape(state, [1, num_states])

    return state, [distance, angle], collisionFlag

def set_reward(to_goal, prev, goal, step):
    dist_to_goal = to_goal[0]
    prev_dist = prev[0]

    angle_to_goal = to_goal[1]
    prev_angle = prev[1]
    rate = (prev_dist - dist_to_goal)

    if col == True:
        return -200.
    elif goal:
        return 100.
    else:
        return rate*1.2*7*(5/(step+1)) + angular_punish + linear_punish + laser_reward + laser_dist_reward
        print(angular_punish, linear_punish, laser_reward, laser_dist_reward)

def get_goal(to_goal):
    dist_to_goal = to_goal[0]

    goal = False

    if dist_to_goal <= 0.1:
        goal = True

    return goal

if __name__ == '__main__':
    # rospy.init_node('rl_ddpg_test_node')
    gazebo = simulation.Simulation(arena_dims=arena_dims)
    sleep(.5)

    my_log = logger.Logger() # ['episode', 'x', 'y', 'theta', 'goal', 'state', 'action', 'reward']

    ou_noise = OUNoise(2, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

    actor_model = get_actor()
    critic_model = get_critic()

    target_actor = get_actor()
    target_critic = get_critic()

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%m_%H-%M-%S")

    try:
        # backup OG weights just to make sure we don't screw things up
        shutil.copyfile('actor.h5', 'actor_backup_' + timestampStr +'.h5')
        shutil.copyfile('critic.h5', 'critic_backup_' + timestampStr +'.h5')

        actor_model.load_weights("actor.h5")
        critic_model.load_weights("critic.h5")

        target_critic.load_weights("critic.h5")
        target_actor.load_weights("actor.h5")
    except:
        print("Weight files not found.")
        # Making the weights equal initially
        target_actor.set_weights(actor_model.get_weights())
        target_critic.set_weights(critic_model.get_weights())

    critic_optimizer = Adam(critic_lr)
    actor_optimizer = Adam(actor_lr)

    buffer = Buffer(buffer_size, batch_size)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # goals = [[1,1], [3.5,1.5], [3,3], [1.5,1.5], [2,2], [1,2], [2.5,1.5], [2.5,2.5], [1,3], [3.5,2.5],
    # [2,3], [1.5,2.5], [2,1], [1.5,3.5], [3,1], [2.5,3.5], [3,2], [2.5,1.5]]
    # goals = [[1,1], [2,2], [1,2], [1,3], [2,3], [2,1], [3,1], [3,2], [3,3], [1.5,3.5], [2, 1.5], [2,2.5], [3.5,1], [2,0.5], [3.5,2.5]]
    # goals = [[1,1], [3,3], [2,2], [1,2], [1,3], [3.5,2.5],
    # [2,3], [2,1], [1.5,3.5], [3,1], [2.5,3.5], [3,2]]
    # goals = [ [1.5, 0.5], [1.5,1.5] ,[2.5,1.5], [2.5,2.5], [2.5,3.5], [2.5,4.5], [3.5,2.5], [3.5, 0.5], [3.5, 2.5], [3.5, 3.5], [4.5, 4.5], [4.5, 3.5], [2.5, 3.5], [2.5,4.5]]


# for my_stage_1
#     goals = [
#     [1,0.5], [1.5,0.5], [2,0.5], [2.5,0.5], [3,0.5], [3.5,0.5], [3.5,1.5], [2.5,1.5], [2,2], [0.5,2], [1.5,2.5], [1.5,3], [2,3], [2.5,3.5], [3.5,3.5]
# ]

    goals = [ [2, 0.5], [0.5, 2], [1.5, 0.5], [0.5, 1.5], [1, 0.5], [0.5, 1], [2.5, 0.5], [0.5, 2.5], [1.5, 2.5], [1.5, 1.5], [2.5, 1.5], [3.5, 1.5], [3.5, 0.5], [1.5, 3.5], [2.5, 3.5], [3.5, 3.5] ]

    spawn_pos = [ [0.5, 0.5], [0.5, 1.5], [0.5, 2.5], [1.5, 0.5], [2.5, 0.5] ]
    goal_index = 0
    gazebo.move_to(x=0.5,y=0.5)

    timeout_counter = 0
    goal_counter = 0
    collision_counter = 0
    linear_punish = 0
    angular_punish = 0
    laser_reward =0
    laser_dist_reward  =0
    prev_ep_success = True
    prev_goal = [0.5,0.5]

    for ep in range(total_episodes):
        goal=False
        goal_pos = random.choice(goals)
        goal_pos = goals[goal_index]
        if prev_ep_success:
            pos = prev_goal
            prev_ep_success = False

        # gazebo.move_to(x=.5, y=.5)
        # goal_pos = [2,2]

        sleep(0.5)
        rospy.loginfo("Current episode: "+ str(ep))
        rospy.loginfo("Current goal: " + str(goal_pos))
        sleep(0.5)
        state, to_goal, col = get_state([[0,0]])
        episodic_reward = 0

        for step in range(max_steps):
            # print(step)
            collisionFlag = False
            prev_to_goal = to_goal
            prev_state = state
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise, step)

            # Recieve state and reward from environment.
            nav.publish(v=0.25*action[0][1], r=action[0][0])
            state, to_goal, col = get_state(action)
            goal = get_goal(to_goal)
            reward = set_reward(to_goal=to_goal, prev= prev_to_goal, goal=goal, step=step)

            buffer.record((prev_state, action[0], reward, state))
            episodic_reward += reward

            if buffer.buffer_counter > batch_size:
                buffer.learn()
                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)
            x,y,theta = nav.returnOdometry()

            tuple = (ep, x, y, theta, goal_pos, state, action, reward)

            # my_log.log(tuple)

            # End this episode when `goal` is True
            if goal:
                rospy.logwarn("Goal!")
                goal_counter +=1
                prev_goal = goal_pos
                goal_index = goal_index +1 if goal_index < len(goals)-1 else 0
                prev_ep_success = True
                nav.publish(v=0,r=0)

                # if goal_counter % 15 ==0:
                #     random.shuffle(goals)

                # gazebo.random_spawn()
                break

            if col:
                rospy.logwarn("Collision!")
                collision_counter +=1
                nav.publish(v=0,r=0)
                gazebo.move_to(x=pos[0], y=pos[1],r=randrange(6))

                # if collision_counter % 30 ==0:
                #     goal_index = goal_index +1 if goal_index < len(goals)-1 else 0
                # gazebo.random_spawn()
                break

            if step >=max_steps-1:
                rospy.logwarn("Time out!")
                timeout_counter +=1
                nav.publish(v=0,r=0)
                gazebo.move_to(x=pos[0], y=pos[1],r=randrange(6))
                # gazebo.random_spawn()
                break

        ep_reward_list.append(episodic_reward)

        # Save weights every 50 episodes
        if ep % 50 ==0:
            actor_model.save_weights("actor.h5")
            critic_model.save_weights("critic.h5")

            # my_log.save('train-log_' + timestampStr + '.csv')

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.savefig('Fig_' + timestampStr + '.png')

    print("Goals hit: " + str(goal_counter))
    print("Collisions: " + str(collision_counter))
    print("Timeouts: " + str(timeout_counter))
