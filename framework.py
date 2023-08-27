#!/usr/bin/env python3

import rl
import tensorflow as tf
import logger2
from datetime import datetime
import math

my_env = {

    'num_episodes' : 3000,
    'state_size' : 14,
    'action_size' : 2,
    'value_size' : 1,

    'arena_dims': [10,10],
    'goal_box' : 0.2,
    'collision_box' : 0.2,

    'actor_lr' : 0.0001,
    'critic_lr' : 0.0001,
    'gamma' : 0.99,
    'tau' : 0.005,
    'buffer_capacity' : 50000,
    'batch_size' : 64

}

class Mover():

    def __init__(self):
        self.nav = rl.nav
        rl.my_env = my_env

        rl.actor_model = rl.get_actor()
        rl.critic_model = rl.get_critic()
        rl.actor_model.load_weights("actor.h5")
        rl.critic_model.load_weights("critic.h5")

        self.my_log = logger2.Logger()
        dateTimeObj = datetime.now()
        self.timestampStr = dateTimeObj.strftime("%d-%m_%H-%M-%S")


    def set_goal(self, goal=[0,0]):
        rl.goal_pos = goal

    def move(self):
        state, to_goal, rl.col = rl.get_state([[0,0]])
        goal = rl.get_goal(to_goal)

        while not (goal or rl.col):
            prev_state = state
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = rl.get_policy(tf_prev_state)
            self.nav.publish(v=0.22*action[0][1], r=action[0][0])
            state, to_goal, rl.col = rl.get_state(action)
            goal = rl.get_goal(to_goal)
            x,y,theta = rl.nav.returnOdometry()
            x_ = 3.5- x
            y_ = 3.5- y
            d= math.sqrt(x_**2 + y_**2)
            tuple = (x,y,d, to_goal, state, action)
            self.my_log.log(tuple)
        self.nav.publish(v=0, r=0)

    def exit(self):
        self.my_log.save('log_' + self.timestampStr + '.csv')
        self.my_log.export_graphs()
