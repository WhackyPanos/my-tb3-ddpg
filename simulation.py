import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler

import random

class Simulation():
    def __init__(self, arena_dims=[10, 10]):

        # model_state_msg = rospy.wait_for_message('/gazebo/model_states', ModelStates)
        #
        # objects = [model_state_msg.name.index(a) for a in model_state_msg.name if 'unit_box' in a]
        #
        # self.object_positions_x = list()
        # self.object_positions_y = list()
        self.arena_dims = arena_dims
        #
        # for i in objects:
        #     self.object_positions_x.append(model_state_msg.pose[i].position.x)
        #     self.object_positions_y.append(model_state_msg.pose[i].position.y)

    def random_spawn(self):
        threshold = 0.2
        obsSize = 1
        #
        # blocked = True
        # while blocked == True:
        #     random_x = random.randint(0, self.arena_dims[0]-1) +obsSize/2
        #     random_y = random.randint(0, self.arena_dims[1]-1) +obsSize/2
        #     blocked = False
        #
        #     for x, y in zip(self.object_positions_x, self.object_positions_y):
        #         if ((abs(random_x - x +obsSize/2) < threshold) and (abs(random_y - y +obsSize/2) < threshold)) \
        #         or ((abs(random_x - x -obsSize/2) < threshold) and (abs(random_y - y -obsSize/2) < threshold)) \
        #         or ((abs(random_x - x +obsSize/2) < threshold) and (abs(random_y - y -obsSize/2) < threshold)) \
        #         or ((abs(random_x - x -obsSize/2) < threshold) and (abs(random_y - y +obsSize/2) < threshold)):
        #             blocked = True
        #
        #             break
        random_x = random.randint(0, self.arena_dims[0]-1) +obsSize/2
        random_y = random.randint(0, self.arena_dims[1]-1) +obsSize/2
        self.move_to(x=random_x, y=random_y)

    def reset_world(self):
        rospy.wait_for_service('/gazebo/reset_world')

    def move_to(self, x=0, y=0, r=None):
        if x > self.arena_dims[0] or y > self.arena_dims[1]:
            print('Invalid coords!')
            return

        state_msg = ModelState()
        state_msg.model_name = 'turtlebot3_burger'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = 0

        if r == None:
            state_msg.pose.orientation.x = 0
            state_msg.pose.orientation.y = 0
            state_msg.pose.orientation.z = 0
            state_msg.pose.orientation.w = 0
        else:
            (w,x,y,z) = quaternion_from_euler(r,0,0)
            state_msg.pose.orientation.x = x
            state_msg.pose.orientation.y = y
            state_msg.pose.orientation.z = z
            state_msg.pose.orientation.w = w

        rospy.wait_for_service('/gazebo/set_model_state')

        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

        except rospy.ServiceException:
            print ("Service call failed")
