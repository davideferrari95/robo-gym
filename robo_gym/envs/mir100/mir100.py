#!/usr/bin/env python3

import copy
from time import *
from math import *
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, mir100_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

class Mir100Env(gym.Env):
    """Mobile Industrial Robots MiR100 base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.

    Attributes:
        mir100 (:obj:): Robot utilities object.
        observation_space (:obj:): Environment observation space.
        action_space (:obj:): Environment action space.
        distance_threshold (float): Minimum distance (m) from target to consider it reached.
        min_target_dist (float): Minimum initial distance (m) between robot and target.
        max_vel (numpy.array): # Maximum allowed linear (m/s) and angular (rad/s) velocity.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.
        laser_len (int): Length of laser data array included in the environment state.

    """

    real_robot = False
    laser_len = 1022
    max_episode_steps = 500 

    def __init__(self, rs_address=None, **kwargs):

        self.mir100 = mir100_utils.Mir100()
        self.elapsed_steps = 0
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=np.full((2), -1.0), high=np.full((2), 1.0), dtype=np.float32)
        self.seed()
        self.distance_threshold = 0.2
        self.min_target_dist = 1.0
        # Maximum linear velocity (m/s) of MiR
        max_lin_vel = 0.5
        # Maximum angular velocity (rad/s) of MiR
        max_ang_vel = 0.7
        self.max_vel = np.array([max_lin_vel, max_ang_vel])

        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, start_pose = None, target_pose = None):
        """Environment reset.

        Args:
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose)==3
        else:
            start_pose = self._get_start_pose()

        rs_state[3:6] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose)==3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[0:3] = target_pose

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.tolist())
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        return 0, False, {}

    def step(self, action):
        
        # Convert from tf to numpy
        if type(action).__name__ == 'ndarray': action = action.astype(np.float32)
        elif type(action).__name__ == 'EagerTensor': action = action.numpy()
        else: print(f'Action {type(action).__name__} Type Not Recognized')

        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Convert environment action to Robot Server action
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action = np.multiply(action, self.max_vel)
        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign reward
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info

    def render(self):
        pass

    def _get_robot_server_state_len(self):
        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """

        # Create a vector of 3 elements 0.0
        target = [0.0] * 3
        mir_pose = [0.0] * 3
        mir_twist = [0.0] * 2
        f_scan = [0.0] * 501
        b_scan = [0.0] * 511
        collision = False
        obstacles = [0.0] * 9
        rs_state = target + mir_pose + mir_twist + f_scan + b_scan + [collision] + obstacles

        return len(rs_state)

    def _get_env_state_len(self):
        """Get length of the environment state.

        Describes the composition of the environment state and returns
        its length.

        Returns:
            int: Length of the environment state

        """

        target_polar_coordinates = [0.0]*2
        mir_twist = [0.0]*2
        laser = [0.0]*self.laser_len
        env_state = target_polar_coordinates + mir_twist + laser

        return len(env_state)

    def _get_start_pose(self):
        """Get initial robot coordinates.

        For the real robot the initial coordinates are its current coordinates
        whereas for the simulated robot the initial coordinates are
        randomly generated.

        Returns:
            numpy.array: [x,y,yaw] robot initial coordinates.

        """

        if self.real_robot:
            # Take current robot position as start position
            start_pose = self.client.get_state_msg().state[3:6]
        else:
            # Create random starting position
            x = self.np_random.uniform(low= -5.0, high= 5.0)
            y = self.np_random.uniform(low= -5.0, high= 5.0)
            yaw = self.np_random.uniform(low= -np.pi, high= np.pi)
            start_pose = [x,y,yaw]

        return start_pose

    def _get_target(self, robot_coordinates):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        target_far_enough = False
        while not target_far_enough:
            x_t = self.np_random.uniform(low= -10.0, high= 10.0)
            y_t = self.np_random.uniform(low= -10.0, high= 10.0)
            yaw_t = 0.0
            target_dist = np.linalg.norm(np.array([x_t,y_t]) - np.array(robot_coordinates[0:2]), axis=-1)

            if target_dist >= self.min_target_dist:
                target_far_enough = True

        return [x_t,y_t,yaw_t]

    def _robot_server_state_to_env_state(self, rs_state):
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # Transform cartesian coordinates of target to polar coordinates
        polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=rs_state[0],\
                                                        y_target=rs_state[1],\
                                                        x_origin=rs_state[3],\
                                                        y_origin=rs_state[4])
        # Rotate origin of polar coordinates frame to be matching with robot frame and normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta - rs_state[5])

        # Get Laser scanners data
        raw_laser_scan = rs_state[8:1020]

        # Downsampling of laser values by picking every n-th value
        if self.laser_len > 0:
            laser = utils.downsample_list_to_len(raw_laser_scan,self.laser_len)
            # Compose environment state
            state = np.concatenate((np.array([polar_r, polar_theta]),rs_state[6:8],laser))
        else:
            # Compose environment state
            state = np.concatenate((np.array([polar_r, polar_theta]),rs_state[6:8]))

        return state.astype(np.float32)

    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Target coordinates range
        max_target_coords = np.array([np.inf,np.pi])
        min_target_coords = np.array([-np.inf,-np.pi])
        # Robot velocity range tolerance
        vel_tolerance = 0.1
        # Robot velocity range used to determine if there is an error in the sensor readings
        max_lin_vel = self.mir100.get_max_lin_vel() + vel_tolerance
        min_lin_vel = self.mir100.get_min_lin_vel() - vel_tolerance
        max_ang_vel = self.mir100.get_max_ang_vel() + vel_tolerance
        min_ang_vel = self.mir100.get_min_ang_vel() - vel_tolerance
        max_vel = np.array([max_lin_vel,max_ang_vel])
        min_vel = np.array([min_lin_vel,min_ang_vel])
        # Laser readings range
        max_laser = np.full(self.laser_len, 29.0)
        min_laser = np.full(self.laser_len, 0.0)
        # Definition of environment observation_space
        max_obs = np.concatenate((max_target_coords,max_vel,max_laser))
        min_obs = np.concatenate((min_target_coords,min_vel,min_laser))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _robot_outside_of_boundary_box(self, robot_coordinates):
        """Check if robot is outside of boundary box.

        Check if the robot is outside of the boundaries defined as a box with
        its center in the origin of the map and sizes width and height.

        Args:
            robot_coordinates (list): [x,y] Cartesian coordinates of the robot.

        Returns:
            bool: True if outside of boundaries.

        """

        # Dimensions of boundary box in m, the box center corresponds to the map origin
        width = 30
        height = 30

        if np.absolute(robot_coordinates[0]) > (width/2) or \
            np.absolute(robot_coordinates[1] > (height/2)):
            return True
        else:
            return False

    def _sim_robot_collision(self, rs_state):
        """Get status of simulated collision sensor.

        Used only for simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is in collision.

        """

        if rs_state[1020] == 1:
            return True
        else:
            return False

    def _min_laser_reading_below_threshold(self, rs_state):
        """Check if any of the laser readings is below a threshold.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if any of the laser readings is below the threshold.

        """

        threshold = 0.15
        if min(rs_state[8:1020]) < threshold:
            return True
        else:
            return False

class NoObstacleNavigationMir100(Mir100Env):
    
    laser_len = 0
    stuck_near_goal = 0
    
    def reset(self, start_pose = None, target_pose = None):
        
        """Environment reset.

        Args:
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.prev_base_reward = None
        
        # Clear Variable
        self.stuck_near_goal = 0
        self.starting_euclidean_dist_2d = None
        
        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose)==3
        else:
            start_pose = self._get_start_pose()

        rs_state[3:6] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose)==3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[0:3] = target_pose

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.tolist())
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}
        linear_power = 0
        angular_power = 0

        # Calculate distance to the target
        target_coords = np.array([rs_state[0], rs_state[1]])
        mir_coords = np.array([rs_state[3],rs_state[4]])
        euclidean_dist_2d = np.linalg.norm(target_coords - mir_coords, axis=-1)
        
        if self.starting_euclidean_dist_2d is None:
            self.starting_euclidean_dist_2d = euclidean_dist_2d
        
        # Reward base
        base_reward = -500*euclidean_dist_2d / self.starting_euclidean_dist_2d
        # base_reward = -50*euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Power used by the motors
        linear_power = abs(action[0] *0.30)
        # angular_power = abs(action[1] *0.03)
        angular_power = abs(action[1] *0.07)
        reward -= linear_power
        reward -= angular_power
        
        # Increment Reward if Path is Straight
        angular_velocity = action[1]
        reward -= abs(angular_velocity * 2)

        # End episode if robot is outside of boundary box
        if self._robot_outside_of_boundary_box(rs_state[3:5]):
            reward = -200.0
            done = True
            info['final_status'] = 'out of boundary'
            
        # If Remain Stuck Near the Goal
        if (euclidean_dist_2d < self.distance_threshold + 0.3):
            self.stuck_near_goal += 1.0
        
        # The episode terminates with success if the distance between the robot
        # and the target is less than the distance threshold.
        if (euclidean_dist_2d < self.distance_threshold):
            reward = 400.0 - self.stuck_near_goal * 0.7
            done = True
            info['final_status'] = 'success'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info

class NoObstacleNavigationMir100Sim(NoObstacleNavigationMir100, Simulation):
    cmd = "roslaunch mir100_robot_server sim_robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        NoObstacleNavigationMir100.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class NoObstacleNavigationMir100Rob(NoObstacleNavigationMir100):
    real_robot = True

class ObstacleAvoidanceMir100(Mir100Env):
    laser_len = 16

    def reset(self, start_pose = None, target_pose = None):
        """Environment reset.

        Args:
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        self.prev_base_reward = None

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose:
            assert len(start_pose)==3
        else:
            start_pose = self._get_start_pose()

        rs_state[3:6] = start_pose

        # Set target position
        if target_pose:
            assert len(target_pose)==3
        else:
            target_pose = self._get_target(start_pose)
        rs_state[0:3] = target_pose

        # Generate obstacles positions
        self._generate_obstacles_positions()
        rs_state[1021:1024] = self.sim_obstacles[0]
        rs_state[1024:1027] = self.sim_obstacles[1]
        rs_state[1027:1030] = self.sim_obstacles[2]

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state = rs_state.tolist())
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        return self.state

    def _reward(self, rs_state, action):
        reward = 0
        done = False
        info = {}
        linear_power = 0
        angular_power = 0

        # Calculate distance to the target
        target_coords = np.array([rs_state[0], rs_state[1]])
        mir_coords = np.array([rs_state[3],rs_state[4]])
        euclidean_dist_2d = np.linalg.norm(target_coords - mir_coords, axis=-1)

        
        # Reward base
        base_reward = -50*euclidean_dist_2d
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Power used by the motors
        linear_power = abs(action[0] *0.30)
        angular_power = abs(action[1] *0.03)
        reward-= linear_power
        reward-= angular_power

        # End episode if robot is collides with an object, if it is too close
        # to an object.
        if not self.real_robot:
            if self._sim_robot_collision(rs_state) or \
            self._min_laser_reading_below_threshold(rs_state) or \
            self._robot_close_to_sim_obstacle(rs_state):
                reward = -200.0
                done = True
                info['final_status'] = 'collision'

        if (euclidean_dist_2d < self.distance_threshold):
            reward = 100
            done = True
            info['final_status'] = 'success'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info

    def _get_start_pose(self):
        """Get initial robot coordinates.

        For the real robot the initial coordinates are its current coordinates
        whereas for the simulated robot the initial coordinates are
        randomly generated.

        Returns:
            numpy.array: [x,y,yaw] robot initial coordinates.

        """

        if self.real_robot:
            # Take current robot position as start position
            start_pose = self.client.get_state_msg().state[3:6]
        else:
            # Create random starting position
            x = self.np_random.uniform(low= -2.0, high= 2.0)
            if np.random.choice(a=[True,False]):
                y = self.np_random.uniform(low= -3.1, high= -2.1)
            else:
                y = self.np_random.uniform(low= 2.1, high= 3.1)
            yaw = self.np_random.uniform(low= -np.pi, high=np.pi)
            start_pose = [x,y,yaw]

        return start_pose

    def _get_target(self, robot_coordinates):
        """Generate coordinates of the target at a minimum distance from the robot.

        Args:
            robot_coordinates (list): [x,y,yaw] coordinates of the robot.

        Returns:
            numpy.array: [x,y,yaw] coordinates of the target.

        """

        target_far_enough = False
        while not target_far_enough:
            x_t = self.np_random.uniform(low= -2.0, high= 2.0)
            if robot_coordinates[1]>0:
                y_t = self.np_random.uniform(low= -3.1, high= -2.1)
            else:
                y_t = self.np_random.uniform(low= 2.1, high= 3.1)
            yaw_t = 0.0
            target_dist = np.linalg.norm(np.array([x_t,y_t]) - np.array(robot_coordinates[0:2]), axis=-1)
            if target_dist >= self.min_target_dist:
                target_far_enough = True

        return [x_t,y_t,yaw_t]

    def _robot_close_to_sim_obstacle(self, rs_state):
        """Check if the robot is too close to one of the obstacles in simulation.

        Check if one of the corner of the robot's base has a distance shorter
        than the safety radius from one of the simulated obstacles. Used only for
        simulated Robot Server.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            bool: True if the robot is too close to an obstacle.

        """

        # Minimum distance from obstacle center
        safety_radius = 0.40

        robot_close_to_obstacle = False
        robot_corners = self.mir100.get_corners_positions(rs_state[3], rs_state[4], rs_state[5])

        for corner in robot_corners:
            for obstacle_coord in self.sim_obstacles:
                if utils.point_inside_circle(corner[0],corner[1],obstacle_coord[0],obstacle_coord[1],safety_radius):
                    robot_close_to_obstacle = True

        return robot_close_to_obstacle

    def _generate_obstacles_positions(self,):
        """Generate random positions for 3 obstacles.

        Used only for simulated Robot Server.

        """

        x_0 = self.np_random.uniform(low= -2.4, high= -1.5)
        y_0 = self.np_random.uniform(low= -1.0, high= 1.0)
        yaw_0 = self.np_random.uniform(low= -np.pi, high=np.pi)
        x_1 = self.np_random.uniform(low= -0.5, high= 0.5)
        y_1 = self.np_random.uniform(low= -1.0, high= 1.0)
        yaw_1 = self.np_random.uniform(low= -np.pi, high=np.pi)
        x_2 = self.np_random.uniform(low= 1.5, high= 2.4)
        y_2 = self.np_random.uniform(low= -1.0, high= 1.0)
        yaw_2 = self.np_random.uniform(low= -np.pi, high=np.pi)

        self.sim_obstacles = [[x_0, y_0, yaw_0],[x_1, y_1, yaw_1],[x_2, y_2, yaw_2]]

class ObstacleAvoidanceMir100Sim(ObstacleAvoidanceMir100, Simulation):
    cmd = "roslaunch mir100_robot_server sim_robot_server.launch world_name:=lab_6x8.world"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidanceMir100.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceMir100Rob(ObstacleAvoidanceMir100):
    real_robot = True


###############################################################################################################################àà


class TrajectoryNavigationMir100(Mir100Env):

    laser_len = 0

    def __init__(self, rs_address=None, **kwargs):

        self.mir100 = mir100_utils.Mir100()

        self.elapsed_steps = 0
        self.min_target_dist = 1.0
        self.observation_space = self._get_observation_space()

        # 1 Parameters for Cubic Polynomial (K)
        self.action_space = spaces.Box(low=np.full((1), 0.1), high=np.full((1), 5), dtype=np.float32)

        self.seed()

        # Maximum linear (m/s) and angular (rad/s) velocities of MiR
        max_lin_vel = 0.5
        max_ang_vel = 0.7
        self.max_vel = np.array([max_lin_vel, max_ang_vel])

        # Connect to Robot Server
        if rs_address: self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")

    def reset(self, start_pose = None, target_pose = None):
        
        """ Environment reset

        Args:
            
            start_pose (list[3] or np.array[3]): [x,y,yaw] initial robot position.
            target_pose (list[3] or np.array[3]): [x,y,yaw] target robot position.

        Returns:
            
            np.array: Environment state.

        """

        self.elapsed_steps = 0

        self.prev_base_reward = None
        self.prev_time = 100

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())

        # Set Robot starting position
        if start_pose: assert len(start_pose) == 3
        else: start_pose = self._get_start_pose()

        # Set target position
        if target_pose: assert len(target_pose) == 3
        else: target_pose = self._get_target(start_pose)

        # Convert Pose To State
        rs_state[3:6] = self.start_pose = start_pose
        rs_state[0:3] = self.target_pose = target_pose
        print('\n------\n')
        print(f'Starting Pose | X = {start_pose[0]:.3f} | Y = {start_pose[1]:.3f} | θ = {degrees(start_pose[2]):.3f}')
        print(f'Target Pose   | X = {target_pose[0]:.3f} | Y = {target_pose[1]:.3f} | θ = {degrees(target_pose[2]):.3f}')

        # Set initial state of the Robot Server
        state_msg = robot_server_pb2.State(state=rs_state.tolist())
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state) == self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # TODO: Plan Trajectory (K Hard-Coded)
        self.trajectory, self.tf = self.plan_trajectory(start_pose, target_pose)

        # Starting Episode Timer
        self.episode_start_time = perf_counter()

        return self.state
    
    def _reward(self, rs_state, trajectory_time, action):

        # Variable Initialization
        reward, done, info = 0, False, {}

        # Out-Of-Time
        if trajectory_time == 0:
            reward = -100
            done = True

        # Base Reward - Positive Reward if New Execution Time is Lower
        base_reward = 50 * (self.prev_time - trajectory_time)
        print(f'Trajectory Time: {trajectory_time}')
        print('\n------\n')

        # Update Previous Time
        self.prev_time = trajectory_time

        # Compute Reward
        if self.prev_base_reward is not None:
            reward = base_reward - self.prev_base_reward
        self.prev_base_reward = base_reward

        # Default Done with IO-SFL
        done = True
        # done = False

        return reward, done, info

    def step(self, action):

        # Convert From TensorFlow to Numpy
        if type(action).__name__ == 'ndarray': action = action.astype(np.float32)
        elif type(action).__name__ == 'EagerTensor': action = action.numpy()
        else: print(f'Action {type(action).__name__} Type Not Recognized')

        # Increase Step
        self.elapsed_steps += 1

        # Check if the action is within the action space
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # IO-SFL - Execute Trajectory
        self.io_sfl(self.start_pose, self.target_pose, action)

        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state

        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()

        # Assign Reward
        reward, done, info = self._reward(rs_state=rs_state, trajectory_time=trajectory_time, action=action)

        return self.state, reward, done, info

    def plan_trajectory(self, starting_pose, target_pose):

        # Initialize Path Planning Parameters
        K, self.dt = 10, 1/50
        self.kp, self.kd = 2.0, 0.0

        # Get Trajectory Parameters
        xi, yi, θi = starting_pose
        xf, yf, θf = target_pose

        # Plan Trajectory
        import robo_gym.envs.mir100.trajectory_planning as tp
        trajectory = tp.plan_trajectory(parametrization='s', method='3rd polynomial', start=starting_pose, target=target_pose, parameters=[K])

        # Compute X(t), Y(t), Vx(t), Vy(t) and tf with Maximum s_dot = 1.0
        x_t, y_t, vx_t, vy_t, tf = tp.trajectory_in_t(trajectory, b, θi, self.dt, ds=1.0, vel_max=self.max_vel)

        # Return Splines and Execution Time
        return x_t, y_t, vx_t, vy_t, tf

    def io_sfl(self, action):

        # Initialize Time Step
        t, dt = 0.0, self.dt

        # Get IO-SFL Parameter
        b = action[0]
        print(f'b = {b}')

        action_timer = perf_counter()

        while 1:

            # Different Thead for functions
            # x = threading.Thread(target = sleeper, args = (5))

            # Get state from Robot Server
            x, y, θ, v, ω = self.client.get_state()[3:8]

            # Compute Actual Xb, Yb, Vbx, Vby
            xb, yb, vbx, vby = (x + b*cos(θ)), (y + b*sin(θ)), (v*cos(θ) - ω*b*sin(θ)), (v*sin(θ) + ω*b*cos(θ))

            # Check Position
            if tp.check_position_from_goal([xb, yb, θ], b_target_pose, distance_threshold=0.1): break
            # if tp.check_position_from_goal([xb,yb,θ], target_pose, distance_threshold=0.1): break

            # Get Position and Velocity from Trajectory in Time
            x_des, y_des, vx_des, vy_des = tp.pos_vel_from_spline([x_t, y_t, vx_t, vy_t], t, boundaries=[0, self.tf])

            # Compute Position and Velocity Errors
            ex, ey, edx, edy = (x_des - xb), (y_des - yb), (vx_des - vbx), (vy_des - vby)

            # Compute Vbx, Vby
            Vbx_des, Vby_des = (vx_des + self.kp*ex + self.kd*edx), (vy_des + self.kp*ey + self.kd*edy)
            # print(f'Vx_des: {Vbx_des}, Vy_des: {Vby_des}')

            # Compute v and ω
            v_des = Vbx_des * cos(θ) + Vby_des * sin(θ)
            ω_des = 1/b * (Vby_des * cos(θ) - Vbx_des * sin(θ))
            # print(f'v = {v_des:.5f} | ω = {ω_des:.5f}')

            # Check Velocity Limits | Cut-Off Tolerance = 0.1
            v, ω = tp.check_velocity_limits([v_des, ω_des], np.add(self.max_vel, 0.1))

            # Wait Time
            if (dt - (perf_counter() - action_timer)) < 0: print('Computation Time Exceeded Cycle Time')
            sleep(max(0, dt - (perf_counter() - action_timer)))

            # Send action to Robot Server
            if not self.client.send_action([v, ω]):
                raise RobotServerError("send_action")

            # Start New Action Timer
            action_timer = perf_counter()

            # Increase t
            t += dt

            # Out-Of-Time
            if (perf_counter() - self.start_time) > 2*self.tf:
                print('Time limit Exceeded')
                return 0

        # Return Trajectory Time
        print(f'Final Time: {self.tf} | Execution Time: {perf_counter() - self.start_time}')
        return (perf_counter() - self.start_time)

class TrajectoryNavigationMir100Sim(TrajectoryNavigationMir100, Simulation):
    cmd = "roslaunch mir100_robot_server sim_robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        TrajectoryNavigationMir100.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
class TrajectoryNavigationMir100Rob(TrajectoryNavigationMir100):
    real_robot = True
