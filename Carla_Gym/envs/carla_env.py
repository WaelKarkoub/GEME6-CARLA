from datetime import datetime
import atexit
import os
import random
import signal
import subprocess
import time
import traceback
import json
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
import carla
import sys

from carla_functions import *

SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/carla/Unreal/CarlaUE4.sh"))
assert os.path.exists(SERVER_BINARY), "CARLA_SERVER environment variable is not set properly. Please check and retry"

RETRIES_ON_ERROR = 4
DISCRETE_ACTIONS = {
    0: [0.0, 0.0],    # Coast
    1: [0.0, -0.5],   # Turn Left
    2: [0.0, 0.5],    # Turn Right
    3: [1.0, 0.0],    # Forward
    4: [-0.5, 0.0],   # Brake
    5: [1.0, -0.5],   # Bear Left & accelerate
    6: [1.0, 0.5],    # Bear Right & accelerate
    7: [-0.5, -0.5],  # Bear Left & decelerate
    8: [-0.5, 0.5],   # Bear Right & decelerate
}

live_carla_processes = set()
def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        os.killpg(pgid, signal.SIGKILL)
    try:
        os.system("killall CarlaUE4")
    except Exception:
        pass

atexit.register(cleanup)

class CarlaEnv(gym.Env):
    def __init__(self):

        # Throttle, Steering
        self.action_space = Box(np.array([-1.0,-1.0]), np.array([1.0,1.0]))

        #Heading angle, xte, velocity, radius (closest), radius (medium), radius (far), distance travelled
        self.observation_space = Box(np.array([-360.0,-5.0,0.0,0.0]), np.array([360,5.0,12.0,np.inf]))
        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._seed = 1
        self.server_port = None
        self.client = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.episode_id = None
        self.measurements_file = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None
        self.server_process = None
        self.world = None
        self.waypoints = None
        self.map = None
        self.vehicle = None
        self.zippedWaypoints = None
        self.velocities = None
        self.radius = None
        self.cam = None
        self.camPos = None
        self.sensor = None

    def init_server(self):
        print("Initializing new Carla server...")
        self.server_port = 2000
        try:
            os.system("killall CarlaUE4")
        except Exception:
            pass
        self.server_process = subprocess.Popen([SERVER_BINARY],preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        time.sleep(30)
        for i in range(RETRIES_ON_ERROR):
            try:
                self.client = carla.Client("localhost", self.server_port)
                self.client.set_timeout(5.0)
                self.world = self.client.get_world()
                print("Successfully connected")
                break
            except Exception as e:
                print("Error connecting: {}, attempt {}".format(e, i))
                time.sleep(2)
        
        live_carla_processes.add(os.getpgid(self.server_process.pid))

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        gem = self.world.get_blueprint_library().find('vehicle.polaris.e6')
        self.map = self.world.get_map()
        while True:
            try:
                self.waypoints = makePath(world)
                self.vehicle = self.world.spawn_actor(gem, self.waypoints[0].transform)
                break
            except Exception as e:
                print("Collision while spawning")
        
        positions = waypoints2tuple(self.waypoints)
        self.cam = world.get_blueprint_library().find('sensor.camera.rgb')
        self.camPos = carla.Transform(carla.Location(x=-8.5, z=2.8))
        self.cam.set_attribute('image_size_x', '1920')
        self.cam.set_attribute('image_size_y', '1080')
        self.cam.set_attribute('fov', '110')
        self.cam.set_attribute('sensor_tick', '0.0')
        self.sensor = world.spawn_actor(self.cam, self.camPos, attach_to=self.vehicle)

        tck,x,y = splineFit(positions)

        data,self.radius = splineEval(x,y,tck)
        self.zippedWaypoints = list(zip(data[0],data[1]))
        newWaypoints,self.velocities,a = velocitySet(data,radius,speedLimit=7)

    def clear_server_state(self):
        print("Clearing Carla server state")

        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self.server_port = None
            self.server_process = None
        try:
            os.system("killall CarlaUE4")
        except Exception:
            pass

    def __del__(self):
        self.clear_server_state()
    
    def reset(self):
        error = None
        for _ in range(RETRIES_ON_ERROR):
            try:
                if not self.server_process:
                    self.init_server()
                return self.reset_env()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                self.clear_server_state()
                error = e
        raise error

    def reset_env(self):
        self.vehicle.destroy()
        self.sensor.destroy()
        self.vehicle = None
        self.sensor = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.radius = None
        self.zippedWaypoints = None
        self.velocities = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        gem = self.world.get_blueprint_library().find('vehicle.polaris.e6')

        while True:
            try:
                self.waypoints = makePath(self.world)
                self.vehicle = self.world.spawn_actor(gem, self.waypoints[0].transform)
                break
            except Exception as e:
                print("Collision while spawning")

        positions = waypoints2tuple(self.waypoints)
        self.sensor = self.world.spawn_actor(self.cam, self.camPos, attach_to=self.vehicle)  

        tck,x,y = splineFit(positions)

        data,self.radius = splineEval(x,y,tck)
        self.zippedWaypoints = list(zip(data[0],data[1]))
        newWaypoints,self.velocities,a = velocitySet(data,radius,speedLimit=7)
        self.world.get_spectator().set_transform(self.sensor.get_transform())


        print("Starting new episode...")

        # Heading angle, xte, velocity, radius (closest), radius (medium), radius (far), distance travelled = self._read_observation()
        # self.prev_measurement = py_measurements
        return self._read_observations()[0]

    def _read_observations(self):
        error = referenceErrors(self.world,self.vehicle,self.zippedWaypoints,self.velocities,self.radius)
        if (error == 0):
            reachedGoal = 1
        else:
            reachedGoal = 0
        if isinstance(error,list):
            xte, velError, angleError,nextWaypoint, index = error[0],error[1],error[2], error[3], error[4]

        py_measurements = {
            "episode_id": self.episode_id,
            "step": self.num_steps,
            "reached_goal": reachedGoal,
            "xte": xte,
            "velocity_error": velError,
            "angle_error": angleError,
            "next_x": nextWaypoint[0],
            "next_y": nextWaypoint[1],
            "radius": self.radius[index]/500,
        }


        obs = (xte,velError,angleError,self.radius[index]/500)
        self.last_obs = obs
        return [obs,py_measurements]

    def step(self, action):
        try:
            obs = self.step_env(action)
            return obs
        except Exception:
            print(
                "Error during step, terminating episode early",
                traceback.format_exc())
            self.clear_server_state()
            return (self.last_obs, 0.0, True, {})
    
    def step_env(self, action):

        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False

        print("steer", steer, "throttle", throttle, "brake", brake)
        control = carla.VehicleControl(
                    throttle = throttle,
                    steer = steer,
                    brake = brake,
                    hand_brake = False,
                    reverse = False,
                    manual_gear_shift = False,
                    gear = 1)
        self.vehicle.apply_control(control)
        self.world.get_spectator().set_transform(self.sensor.get_transform())
        time.sleep(1/30)
        self.world.tick()
        # Process observations
        obs, py_measurements = self._read_observations()

        py_measurements["control"] = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "reverse": reverse,
            "hand_brake": hand_brake,
        }
        reward = self.calculate_reward(py_measurements)
        self.total_reward += reward
        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward
        done = (self.num_steps > 10**12 or
                py_measurements["reached_goal"] or
                (py_measurements["xte"]>1))
        py_measurements["done"] = done
        self.prev_measurement = py_measurements

        self.num_steps += 1
        return (obs, reward, done,py_measurements)

    def calculate_reward(self, current_measurement):
        """
        Calculate the reward based on the effect of the action taken using the previous and the current measurements
        :param current_measurement: The measurement obtained from the Carla engine after executing the current action
        :return: The scalar reward
        """
        reward = 0.0

        dist = np.abs(current_measurement["xte"]) - np.abs(self.prev_measurement["xte"])
        vel = np.abs(current_measurement["velocity_error"]) - np.abs(self.prev_measurement["velocity_error"])

        reward += 10*dist
        reward += 10*vel
        reward -= np.abs(current_measurement["xte"])
        reward -= np.abs(current_measurement["velocity_error"])


        if  np.abs(current_measurement["xte"])> 0.5:
            reward -= 1000

        if np.abs(current_measurement["velocity_error"])> 3:
            reward -= 1000

        return reward