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
# code
RETRIES_ON_ERROR = 4


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

        #np.array([xte,velError,vel,angleError)
        self.observation_space = Box(np.array([-np.inf,-11.0,0.0,-np.inf]), np.array([np.inf,11.0,np.inf,np.inf]))
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

        self.prevLocation = None
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
        self.total_distance = 0
        self.init_server()
        self.reset()

    def init_server(self):
        print("Initializing new Carla server...")
        self.server_port = 2000
        try:
            os.system("killall CarlaUE4")
        except Exception:
            pass
        self.server_process = subprocess.Popen([SERVER_BINARY],preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        time.sleep(10)
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
                self.waypoints = makePath(self.world)
                self.vehicle = self.world.spawn_actor(gem, self.waypoints[0].transform)
                break
            except Exception as e:
                print("Collision while spawning")
        
        positions = waypoints2tuple(self.waypoints)
        self.cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camPos = carla.Transform(carla.Location(x=-8.5, z=2.8))
        self.cam.set_attribute('image_size_x', '1920')
        self.cam.set_attribute('image_size_y', '1080')
        self.cam.set_attribute('fov', '110')
        self.cam.set_attribute('sensor_tick', '0.0')
        self.sensor = self.world.spawn_actor(self.cam, self.camPos, attach_to=self.vehicle)
        tck,x,y = splineFit(positions)

        data,self.radius = splineEval(x,y,tck)
        self.zippedWaypoints = list(zip(data[0],data[1]))
        newWaypoints,self.velocities,a = velocitySet(data,self.radius,speedLimit=7)

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
        return self.reset_env()

    def reset_env(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.sensor.destroy()
        self.total_distance = 0
        self.prevLocation = None
        self.vehicle = None
        self.sensor = None
        self.num_steps = 0
        self.total_reward = 0
        self.prevLocation = None
        self.currenLocation = None
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
        self.loc = self.vehicle.get_location()
        self.loc = (self.loc.x,self.loc.y)
        newWaypoints,self.velocities,a = velocitySet(data,self.radius,speedLimit=7)
        print("Starting new episode...")

        # Heading angle, xte, velocity, radius (closest), radius (medium), radius (far), distance travelled = self._read_observation()
        self.prev_measurement = self._read_observations()[1]
        
        self.render()
        time.sleep(0.5)
        return self._read_observations()[0]

    def _read_observations(self):
        error = referenceErrors(self.world,self.vehicle,self.zippedWaypoints,self.velocities,self.radius)
        self.vel = self.vehicle.get_velocity()
        self.vel = np.sqrt(self.vel.x**2 + self.vel.y**2 + self.vel.z**2 )

        self.acc = self.vehicle.get_acceleration()
        self.acc = np.sqrt(self.acc.x**2 + self.acc.y**2 + self.acc.z**2 )

        if (error == 0):
            reachedGoal = 1
        else:
            reachedGoal = 0
        
        

        if isinstance(error,tuple):
            xte, velError, angleError,self.nextWaypoint, index = error[0],error[1],error[2], error[3], error[4]
        if reachedGoal == 0:
            py_measurements = {
                "episode_id": self.episode_id,
                "step": self.num_steps,
                "reached_goal": reachedGoal,
                "xte": xte,
                "velocity": self.vel,
                "velocity_error": velError,
                "angle_error": angleError,
                "next_x": self.nextWaypoint[0],
                "next_y": self.nextWaypoint[1],
                "radius": self.radius[index]/500,
                "acceleration":self.acc,
            }
            obs = np.array([xte,velError,self.vel,angleError])
        else:
            obs = self.last_obs
            py_measurements = {
                "episode_id": self.episode_id,
                "step": self.num_steps,
                "reached_goal": reachedGoal,
                "xte": obs[0],
                "velocity": self.vel,
                "velocity_error": obs[1],
                "angle_error": obs[2],
                "next_x": self.map.get_waypoint(self.vehicle.get_location()).transform.location.x,
                "next_y":self.map.get_waypoint(self.vehicle.get_location()).transform.location.x,
                "radius": obs[3],
                "acceleration":self.acc,
            }

        self.last_obs = obs
        return [obs,py_measurements]

    def step(self, action):
        try:
            obs = self.step_env(action)
            return obs[0],obs[1], obs[2], {}
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
        self.loc = self.vehicle.get_location()
        self.loc = (self.loc.x,self.loc.y)

        print("steer: {}, throttle: {}, brake: {}".format(steer, throttle, brake))
        control = carla.VehicleControl(
                    throttle = throttle,
                    steer = steer,
                    brake = brake,
                    hand_brake = False,
                    reverse = False,
                    manual_gear_shift = False,
                    gear = 1)
        self.vehicle.apply_control(control)
        self.render()
        

        if self.num_steps > 1:
            self.total_distance += np.sqrt((self.loc[0]-self.prevLocation[0])**2 + (self.loc[1]-self.prevLocation[1])**2)

        self.prevLocation = self.loc
        
        print("total distance: {}".format(self.total_distance))
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
        done = (self.num_steps > 10**5 or
                py_measurements["reached_goal"] or
                (py_measurements["xte"]>2))
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


        # if np.abs(current_measurement["xte"]) < np.abs(self.prev_measurement["xte"]):
        #     reward += 10
        
        # elif np.abs(current_measurement["xte"]) > np.abs(self.prev_measurement["xte"]):
        #     reward -= 10
        
        # else:
        #     pass

        # if np.abs(current_measurement["velocity_error"]) < np.abs(self.prev_measurement["velocity_error"]):
        #     reward += 20
        
        # elif np.abs(current_measurement["velocity_error"]) > np.abs(self.prev_measurement["velocity_error"]):
        #     reward -= 20
        
        # else:
        #     pass
        
        reward -= 10*np.abs(current_measurement["xte"])
        reward -= np.abs(current_measurement["velocity_error"])
        reward -= current_measurement["angle_error"]/5.0

        if  np.abs(current_measurement["xte"])> 0.3:
            reward -= 20
        if  np.abs(current_measurement["xte"])<= 0.3:
            reward += 20

        if np.abs(current_measurement["velocity_error"])> 2:
            reward -= 10
        if np.abs(current_measurement["velocity_error"])<= 2:
            reward += 10
        
        if (self.total_distance >= 5):
            print("\033[92m Travelled 5 meters \x1b[0m")
            reward += 30 
            self.total_distance = 0
        if self.vel <= 0.0000001:
            reward -= 1000
        return reward
    
    def render(self,mode="human"):
        self.world.get_spectator().set_transform(self.sensor.get_transform())
        time.sleep(1/31)
        self.world.tick()
        