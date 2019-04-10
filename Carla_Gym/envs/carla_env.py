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
sys.path.insert(0, '~/Desktop/golfcart')

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
atexit.register(cleanup)

class CarlaEnv(gym.Env):
    def __init__(self):

        # Steering, Throttle
        self.action_space = Box(np.array([-1.0,-1.0]), np.array([1.0,1.0]))

        #Heading angle, xte, velocity, radius (closest), radius (medium), radius (far), distance travelled
        self.observation_space = Box(np.array([-360.0,-5.0,0.0,0.0,0.0,0.0,-np.inf]), np.array([360,5.0,12.0,np.inf,np.inf,np.inf,np.inf]))
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
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None
        self.server_process = None
        self.world = None
        self.waypoint = None


    def init_server(self):
        print("Initializing new Carla server...")
        self.server_port = 2000
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
    def clear_server_state(self):
        print("Clearing Carla server state")

        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self.server_port = None
            self.server_process = None
    
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
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        # Create a CarlaSettings object. This object is a wrapper around
        # the CarlaSettings.ini file. Here we set the configuration we
        # want for the new episode.
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        # Setup start and end positions

        print(
            "Start pos {} ({}), end {} ({})".format(
                self.scenario["start_pos_id"], self.start_coord,
                self.scenario["end_pos_id"], self.end_coord))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print("Starting new episode...")

        Heading angle, xte, velocity, radius (closest), radius (medium), radius (far), distance travelled = self._read_observation()
        self.prev_measurement = py_measurements
        return self.encode_obs(self.preprocess_image(image), py_measurements)

        def encode_obs(self, image, py_measurements):
            prev_image = self.prev_image
            self.prev_image = image
            if prev_image is None:
                prev_image = image

            if self.config["use_image_only_observations"]:
                obs = image
            else:
                obs = (
                    image,
                    COMMAND_ORDINAL[py_measurements["next_command"]],
                    [py_measurements["forward_speed"],
                    py_measurements["distance_to_goal"]])
            self.last_obs = obs
            return obs