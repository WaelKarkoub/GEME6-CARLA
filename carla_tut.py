import carla
import random
import time
from scipy.interpolate import *
import numpy as np 
from carla_functions import *
from scipy.spatial import distance
import subprocess
import os
import signal


SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/carla/Unreal/CarlaUE4.sh"))
server_process = subprocess.Popen([SERVER_BINARY],preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
print(server_process.pid)
time.sleep(30)

for i in range(4):   
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        print("Successfully connected")
        break
    except Exception as e:
        print("Error connecting: {}, attempt {}".format(e, i))
        time.sleep(2)
    
        


settings = world.get_settings()
settings.synchronous_mode = True

world.apply_settings(settings)
print(world.get_settings())

gem = world.get_blueprint_library().find('vehicle.polaris.e6')
print(gem)


# vehicle.set_autopilot(True)
map = world.get_map()

while True:
    try:
        waypoints = makePath(world)
        vehicle = world.spawn_actor(gem, waypoints[0].transform)
        break
    except Exception as e:
        print("Collision while spawning")

cam = world.get_blueprint_library().find('sensor.camera.rgb')
camPos = carla.Transform(carla.Location(x=-8.5, z=2.8))
cam.set_attribute('image_size_x', '1920')
cam.set_attribute('image_size_y', '1080')
cam.set_attribute('fov', '110')
cam.set_attribute('sensor_tick', '0.0')
sensor = world.spawn_actor(cam, camPos, attach_to=vehicle)
positions = waypoints2tuple(waypoints)


tck,x,y = splineFit(positions)

data,radius = splineEval(x,y,tck)
zippedWaypoints = list(zip(data[0],data[1]))
newWaypoints,velocities,a = velocitySet(data,radius,speedLimit=7)

while True:
    error = referenceErrors(world,vehicle,zippedWaypoints,velocities,radius)
    if error == 0:
        break
    xte, velError, angle = error[0],error[1],error[2]
    controller(vehicle,xte,velError,angle)
    world.get_spectator().set_transform(sensor.get_transform())

    startTime = time.time()
    t = 0
    time.sleep(1/30)
        
    world.tick()

time.sleep(2)
os.killpg(server_process.pid, signal.SIGKILL)
