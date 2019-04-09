
from scipy.interpolate import *
import numpy as np 
import carla
import random
from scipy.spatial import distance


def waypoints2tuple(waypoints):
    """Converts waypoints (carla.Waypoint) to a list of (x,y,z)

    Arguments:
        waypoints {[carla.Waypoint]} -- List of carla Waypoints

    Returns:
        [List] -- List of tuples (x,y,z)
    """

    tupleWaypoint = []

    for waypoint in waypoints:
        x = waypoint.transform.location.x
        y = waypoint.transform.location.y
        z = waypoint.transform.location.z
        tupleWaypoint.append((x, y, z))

    return tupleWaypoint

def makePath(world, distance=1, max_distance=1000):
    """Function to make a set of connected waypoints

    Arguments:
        map {client.get_world().get_map()} -- uses Carla maps API

    Keyword Arguments:
        distance {float/int} -- Distance between each waypoint in m (approx.)(default: {1.0})
        max_distance {float/int} -- Total path distance in m (default: {200})

    Returns:
        [carla.Waypoints] -- returns a list of waypoints
    """
    map = world.get_map()

    spawn_points = world.get_map().get_spawn_points()
    point = random.choice(spawn_points)
    # loc = carla.Location(x=44, y=-6, z=0)
    point = map.get_waypoint(point.location)
    total_distance = 0
    waypoints = [point]
    while total_distance < max_distance:
        nextPoint = random.choice(point.next(distance))
        waypoints.append(nextPoint)
        point = nextPoint

        total_distance += distance

    return waypoints

def splineFit(positions):
    """Function to fit a cubic spline

    Arguments:
        positions {[List of waypoints in tuple form]} -- waypoints

    Returns:
        [Spline coefficients] -- [description]
    """

    x, y = [], []

    for point in positions:
        x.append(point[0])
        y.append(point[1])

    tck, u = splprep([x, y], k=5,s=0)
    return tck,x,y

def splineEval(x, y, tck,res=0.001):
    """Evaluates the spline function, first and second derivatives
    
    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
        tck {[type]} -- [description]
    
    Returns:
        [tuple] -- (evaluated)
    """
    unew = np.arange(0, 1+res, res)
    x, y = splev(unew, tck, der=0)

    def define_circle(p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return (None, np.inf)

        # Center of circle
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        return radius
    
    radius = []
    for i, p in enumerate(x):
        if i == 0:
            radius.append(10000000)
            continue
        if i == len(x)-1:
            radius.append(10000000)
            break

        p1 = [x[i-1],y[i-1]]
        p2 = [p,y[i]]
        p3 = [x[i+1],y[i+1]]
        r = define_circle(p1,p2,p3)
        if isinstance(r, tuple):
            r = radius[i-1]
        radius.append(r)


    return ([x,y], radius)

def velocitySet(waypoints,radius,speedLimit=11,acc=2.5):
    newData = []
    velocity = []
    a =[]
    for i,r in enumerate(radius):
        acceleration = (speedLimit**2.0)/np.abs(r)
        a.append(acceleration)
        if acceleration <= acc:
            vel = speedLimit
            
        else:
            vel = np.sqrt(np.abs(r)*acc)

        velocity.append(vel)
        newData.append(([waypoints[0][i],waypoints[1][i]],vel))
    
    return newData,velocity,a

def referenceErrors(world,vehicle,waypoints,velocities,radius):
    def closest_node(node, nodes):

        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index],closest_index

    def headingAngle(currentLoc,nextLoc,yaw):
        # currentVec = [np.cos(yaw),np.sin(yaw)]
        # nextVec = [(nextLoc[0]-currentLoc[0]),(nextLoc[1]-currentLoc[1])]
        # def unit_vector(vector):
        #     """ Returns the unit vector of the vector.  """
        #     return vector / np.linalg.norm(vector)

        # def angle_between(v1, v2):

        #     v1_u = unit_vector(v1)
        #     v2_u = unit_vector(v2)
        #     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    
        return np.arctan2((nextLoc[1]-currentLoc[1]),(nextLoc[0]-currentLoc[0])) #angle_between(nextVec,currentVec)
    map = world.get_map()


    loc = vehicle.get_location()
    refPosition = map.get_waypoint(loc)
    refPosition = (refPosition.transform.location.x, refPosition.transform.location.y)
    loc = (loc.x,loc.y)
    vel = vehicle.get_velocity()
    point, index = closest_node(loc, waypoints)
    vel = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2 )
    xte =  tuple(np.abs(np.subtract(point, loc)))
    xte = np.sqrt(xte[0]**2 + xte[1]**2)
    print("xte: {}".format(xte))
    velError = vel - velocities[index+4]
    yaw = vehicle.get_transform().rotation.yaw
    totalDistance = 0
    # print(yaw)
    try:
        for i,r in enumerate(radius[index:]):
            print("index: {}, totalDistance: {}, Radius: {}".format(index+i,totalDistance,r))
            if r > 500:
                totalDistance = totalDistance + np.sqrt((waypoints[index+i][0]-waypoints[index][0])**2 + (waypoints[index+i][1]-waypoints[index+i][1])**2)

                if totalDistance > 5:
                    if i > 9:
                        nextWaypoint = waypoints[index+i]
                        break
                    else:
                        nextWaypoint = waypoints[index+9]
                        break
            
            else:
                nextWaypoint = waypoints[index+9]
                break
    except Exception:
        return 0
    world.debug.draw_point(carla.Location(x=nextWaypoint[0], y=nextWaypoint[1], z=vehicle.get_location().z),size= 0.3, color=carla.Color(r=0, g=0, b=255), life_time=10, persistent_lines=True)

    angle = headingAngle(loc,nextWaypoint,yaw)
    angle = 180*angle/np.pi



    angleError = angle - yaw
    if angleError>180:
        angleError  = angleError - 360
    elif angleError < -180:
        angleError = angleError + 360

    print("Angle: {}, Yaw: {}, Error: {}".format(angle,yaw,angleError))
    return xte, velError, angleError

def controller(vehicle,xte,velError,angle):
     
    # if angle < 0:
    #     angle = angle + 180
    # elif angle == 0:
    #     angle = 0
    # else:
    #     angle = angle - 180


    if angle > 50.0:
        angle = 50.0
    
    if angle < -50:
        angle = -50

    steering = angle*1/50.0

    if velError > 2:
        velError = 2
    if velError < -2:
        velError = -2

    prop = -(1.0/2.0)*velError

    if prop < 0:
        brake = np.abs(prop)
        gas = 0
    elif prop > 0:
        gas = np.abs(prop)
        brake = 0
    else:
        gas, brake = 0, 0
    print("Gas: {}, Break: {}, Steering: {}".format(gas,brake,steering))
    control = carla.VehicleControl(
        throttle = gas,
        steer = steering,
        brake = brake,
        hand_brake = False,
        reverse = False,
        manual_gear_shift = False,
        gear = 1)
    
    vehicle.apply_control(control)
