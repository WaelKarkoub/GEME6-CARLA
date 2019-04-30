
from scipy.interpolate import *
import numpy as np 
import carla
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt

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

    try:
        velError = vel - velocities[index+4]
    except Exception:
        velError = vel - velocities[index]
        print("Velocity index out of range")
        

    yaw = vehicle.get_transform().rotation.yaw
    totalDistance = 0
    # print(yaw)
    nextWaypoint = waypoints[-1]
    z = 0
    try:
        for i,r in enumerate(radius[index:]):
#            print("index: {}, totalDistance: {}, Radius: {}".format(index+i,totalDistance,r))
            if r > 500:
                totalDistance = totalDistance + np.sqrt((waypoints[index+i][0]-waypoints[index][0])**2 + (waypoints[index+i][1]-waypoints[index+i][1])**2)

                if totalDistance > 3:
                    if i > 4:
                        nextWaypoint = waypoints[index+i]
                        z = i
                        break
                    else:
                        nextWaypoint = waypoints[index+4]
                        z = 4
                        break
            
            else:
                nextWaypoint = waypoints[index+4]
                z = 4
                break
    except Exception:
        return 0
    
    if index == len(waypoints)-1:
        return 0
    world.debug.draw_point(carla.Location(x=nextWaypoint[0], y=nextWaypoint[1], z=vehicle.get_location().z),size= 0.3, color=carla.Color(r=0, g=0, b=255), life_time=10, persistent_lines=True)

    angle = headingAngle(loc,nextWaypoint,yaw)
    angle = 180*angle/np.pi



    angleError = angle - yaw
    if angleError>180:
        angleError  = angleError - 360
    elif angleError < -180:
        angleError = angleError + 360
    return xte, velError, angleError, nextWaypoint, index+z

def drawWaypoints(waypoints):
    x = []
    y = []

    for point in waypoints:
        x.append(point[0])
        y.append(point[1])
    
    points = [x,y]

    plt.plot(x,y)
    plt.show()

    return points

