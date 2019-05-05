import time
import numpy as np 
from carla_functions import *
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open('stored_data/waypoints2.pkl',"rb") as hand:
    positions  = pickle.load(hand)

tck,x,y = splineFit(positions)

data,radius = splineEval(x,y,tck)
zippedWaypoints = list(zip(data[0],data[1]))
newWaypoints,velocities,a = velocitySet(data,radius,speedLimit=7)

with open("stored_data/car_history_H2_2.pkl","rb") as hand:
    history = pickle.load(hand)
# points = drawWaypoints(zippedWaypoints)

y = history[1][3:]
x = history[0][3:]

y = [-i for i in y]
data_y = [-i for i in data[1]]
fig, ax = plt.subplots()

ax.plot(data[0],data_y, 'b', linewidth = 1, label = "Generated Path")
scat, = ax.plot([],[],'ro', label = "Vehicle Position", markersize = 1)

def update(ifrm, xa, ya):
    scat.set_data(xa[:ifrm], ya[:ifrm])



# plt.title("Path Tracking - H2")
# plt.xlabel("X position")
# plt.ylabel("Y position")
# plt.legend()
# ani = animation.FuncAnimation(fig, update, frames=len(x), fargs =(x,y),interval=1)
# fn = 'Results/Path Tracking_H2_2'
# ani.save(fn+'.mp4',writer='ffmpeg',fps=12) 

# plt.plot(data[0],data_y, 'b', linewidth = 1, label = "Generated Path")
# plt.scatter(x[::1],y[::1],c="r", s = 3, label = "Vehicle Position")
# plt.title("Path Tracking - H2",fontsize=14)
# plt.xlabel("X position (m)",fontsize=12)
# plt.ylabel("Y position (m)",fontsize=12)
# plt.legend()
# plt.show()

with open("stored_data/car_history_H2_2_xte.pkl","rb") as hand:
    history_xte = np.array(pickle.load(hand))[1:]

with open("stored_data/car_history_H2_2_vel.pkl","rb") as hand:
    history_vel = np.absolute(np.array(pickle.load(hand))[1:]
)
# xte_mean = history_xte.mean()
# xte_std = np.std(history_xte,ddof=1)
# u = 2*xte_std
# max_xte = history_xte.max()
# print("mean: {}, std: {}, u: {}, max: {}".format(xte_mean,xte_std,u,max_xte))

# vel_mean = history_vel.mean()
# vel_std = np.std(history_vel,ddof=1)
# u = 2*vel_std
# max_vel = history_vel.max()
# print(history_vel)
# print("mean: {}, std: {}, u: {}, max: {}".format(vel_mean,vel_std,u,max_vel))

# num_bins = 15
# n, bins, patches = plt.hist(history_xte, num_bins,edgecolor='black', facecolor='blue', alpha=0.7,linewidth=1)
# plt.title("Cross-Track Error - LQR",fontsize=14)
# plt.xlabel("Error",fontsize=12)
# plt.ylabel("Frequency",fontsize=12)
# plt.show()
def update(i, xs, ys):

    # Read temperature (Celsius) from TMP102
    temp_c = round(tmp102.read_temp(), 2)

    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(temp_c)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('TMP102 Temperature over Time')
    plt.ylabel('Temperature (deg C)')

plt.title("Path Tracking - H2")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()
ani = animation.FuncAnimation(fig, update, frames=len(x), fargs =(x,y),interval=1)
fn = 'Results/Path Tracking_H2_2_xte'
ani.save(fn+'.mp4',writer='ffmpeg',fps=12) 