import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('/home/pradeep/catkin_ws/src/fredbots/src/scripts/new/frozen_lake/diff_target/space_gamer_dnd/rewards.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    xs_avg = []
    ys_avg = []
    x_temp = []
    y_temp = []
    for i,line in enumerate(lines):
        if len(line) > 1:
            x, y = line.split(',')
            if i%50 == 0:
                xs_avg.append(np.mean(x_temp))
                ys_avg.append(np.mean(y_temp))
                x_temp, y_temp = [], []
            xs.append(float(x))
            ys.append(float(y))
            x_temp.append(float(x))
            y_temp.append(float(y))
        # time.sleep(1)

    ax1.clear()
    ax1.plot(xs, ys, 'g', alpha=0.5)
    ax1.plot(xs_avg, ys_avg, 'b')

# animate slowly
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()