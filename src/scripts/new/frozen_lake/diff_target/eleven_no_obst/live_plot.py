import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
# ax2 = fig.add_subplot(1,1,1)


def animate(i):
    graph_data = open('/home/pradeep/catkin_ws/src/fredbots/src/scripts/new/frozen_lake/diff_target/eleven_no_obst/rewards.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    xs_avg = []
    ys_avg = []
    x_mean = 0
    y_mean = 0
    x_t_mean = []
    y_t_mean = []
    count = 0
    for i, line in enumerate(lines):
        if len(line) > 1:
            count += 1
            x, y = line.split(',')
            if i%10 == 0:
                xs_avg.append(x_mean/10)
                ys_avg.append(y_mean/10)
                x_mean = 0
                y_mean = 0
            xs.append(float(x))
            ys.append(float(y))
            x_mean += float(x)
            y_mean += float(y)
            x_t_mean.append(count)
            y_t_mean.append(np.mean(ys))

        # time.sleep(1)

    ax1.clear()
    ax1.plot(xs, ys, 'g', alpha=0.5)
    ax1.plot(xs_avg, ys_avg, 'b')
    ax1.plot(x_t_mean, y_t_mean, 'r')

# animate slowly
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()