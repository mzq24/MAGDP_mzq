import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_polyline(ax, polyline, color='k', line='-', alpha=0.2):
    ax.plot(polyline[:,0], polyline[:,1], linestyle=line, color=color, alpha=alpha)

def plot_polylines(ax, polylines, color='k', line='-', alpha=0.2):
    for polyline in polylines:
        plot_polyline(ax, polyline, line=line, color=color, alpha=alpha)

def plot_centerline(ax, centerline, color='k', line='--', marker=None, alpha=0.2):
    ax.plot(centerline[:,0], centerline[:,1], linestyle=line, marker=marker, color=color, alpha=alpha)

def scatter_neighbors(ax, neighbors):
    ax.scatter(neighbors[:,1], neighbors[:,2])

def scatter_goal(ax, goal, color='darkcyan', alpha=0.5, label='Goal'):
    ax.scatter(goal[:,0], goal[:,1], marker='*', color=color, alpha=alpha, label=label)

def scatter_pos(ax, positions, color='blue', size = 2, alpha=0.5, label='Pos'):
    ax.scatter(positions[:,0], positions[:,1], s=size, marker='o', color=color, alpha=alpha, label=label)

def plot_trajectory(ax, trajectory, color='g', alpha=1.0):
    ax.plot(trajectory[:,0], trajectory[:,1], color=color, alpha=alpha)

def plot_history(ax, hist, color='r', alpha=1.0):
    plot_trajectory(ax, hist, color=color, alpha=alpha)

def plot_future(ax, fut, color='g', alpha=1.0):
    plot_trajectory(ax, fut, color=color, alpha=alpha)

if __name__ == '__main__':
    fig, axs = plt.subplots(1,2)