import matplotlib
matplotlib.use('Agg')
from matplotlib import patches

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import partial


def plot_value(env, dataset, value_fn, fig, ax, N=20, random=False, title=None):
    observations = env.XY(n=N)

    if random:
        base_observations = np.copy(dataset['observations'][np.random.choice(dataset.size, len(observations))])
    else:
        base_observation = np.copy(dataset['observations'][0])
        base_observations = np.tile(base_observation, (observations.shape[0], 1))

    base_observations[:, :2] = observations

    values = value_fn(base_observations)

    x, y = observations[:, 0], observations[:, 1]
    x = x.reshape(N, N)
    y = y.reshape(N, N)
    values = values.reshape(N, N)
    mesh = ax.pcolormesh(x, y, values, cmap='viridis')
    env.draw(ax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mesh, cax=cax, orientation='vertical')

    if title:
        ax.set_title(title)


def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image


def value_image(env, dataset, value_fn):
    """
    Visualize the value function.
    Args:
        env: The environment.
        value_fn: a function with signature value_fn([# states, state_dim]) -> [#states, 1]
    Returns:
        A numpy array of the image.
    """
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_value(env, dataset, value_fn, fig, plt.gca())
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image


def gcvalue_image(env, dataset, value_fn):
    """
    Visualize the value function for a goal-conditioned policy.

    Args:
        env: The environment.
        value_fn: a function with signature value_fn(goal, observations) -> values
    """
    base_observation = dataset['observations'][0]

    point1, point2, point3, point4 = env.four_goals()
    point3 = (32.75, 24.75)

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    points = [point1, point2, point3, point4]
    for i, point in enumerate(points):
        point = np.array(point)
        ax = fig.add_subplot(2, 2, i + 1)

        goal_observation = base_observation.copy()
        goal_observation[:2] = point

        plot_value(env, dataset, partial(value_fn, goal_observation), fig, ax)
        
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(point[0], point[1])) 
        ax.scatter(point[0], point[1], s=50, c='red', marker='*')

    image = get_canvas_image(canvas)
    plt.close(fig)
    return image