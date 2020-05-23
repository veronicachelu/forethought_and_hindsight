import matplotlib.pyplot as plt
import numpy as np
import os

def plot_grid(env, logs=None, env_type="discrete", vmin=None, vmax=None):
    if env_type == "chain":
        if logs is not None:
            plot_chain(env, 0, 1)
            plt.savefig(os.path.join(logs, "environment.png"))
        else:
            plot_chain(env, vmin, vmax)
    else:
        if env_type == "continuous":
            # mdp = env._mdp_tilings
            sX = sY = None
            # sX = env._sX_tilings
            # sY = env._sY_tilings
            g = env._g_tilings
        elif env_type == "puddleworld":
            sX = sY = None
            mdp = env._mdp_tilings
            g = env._g_tilings
        else:
            mdp = env._mdp
            # sX = env._sX
            # sY = env._sY
            g = env._g
        plt.figure(figsize=(3, 3))
        plt.imshow(mdp > -1, interpolation="nearest", cmap='gray')
        ax = plt.gca()
        ax.grid(0)
        plt.xticks([])
        plt.yticks([])
        # plt.title("Environment")

        for sX, sY in env._starting_states:
        # if sX is not None and sY is not None:
            plt.text(
                sY, sX,
                r"$\mathbf{S}$", ha='center', va='center')
        for i, j in g:
            plt.text(
                j, i,
                r"$\mathbf{G}$", ha='center', va='center')
        h, w = mdp.shape[0], mdp.shape[1]
        if not env_type == "continuous":
            for y in range(0, h - 1):
                plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
                # plt.plot([1, w], [y + 0.5, y + 0.5], '-k', lw=1)
            for x in range(0, w - 1):
                plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-k', lw=2)
                # plt.plot([x + 1, x + 1], [0, h], '-k', lw=1)

        if logs is not None:
            plt.savefig(os.path.join(logs, "environment.png"))


def plot_chain(env, vmin, vmax):
    # plt.figure(figsize=(10, 10))
    mdp = np.zeros(shape=(1, env._nS))
    plt.imshow(mdp, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.grid(0)
    plt.xticks([])
    plt.yticks([])
    plt.text(
        env._start_state, 0,
        r"$\mathbf{S}$", ha='center', va='center')
    for i in env._end_states:
        plt.text(
            i, 0,
            r"$\mathbf{G}$", ha='center', va='center')
    h, w = 1, env._nS
    for y in range(0, h - 1):
        plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        # plt.plot([1, w], [y + 0.5, y + 0.5], '-k', lw=1)
    for x in range(0, w - 1):
        plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-k', lw=2)
        # plt.plot([x + 1, x + 1], [0, h], '-k', lw=1)


map_from_action_to_subplot = lambda a: (2, 6, 8, 4)[a]
map_from_action_to_name = lambda a: ("up", "right", "down", "left")[a]


def plot_v(env, values, logs=None, colormap='Blues',
           filename="v.png",
           true_v=None,
           env_type="discrete",
           policy=None,
           ):#vmin=-1, vmax=10):
    plt.clf()
    vmin = np.min(values)
    vmax = np.max(values)
    # plot_grid(env, env_type=env_type, vmin=vmin, vmax=vmax)
    # if policy is not None:
    #     plot_policy(env, policy, env_type=env_type)
    # if true_v is not None:
    #     vmin = np.min(true_v)
    #     vmax = np.max(true_v)
    # else:
    values = values[None, ...] if len(values.shape) == 1 else values
    plt.imshow(values, interpolation="nearest",
               cmap=colormap, vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])
    if logs is not None:
        plt.savefig(os.path.join(logs, filename))

def plot_gain(env, values, logs=None, colormap='Blues',
           filename="v.png",
           true_v=None,
           env_type="discrete",
           policy=None,
           ):#vmin=-1, vmax=10):
    plt.clf()
    vmin = np.min(values)
    vmax = np.max(values)
    # plot_grid(env, env_type=env_type, vmin=vmin, vmax=vmax)
    # if policy is not None:
    #     plot_policy(env, policy, env_type=env_type)
    # if true_v is not None:
    #     vmin = np.min(true_v)
    #     vmax = np.max(true_v)
    # else:
    values = values[None, ...] if len(values.shape) == 1 else values
    plt.imshow(values, interpolation="nearest",
               cmap=colormap, vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])
    if logs is not None:
        plt.savefig(os.path.join(logs, filename))

def plot_error(env, values, logs=None, colormap='Greys',
           filename="error.png",
           eta_pi=None,
           env_type="discrete",
           non_gridworld=False,
           policy=None,):
    plt.clf()
    vmin = np.min(values)
    vmax = np.max(values)
    # if not non_gridworld:
    #     plot_grid(env, env_type=env_type, vmin=vmin, vmax=vmax)
    # if policy is not None:
    #     plot_policy(env, policy, env_type=env_type)
    # if true_v is not None:
    #     vmin = np.min(true_v)
    #     vmax = np.max(true_v)
    # else:
    # values *= eta_pi
    values = values[None, ...] if len(values.shape) == 1 else values
    plt.imshow(values, interpolation="nearest",
               cmap=colormap)
    if not non_gridworld:
        plt.yticks([])
        plt.xticks([])
        plt.colorbar(ticks=[vmin, vmax])
    # normalized_eta_pi = (eta_pi - np.min(eta_pi))/(np.max(eta_pi) - np.min(eta_pi))
    # for i in range(values.shape[0]):
    #     for j in range(values.shape[1]):
    #         t = plt.annotate(r"$\cdot$", xy=(j, i), ha='center', va='center')
    #         t.set_alpha(normalized_eta_pi[i][j])

    if logs is not None:
        plt.savefig(os.path.join(logs, filename))


def plot_eta_pi(env, values, logs=None, colormap='Greys',
           filename="eta_pi.png",
           true_v=None,
           env_type="discrete",
           policy=None,
           ):
    plt.clf()
    vmin = np.min(values)
    vmax = np.max(values)
    # plot_grid(env, env_type=env_type, vmin=vmin, vmax=vmax)

    # if policy is not None:
    #     plot_policy(env, policy, env_type=env_type)

    values = values[None, ...] if len(values.shape) == 1 else values
    plt.imshow(values, interpolation="nearest",
               cmap=colormap, vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])
    if logs is not None:
        plt.savefig(os.path.join(logs, filename))

def plot_state_value(action_values):
    q = action_values
    fig = plt.figure(figsize=(4, 4))
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    v = 0.9 * np.max(q, axis=-1) + 0.1 * np.mean(q, axis=-1)
    plot_v(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def plot_action_values(action_values):
    q = action_values
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    vmin = np.min(action_values)
    vmax = np.max(action_values)
    dif = vmax - vmin
    for a in [0, 1, 2, 3]:
        plt.subplot(3, 3, map_from_action_to_subplot(a))

        plot_v(q[..., a], vmin=vmin - 0.05 * dif, vmax=vmax + 0.05 * dif)
        action_name = map_from_action_to_name(a)
        plt.title(r"$q(s, \mathrm{" + action_name + r"})$")

    plt.subplot(3, 3, 5)
    v = 0.9 * np.max(q, axis=-1) + 0.1 * np.mean(q, axis=-1)
    plot_v(v, colormap='summer', vmin=vmin, vmax=vmax)
    plt.title("$v(s)$")


def smooth(x, window=10):
    return x[:window * (len(x) // window)].reshape(len(x) // window, window).mean(axis=1)


def plot_stats(stats, window=10):
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    xline = range(0, len(stats.episode_lengths), window)
    plt.plot(xline, smooth(stats.episode_lengths, window=window))
    plt.ylabel('Episode Length')
    plt.xlabel('Episode Count')
    plt.subplot(122)
    plt.plot(xline, smooth(stats.episode_rewards, window=window))
    plt.ylabel('Episode Return')
    plt.xlabel('Episode Count')


def plot_policy(env, policy, logs=None, env_type="discrete"):
    plot_grid(env, env_type=env_type)
    # plt.title('Policy Visualization')
    if env_type == "chain":
        action_names = [r"$\rightarrow$"]
        policy = policy[None, ...]
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                if j != env._start_state and j not in env._end_states:
                    action_name = action_names[np.argmax(policy[i][j])]
                    plt.text(j, i, action_name, ha='center', va='center')
    else:
        action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
        for i in range(policy.shape[0]):
            for j in range(policy.shape[1]):
                action_name = action_names[np.argmax(policy[i][j])]
                plt.text(j, i, action_name, ha='center', va='center')
    if logs is not None:
        plt.savefig(os.path.join(logs, "pi.png"))


def plot_greedy_policy(grid, q):
    action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
    greedy_actions = np.argmax(q, axis=2)
    grid.plot_grid()
    plt.hold('on')
    plt.title('Greedy Policy')
    for i in range(9):
        for j in range(10):
            action_name = action_names[greedy_actions[i, j]]
            plt.text(j, i, action_name, ha='center', va='center')

def plot_pi(env, pi, values, logs=None, filename=None):
    vmin = np.min(values)
    vmax = np.max(values)
    # if not non_gridworld:
    plot_grid(env, env_type="discrete")
    # if policy is not None:
    #     plot_policy(env, policy, env_type=env_type)
    # if true_v is not None:
    #     vmin = np.min(true_v)
    #     vmax = np.max(true_v)
    # else:
    # values *= eta_pi
    # values = values[None, ...] if len(values.shape) == 1 else values
    # plt.imshow(values, interpolation="nearest",
    #            cmap='Greys')

    # plt.title('Policy Visualization')
    action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
    for i in range(pi.shape[0]):
        for j in range(pi.shape[1]):
            action_name = action_names[int(pi[i][j])]
            plt.text(j, i, action_name, ha='center', va='center')
    if logs is not None:
        plt.savefig(os.path.join(logs, filename))