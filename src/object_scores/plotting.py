import pynapple as nap
import numpy as np
import matplotlib.pyplot as plt
from .scores import compute_rate_map, compute_spikes_near_object_ratio_one_cluster, compute_spikes_away_from_object_ratio_one_cluster, gaussian_filter_nan

color_scheme = 'terrain'

def plot_rate_map(session, cluster_id, sigma=2.5, minmax=None, plot_object=False, object_position=None):

    tc = compute_rate_map(session, cluster_id=cluster_id, minmax=minmax)
    smooth_tc = gaussian_filter_nan(tc, [sigma,sigma])

    fig, ax = plt.subplots()
    ax.imshow(smooth_tc, extent=(0, 100, 0, 100))

    ax.set_xlabel("x-Position (cm)")
    ax.set_ylabel("y-Position (cm)")

    object_position = session.object_position

    if plot_object is True or object_position is not None:

        ax.scatter(object_position[0], object_position[1], s=50, c='red')
        from matplotlib.patches import Circle
        center = object_position
        radius = 18
        circle = Circle(center, radius, fill=False, color='red')
        ax.add_patch(circle)

    return fig


def _plot_ori(spikes, Px, Py, object_position, ori_score, On, An, mask_cm, fig, ax1, ax2, session_name, sigma=1):

    tc = compute_rate_map(spikes, Px, Py, minmax=(np.nanmin(Px), np.nanmax(Px), np.nanmin(Py), np.nanmax(Py)))
    smooth_tc = gaussian_filter_nan(tc, [sigma,sigma])

    tcax = ax1.imshow(smooth_tc, extent=(np.nanmin(Px), np.nanmax(Px), np.nanmin(Py), np.nanmax(Py)), cmap=color_scheme)

    ax1.set_xlabel("x-Position (cm)")
    ax1.set_ylabel(f"{session_name}\nORI: {ori_score:.2f}", rotation=90)

    ax1.set_title("Rate map")

    ax1.scatter(object_position[0], object_position[1], s=50, c='black')
    from matplotlib.patches import Circle
    center = object_position
    radius = 18
    circle = Circle(center, radius, fill=False, color='black')
    ax1.add_patch(circle)

    fig.colorbar(tcax, ax=ax1)

    ax2.set_xlabel("x-Position (cm)")
    ax2.set_xlim(np.nanmin(Px), np.nanmax(Px))
    ax2.set_ylim(np.nanmin(Py), np.nanmax(Py))
    ax2.set_aspect(1)
    ax2.set_title("Average firing near and away from object")
    ax2.scatter(object_position[0], object_position[1], s=50, c='black')
    from matplotlib.patches import Circle
    center = object_position
    radius = 18
    circle = Circle(center, radius, fill=False, color='black')
    ax2.add_patch(circle)
    ax2.text(object_position[0]-radius/1.5, object_position[1], f'O_n = {On:.2f}')
    ax2.text(20,20, f'A_n = {An:.2f}')


def _plot_information_content(spikes, Px, Py, fig, ax, session_name, score, sigma=1):

    tc = compute_rate_map(spikes, Px, Py, minmax=(np.nanmin(Px), np.nanmax(Px), np.nanmin(Py), np.nanmax(Py)))
    smooth_tc = gaussian_filter_nan(tc, [sigma,sigma])

    tcax = ax.imshow(smooth_tc, extent=(np.nanmin(Px), np.nanmax(Px), np.nanmin(Py), np.nanmax(Py)), cmap=color_scheme)

    ax.set_xlabel("x-Position (cm)")
    ax.set_ylabel("y-Position (cm)")

    ax.set_title(f"Rate map for session: {session_name}.\nInformation content score: {score:.2f}.")

    fig.colorbar(tcax, ax=ax)


def _plot_increase(scores_info, of_spikes, of_Px, of_Py, obj_spikes, obj_Px, obj_Py, object_position, cluster_id, mask_cm, sigma):

    from matplotlib.patches import Circle
    from matplotlib import colors
    from scipy.stats import norm

    fig, axes = plt.subplots(2,2, figsize=(10,10))

    of_tc = compute_rate_map(of_spikes[cluster_id],of_Px, of_Py)
    smooth_of_tc = gaussian_filter_nan(of_tc, [sigma,sigma])

    obj_tc = compute_rate_map(obj_spikes[cluster_id],obj_Px, obj_Py)
    smooth_obj_tc = gaussian_filter_nan(obj_tc, [sigma,sigma])

    both_tcs = np.array([smooth_of_tc, smooth_obj_tc])
    
    color_norm = colors.Normalize(vmin=np.nanmin(both_tcs), vmax=np.nanmax(both_tcs))

    extent  = (np.nanmin(of_Px), np.nanmax(of_Px), np.nanmin(of_Py), np.nanmax(of_Py))

    tcax = axes[0,0].imshow(smooth_of_tc, extent=extent, cmap=color_scheme, norm=color_norm)

    axes[0,0].set_xlabel("x-Position (cm)")
    axes[0,0].set_ylabel("y-Position (cm)")

    axes[0,0].scatter(object_position[0], object_position[1], s=50, c='black')

    center = object_position
    radius = mask_cm
    circle = Circle(center, radius, fill=False, color='black')
    axes[0,0].add_patch(circle)

    color_2 = axes[0,1].imshow(smooth_obj_tc, extent=extent, cmap=color_scheme, norm=color_norm)

    axes[0,1].set_xlabel("x-Position (cm)")
    axes[0,1].set_ylabel("y-Position (cm)")

    axes[0,1].scatter(object_position[0], object_position[1], s=50, c='black')

    fig.colorbar(tcax, ax=axes[0,0], norm=color_norm)
    fig.colorbar(color_2, ax=axes[0,1], norm=color_norm)

    center = object_position
    radius = mask_cm
    circle = Circle(center, radius, fill=False, color='black')
    axes[0,1].add_patch(circle)

    for i, grad in enumerate(scores_info[cluster_id]):
        if i == 0:
            axes[1,0].scatter(0, grad[1], 2, 'C0', zorder=1000)
            axes[1,0].scatter(1, grad[2], 2, 'C0', zorder=1000)
            axes[1,0].plot(grad[1:3], 'C0', label='real data', zorder=10000, alpha=1)
        else:
            axes[1,0].scatter(0, grad[1], 2, 'C1')
            axes[1,0].scatter(1, grad[2], 2, 'C1')
            axes[1,0].plot(grad[1:3], 'C1', zorder=1000, alpha=0.2)

    axes[1,0].legend()
    axes[1,0].set_xlabel('Session')
    axes[1,0].set_ylabel('Normalised firing')
    axes[1,0].set_xticks([0,1], ['of', 'obj'])

    all_scores = np.nan_to_num(scores_info[cluster_id][:,0])

    mu, std = norm.fit(all_scores)

    axes[1,1].hist(all_scores, bins=30, alpha=0.6, label='Score distribution')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axes[1,1].plot(x,p,'k', linewidth=2, label='Normal approx')
    threshold = norm.ppf(0.95, loc=mu, scale=std)
    axes[1,1].axvline(threshold, color='b', linestyle='dashed', label='top 5% of normal')
    axes[1,1].axvline(scores_info[cluster_id][0,0], color='r', linestyle='dashed', label='Real data')

    axes[1,1].legend()

    return fig