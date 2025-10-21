import numpy as np
import pynapple as nap
from scipy.ndimage import gaussian_filter
from scipy.stats import norm


def _compute_information_content(spikes, Px, Py, cluster_ids, bins=None, n_shuffles=10):

    if bins is None:
        bins = (40, 40)

    if cluster_ids is not None:
        spikes = spikes[cluster_ids]

    information_content = {}
    for cluster_id, spike_train in spikes.items():
        firing_rate = spike_train.rate
        information_content[cluster_id] = _compute_information_content_one_cluster(spike_train, Px, Py, firing_rate, bins=bins)

    return information_content


def _compute_information_content_one_cluster(spikes, Px, Py, firing_rate, bins):

    occupancy_map, _, _ = np.histogram2d(Px.values, Py.values, bins=bins)
    prob_of_occupancy = np.transpose(occupancy_map)/len(Px)

    firing_rate_per_bin = nap.compute_2d_tuning_curves(
        nap.TsGroup([spikes]),
        np.stack([Px, Py], axis=1),
        nb_bins=bins,
    )[0][0]

    normalised_firing_rate = firing_rate_per_bin/firing_rate

    information_content = np.sum(
        np.nan_to_num( # deals with lambda_i = 0
            prob_of_occupancy*normalised_firing_rate*np.log2(normalised_firing_rate)
        )
    )

    return information_content


def _compute_increase(of_spikes, of_Px, of_Py, obj_spikes, obj_Px, obj_Py, object_position, cluster_id, mask_cm, n_shuffles=100):

    of_spikes_cluster = of_spikes[cluster_id]
    obj_spikes_cluster = obj_spikes[cluster_id]

    ratios = []
    for i in range(n_shuffles):

        if i == 0:
            shuffled_of_spikes = of_spikes_cluster
            shuffled_obj_spikes = obj_spikes_cluster
        else:
            end_time = of_spikes.time_support.end[0]
            shuffled_of_spikes = nap.shift_timestamps(of_spikes_cluster, min_shift=20, max_shift=end_time-20)
            shuffled_obj_spikes = nap.shift_timestamps(obj_spikes_cluster, min_shift=20, max_shift=end_time-20)

        of_ratio = compute_spikes_near_object_ratio(shuffled_of_spikes, of_Px, of_Py, object_position, mask_cm=mask_cm)
        obj_ratio = compute_spikes_near_object_ratio(shuffled_obj_spikes, obj_Px, obj_Py, object_position, mask_cm=mask_cm)

        normalised_of_ratio = of_ratio/(of_ratio+obj_ratio)
        normalised_obj_ratio = obj_ratio/(of_ratio+obj_ratio)
        normalised_gradient = normalised_obj_ratio - normalised_of_ratio

        ratios.append([normalised_gradient, normalised_of_ratio, normalised_obj_ratio])

    mu, std = norm.fit(np.nan_to_num(np.array(ratios)[1:,0]))
    #threshold = norm.ppf(0.95, loc=mu, scale=std)
    return norm.cdf(ratios[0][0], loc=mu, scale=std), np.array(ratios)



def compute_spikes_near_object_ratio(spikes, Px, Py, object_position, mask_cm):

    near_object = (np.sqrt(np.pow(Px.values - object_position[0],2) + np.pow(Py.values - object_position[1],2)) < mask_cm)
    
    is_near_object = nap.Tsd(t=Px.times(), d=near_object)

    spike_near_object = spikes.value_from(is_near_object)

    spike_near_object = spikes.value_from(is_near_object)
    time_spent_near_object = np.sum(is_near_object.values)*np.mean(np.diff(is_near_object.times()))

    return np.sum(spike_near_object.values)/time_spent_near_object


# from Wolf
def gaussian_filter_nan(X, sigma, mode="reflect", keep=True):
    # Check if input is xarray DataArray or Dataset (duck typing)
    is_xarray = hasattr(X, "values") and hasattr(X, "dims") and hasattr(X, "coords")

    # Extract raw numpy array
    data = X.values if is_xarray else X

    V = data.copy()
    V[np.isnan(data)] = 0
    VV = gaussian_filter(V, sigma=sigma, mode=mode, truncate=6)

    W = np.ones_like(data)
    W[np.isnan(data)] = 0
    WW = gaussian_filter(W, sigma=sigma, mode=mode, truncate=6)

    Y = VV / WW
    if keep:
        Y[np.isnan(data)] = np.nan

    if is_xarray:
        # Rebuild xarray with same dims and coords
        import xarray as xr

        return xr.DataArray(Y, dims=X.dims, coords=X.coords, attrs=X.attrs)
    else:
       return Y

def compute_rate_map(spikes, Px, Py, minmax=None):

    minmax = (np.nanmin(Px), np.nanmax(Px), np.nanmin(Py), np.nanmax(Py))

    tc = nap.compute_2d_tuning_curves(
        nap.TsGroup([spikes]),
        np.stack([Px, Py], axis=1),
        nb_bins=(20,20),
        minmax=minmax,
        # range=bin_config["bounds"],
        # epochs=session["moving"].intersect(epochs),
    )[0][0]

    tc = np.transpose(tc[:,::-1])

    return tc



def _compute_ori(spikes, Px, Py, object_position, cluster_ids=None, mask_cm=18):

    if cluster_ids is not None:
        spikes = spikes[cluster_ids]

    ori = {}
    ori_info = {}
    for cluster_id in spikes:
        ori[cluster_id], ori_info[cluster_id] = compute_ori_one_cluster(spikes[cluster_id], Px, Py, object_position, mask_cm=mask_cm)
        
    return ori, ori_info


def compute_ori_one_cluster(spikes, Px, Py, object_position, mask_cm):

    On = compute_spikes_near_object_ratio_one_cluster(spikes, Px, Py, object_position, mask_cm)
    An = compute_spikes_away_from_object_ratio_one_cluster(spikes, Px, Py, object_position, mask_cm)

    return (On - An)/(On + An), (On, An)


def compute_spikes_near_object_ratio_one_cluster(spikes, Px, Py, object_position, mask_cm):

    near_object = (np.sqrt(np.pow(Px.values - object_position[0],2) + np.pow(Py.values - object_position[1],2)) < mask_cm)
    
    is_near_object = nap.Tsd(t=Px.times(), d=near_object)
    spike_near_object = spikes.value_from(is_near_object)
    time_spent_near_object = np.sum(is_near_object.values)*np.mean(np.diff(is_near_object.times()))

    return np.sum(spike_near_object.values)/time_spent_near_object


def compute_spikes_away_from_object_ratio_one_cluster(spikes, Px, Py, object_position, mask_cm):

    near_object = (np.sqrt(np.pow(Px.values - object_position[0],2) + np.pow(Py.values - object_position[1],2)) > mask_cm)
    
    is_near_object = nap.Tsd(t=Px.times(), d=near_object)
    spike_near_object = spikes.value_from(is_near_object)
    time_spent_near_object = np.sum(is_near_object.values)*np.mean(np.diff(is_near_object.times()))

    return np.sum(spike_near_object.values)/time_spent_near_object