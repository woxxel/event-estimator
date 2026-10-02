import numpy as np
from .utils import estimate_stats_from_one_sided_process


def get_events(neuron_activity, f=15, Ns_thr=1, prctile=20):
    """
    calculates the firing rate from a an array of spike probabilities ('S' from CaImAn)
    by thresholding data according to multiples sd_r of estimated variance

    Ns_thr:
      - minimum number of non-zero entries in S


    returns
        - firing rate (in spikes/sec, depending on provided framerate f)
        - calculated spiking threshold
        - array with number of spikes per frame

    """
    max_burst_rate = 200.0  # spikes per second
    max_spikes_per_frame = max_burst_rate // f

    cutoff_thr = 10 ** (-2)
    neuron_activity[neuron_activity < (neuron_activity.max() * cutoff_thr)] = 0
    Ns = (neuron_activity > 0).sum()
    if Ns < Ns_thr:
        return np.zeros_like(neuron_activity), 0, np.nan

    baseline, _ = estimate_stats_from_one_sided_process(
        neuron_activity,
        baseline_mode="percentile",
        prctile=prctile,
        only_nonzero_entries=True,
    )

    activity = np.clip(neuron_activity / baseline, 0, max_spikes_per_frame)
    activity[np.logical_and(activity > 0.1, activity < 1)] = 1.0
    activity = np.floor(activity)

    return (
        activity,
        np.mean(activity) * f,
        baseline,
    )
