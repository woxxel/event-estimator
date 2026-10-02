import numpy as np
import scipy.stats as sstats


def obtain_significant_events_from_one_sided_process(
    data, sd_r=-1, baseline_mode="hsm", sd_mode="iqr", **kwargs
):
    """
    estimates the standard deviation of a one-sided process (e.g. spike train)
    by using the half-sample mode (hsm) or the median absolute deviation (mad)

    data:           process from which stats are inferred (spike train, firing map, etc)
    baseline_mode:  'hsm' for half-sample mode, 'percentile' for percentile-based estimation,
    SD_mode:        'iqr' for interquartile range, 'mad' for median absolute deviation
    kwargs:         additional parameters
            for "hsm":        no additional parameters required
            for "percentile": 'prctile' (default 50) to specify the percentile to use

    returns:
        - estimated standard deviation
        - baseline value
    """

    baseline, sd = estimate_stats_from_one_sided_process(
        data, baseline_mode=baseline_mode, sd_mode=sd_mode, **kwargs
    )

    # either use provided sd_r,
    # or calculate multiples of variance to ensure
    # p-value of 0.01 (including correction for multiple comparisons)
    p_value = kwargs.get("p_value", 0.1)
    sd_r = sstats.norm.ppf((1 - p_value) ** (1 / len(data))) if (sd_r == -1) else sd_r

    ## calculate threshold for significant events
    threshold = baseline + sd_r * sd
    significant_events = data / threshold
    significant_events[significant_events < 1] = 0

    return significant_events, threshold, sd_r


def estimate_stats_from_one_sided_process(
    data, baseline_mode="hsm", sd_mode="iqr", only_nonzero_entries=False, **kwargs
):
    """
    estimates the standard deviation of a one-sided process (e.g. spike train)
    by using the half-sample mode (hsm) or the median absolute deviation (mad)

    data:           process from which stats are inferred (spike train, firing map, etc)
    baseline_mode:  'hsm' for half-sample mode, 'percentile' for percentile-based estimation,
    SD_mode:        'iqr' for interquartile range, 'mad' for median absolute deviation
    kwargs:         additional parameters
            for "hsm":        no additional parameters required
            for "percentile": 'prctile' (default 50) to specify the percentile to use


    returns:
        - estimated standard deviation
    """

    # estimate noise level by using median or half-sampling mode method (assuming most entries are not actual spikes)
    if only_nonzero_entries:
        floor = data.max() * 5 * 10 ** (-2)
        data = data[data > floor]

    if baseline_mode == "hsm":
        baseline = calculate_hsm(data)
    elif baseline_mode == "percentile":
        prctile = kwargs.get("prctile", 50)
        baseline = np.percentile(data, prctile)

        baseline = max(baseline, 10 ** (-6))
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    # and use values below baseline to estimate variance from negative half-gaussian distribution
    if sd_mode is None:
        # if no sd_mode is specified, return baseline and 0 as sd
        return baseline, 0

    ### restrict data to one side
    data = data - baseline
    data = -data[data <= 0]
    datapoints = len(data)

    ### calculate standard deviation
    if sd_mode == "iqr":
        ## pretty much equals to "median absolute deviation" (mad)
        data.sort()
        # approximate standard deviation from inter quartile range
        Ns = round(datapoints * 0.5)  # 25 quartile is at half of data points
        # estimate (iqr_75 - iqr_25) from 2*(median - iqr_25)
        iqr = 2 * data[-Ns]
        sd = iqr / 1.349  # iqr relates to SD via a factor of 1.349 (theory)
    elif sd_mode == "var":
        # sd = np.sqrt(
        #     np.var(data, ddof=1) / (1 - 2 / np.pi)
        # )  # variance of one-sided process
        sd = np.sqrt((data**2).sum() / (datapoints * (1 - 2 / np.pi)))
        # print(sd, "sd from variance ")
    else:
        raise ValueError(f"Unknown sd_mode: {sd_mode}")

    return baseline, sd


def find_modes(data, axis=None, sort_it=True):

    if axis is not None:

        def fnc(x):
            return find_modes(x, sort_it=sort_it)

        dataMode = np.apply_along_axis(fnc, axis, data)
    else:
        data = data[np.isfinite(data)]
        if sort_it:
            data = np.sort(data)

        dataMode = calculate_hsm(data)

    return dataMode


def calculate_hsm(data, sort_it=True):
    ### adapted from caiman
    ### Robust estimator of the mode of a data set using the half-sample mode.
    ### versionadded: 1.0.3

    ### Create the function that we can use for the half-sample mode
    ### sorting done as first step, if not specified else

    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.nan
    if np.all(data == data[0]):
        return data[0]

    data = data[data > 0]  # remove 0 entries
    if sort_it:
        data = np.sort(data)

    # switch through different cases, depending on number of remaining datapoints:
    # for size <= 3, return result
    # for size > 3, find flattest part of the data over length size/2 and call function recursively
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:
        wMin = np.inf
        N = data.size // 2 + data.size % 2
        for i in range(N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i
        return calculate_hsm(data[j : j + N])
