# power-law-exponent
#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import pandas as pd
import scipy.stats as stats
import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PARAMETERS
# ============================================================
TR = 2.3
LOW_FREQ = 0.01
HIGH_FREQ = 0.25
MIN_LENGTH = 100
VAR_EPS = 1e-8
BINS_PER_DECADE = 10
REG_METHOD = "linregress"  # linregress, siegelslopes, theilslopes

# ============================================================
# PATHS
# ============================================================
voxelwise_dir = Path('/home/stubanadean/voxelwise_timeseries_nilearn')
output_dir = Path('/home/stubanadean/ple_results')
output_dir.mkdir(exist_ok=True)

layers = ['Interoception', 'Exteroception', 'Cognition']

# ============================================================
# FUNCTIONS
# ============================================================
def compute_avgbins(log_freq_band, log_power_band, bins_per_decade=BINS_PER_DECADE):
    decades = log_freq_band.max() - log_freq_band.min()
    n_bins = int(bins_per_decade * decades)
    log_bins = np.linspace(log_freq_band.min(), log_freq_band.max(), n_bins + 1)
    digitized = np.digitize(log_freq_band, log_bins)
    binned_freq, binned_log_power = [], []
    for bin_idx in range(1, len(log_bins)):
        indices = np.where(digitized == bin_idx)[0]
        if len(indices) > 0:
            geom_mean_freq = 10 ** np.mean(log_freq_band[indices])
            mean_log_power = np.mean(log_power_band[indices])
            binned_freq.append(geom_mean_freq)
            binned_log_power.append(mean_log_power)
    return np.log10(np.array(binned_freq)), np.array(binned_log_power)

def compute_ple_per_voxel(ts, TR, low=LOW_FREQ, high=HIGH_FREQ, bins_per_decade=BINS_PER_DECADE, reg_method=REG_METHOD):
    if len(ts) < MIN_LENGTH or np.std(ts) < VAR_EPS:
        return np.nan, np.nan
    ts = ts - np.mean(ts)
    nfft = len(ts)
    rfft = np.fft.rfft(ts, n=nfft)
    rfft = np.abs(rfft)**2
    rfft[rfft == 0] = np.finfo(float).eps

    freq = np.fft.rfftfreq(nfft, d=TR)[1:]
    rfft = rfft[1:]
    idx = np.where((freq >= low) & (freq <= high))[0]
    freq_band = freq[idx]
    power_band = rfft[idx]

    log_freq = np.log10(freq_band)
    log_power = np.log10(power_band)

    # Original slope
    if reg_method == "linregress":
        slope = stats.linregress(log_freq, log_power)[0]
    else:
        slope = np.nan
    ple_original = -slope

    # Binned slope
    avg_log_freq, avg_log_power = compute_avgbins(log_freq, log_power, bins_per_decade=bins_per_decade)
    if reg_method == "linregress":
        slope = stats.linregress(avg_log_freq, avg_log_power)[0]
    else:
        slope = np.nan
    ple_binned = -slope

    return ple_original, ple_binned

# ============================================================
# RUN PLE ON ALL LAYERS AND ROIs
# ============================================================
results = []

for layer in layers:
    layer_dir = voxelwise_dir / layer
    roi_files = list(layer_dir.glob("*_voxelwise_timeseries.npy"))

    for ts_file in roi_files:
        roi_name = ts_file.name.replace("_voxelwise_timeseries.npy", "")
        ts_data = np.load(ts_file)  # time x voxels
        T, V = ts_data.shape

        if T < MIN_LENGTH:
            print(f"Skipping {roi_name}: too short (T={T})")
            continue

        voxel_std = np.std(ts_data, axis=0)
        valid_voxels = voxel_std > VAR_EPS
        ts_data = ts_data[:, valid_voxels]

        if ts_data.shape[1] == 0:
            print(f"Skipping {roi_name}: no valid voxels")
            continue

        ple_orig_list = []
        ple_bin_list = []

        for v in range(ts_data.shape[1]):
            ple_orig, ple_bin = compute_ple_per_voxel(ts_data[:, v], TR)
            ple_orig_list.append(ple_orig)
            ple_bin_list.append(ple_bin)

        ple_orig_arr = np.array(ple_orig_list)
        ple_bin_arr = np.array(ple_bin_list)

        results.append({
            "Layer": layer,
            "ROI": roi_name,
            "Timepoints": T,
            "Num_voxels_used": ts_data.shape[1],
            "Mean_PLE_orig": np.nanmean(ple_orig_arr),
            "Median_PLE_orig": np.nanmedian(ple_orig_arr),
            "Std_PLE_orig": np.nanstd(ple_orig_arr),
            "Min_PLE_orig": np.nanmin(ple_orig_arr),
            "Max_PLE_orig": np.nanmax(ple_orig_arr),
            "Mean_PLE_binned": np.nanmean(ple_bin_arr),
            "Median_PLE_binned": np.nanmedian(ple_bin_arr),
            "Std_PLE_binned": np.nanstd(ple_bin_arr),
            "Min_PLE_binned": np.nanmin(ple_bin_arr),
            "Max_PLE_binned": np.nanmax(ple_bin_arr)
        })

        print(f"{layer} | {roi_name}")
        print(f"  Voxels used        : {ts_data.shape[1]}")
        print(f"  Mean PLE (orig)    : {np.nanmean(ple_orig_arr):.3f}")
        print(f"  Mean PLE (binned)  : {np.nanmean(ple_bin_arr):.3f}")
        print("-"*50)

# ============================================================
# SAVE RESULTS TO CSV
# ============================================================
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
outpath = output_dir / f"ple_summary_{timestamp}.csv"
pd.DataFrame(results).to_csv(outpath, index=False)
print(f"\nSaved results to: {outpath}")
