import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re


def read_radial_velocities(file_path):
    velocities = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            _, vr = parts
            velocities.append(float(vr))
    return np.array(velocities)


def robustDispersionWithThreshold(val, threshold=2):
    validValues = val[~pd.isnull(val)]
    nbVal = len(validValues)

    if nbVal >= max(threshold, 2):
        p1585 = np.nanpercentile(validValues, 15.85, method="linear")
        p8415 = np.nanpercentile(validValues, 84.15, method="linear")
        dispVal = (p8415 - p1585) / 2.0

        if nbVal >= 5:
            medVal = np.nanmedian(validValues)
            wingLow = (p1585 - medVal)
            wingUpp = (p8415 - medVal)

            norm = np.sqrt(2.0 * np.pi) / np.exp(-0.5)
            norm *= np.sqrt((0.1585 * 0.683) / nbVal)

            dispLow = wingLow * norm
            dispUpp = wingUpp * norm
        else:
            dispLow = np.nan
            dispUpp = np.nan
    else:
        dispVal = dispLow = dispUpp = np.nan

    return dispVal, dispLow, dispUpp


def load_simulation_data(sim_name, sim_files):
    selected_paths = [p for p in sim_files if f"/{sim_name}/" in p]
    data = []

    for path in selected_paths:
        try:
            snr = int(re.search(r'snr(\d+)', os.path.basename(path)).group(1))
            residuals = read_radial_velocities(path)

            disp, disp_low, disp_upp = robustDispersionWithThreshold(residuals)
            median, med_low, med_upp = medianWithThreshold(residuals)

            data.append({
                'simulation': sim_name,
                'snr': snr,
                'disp': disp,
                'disp_low': disp_low,
                'disp_upp': disp_upp,
                'median': median,
                'med_low': med_low,
                'med_upp': med_upp,
                'n': len(residuals),
                'path': path
            })
        except Exception as e:
            print(f"Failed on {path}: {e}")

    return pd.DataFrame(data).sort_values(by="snr").reset_index(drop=True)

def plot_histograms_from_dict(sim_name, sim_data_dict, max_histograms=5):
    sim_df = sim_data_dict[sim_name]

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    axes = axes.flatten()

    for i, (_, row) in enumerate(sim_df.iterrows()):
        if i >= max_histograms:
            break

        data = read_radial_velocities(row['path'])
        median = np.nanmedian(data)
        disp = row['disp']
        hist_range = (median - 5 * disp, median + 5 * disp)

        ax = axes[i]
        ax.hist(data, bins=30, range=hist_range, edgecolor='black', alpha=0.8)
        ax.axvline(median, color='red', linestyle='--', label=f"Median = {median:.3f}")
        ax.axvline(median + disp, color='green', linestyle=':', label=f"+Disp")
        ax.axvline(median - disp, color='green', linestyle=':', label=f"-Disp")
        ax.set_title(f"{sim_name}: SNR {row['snr']}")
        ax.set_xlabel("Vr [km/s]")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


########### Gaia ######################################

def load_gaia(path):

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .parquet")
    

def medianWithThreshold(val, threshold=1):
    validValues = val[~pd.isnull(val)]
    nbVal = len(validValues)

    if nbVal >= threshold:
        medVal = np.nanmedian(validValues)
        if nbVal >= 5:
            p1585 = np.nanpercentile(validValues, 15.85, method="linear")
            p8415 = np.nanpercentile(validValues, 84.15, method="linear")

            wingLow = p1585 - medVal
            wingUpp = p8415 - medVal

            medLow = wingLow * np.sqrt(np.pi / (2.0 * nbVal))
            medUpp = wingUpp * np.sqrt(np.pi / (2.0 * nbVal))
        else:
            medLow = medUpp = np.nan
    else:
        medVal = medLow = medUpp = np.nan

    return medVal, medLow, medUpp


def summarize_gaia_uncertainty(df, snr_bins):
    summary = []

    for (low, high) in snr_bins:
        group = df[(df["rv_expected_sig_to_noise"] > low) & 
                   (df["rv_expected_sig_to_noise"] <= high)]

        rv_errors = group["radial_velocity_error"].dropna()
        snr_mean = group["rv_expected_sig_to_noise"].mean()

        median, medLow, medUpp = medianWithThreshold(rv_errors)

        summary.append({
            "snr": snr_mean,
            "median_error": median,
            "med_low": medLow,
            "med_upp": medUpp,
            "count": len(rv_errors),
            "snr_range": f"{low}-{high}"
        })

    return pd.DataFrame(summary)