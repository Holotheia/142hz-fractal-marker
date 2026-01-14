#!/usr/bin/env python3
"""
Analyse multi-fréquence - Cogitate iEEG
Teste plusieurs bandes gamma pour comparaison
"""

import numpy as np
from scipy import signal
from scipy.stats import mannwhitneyu
from pathlib import Path

# Réutiliser les fonctions du script principal
from analyse_cogitate_142Hz import load_brainvision, load_events, categorize_events

PHI = (1 + np.sqrt(5)) / 2
D_STAR = 2.3107
F_PHI = 432 / (PHI ** D_STAR)

# Bandes à tester
BANDS = {
    'Low Gamma (30-50 Hz)': (30, 50),
    'Mid Gamma (50-80 Hz)': (50, 80),
    'High Gamma (80-120 Hz)': (80, 120),
    'f_Φ band (135-150 Hz)': (135, 150),
    'Ultra High (150-200 Hz)': (150, 200)
}


def compute_band_power(data, fs, band_low, band_high):
    """Calcule la puissance dans une bande de fréquence"""
    nyq = fs / 2
    if band_high >= nyq:
        band_high = nyq * 0.95
    if band_low >= band_high:
        return None

    b, a = signal.butter(4, [band_low/nyq, band_high/nyq], btype='band')

    if data.ndim > 1:
        data_mean = np.nanmean(data, axis=0)
    else:
        data_mean = data

    try:
        filtered = signal.filtfilt(b, a, data_mean)
        power = np.mean(filtered**2)
        return power
    except:
        return None


def analyze_bands(subject_path, subject_id):
    """Analyse toutes les bandes pour un sujet"""
    print(f"\n{'='*60}")
    print(f"SUJET: {subject_id}")
    print(f"{'='*60}")

    ieeg_dir = subject_path / 'ses-1' / 'ieeg'
    vhdr_files = list(ieeg_dir.glob('*_task-Dur_ieeg.vhdr'))
    events_files = list(ieeg_dir.glob('*_task-Dur_events.tsv'))

    if not vhdr_files or not events_files:
        return None

    data, fs, n_channels = load_brainvision(vhdr_files[0])
    events = load_events(events_files[0])
    conscious, unconscious = categorize_events(events)

    results = {}
    epoch_duration = 1.0

    for band_name, (low, high) in BANDS.items():
        powers_con = []
        powers_uncon = []

        for evt in conscious[:50]:
            start = int(evt['onset'] * fs)
            end = start + int(epoch_duration * fs)
            if end > data.shape[1]:
                continue
            p = compute_band_power(data[:, start:end], fs, low, high)
            if p is not None:
                powers_con.append(p)

        for evt in unconscious[:50]:
            start = int(evt['onset'] * fs)
            end = start + int(epoch_duration * fs)
            if end > data.shape[1]:
                continue
            p = compute_band_power(data[:, start:end], fs, low, high)
            if p is not None:
                powers_uncon.append(p)

        if powers_con and powers_uncon:
            mean_con = np.mean(powers_con)
            mean_uncon = np.mean(powers_uncon)
            ratio = mean_con / mean_uncon if mean_uncon > 0 else 1
            stat, p_val = mannwhitneyu(powers_con, powers_uncon, alternative='two-sided')

            results[band_name] = {
                'mean_conscious': mean_con,
                'mean_unconscious': mean_uncon,
                'ratio': ratio,
                'p_value': p_val,
                'significant': p_val < 0.05
            }

            marker = "***" if p_val < 0.05 else ""
            print(f"  {band_name}: ratio={ratio:.3f}, p={p_val:.4f} {marker}")

    return results


def main():
    base_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/datasets/cogitate_ieeg/bids_ecog/mnt/beegfs/workspace/2023-0385-Cogitatedatarelease/CURATE/COG_ECOG_EXP1_BIDS_SAMPLE")

    subjects = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('sub-')])

    print("=" * 70)
    print("ANALYSE MULTI-FRÉQUENCE - COMPARAISON BANDES GAMMA")
    print("=" * 70)
    print(f"Prédiction f_Φ = {F_PHI:.2f} Hz")
    print(f"Bandes testées: {list(BANDS.keys())}")

    all_results = {band: [] for band in BANDS}

    for subject_path in subjects:
        res = analyze_bands(subject_path, subject_path.name)
        if res:
            for band, data in res.items():
                all_results[band].append(data)

    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ - TOUTES LES BANDES")
    print("=" * 70)
    print(f"{'Bande':<30} {'Ratio moyen':<15} {'p combinée':<15} {'Sig?'}")
    print("-" * 70)

    for band_name in BANDS:
        if all_results[band_name]:
            ratios = [r['ratio'] for r in all_results[band_name]]
            p_values = [r['p_value'] for r in all_results[band_name]]

            mean_ratio = np.mean(ratios)

            # Fisher combined p
            chi2_stat = -2 * sum(np.log(p + 1e-10) for p in p_values)
            from scipy.stats import chi2
            combined_p = 1 - chi2.cdf(chi2_stat, 2 * len(p_values))

            sig = "OUI" if combined_p < 0.05 else "non"
            marker = "<<<" if band_name == 'f_Φ band (135-150 Hz)' else ""

            print(f"{band_name:<30} {mean_ratio:<15.3f} {combined_p:<15.4f} {sig} {marker}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
