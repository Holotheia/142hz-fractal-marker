#!/usr/bin/env python3
"""
Analyse Cogitate iEEG - Test 1: Multi-band analysis (f1 = 102 Hz)
==================================================================

This script tests the prediction that f1 = 102 Hz (D=3, ordinary
conscious perception) shows stronger high-gamma power during conscious
vs unconscious processing in intracranial EEG data.

Method: Multi-band comparison across Low Gamma (30-50 Hz),
Mid Gamma (50-80 Hz), High Gamma (80-120 Hz), and f2 band (135-150 Hz).

Expected result: Only the 80-120 Hz band (containing f1 = 102 Hz)
should differentiate conscious from unconscious perception.

Author: Aurelie Assouline (Holotheia.ai)
Date: January 2026
"""

import numpy as np
from scipy import signal
from scipy.stats import ttest_ind, mannwhitneyu
import json
import os
from pathlib import Path

# Constantes HOLOTHEIA
PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
D_STAR = 2.3107  # Dimension fractale optimale
F_PHI = 432 / (PHI ** D_STAR)  # ≈ 142.09 Hz

F1 = 432 / (PHI ** 3)  # ≈ 102 Hz - perception

# Bandes à tester (comme dans l'article, Table 4.2)
BANDS = {
    'Low Gamma (30-50 Hz)': (30, 50),
    'Mid Gamma (50-80 Hz)': (50, 80),
    'High Gamma (80-120 Hz)': (80, 120),
    'f2 band (135-150 Hz)': (135, 150),
}

print("=" * 70)
print("ANALYSE COGITATE iEEG - TEST 1: Multi-band (f1 = 102 Hz)")
print("=" * 70)
print(f"\nConstantes théoriques:")
print(f"  φ (golden ratio) = {PHI:.10f}")
print(f"  D* (dimension optimale) = {D_STAR}")
print(f"  f1 = 432 / φ³ = {F1:.2f} Hz (perception)")
print(f"  f2 = 432 / φ^D* = {F_PHI:.2f} Hz (integration)")
print(f"\nBandes testées: {list(BANDS.keys())}")
print("=" * 70)


def load_brainvision(vhdr_path):
    """Charge un fichier BrainVision (.vhdr/.eeg/.vmrk)"""
    vhdr_path = Path(vhdr_path)
    eeg_path = vhdr_path.with_suffix('.eeg')

    # Lire le header
    with open(vhdr_path, 'r') as f:
        header = f.read()

    # Extraire les paramètres
    n_channels = None
    sampling_rate = None
    binary_format = 'IEEE_FLOAT_32'

    for line in header.split('\n'):
        if 'NumberOfChannels' in line:
            n_channels = int(line.split('=')[1])
        if 'SamplingInterval' in line:
            interval_us = float(line.split('=')[1])
            sampling_rate = 1e6 / interval_us
        if 'BinaryFormat' in line:
            binary_format = line.split('=')[1].strip()

    if n_channels is None or sampling_rate is None:
        raise ValueError(f"Cannot parse header: {vhdr_path}")

    # Déterminer le dtype
    if binary_format == 'IEEE_FLOAT_32':
        dtype = np.float32
    elif binary_format == 'INT_16':
        dtype = np.int16
    else:
        dtype = np.float32

    # Charger les données
    data = np.fromfile(eeg_path, dtype=dtype)

    # Reshape en (channels, samples)
    n_samples = len(data) // n_channels
    data = data[:n_samples * n_channels].reshape(n_channels, n_samples, order='F')

    return data, sampling_rate, n_channels


def load_events(events_path):
    """Charge les événements depuis un fichier TSV"""
    events = []
    with open(events_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    onset_idx = header.index('onset')
    duration_idx = header.index('duration')
    trial_type_idx = header.index('trial_type')

    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) > max(onset_idx, duration_idx, trial_type_idx):
            onset = float(parts[onset_idx])
            duration = float(parts[duration_idx])
            trial_type = parts[trial_type_idx]
            events.append({
                'onset': onset,
                'duration': duration,
                'trial_type': trial_type
            })

    return events


def categorize_events(events):
    """Catégorise les événements en conscient vs non-conscient"""
    conscious = []  # Perception avec réponse correcte
    unconscious = []  # Stimuli irrelevants ou faux

    for evt in events:
        tt = evt['trial_type']
        if 'stimulus onset' not in tt:
            continue

        # Perception consciente active: Hit ou CorrRej sur cible relevante
        if 'Relevant' in tt and ('Hit' in tt or 'CorrRej' in tt):
            conscious.append(evt)
        # Stimuli non-pertinents (traitement moins conscient)
        elif 'Irrelevant' in tt:
            unconscious.append(evt)
        # Faux stimuli (baseline)
        elif '/false/' in tt.lower():
            unconscious.append(evt)

    return conscious, unconscious


def compute_psd_142Hz(data, fs, f_target=142.09, band_low=135, band_high=150):
    """
    Calcule la puissance spectrale autour de 142 Hz (kept for backward compat)
    """
    return compute_band_power(data, fs, band_low, band_high)


def compute_band_power(data, fs, band_low, band_high):
    """
    Compute mean power in a frequency band from raw epoch data.

    Returns:
        peak_freq, peak_power, z_score, psd, freqs
    """
    nyq = fs / 2

    # High-pass filter at 1 Hz (remove drift), no bandpass to allow multi-band
    b_hp, a_hp = signal.butter(4, 1.0 / nyq, btype='high')

    # Moyenne sur les canaux si multi-canal
    if data.ndim > 1:
        data_mean = np.nanmean(data, axis=0)
    else:
        data_mean = data

    try:
        filtered = signal.filtfilt(b_hp, a_hp, data_mean)
    except:
        return None, None, None, None, None

    # PSD avec Welch
    nperseg = min(int(fs * 2), len(filtered) // 4)
    if nperseg < 256:
        nperseg = min(256, len(filtered))

    freqs, psd = signal.welch(filtered, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # Band of interest
    mask_interest = (freqs >= band_low) & (freqs <= band_high)
    if not np.any(mask_interest):
        return None, None, None, psd, freqs

    # Mean power in band (not just peak)
    band_power = np.mean(psd[mask_interest])

    psd_interest = psd[mask_interest]
    freqs_interest = freqs[mask_interest]
    peak_idx = np.argmax(psd_interest)
    peak_freq = freqs_interest[peak_idx]

    # Baseline: adjacent band below (same width)
    band_width = band_high - band_low
    baseline_low = max(1, band_low - band_width)
    baseline_high = band_low
    mask_baseline = (freqs >= baseline_low) & (freqs < baseline_high)

    if np.any(mask_baseline):
        baseline = psd[mask_baseline]
        z_score = (band_power - np.mean(baseline)) / (np.std(baseline) + 1e-10)
    else:
        z_score = 0

    return peak_freq, band_power, z_score, psd, freqs


def analyze_subject(subject_path, subject_id):
    """Analyse un sujet complet"""
    print(f"\n{'='*60}")
    print(f"ANALYSE SUJET: {subject_id}")
    print(f"{'='*60}")

    # Trouver les fichiers
    ieeg_dir = subject_path / 'ses-1' / 'ieeg'
    vhdr_files = list(ieeg_dir.glob('*_task-Dur_ieeg.vhdr'))
    events_files = list(ieeg_dir.glob('*_task-Dur_events.tsv'))

    if not vhdr_files or not events_files:
        print(f"  [ERREUR] Fichiers manquants pour {subject_id}")
        return None

    vhdr_path = vhdr_files[0]
    events_path = events_files[0]

    # Charger les données
    print(f"  Chargement: {vhdr_path.name}")
    try:
        data, fs, n_channels = load_brainvision(vhdr_path)
        print(f"  - Sampling rate: {fs} Hz")
        print(f"  - Channels: {n_channels}")
        print(f"  - Duration: {data.shape[1]/fs:.1f} s")
    except Exception as e:
        print(f"  [ERREUR] Chargement: {e}")
        return None

    # Charger les événements
    events = load_events(events_path)
    conscious, unconscious = categorize_events(events)
    print(f"  - Événements conscients: {len(conscious)}")
    print(f"  - Événements inconscients: {len(unconscious)}")

    if len(conscious) < 5 or len(unconscious) < 5:
        print(f"  [WARN] Pas assez d'événements")

    # Analyser les epochs par condition ET par bande
    epoch_duration = 1.0  # 1 seconde après stimulus

    band_results = {}

    for band_name, (b_low, b_high) in BANDS.items():
        z_conscious_list = []
        z_unconscious_list = []

        # Epochs conscients
        for evt in conscious[:50]:
            start_sample = int(evt['onset'] * fs)
            end_sample = start_sample + int(epoch_duration * fs)
            if end_sample > data.shape[1]:
                continue
            epoch = data[:, start_sample:end_sample]
            _, _, z, _, _ = compute_band_power(epoch, fs, b_low, b_high)
            if z is not None:
                z_conscious_list.append(z)

        # Epochs inconscients
        for evt in unconscious[:50]:
            start_sample = int(evt['onset'] * fs)
            end_sample = start_sample + int(epoch_duration * fs)
            if end_sample > data.shape[1]:
                continue
            epoch = data[:, start_sample:end_sample]
            _, _, z, _, _ = compute_band_power(epoch, fs, b_low, b_high)
            if z is not None:
                z_unconscious_list.append(z)

        # Statistique pour cette bande
        if len(z_conscious_list) >= 3 and len(z_unconscious_list) >= 3:
            stat, p_value = mannwhitneyu(z_conscious_list, z_unconscious_list, alternative='greater')
        else:
            p_value = 1.0

        mean_z_con = np.mean(z_conscious_list) if z_conscious_list else 0
        mean_z_uncon = np.mean(z_unconscious_list) if z_unconscious_list else 0
        significant = p_value < 0.05 and mean_z_con > mean_z_uncon

        band_results[band_name] = {
            'p_value': p_value,
            'mean_z_conscious': mean_z_con,
            'mean_z_unconscious': mean_z_uncon,
            'difference': mean_z_con - mean_z_uncon,
            'significant': significant,
            'n_conscious': len(z_conscious_list),
            'n_unconscious': len(z_unconscious_list),
        }

    # Affichage multi-bande
    print(f"\n  RÉSULTATS {subject_id} (multi-bande):")
    print(f"  {'Bande':<25} {'Conscient':>10} {'Inconscient':>12} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*65}")
    for band_name, br in band_results.items():
        sig_mark = '**' if br['significant'] else ''
        print(f"  {band_name:<25} {br['mean_z_conscious']:>10.3f} {br['mean_z_unconscious']:>12.3f} {br['p_value']:>10.4f} {sig_mark:>5}")

    return {
        'subject': subject_id,
        'bands': band_results
    }


def main():
    """Analyse principale"""

    # Chemin vers les données BIDS
    base_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/datasets/cogitate_ieeg/bids_ecog/mnt/beegfs/workspace/2023-0385-Cogitatedatarelease/CURATE/COG_ECOG_EXP1_BIDS_SAMPLE")

    if not base_path.exists():
        print(f"[ERREUR] Chemin non trouvé: {base_path}")
        return

    # Trouver tous les sujets
    subjects = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    print(f"\nSujets trouvés: {len(subjects)}")
    for s in subjects:
        print(f"  - {s.name}")

    # Analyser chaque sujet
    all_results = []
    for subject_path in subjects:
        result = analyze_subject(subject_path, subject_path.name)
        if result:
            all_results.append(result)

    # Résumé global — multi-bande
    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL - ANALYSE MULTI-BANDE")
    print("=" * 70)

    if all_results:
        from scipy.stats import chi2

        print(f"\nNombre de sujets analysés: {len(all_results)}")

        # Combiner les p-values par bande (Fisher's method)
        print(f"\n{'Bande':<25} {'p combinée':>12} {'Significatif':>14}")
        print("-" * 55)

        for band_name in BANDS:
            p_values = []
            for r in all_results:
                if band_name in r['bands']:
                    p_values.append(r['bands'][band_name]['p_value'])

            if p_values:
                chi2_stat = -2 * sum(np.log(p + 1e-10) for p in p_values)
                combined_p = 1 - chi2.cdf(chi2_stat, 2 * len(p_values))
                sig = 'YES' if combined_p < 0.05 else 'No'
                print(f"  {band_name:<25} {combined_p:>10.4f}   {sig:>10}")

        # Interprétation
        print("\n" + "-" * 70)
        print("INTERPRÉTATION:")
        print("-" * 70)

        # Check High Gamma specifically
        hg_pvals = [r['bands']['High Gamma (80-120 Hz)']['p_value']
                     for r in all_results if 'High Gamma (80-120 Hz)' in r['bands']]
        if hg_pvals:
            chi2_stat = -2 * sum(np.log(p + 1e-10) for p in hg_pvals)
            hg_combined_p = 1 - chi2.cdf(chi2_stat, 2 * len(hg_pvals))

            if hg_combined_p < 0.05:
                print(f"\nThe 80-120 Hz band (containing f1 = {F1:.0f} Hz) shows")
                print(f"significant differentiation (p = {hg_combined_p:.4f}).")
                print(f"Prediction f1 = 102 Hz: VALIDATED")
            else:
                print(f"\nThe 80-120 Hz band does not reach significance (p = {hg_combined_p:.4f}).")

        # Check f2 band
        f2_pvals = [r['bands']['f2 band (135-150 Hz)']['p_value']
                     for r in all_results if 'f2 band (135-150 Hz)' in r['bands']]
        if f2_pvals:
            chi2_stat = -2 * sum(np.log(p + 1e-10) for p in f2_pvals)
            f2_combined_p = 1 - chi2.cdf(chi2_stat, 2 * len(f2_pvals))
            print(f"\nThe 135-150 Hz band (f2 = {F_PHI:.0f} Hz): p = {f2_combined_p:.4f}")
            print(f"(Not expected to be significant on perception task — f2 indexes integration cost)")

    else:
        print("Aucun résultat à analyser.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
