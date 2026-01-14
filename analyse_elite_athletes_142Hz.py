#!/usr/bin/env python3
"""
Analyse Elite Athletes EEG - Test Prédiction 142 Hz
Dataset: Control subject - Attention & Concentration tasks
Sampling rate: 1000 Hz (Nyquist = 500 Hz)
"""

import numpy as np
from scipy import signal
from scipy.stats import mannwhitneyu, ttest_ind
from pathlib import Path
import struct

PHI = (1 + np.sqrt(5)) / 2
D_STAR = 2.3107
F1 = 432 / (PHI ** 3)        # 102 Hz - perception
F2 = 432 / (PHI ** D_STAR)   # 142 Hz - integration

print(f"Constantes théoriques:")
print(f"  φ = {PHI:.10f}")
print(f"  D* = {D_STAR}")
print(f"  f₁ = 432/φ³ = {F1:.2f} Hz (perception)")
print(f"  f₂ = 432/φ^D* = {F2:.2f} Hz (intégration)")


def read_cdt_file(cdt_path, dpo_path):
    """Lit un fichier CDT (Curry Data format) avec son descripteur"""

    # Lire les paramètres du fichier .dpo
    params = {}
    with open(dpo_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                params[key.strip()] = value.strip()

    n_channels = int(params.get('NumChannels', 67))
    n_samples = int(params.get('NumSamples', 0))
    fs = float(params.get('SampleFreqHz', 1000))
    data_format = int(params.get('DataFormat', 6))

    print(f"\nParamètres CDT:")
    print(f"  Channels: {n_channels}")
    print(f"  Samples: {n_samples}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Data format: {data_format}")
    print(f"  Duration: {n_samples/fs:.1f} seconds")

    # Lire les données binaires
    # Format 6 = 32-bit float
    with open(cdt_path, 'rb') as f:
        if data_format == 6:
            data = np.fromfile(f, dtype=np.float32)
        else:
            data = np.fromfile(f, dtype=np.float32)

    # Reshape: samples x channels (SAMP order)
    try:
        data = data.reshape((n_samples, n_channels))
        data = data.T  # channels x samples
    except:
        print(f"  Reshape failed, using alternative method")
        n_samples_actual = len(data) // n_channels
        data = data[:n_samples_actual * n_channels].reshape((n_samples_actual, n_channels)).T

    return data, fs, n_channels


def compute_band_power(data, fs, band_low, band_high):
    """Calcule la puissance dans une bande de fréquence"""
    nyq = fs / 2
    if band_high >= nyq:
        band_high = nyq * 0.95

    b, a = signal.butter(4, [band_low/nyq, band_high/nyq], btype='band')

    # Moyenne sur tous les canaux EEG (exclure les 3 derniers = autres)
    data_eeg = data[:64, :]  # 64 canaux EEG

    powers = []
    for ch in range(min(64, data_eeg.shape[0])):
        try:
            filtered = signal.filtfilt(b, a, data_eeg[ch, :])
            powers.append(np.mean(filtered**2))
        except:
            pass

    return np.mean(powers) if powers else None


def compute_psd_peak(data, fs, target_freq, bandwidth=10):
    """Calcule le pic PSD autour d'une fréquence cible"""
    data_eeg = data[:64, :]

    # PSD Welch sur tous les canaux
    all_peaks = []
    all_z_scores = []

    for ch in range(min(32, data_eeg.shape[0])):  # Premiers 32 canaux pour rapidité
        try:
            freqs, psd = signal.welch(data_eeg[ch, :], fs=fs, nperseg=int(fs*2))

            # Bande cible
            mask_target = (freqs >= target_freq - bandwidth/2) & (freqs <= target_freq + bandwidth/2)
            # Baseline (hors bande)
            mask_baseline = (freqs >= target_freq - 30) & (freqs < target_freq - bandwidth/2)

            if np.sum(mask_target) > 0 and np.sum(mask_baseline) > 0:
                peak_power = psd[mask_target].max()
                baseline_mean = psd[mask_baseline].mean()
                baseline_std = psd[mask_baseline].std()

                if baseline_std > 0:
                    z_score = (peak_power - baseline_mean) / baseline_std
                    all_z_scores.append(z_score)
                    all_peaks.append(peak_power)
        except:
            pass

    return np.mean(all_z_scores) if all_z_scores else None, np.mean(all_peaks) if all_peaks else None


def segment_by_task(data, fs, segment_duration=30):
    """Segmente les données en epochs"""
    n_samples_per_segment = int(segment_duration * fs)
    n_segments = data.shape[1] // n_samples_per_segment

    segments = []
    for i in range(n_segments):
        start = i * n_samples_per_segment
        end = start + n_samples_per_segment
        segments.append(data[:, start:end])

    return segments


def main():
    base_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/datasets/elite_athletes/Control_Sub723")

    experiments = {
        'ABT': 'Experiment1_ABT/ABT_EEG/Sub723_ABT_EEG',  # Attention-Based Task
        'CCT': 'Experiment2_CCT/CCT_EEG/Sub723_CCT_EEG',  # Concentration/Cognitive Task
    }

    print("\n" + "="*70)
    print("ANALYSE ELITE ATHLETES - TEST PRÉDICTION 142 Hz")
    print("="*70)

    all_results = {}

    for exp_name, exp_path in experiments.items():
        cdt_path = base_path / f"{exp_path}.cdt"
        dpo_path = base_path / f"{exp_path}.cdt.dpo"

        if not cdt_path.exists():
            print(f"\n{exp_name}: Fichier non trouvé")
            continue

        print(f"\n{'='*60}")
        print(f"EXPÉRIENCE: {exp_name}")
        print(f"{'='*60}")

        # Charger les données
        data, fs, n_ch = read_cdt_file(cdt_path, dpo_path)

        print(f"\nData shape: {data.shape}")
        print(f"Nyquist frequency: {fs/2} Hz")

        # Analyse globale
        print(f"\n--- Analyse PSD globale ---")

        # Test f₁ = 102 Hz
        z_102, power_102 = compute_psd_peak(data, fs, 102, bandwidth=20)
        print(f"  f₁ = 102 Hz: z-score = {z_102:.3f}" if z_102 else "  f₁ = 102 Hz: N/A")

        # Test f₂ = 142 Hz
        z_142, power_142 = compute_psd_peak(data, fs, 142, bandwidth=15)
        print(f"  f₂ = 142 Hz: z-score = {z_142:.3f}" if z_142 else "  f₂ = 142 Hz: N/A")

        # Analyse par segments (30 secondes)
        print(f"\n--- Analyse par segments (30s) ---")
        segments = segment_by_task(data, fs, segment_duration=30)
        print(f"  Nombre de segments: {len(segments)}")

        z_scores_102 = []
        z_scores_142 = []

        for i, seg in enumerate(segments):
            z102, _ = compute_psd_peak(seg, fs, 102, bandwidth=20)
            z142, _ = compute_psd_peak(seg, fs, 142, bandwidth=15)

            if z102 is not None:
                z_scores_102.append(z102)
            if z142 is not None:
                z_scores_142.append(z142)

        if z_scores_102:
            print(f"\n  f₁ = 102 Hz:")
            print(f"    Mean z-score: {np.mean(z_scores_102):.3f} ± {np.std(z_scores_102):.3f}")
            print(f"    Range: [{min(z_scores_102):.3f}, {max(z_scores_102):.3f}]")

        if z_scores_142:
            print(f"\n  f₂ = 142 Hz:")
            print(f"    Mean z-score: {np.mean(z_scores_142):.3f} ± {np.std(z_scores_142):.3f}")
            print(f"    Range: [{min(z_scores_142):.3f}, {max(z_scores_142):.3f}]")

        # Comparaison première vs dernière moitié (effet d'entraînement?)
        mid_point = len(segments) // 2
        if mid_point > 0 and len(z_scores_142) >= mid_point * 2:
            first_half_142 = z_scores_142[:mid_point]
            second_half_142 = z_scores_142[mid_point:]

            stat, p_val = mannwhitneyu(first_half_142, second_half_142, alternative='two-sided')

            print(f"\n  Évolution temporelle (142 Hz):")
            print(f"    1ère moitié: {np.mean(first_half_142):.3f}")
            print(f"    2ème moitié: {np.mean(second_half_142):.3f}")
            print(f"    p-value: {p_val:.4f}")

        all_results[exp_name] = {
            'z_102_global': z_102,
            'z_142_global': z_142,
            'z_102_segments': z_scores_102,
            'z_142_segments': z_scores_142,
        }

    # Résumé comparatif
    print("\n" + "="*70)
    print("RÉSUMÉ COMPARATIF ABT vs CCT")
    print("="*70)

    if 'ABT' in all_results and 'CCT' in all_results:
        print(f"\n{'Métrique':<30} {'ABT':<20} {'CCT':<20}")
        print("-"*70)

        for freq in [102, 142]:
            key = f'z_{freq}_global'
            abt_val = all_results['ABT'].get(key)
            cct_val = all_results['CCT'].get(key)

            abt_str = f"{abt_val:.3f}" if abt_val else "N/A"
            cct_str = f"{cct_val:.3f}" if cct_val else "N/A"

            print(f"f = {freq} Hz (z-score global)    {abt_str:<20} {cct_str:<20}")

        # Test statistique ABT vs CCT pour 142 Hz
        abt_142 = all_results['ABT'].get('z_142_segments', [])
        cct_142 = all_results['CCT'].get('z_142_segments', [])

        if abt_142 and cct_142:
            stat, p_val = mannwhitneyu(abt_142, cct_142, alternative='two-sided')
            print(f"\n142 Hz - Test ABT vs CCT:")
            print(f"  ABT mean: {np.mean(abt_142):.3f}")
            print(f"  CCT mean: {np.mean(cct_142):.3f}")
            print(f"  p-value: {p_val:.4f}")
            print(f"  Significatif: {'OUI' if p_val < 0.05 else 'Non'}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    # Vérifier si 142 Hz montre un signal positif
    all_142 = []
    for exp in all_results.values():
        if exp.get('z_142_segments'):
            all_142.extend(exp['z_142_segments'])

    if all_142:
        mean_142 = np.mean(all_142)
        positive_ratio = sum(1 for z in all_142 if z > 0) / len(all_142)

        print(f"\nRésultats globaux 142 Hz:")
        print(f"  Mean z-score: {mean_142:.3f}")
        print(f"  Segments positifs: {positive_ratio*100:.1f}%")

        if mean_142 > 1.5:
            print(f"\n→ SIGNAL POSITIF à 142 Hz détecté!")
        elif mean_142 > 0:
            print(f"\n→ Signal faible mais présent à 142 Hz")
        else:
            print(f"\n→ Pas de signal significatif à 142 Hz")


if __name__ == "__main__":
    main()
