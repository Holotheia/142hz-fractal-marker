#!/usr/bin/env python3
"""
Analyse Neural Efficiency - Test 4: Ratio CCT/ABT at 142 Hz
=============================================================

This script tests the prediction that athletes and controls show
OPPOSITE modulation patterns of 142 Hz between tasks:
  - Athletes: ratio CCT/ABT < 1 (142 Hz DECREASES during concentration)
  - Controls: ratio CCT/ABT > 1 (142 Hz INCREASES during concentration)

Method: For each subject, compute ratio = z_142_CCT / z_142_ABT,
then compare ratios between groups using Mann-Whitney U test.

Expected result: Significant difference in ratios (p < 0.05)
"""

import numpy as np
from scipy import signal
from scipy.stats import mannwhitneyu, ttest_ind
from pathlib import Path
import json

PHI = (1 + np.sqrt(5)) / 2
D_STAR = 2.3107
F1 = 432 / (PHI ** 3)        # 102 Hz
F2 = 432 / (PHI ** D_STAR)   # 142 Hz


def read_cdt_file(cdt_path, dpo_path):
    """Lit un fichier CDT avec son descripteur"""
    params = {}
    with open(dpo_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                params[key.strip()] = value.strip()

    n_channels = int(params.get('NumChannels', 67))
    n_samples = int(params.get('NumSamples', 0))
    fs = float(params.get('SampleFreqHz', 1000))

    with open(cdt_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    try:
        data = data.reshape((n_samples, n_channels)).T
    except:
        n_samples_actual = len(data) // n_channels
        data = data[:n_samples_actual * n_channels].reshape((n_samples_actual, n_channels)).T

    return data, fs


def compute_psd_peak(data, fs, target_freq, bandwidth=15):
    """Calcule le z-score du pic PSD autour d'une fréquence"""
    data_eeg = data[:64, :]

    all_z_scores = []
    for ch in range(min(32, data_eeg.shape[0])):
        try:
            freqs, psd = signal.welch(data_eeg[ch, :], fs=fs, nperseg=int(fs*2))
            mask_target = (freqs >= target_freq - bandwidth/2) & (freqs <= target_freq + bandwidth/2)
            mask_baseline = (freqs >= target_freq - 30) & (freqs < target_freq - bandwidth/2)

            if np.sum(mask_target) > 0 and np.sum(mask_baseline) > 0:
                peak_power = psd[mask_target].max()
                baseline_mean = psd[mask_baseline].mean()
                baseline_std = psd[mask_baseline].std()

                if baseline_std > 0:
                    z_score = (peak_power - baseline_mean) / baseline_std
                    all_z_scores.append(z_score)
        except:
            pass

    return np.mean(all_z_scores) if all_z_scores else None


def analyze_subject(subject_path, subject_id):
    """Analyse un sujet complet"""
    results = {'id': subject_id, 'ABT': {}, 'CCT': {}}

    experiments = {
        'ABT': f'Experiment1_ABT/ABT_EEG/{subject_id}_ABT_EEG',
        'CCT': f'Experiment2_CCT/CCT_EEG/{subject_id}_CCT_EEG',
    }

    for exp_name, exp_path in experiments.items():
        cdt_path = subject_path / f"{exp_path}.cdt"
        dpo_path = subject_path / f"{exp_path}.cdt.dpo"
        dpa_path = subject_path / f"{exp_path}.cdt.dpa"

        if not cdt_path.exists():
            continue

        # Try .dpo first, then .dpa
        desc_path = dpo_path if dpo_path.exists() else dpa_path

        try:
            data, fs = read_cdt_file(cdt_path, desc_path)

            z_102 = compute_psd_peak(data, fs, 102, bandwidth=20)
            z_142 = compute_psd_peak(data, fs, 142, bandwidth=15)

            results[exp_name] = {
                'z_102': z_102,
                'z_142': z_142,
                'duration': data.shape[1] / fs
            }

            print(f"    {exp_name}: z_102={z_102:.1f}, z_142={z_142:.1f}")

        except Exception as e:
            print(f"    {exp_name}: Erreur - {e}")

    return results


def main():
    base_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/datasets/elite_athletes")

    # Find all extracted subjects
    subjects = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith(('Athlete_', 'Control_')):
            subjects.append(folder)

    print("="*70)
    print("ANALYSE NEURAL EFFICIENCY - RATIO CCT/ABT à 142 Hz")
    print("="*70)
    print(f"f₁ = {F1:.2f} Hz (perception)")
    print(f"f₂ = {F2:.2f} Hz (intégration/concentration)")
    print(f"Sujets trouvés: {len(subjects)}")

    athletes_142 = []
    controls_142 = []
    athletes_102 = []
    controls_102 = []

    all_results = []

    for subject_path in sorted(subjects):
        subject_id = subject_path.name.split('_')[1]
        group = subject_path.name.split('_')[0]

        print(f"\n{group} {subject_id}:")

        result = analyze_subject(subject_path, subject_id)
        result['group'] = group
        result['subject_id'] = subject_path.name
        all_results.append(result)

        # Collect z-scores for CCT (concentration task)
        if 'z_142' in result.get('CCT', {}) and result['CCT']['z_142'] is not None:
            if group == 'Athlete':
                athletes_142.append(result['CCT']['z_142'])
                if result['CCT'].get('z_102'):
                    athletes_102.append(result['CCT']['z_102'])
            else:
                controls_142.append(result['CCT']['z_142'])
                if result['CCT'].get('z_102'):
                    controls_102.append(result['CCT']['z_102'])

    # RATIO ANALYSIS: CCT/ABT at 142 Hz (as described in paper)
    print("\n" + "="*70)
    print("ANALYSE DES RATIOS CCT/ABT à 142 Hz")
    print("="*70)

    athletes_ratios = []
    controls_ratios = []

    print(f"\n{'Sujet':<25} {'ABT z_142':>10} {'CCT z_142':>10} {'Ratio':>8} {'Groupe':>10}")
    print("-" * 70)

    for r in all_results:
        abt_142 = r.get('ABT', {}).get('z_142')
        cct_142 = r.get('CCT', {}).get('z_142')

        if abt_142 is not None and cct_142 is not None and abt_142 > 0.1:
            ratio = cct_142 / abt_142
            subject_id = r.get('subject_id', '?')
            group = r['group']

            print(f"  {subject_id:<23} {abt_142:>10.1f} {cct_142:>10.1f} {ratio:>8.2f} {group:>10}")

            if group == 'Athlete':
                athletes_ratios.append(ratio)
            else:
                controls_ratios.append(ratio)

    # Statistical comparison of ratios
    print("\n" + "="*70)
    print("COMPARAISON STATISTIQUE - RATIO CCT/ABT")
    print("="*70)

    if athletes_ratios and controls_ratios:
        print(f"\nAthlètes (n={len(athletes_ratios)}):")
        print(f"  Ratio moyen: {np.mean(athletes_ratios):.2f} ± {np.std(athletes_ratios):.2f}")
        print(f"  Interprétation: 142 Hz {'diminue' if np.mean(athletes_ratios) < 1 else 'augmente'}"
              f" de {abs(1-np.mean(athletes_ratios))*100:.0f}% pendant la concentration")

        print(f"\nContrôles (n={len(controls_ratios)}):")
        print(f"  Ratio moyen: {np.mean(controls_ratios):.2f} ± {np.std(controls_ratios):.2f}")
        print(f"  Interprétation: 142 Hz {'diminue' if np.mean(controls_ratios) < 1 else 'augmente'}"
              f" de {abs(1-np.mean(controls_ratios))*100:.0f}% pendant la concentration")

        stat, p_val = mannwhitneyu(athletes_ratios, controls_ratios, alternative='two-sided')
        print(f"\nMann-Whitney p = {p_val:.4f}")
        print(f"Significatif: {'OUI' if p_val < 0.05 else 'Non'}")

    # Also show direct CCT comparison
    print("\n" + "="*70)
    print("COMPARAISON DIRECTE - CCT à 142 Hz")
    print("="*70)

    if athletes_142 and controls_142:
        print(f"\n142 Hz (prédiction intégration):")
        print(f"  Athlètes (n={len(athletes_142)}): {np.mean(athletes_142):.1f} ± {np.std(athletes_142):.1f}")
        print(f"  Contrôles (n={len(controls_142)}): {np.mean(controls_142):.1f} ± {np.std(controls_142):.1f}")

        if len(athletes_142) > 1 and len(controls_142) > 1:
            stat, p_direct = mannwhitneyu(athletes_142, controls_142, alternative='two-sided')
            print(f"  p-value: {p_direct:.4f}")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if athletes_ratios and controls_ratios:
        a_mean = np.mean(athletes_ratios)
        c_mean = np.mean(controls_ratios)
        if a_mean < 1 and c_mean > 1 and p_val < 0.05:
            print(f"\nOpposite modulation patterns confirmed:")
            print(f"  Athletes: 142 Hz DECREASES during concentration (ratio = {a_mean:.2f})")
            print(f"  Controls: 142 Hz INCREASES during concentration (ratio = {c_mean:.2f})")
            print(f"  p = {p_val:.4f}")
            print(f"\nThis supports the Neural Efficiency Hypothesis:")
            print(f"  142 Hz indexes integration COST, not integration itself.")
            print(f"\nPrediction: VALIDATED")
        elif p_val < 0.05:
            print(f"\nSignificant difference in ratios (p = {p_val:.4f})")
            print(f"  Athletes ratio: {a_mean:.2f}, Controls ratio: {c_mean:.2f}")
        else:
            print(f"\nNo significant difference in ratios (p = {p_val:.4f})")

    # Save results
    output_file = base_path.parent / "resultats_athletes_vs_controls.json"
    with open(output_file, 'w') as f:
        json.dump({
            'athletes_ratios': [float(x) for x in athletes_ratios],
            'controls_ratios': [float(x) for x in controls_ratios],
            'athletes_142_cct': [float(x) for x in athletes_142],
            'controls_142_cct': [float(x) for x in controls_142],
        }, f, indent=2)
    print(f"\nRésultats sauvegardés: {output_file}")


if __name__ == "__main__":
    main()
