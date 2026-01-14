#!/usr/bin/env python3
"""
Analyse comparative Athlètes vs Contrôles - Test 142 Hz
Hypothèse: Les athlètes (entraînement concentration) ont un signal plus fort à 142 Hz
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
    print("ANALYSE COMPARATIVE ATHLÈTES vs CONTRÔLES - 142 Hz")
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

    # Statistical comparison
    print("\n" + "="*70)
    print("COMPARAISON STATISTIQUE - TÂCHE CONCENTRATION (CCT)")
    print("="*70)

    if athletes_142 and controls_142:
        print(f"\n142 Hz (prédiction intégration):")
        print(f"  Athlètes (n={len(athletes_142)}): {np.mean(athletes_142):.1f} ± {np.std(athletes_142):.1f}")
        print(f"  Contrôles (n={len(controls_142)}): {np.mean(controls_142):.1f} ± {np.std(controls_142):.1f}")

        if len(athletes_142) > 1 and len(controls_142) > 1:
            stat, p_val = mannwhitneyu(athletes_142, controls_142, alternative='two-sided')
            print(f"  p-value: {p_val:.4f}")
            print(f"  Différence significative: {'OUI' if p_val < 0.05 else 'Non'}")

            # Effect size
            diff = np.mean(athletes_142) - np.mean(controls_142)
            pooled_std = np.sqrt((np.std(athletes_142)**2 + np.std(controls_142)**2) / 2)
            effect_size = diff / pooled_std if pooled_std > 0 else 0
            print(f"  Taille d'effet (Cohen's d): {effect_size:.3f}")

    if athletes_102 and controls_102:
        print(f"\n102 Hz (perception):")
        print(f"  Athlètes (n={len(athletes_102)}): {np.mean(athletes_102):.1f} ± {np.std(athletes_102):.1f}")
        print(f"  Contrôles (n={len(controls_102)}): {np.mean(controls_102):.1f} ± {np.std(controls_102):.1f}")

        if len(athletes_102) > 1 and len(controls_102) > 1:
            stat, p_val = mannwhitneyu(athletes_102, controls_102, alternative='two-sided')
            print(f"  p-value: {p_val:.4f}")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if athletes_142 and controls_142:
        athlete_mean = np.mean(athletes_142)
        control_mean = np.mean(controls_142)

        if athlete_mean > control_mean:
            print("\n→ Les ATHLÈTES montrent un signal 142 Hz PLUS ÉLEVÉ que les contrôles")
            print("  Cela suggère que l'entraînement à la concentration")
            print("  renforce l'activité à la fréquence d'intégration f₂ = 142 Hz")
        else:
            print("\n→ Pas de différence claire entre athlètes et contrôles à 142 Hz")

    # Save results
    output_file = base_path.parent / "resultats_athletes_vs_controls.json"
    with open(output_file, 'w') as f:
        json.dump({
            'athletes_142': [float(x) for x in athletes_142],
            'controls_142': [float(x) for x in controls_142],
            'athletes_102': [float(x) for x in athletes_102],
            'controls_102': [float(x) for x in controls_102],
        }, f, indent=2)
    print(f"\nRésultats sauvegardés: {output_file}")


if __name__ == "__main__":
    main()
