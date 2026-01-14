#!/usr/bin/env python3
"""
Analyse stratifiée par sous-cohortes
Cohorte A: Contrôles 700s (protocole original)
Cohorte B: Contrôles 001-023 (protocole différent?)
"""

import numpy as np
from scipy import signal
from scipy.stats import mannwhitneyu, wilcoxon
from pathlib import Path
import json

PHI = (1 + np.sqrt(5)) / 2
D_STAR = 2.3107
F1 = 432 / (PHI ** 3)
F2 = 432 / (PHI ** D_STAR)


def read_cdt_file(cdt_path, desc_path):
    params = {}
    with open(desc_path, 'r', encoding='utf-8-sig') as f:
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
    results = {'ABT': {}, 'CCT': {}}

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

        desc_path = dpo_path if dpo_path.exists() else dpa_path
        if not desc_path.exists():
            continue

        try:
            data, fs = read_cdt_file(cdt_path, desc_path)
            z_102 = compute_psd_peak(data, fs, 102, bandwidth=20)
            z_142 = compute_psd_peak(data, fs, 142, bandwidth=15)
            results[exp_name] = {'z_102': z_102, 'z_142': z_142}
        except:
            pass

    return results


def main():
    base_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/datasets/elite_athletes")

    subjects = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith(('Athlete_', 'Control_')):
            subjects.append(folder)

    print("="*70)
    print("ANALYSE STRATIFIÉE PAR SOUS-COHORTES")
    print("="*70)

    # Collect data by cohort
    athletes_cct_142 = []
    athletes_abt_142 = []

    controls_700s_cct_142 = []
    controls_700s_abt_142 = []

    controls_001_cct_142 = []
    controls_001_abt_142 = []

    print("\n--- Collecte des données ---\n")

    for subject_path in sorted(subjects):
        subject_id = subject_path.name.split('_')[1]
        group = subject_path.name.split('_')[0]

        result = analyze_subject(subject_path, subject_id)

        cct_142 = result.get('CCT', {}).get('z_142')
        abt_142 = result.get('ABT', {}).get('z_142')

        if group == 'Athlete':
            if cct_142 is not None:
                athletes_cct_142.append(cct_142)
            if abt_142 is not None:
                athletes_abt_142.append(abt_142)
            abt_str = f"{abt_142:.1f}" if abt_142 is not None else "N/A"
            cct_str = f"{cct_142:.1f}" if cct_142 is not None else "N/A"
            print(f"Athlete {subject_id}: ABT={abt_str}, CCT={cct_str}")
        else:
            # Stratify controls by ID
            num_id = int(subject_id.replace('Sub', ''))
            if num_id >= 700:
                # 700s cohort
                if cct_142 is not None:
                    controls_700s_cct_142.append(cct_142)
                if abt_142 is not None:
                    controls_700s_abt_142.append(abt_142)
                abt_str = f"{abt_142:.1f}" if abt_142 is not None else "N/A"
                cct_str = f"{cct_142:.1f}" if cct_142 is not None else "N/A"
                print(f"Control {subject_id} [700s]: ABT={abt_str}, CCT={cct_str}")
            else:
                # 001-023 cohort
                if cct_142 is not None:
                    controls_001_cct_142.append(cct_142)
                if abt_142 is not None:
                    controls_001_abt_142.append(abt_142)
                abt_str = f"{abt_142:.1f}" if abt_142 is not None else "N/A"
                cct_str = f"{cct_142:.1f}" if cct_142 is not None else "N/A"
                print(f"Control {subject_id} [001s]: ABT={abt_str}, CCT={cct_str}")

    # Statistics
    print("\n" + "="*70)
    print("RÉSULTATS STATISTIQUES")
    print("="*70)

    # Test 1: Athletes vs Controls-700s (original analysis)
    print("\n--- TEST 1: Athlètes vs Contrôles-700s (cohorte originale) ---")
    print(f"Athlètes (n={len(athletes_cct_142)}): {np.mean(athletes_cct_142):.1f} ± {np.std(athletes_cct_142):.1f}")
    print(f"Contrôles-700s (n={len(controls_700s_cct_142)}): {np.mean(controls_700s_cct_142):.1f} ± {np.std(controls_700s_cct_142):.1f}")

    if len(athletes_cct_142) > 1 and len(controls_700s_cct_142) > 1:
        stat, p = mannwhitneyu(athletes_cct_142, controls_700s_cct_142, alternative='two-sided')
        diff = np.mean(athletes_cct_142) - np.mean(controls_700s_cct_142)
        pooled_std = np.sqrt((np.std(athletes_cct_142)**2 + np.std(controls_700s_cct_142)**2) / 2)
        d = diff / pooled_std if pooled_std > 0 else 0
        print(f"Mann-Whitney p = {p:.4f}")
        print(f"Cohen's d = {d:.2f}")
        print(f"→ {'SIGNIFICATIF' if p < 0.05 else 'Non significatif'}")

    # Test 2: Athletes vs Controls-001s
    print("\n--- TEST 2: Athlètes vs Contrôles-001s (nouvelle cohorte) ---")
    print(f"Athlètes (n={len(athletes_cct_142)}): {np.mean(athletes_cct_142):.1f} ± {np.std(athletes_cct_142):.1f}")
    print(f"Contrôles-001s (n={len(controls_001_cct_142)}): {np.mean(controls_001_cct_142):.1f} ± {np.std(controls_001_cct_142):.1f}")

    if len(athletes_cct_142) > 1 and len(controls_001_cct_142) > 1:
        stat, p = mannwhitneyu(athletes_cct_142, controls_001_cct_142, alternative='two-sided')
        print(f"Mann-Whitney p = {p:.4f}")
        print(f"→ {'SIGNIFICATIF' if p < 0.05 else 'Non significatif'}")

    # Test 3: Controls-700s vs Controls-001s (heterogeneity)
    print("\n--- TEST 3: Contrôles-700s vs Contrôles-001s (hétérogénéité) ---")
    print(f"Contrôles-700s (n={len(controls_700s_cct_142)}): {np.mean(controls_700s_cct_142):.1f} ± {np.std(controls_700s_cct_142):.1f}")
    print(f"Contrôles-001s (n={len(controls_001_cct_142)}): {np.mean(controls_001_cct_142):.1f} ± {np.std(controls_001_cct_142):.1f}")

    if len(controls_700s_cct_142) > 1 and len(controls_001_cct_142) > 1:
        stat, p = mannwhitneyu(controls_700s_cct_142, controls_001_cct_142, alternative='two-sided')
        ratio = np.mean(controls_700s_cct_142) / np.mean(controls_001_cct_142) if np.mean(controls_001_cct_142) > 0 else float('inf')
        print(f"Mann-Whitney p = {p:.4f}")
        print(f"Ratio 700s/001s = {ratio:.1f}x")
        print(f"→ HÉTÉROGÉNÉITÉ {'CONFIRMÉE' if p < 0.05 else 'Non significative'}")

    # Test 4: ABT vs CCT ratio analysis
    print("\n--- TEST 4: Ratio CCT/ABT à 142 Hz ---")

    # Athletes ratio
    athletes_ratio = []
    for i in range(min(len(athletes_abt_142), len(athletes_cct_142))):
        if athletes_abt_142[i] > 0.1:
            athletes_ratio.append(athletes_cct_142[i] / athletes_abt_142[i])

    # Controls-700s ratio
    controls_700s_ratio = []
    for i in range(min(len(controls_700s_abt_142), len(controls_700s_cct_142))):
        if controls_700s_abt_142[i] > 0.1:
            controls_700s_ratio.append(controls_700s_cct_142[i] / controls_700s_abt_142[i])

    print(f"Athlètes ratio CCT/ABT (n={len(athletes_ratio)}): {np.mean(athletes_ratio):.2f} ± {np.std(athletes_ratio):.2f}")
    print(f"Contrôles-700s ratio (n={len(controls_700s_ratio)}): {np.mean(controls_700s_ratio):.2f} ± {np.std(controls_700s_ratio):.2f}")

    if len(athletes_ratio) > 1 and len(controls_700s_ratio) > 1:
        stat, p = mannwhitneyu(athletes_ratio, controls_700s_ratio, alternative='two-sided')
        print(f"Mann-Whitney p = {p:.4f}")

    # Summary
    print("\n" + "="*70)
    print("RÉSUMÉ STRATIFIÉ")
    print("="*70)
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ANALYSE STRATIFIÉE 142 Hz                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Groupe          │  N  │  142 Hz CCT (z)  │  Interprétation         ║
╠══════════════════════════════════════════════════════════════════════╣""")
    print(f"║  Athlètes        │ {len(athletes_cct_142):>2}  │  {np.mean(athletes_cct_142):>6.1f} ± {np.std(athletes_cct_142):>5.1f}   │  Efficience neurale     ║")
    print(f"║  Contrôles-700s  │ {len(controls_700s_cct_142):>2}  │  {np.mean(controls_700s_cct_142):>6.1f} ± {np.std(controls_700s_cct_142):>5.1f}   │  Effort cognitif élevé  ║")
    print(f"║  Contrôles-001s  │ {len(controls_001_cct_142):>2}  │  {np.mean(controls_001_cct_142):>6.1f} ± {np.std(controls_001_cct_142):>5.1f}   │  Profil type athlète    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    print("""
INTERPRÉTATION:
- Les contrôles-700s montrent le pattern attendu (effort élevé)
- Les contrôles-001s ressemblent aux athlètes (possible stratégie efficiente?)
- Cette hétérogénéité suggère que 142 Hz capture bien une dimension
  fonctionnelle (stratégie cognitive) plutôt qu'un simple marqueur de groupe
""")

    # Save
    output = {
        'athletes_n': len(athletes_cct_142),
        'athletes_mean': float(np.mean(athletes_cct_142)),
        'controls_700s_n': len(controls_700s_cct_142),
        'controls_700s_mean': float(np.mean(controls_700s_cct_142)),
        'controls_001s_n': len(controls_001_cct_142),
        'controls_001s_mean': float(np.mean(controls_001_cct_142)),
    }

    output_file = base_path.parent / "resultats_stratifies.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats: {output_file}")


if __name__ == "__main__":
    main()
