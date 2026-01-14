#!/usr/bin/env python3
"""
Analyse comparative ABT vs CCT - Test de dissociation 102 Hz / 142 Hz
Hypothèse:
  - 102 Hz présent dans les deux tâches (perception générale)
  - 142 Hz spécifique à CCT (concentration/intégration)
"""

import numpy as np
from scipy import signal
from scipy.stats import wilcoxon, mannwhitneyu
from pathlib import Path
import json

PHI = (1 + np.sqrt(5)) / 2
D_STAR = 2.3107
F1 = 432 / (PHI ** 3)        # 102 Hz
F2 = 432 / (PHI ** D_STAR)   # 142 Hz


def read_cdt_file(cdt_path, desc_path):
    """Lit un fichier CDT avec son descripteur (.dpo ou .dpa)"""
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
    """Analyse un sujet pour ABT et CCT"""
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

        desc_path = dpo_path if dpo_path.exists() else dpa_path
        if not desc_path.exists():
            continue

        try:
            data, fs = read_cdt_file(cdt_path, desc_path)

            z_102 = compute_psd_peak(data, fs, 102, bandwidth=20)
            z_142 = compute_psd_peak(data, fs, 142, bandwidth=15)

            results[exp_name] = {
                'z_102': z_102,
                'z_142': z_142,
            }
        except Exception as e:
            pass

    return results


def main():
    base_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/datasets/elite_athletes")

    subjects = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith(('Athlete_', 'Control_')):
            subjects.append(folder)

    print("="*70)
    print("ANALYSE DISSOCIATION ABT vs CCT")
    print("="*70)
    print(f"Hypothèse: 142 Hz spécifique à la concentration (CCT)")
    print(f"           102 Hz présent dans les deux tâches")
    print(f"\nf₁ = {F1:.2f} Hz (perception)")
    print(f"f₂ = {F2:.2f} Hz (intégration/concentration)")
    print(f"\nSujets trouvés: {len(subjects)}")

    # Collecter données appariées
    paired_102_abt = []
    paired_102_cct = []
    paired_142_abt = []
    paired_142_cct = []

    athletes_ratio_142 = []  # CCT/ABT ratio for 142 Hz
    controls_ratio_142 = []

    print("\n" + "-"*70)
    print("DONNÉES PAR SUJET (ABT vs CCT)")
    print("-"*70)
    print(f"{'Sujet':<20} {'102 ABT':>10} {'102 CCT':>10} {'142 ABT':>10} {'142 CCT':>10} {'Ratio 142':>12}")
    print("-"*70)

    for subject_path in sorted(subjects):
        subject_id = subject_path.name.split('_')[1]
        group = subject_path.name.split('_')[0]

        result = analyze_subject(subject_path, subject_id)

        # Only include if both ABT and CCT available
        if result['ABT'] and result['CCT']:
            z102_abt = result['ABT'].get('z_102')
            z102_cct = result['CCT'].get('z_102')
            z142_abt = result['ABT'].get('z_142')
            z142_cct = result['CCT'].get('z_142')

            if all(v is not None for v in [z102_abt, z102_cct, z142_abt, z142_cct]):
                paired_102_abt.append(z102_abt)
                paired_102_cct.append(z102_cct)
                paired_142_abt.append(z142_abt)
                paired_142_cct.append(z142_cct)

                # Ratio CCT/ABT pour 142 Hz (>1 = plus fort en CCT)
                ratio_142 = z142_cct / z142_abt if z142_abt > 0.1 else np.nan

                if not np.isnan(ratio_142):
                    if group == 'Athlete':
                        athletes_ratio_142.append(ratio_142)
                    else:
                        controls_ratio_142.append(ratio_142)

                print(f"{group}_{subject_id:<12} {z102_abt:>10.1f} {z102_cct:>10.1f} {z142_abt:>10.1f} {z142_cct:>10.1f} {ratio_142:>12.2f}")

    # Analyse statistique
    print("\n" + "="*70)
    print("ANALYSE STATISTIQUE - TEST APPARIÉ (Wilcoxon)")
    print("="*70)

    n_paired = len(paired_102_abt)
    print(f"\nSujets avec données complètes ABT+CCT: {n_paired}")

    if n_paired >= 5:
        # Test 102 Hz: ABT vs CCT
        print(f"\n--- 102 Hz (perception générale) ---")
        print(f"  ABT: {np.mean(paired_102_abt):.1f} ± {np.std(paired_102_abt):.1f}")
        print(f"  CCT: {np.mean(paired_102_cct):.1f} ± {np.std(paired_102_cct):.1f}")

        stat, p_102 = wilcoxon(paired_102_abt, paired_102_cct)
        print(f"  Wilcoxon p-value: {p_102:.4f}")
        print(f"  Différence ABT vs CCT: {'OUI' if p_102 < 0.05 else 'NON'}")

        # Test 142 Hz: ABT vs CCT
        print(f"\n--- 142 Hz (intégration/concentration) ---")
        print(f"  ABT: {np.mean(paired_142_abt):.1f} ± {np.std(paired_142_abt):.1f}")
        print(f"  CCT: {np.mean(paired_142_cct):.1f} ± {np.std(paired_142_cct):.1f}")

        stat, p_142 = wilcoxon(paired_142_abt, paired_142_cct)
        print(f"  Wilcoxon p-value: {p_142:.4f}")
        print(f"  Différence ABT vs CCT: {'OUI' if p_142 < 0.05 else 'NON'}")

        # Test de dissociation
        print("\n" + "="*70)
        print("TEST DE DISSOCIATION")
        print("="*70)

        # Calcul des ratios CCT/ABT
        ratios_102 = [c/a if a > 0.1 else np.nan for a, c in zip(paired_102_abt, paired_102_cct)]
        ratios_142 = [c/a if a > 0.1 else np.nan for a, c in zip(paired_142_abt, paired_142_cct)]

        ratios_102_clean = [r for r in ratios_102 if not np.isnan(r)]
        ratios_142_clean = [r for r in ratios_142 if not np.isnan(r)]

        print(f"\nRatio CCT/ABT (>1 = plus fort en concentration):")
        print(f"  102 Hz: {np.mean(ratios_102_clean):.2f} ± {np.std(ratios_102_clean):.2f}")
        print(f"  142 Hz: {np.mean(ratios_142_clean):.2f} ± {np.std(ratios_142_clean):.2f}")

        if len(ratios_102_clean) >= 5 and len(ratios_142_clean) >= 5:
            # Test si les ratios sont différents
            stat, p_dissoc = wilcoxon(ratios_102_clean[:len(ratios_142_clean)], ratios_142_clean[:len(ratios_102_clean)])
            print(f"\n  Test dissociation (ratios différents): p = {p_dissoc:.4f}")

            if p_dissoc < 0.05:
                print("  → DISSOCIATION CONFIRMÉE: 142 Hz se comporte différemment de 102 Hz")
            else:
                print("  → Pas de dissociation claire")

    # Analyse athlètes vs contrôles pour le ratio
    print("\n" + "="*70)
    print("RATIO 142 Hz CCT/ABT - ATHLÈTES vs CONTRÔLES")
    print("="*70)

    if athletes_ratio_142 and controls_ratio_142:
        print(f"\nRatio CCT/ABT à 142 Hz:")
        print(f"  Athlètes (n={len(athletes_ratio_142)}): {np.mean(athletes_ratio_142):.2f} ± {np.std(athletes_ratio_142):.2f}")
        print(f"  Contrôles (n={len(controls_ratio_142)}): {np.mean(controls_ratio_142):.2f} ± {np.std(controls_ratio_142):.2f}")

        if len(athletes_ratio_142) > 1 and len(controls_ratio_142) > 1:
            stat, p_ratio = mannwhitneyu(athletes_ratio_142, controls_ratio_142, alternative='two-sided')
            print(f"  Mann-Whitney p-value: {p_ratio:.4f}")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if n_paired >= 5:
        mean_ratio_102 = np.mean(ratios_102_clean)
        mean_ratio_142 = np.mean(ratios_142_clean)

        print(f"\n📊 Résultats clés:")
        print(f"   • 102 Hz ratio CCT/ABT = {mean_ratio_102:.2f}")
        print(f"   • 142 Hz ratio CCT/ABT = {mean_ratio_142:.2f}")

        if mean_ratio_142 < mean_ratio_102:
            print(f"\n✓ OBSERVATION: 142 Hz diminue plus en CCT que 102 Hz")
            print(f"  → Compatible avec l'efficience neurale (experts = moins d'activité)")
        elif mean_ratio_142 > mean_ratio_102:
            print(f"\n✓ OBSERVATION: 142 Hz augmente relativement plus en CCT")
            print(f"  → Compatible avec activation spécifique concentration")

    # Save results
    output = {
        'n_subjects': n_paired,
        'ratio_102_cct_abt': float(np.mean(ratios_102_clean)) if ratios_102_clean else None,
        'ratio_142_cct_abt': float(np.mean(ratios_142_clean)) if ratios_142_clean else None,
        'p_102_abt_vs_cct': float(p_102) if n_paired >= 5 else None,
        'p_142_abt_vs_cct': float(p_142) if n_paired >= 5 else None,
    }

    output_file = base_path.parent / "resultats_ABT_vs_CCT.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats sauvegardés: {output_file}")


if __name__ == "__main__":
    main()
