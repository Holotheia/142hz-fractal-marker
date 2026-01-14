#!/usr/bin/env python3
"""
Analyse temporelle 142 Hz - Début vs Fin de CCT
Manipulation interne: charge cognitive croissante avec fatigue
Hypothèse: Si 142 Hz indexe le coût d'intégration, il devrait augmenter avec la fatigue
"""

import numpy as np
from scipy import signal
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel
from pathlib import Path
import json

PHI = (1 + np.sqrt(5)) / 2
D_STAR = 2.3107
F2 = 432 / (PHI ** D_STAR)   # 142 Hz


def read_cdt_file(cdt_path, desc_path):
    """Lit un fichier CDT avec son descripteur"""
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


def compute_psd_segment(data, fs, target_freq, bandwidth=15):
    """Calcule le z-score PSD pour un segment"""
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


def analyze_temporal(subject_path, subject_id, n_segments=3):
    """Analyse temporelle d'un sujet CCT - découpage en segments"""
    cdt_path = subject_path / f"Experiment2_CCT/CCT_EEG/{subject_id}_CCT_EEG.cdt"
    dpo_path = subject_path / f"Experiment2_CCT/CCT_EEG/{subject_id}_CCT_EEG.cdt.dpo"
    dpa_path = subject_path / f"Experiment2_CCT/CCT_EEG/{subject_id}_CCT_EEG.cdt.dpa"

    if not cdt_path.exists():
        return None

    desc_path = dpo_path if dpo_path.exists() else dpa_path
    if not desc_path.exists():
        return None

    try:
        data, fs = read_cdt_file(cdt_path, desc_path)

        # Diviser en n segments temporels
        n_samples = data.shape[1]
        segment_size = n_samples // n_segments

        z_scores_142 = []
        z_scores_102 = []

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else n_samples

            segment_data = data[:, start:end]

            z_142 = compute_psd_segment(segment_data, fs, 142, bandwidth=15)
            z_102 = compute_psd_segment(segment_data, fs, 102, bandwidth=20)

            z_scores_142.append(z_142)
            z_scores_102.append(z_102)

        return {
            'z_142': z_scores_142,
            'z_102': z_scores_102,
            'duration': n_samples / fs
        }

    except Exception as e:
        return None


def main():
    base_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/datasets/elite_athletes")

    subjects = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.startswith(('Athlete_', 'Control_')):
            subjects.append(folder)

    print("="*70)
    print("ANALYSE TEMPORELLE - ÉVOLUTION 142 Hz PENDANT CCT")
    print("="*70)
    print("Hypothèse: 142 Hz augmente avec la fatigue (coût d'intégration)")
    print("           Si vrai → confirme que 142 Hz = charge, pas performance")
    print(f"\nf₂ = {F2:.2f} Hz")
    print(f"Sujets: {len(subjects)}")

    # Analyse en 3 segments: début, milieu, fin
    n_segments = 3

    athletes_early_142 = []
    athletes_late_142 = []
    controls_early_142 = []
    controls_late_142 = []

    athletes_slope_142 = []  # pente temporelle
    controls_slope_142 = []

    print("\n" + "-"*70)
    print(f"{'Sujet':<20} {'Début':>10} {'Milieu':>10} {'Fin':>10} {'Δ(Fin-Début)':>12} {'Pente':>10}")
    print("-"*70)

    all_early = []
    all_late = []
    all_slopes = []

    for subject_path in sorted(subjects):
        subject_id = subject_path.name.split('_')[1]
        group = subject_path.name.split('_')[0]

        result = analyze_temporal(subject_path, subject_id, n_segments)

        if result and all(v is not None for v in result['z_142']):
            early = result['z_142'][0]
            mid = result['z_142'][1]
            late = result['z_142'][2]

            delta = late - early
            slope = delta / (result['duration'] / 60)  # per minute

            print(f"{group}_{subject_id:<12} {early:>10.1f} {mid:>10.1f} {late:>10.1f} {delta:>12.1f} {slope:>10.2f}")

            all_early.append(early)
            all_late.append(late)
            all_slopes.append(slope)

            if group == 'Athlete':
                athletes_early_142.append(early)
                athletes_late_142.append(late)
                athletes_slope_142.append(slope)
            else:
                controls_early_142.append(early)
                controls_late_142.append(late)
                controls_slope_142.append(slope)

    # Statistiques
    print("\n" + "="*70)
    print("ANALYSE STATISTIQUE")
    print("="*70)

    n_total = len(all_early)
    print(f"\nSujets avec données complètes: {n_total}")

    if n_total >= 5:
        # Test apparié: début vs fin (tous sujets)
        print(f"\n--- TEST GLOBAL: Début vs Fin ---")
        print(f"  Début: {np.mean(all_early):.1f} ± {np.std(all_early):.1f}")
        print(f"  Fin:   {np.mean(all_late):.1f} ± {np.std(all_late):.1f}")

        stat, p_global = wilcoxon(all_early, all_late)
        print(f"  Wilcoxon p-value: {p_global:.4f}")
        print(f"  Évolution significative: {'OUI' if p_global < 0.05 else 'NON'}")

        mean_slope = np.mean(all_slopes)
        print(f"  Pente moyenne: {mean_slope:+.2f} z-score/min")
        if mean_slope > 0:
            print(f"  → 142 Hz AUGMENTE avec le temps (fatigue = plus de coût)")
        else:
            print(f"  → 142 Hz DIMINUE avec le temps (adaptation)")

    # Comparaison athlètes vs contrôles
    print(f"\n--- ATHLÈTES vs CONTRÔLES: Pente temporelle ---")

    if athletes_slope_142 and controls_slope_142:
        print(f"  Athlètes (n={len(athletes_slope_142)}): {np.mean(athletes_slope_142):+.2f} ± {np.std(athletes_slope_142):.2f} /min")
        print(f"  Contrôles (n={len(controls_slope_142)}): {np.mean(controls_slope_142):+.2f} ± {np.std(controls_slope_142):.2f} /min")

        if len(athletes_slope_142) > 1 and len(controls_slope_142) > 1:
            stat, p_slope = mannwhitneyu(athletes_slope_142, controls_slope_142, alternative='two-sided')
            print(f"  Mann-Whitney p-value: {p_slope:.4f}")

            if np.mean(athletes_slope_142) < np.mean(controls_slope_142):
                print(f"  → Athlètes: moins d'augmentation avec fatigue (résistance)")
            else:
                print(f"  → Contrôles: moins d'augmentation avec fatigue")

    # Test de la prédiction clé
    print("\n" + "="*70)
    print("TEST DE LA PRÉDICTION CAUSALE")
    print("="*70)

    if n_total >= 5:
        # Proportion de sujets avec augmentation
        n_increase = sum(1 for s in all_slopes if s > 0)
        pct_increase = 100 * n_increase / n_total

        print(f"\n  Sujets avec ↑ 142 Hz pendant CCT: {n_increase}/{n_total} ({pct_increase:.0f}%)")

        if pct_increase > 60:
            print(f"\n  ✓ CONFIRMATION: 142 Hz suit la charge cognitive")
            print(f"    → Plus on maintient la concentration, plus le coût augmente")
            print(f"    → 142 Hz indexe le COÛT d'intégration, pas la performance")
        elif pct_increase < 40:
            print(f"\n  ✓ OBSERVATION: 142 Hz diminue avec le temps")
            print(f"    → Possible adaptation / efficience progressive")
        else:
            print(f"\n  ⚠ Résultat mixte: variabilité inter-individuelle importante")

    # Conclusion
    print("\n" + "="*70)
    print("CONCLUSION POUR LA PUBLICATION")
    print("="*70)

    if n_total >= 5 and p_global < 0.10:
        print(f"""
📊 Résultat clé:
   142 Hz montre une évolution temporelle pendant la concentration
   (p = {p_global:.4f})

   Cela confirme que 142 Hz est sensible à la CHARGE cognitive,
   pas seulement à l'ÉTAT de concentration.

🎯 Implication:
   "142 Hz indexes integration cost, not task engagement per se"
""")

    # Save results
    output = {
        'n_subjects': n_total,
        'mean_early_142': float(np.mean(all_early)) if all_early else None,
        'mean_late_142': float(np.mean(all_late)) if all_late else None,
        'p_early_vs_late': float(p_global) if n_total >= 5 else None,
        'mean_slope_athletes': float(np.mean(athletes_slope_142)) if athletes_slope_142 else None,
        'mean_slope_controls': float(np.mean(controls_slope_142)) if controls_slope_142 else None,
    }

    output_file = base_path.parent / "resultats_temporal_142Hz.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nRésultats sauvegardés: {output_file}")


if __name__ == "__main__":
    main()
