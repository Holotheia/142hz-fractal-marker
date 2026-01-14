#!/usr/bin/env python3
"""
Analyse Cogitate iEEG - Test de la prédiction à 142 Hz
======================================================

Ce script analyse les données iEEG du projet Cogitate pour tester
la prédiction HOLOTHEIA : un pic spectral à f_Φ = 142 Hz devrait
être présent lors de la perception consciente.

Prédiction théorique:
    f_Φ = 432 / φ^D* = 432 / 1.618^2.31 ≈ 142 Hz

Auteur: Aurélie Assouline (Holotheia.ai)
Date: Janvier 2026
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

print("=" * 70)
print("ANALYSE COGITATE iEEG - TEST PRÉDICTION 142 Hz")
print("=" * 70)
print(f"\nConstantes théoriques:")
print(f"  φ (golden ratio) = {PHI:.10f}")
print(f"  D* (dimension optimale) = {D_STAR}")
print(f"  f_Φ = 432 / φ^D* = {F_PHI:.2f} Hz")
print(f"\nBande d'intérêt: 135-150 Hz")
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
    Calcule la puissance spectrale autour de 142 Hz

    Returns:
        peak_freq: fréquence du pic dans la bande
        peak_power: puissance au pic
        z_score: z-score par rapport à la baseline (100-135 Hz)
        psd: spectre complet
        freqs: vecteur de fréquences
    """
    # Band-pass filter 80-200 Hz pour isoler le high-gamma
    nyq = fs / 2
    if nyq < 200:
        high_freq = nyq * 0.9
    else:
        high_freq = 200

    b, a = signal.butter(4, [80/nyq, high_freq/nyq], btype='band')

    # Moyenne sur les canaux si multi-canal
    if data.ndim > 1:
        data_mean = np.nanmean(data, axis=0)
    else:
        data_mean = data

    # Filtrer
    try:
        filtered = signal.filtfilt(b, a, data_mean)
    except:
        return None, None, None, None, None

    # PSD avec Welch
    nperseg = min(int(fs * 2), len(filtered) // 4)
    if nperseg < 256:
        nperseg = min(256, len(filtered))

    freqs, psd = signal.welch(filtered, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    # Bande d'intérêt (135-150 Hz)
    mask_interest = (freqs >= band_low) & (freqs <= band_high)
    if not np.any(mask_interest):
        return None, None, None, psd, freqs

    psd_interest = psd[mask_interest]
    freqs_interest = freqs[mask_interest]

    peak_idx = np.argmax(psd_interest)
    peak_freq = freqs_interest[peak_idx]
    peak_power = psd_interest[peak_idx]

    # Baseline (100-135 Hz) pour z-score
    mask_baseline = (freqs >= 100) & (freqs < band_low)
    if np.any(mask_baseline):
        baseline = psd[mask_baseline]
        z_score = (peak_power - np.mean(baseline)) / (np.std(baseline) + 1e-10)
    else:
        z_score = 0

    return peak_freq, peak_power, z_score, psd, freqs


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

    # Analyser les epochs par condition
    results_conscious = []
    results_unconscious = []

    epoch_duration = 1.0  # 1 seconde après stimulus

    # Epochs conscients
    for evt in conscious[:50]:  # Max 50 pour rapidité
        start_sample = int(evt['onset'] * fs)
        end_sample = start_sample + int(epoch_duration * fs)

        if end_sample > data.shape[1]:
            continue

        epoch = data[:, start_sample:end_sample]
        peak_f, peak_p, z, _, _ = compute_psd_142Hz(epoch, fs)

        if peak_f is not None:
            results_conscious.append({
                'peak_freq': peak_f,
                'peak_power': peak_p,
                'z_score': z
            })

    # Epochs inconscients
    for evt in unconscious[:50]:
        start_sample = int(evt['onset'] * fs)
        end_sample = start_sample + int(epoch_duration * fs)

        if end_sample > data.shape[1]:
            continue

        epoch = data[:, start_sample:end_sample]
        peak_f, peak_p, z, _, _ = compute_psd_142Hz(epoch, fs)

        if peak_f is not None:
            results_unconscious.append({
                'peak_freq': peak_f,
                'peak_power': peak_p,
                'z_score': z
            })

    # Statistiques
    if results_conscious and results_unconscious:
        z_conscious = [r['z_score'] for r in results_conscious]
        z_unconscious = [r['z_score'] for r in results_unconscious]

        mean_z_con = np.mean(z_conscious)
        mean_z_uncon = np.mean(z_unconscious)

        # Test statistique
        if len(z_conscious) >= 3 and len(z_unconscious) >= 3:
            stat, p_value = mannwhitneyu(z_conscious, z_unconscious, alternative='greater')
        else:
            p_value = 1.0

        freq_conscious = np.mean([r['peak_freq'] for r in results_conscious])
        freq_unconscious = np.mean([r['peak_freq'] for r in results_unconscious])

        print(f"\n  RÉSULTATS {subject_id}:")
        print(f"  - Z-score moyen (conscient): {mean_z_con:.3f}")
        print(f"  - Z-score moyen (inconscient): {mean_z_uncon:.3f}")
        print(f"  - Différence: {mean_z_con - mean_z_uncon:.3f}")
        print(f"  - p-value (Mann-Whitney): {p_value:.4f}")
        print(f"  - Fréquence pic conscient: {freq_conscious:.1f} Hz")
        print(f"  - Fréquence pic inconscient: {freq_unconscious:.1f} Hz")

        significant = p_value < 0.05 and mean_z_con > mean_z_uncon
        print(f"  - SIGNIFICATIF: {'OUI ✓' if significant else 'NON'}")

        return {
            'subject': subject_id,
            'n_conscious': len(results_conscious),
            'n_unconscious': len(results_unconscious),
            'mean_z_conscious': mean_z_con,
            'mean_z_unconscious': mean_z_uncon,
            'difference': mean_z_con - mean_z_uncon,
            'p_value': p_value,
            'freq_conscious': freq_conscious,
            'freq_unconscious': freq_unconscious,
            'significant': significant
        }

    return None


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

    # Résumé global
    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL - TEST PRÉDICTION 142 Hz")
    print("=" * 70)

    if all_results:
        n_significant = sum(1 for r in all_results if r['significant'])
        mean_diff = np.mean([r['difference'] for r in all_results])
        mean_z_con = np.mean([r['mean_z_conscious'] for r in all_results])
        mean_z_uncon = np.mean([r['mean_z_unconscious'] for r in all_results])

        # Meta-analyse: combiner les p-values (Fisher's method)
        p_values = [r['p_value'] for r in all_results]
        chi2_stat = -2 * sum(np.log(p + 1e-10) for p in p_values)
        from scipy.stats import chi2
        combined_p = 1 - chi2.cdf(chi2_stat, 2 * len(p_values))

        print(f"\nNombre de sujets analysés: {len(all_results)}")
        print(f"Sujets avec effet significatif: {n_significant}/{len(all_results)}")
        print(f"\nMoyennes globales:")
        print(f"  - Z-score conscient: {mean_z_con:.3f}")
        print(f"  - Z-score inconscient: {mean_z_uncon:.3f}")
        print(f"  - Différence moyenne: {mean_diff:.3f}")
        print(f"  - p-value combinée (Fisher): {combined_p:.6f}")

        # Interprétation
        print("\n" + "-" * 70)
        print("INTERPRÉTATION:")
        print("-" * 70)

        if combined_p < 0.05 and mean_diff > 0:
            print("★★★ RÉSULTAT POSITIF ★★★")
            print("La puissance à 142 Hz est significativement plus élevée")
            print("lors de la perception consciente.")
            print("→ La prédiction HOLOTHEIA est SUPPORTÉE par ces données.")
        elif combined_p < 0.10 and mean_diff > 0:
            print("◇ TENDANCE POSITIVE ◇")
            print("Une tendance vers plus de puissance à 142 Hz")
            print("est observée en condition consciente.")
            print("→ Résultats encourageants mais non conclusifs.")
        elif mean_diff > 0:
            print("○ DIRECTION ATTENDUE ○")
            print("La différence va dans le sens prédit mais n'est pas significative.")
            print("→ Plus de données nécessaires pour conclure.")
        else:
            print("✗ RÉSULTAT NÉGATIF ✗")
            print("Aucune différence significative observée.")
            print("→ La prédiction n'est pas supportée par ces données.")

        print(f"\nPrédiction théorique: f_Φ = {F_PHI:.2f} Hz")
        mean_freq = np.mean([r['freq_conscious'] for r in all_results])
        print(f"Fréquence observée (conscient): {mean_freq:.1f} Hz")
        print(f"Écart: {abs(mean_freq - F_PHI):.1f} Hz ({abs(mean_freq - F_PHI)/F_PHI*100:.1f}%)")

        # Sauvegarder les résultats
        output_path = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/resultats_cogitate_142Hz.json")

        output = {
            'prediction': {
                'f_phi': F_PHI,
                'D_star': D_STAR,
                'phi': PHI
            },
            'results_by_subject': all_results,
            'global': {
                'n_subjects': len(all_results),
                'n_significant': n_significant,
                'mean_z_conscious': mean_z_con,
                'mean_z_unconscious': mean_z_uncon,
                'mean_difference': mean_diff,
                'combined_p_value': combined_p,
                'significant': bool(combined_p < 0.05 and mean_diff > 0)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nRésultats sauvegardés: {output_path}")

    else:
        print("Aucun résultat à analyser.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
