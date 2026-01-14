#!/usr/bin/env python3
"""
Analyse iEEG pour tester la prédiction f_Φ = 147 Hz

Ce script analyse les données iEEG (ECoG/sEEG) pour détecter un pic spectral
à 147 Hz corrélé avec les états de conscience.

Dataset cible: Open multi-center iEEG dataset (consciousness study)
https://www.nature.com/articles/s41597-025-04833-z

Auteure: Aurélie Assouline
Date: Janvier 2026
ORCID: 0009-0004-8557-8772
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr, ttest_ind, zscore
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constantes HOLOTHEIA
PHI = 1.618033988749895
D_STAR = 2.3107
F_PHI = 432 / (PHI ** D_STAR)  # ≈ 147 Hz

print(f"Fréquence prédite: f_Φ = {F_PHI:.2f} Hz")


class AnalyseHolotheiaIEEG:
    """Analyse iEEG pour tester la prédiction 147 Hz."""

    def __init__(self, sampling_rate=1000):
        """
        Args:
            sampling_rate: Fréquence d'échantillonnage en Hz
        """
        self.fs = sampling_rate
        self.f_phi = F_PHI
        self.band_of_interest = (140, 155)  # Bande autour de 147 Hz
        self.control_band = (120, 135)  # Bande contrôle

    def preprocess(self, data, notch_freq=50):
        """
        Prétraitement du signal iEEG.

        Args:
            data: Signal brut (n_channels, n_samples) ou (n_samples,)
            notch_freq: Fréquence du secteur à filtrer (50 ou 60 Hz)

        Returns:
            Signal prétraité
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        processed = np.zeros_like(data)

        for i in range(data.shape[0]):
            # Filtre notch pour le secteur et ses harmoniques
            signal_clean = data[i].copy()
            for harmonic in [1, 2, 3]:  # 50, 100, 150 Hz
                freq = notch_freq * harmonic
                if freq < self.fs / 2:
                    b, a = signal.iirnotch(freq, Q=30, fs=self.fs)
                    signal_clean = signal.filtfilt(b, a, signal_clean)

            # Filtre passe-haut pour enlever les dérives
            b, a = signal.butter(4, 1, btype='high', fs=self.fs)
            signal_clean = signal.filtfilt(b, a, signal_clean)

            processed[i] = signal_clean

        return processed if processed.shape[0] > 1 else processed[0]

    def compute_psd(self, data, nperseg=None):
        """
        Calcule la densité spectrale de puissance.

        Args:
            data: Signal prétraité
            nperseg: Longueur de segment pour Welch

        Returns:
            freqs, psd
        """
        if nperseg is None:
            nperseg = min(self.fs * 2, len(data))

        freqs, psd = signal.welch(data, fs=self.fs, nperseg=nperseg,
                                   noverlap=nperseg//2)
        return freqs, psd

    def extract_band_power(self, freqs, psd, band):
        """
        Extrait la puissance dans une bande de fréquence.

        Args:
            freqs: Vecteur de fréquences
            psd: Densité spectrale
            band: Tuple (f_low, f_high)

        Returns:
            Puissance moyenne dans la bande
        """
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.mean(psd[mask]) if np.any(mask) else 0

    def detect_peak(self, freqs, psd, target_freq=None, bandwidth=10):
        """
        Détecte un pic significatif autour de la fréquence cible.

        Args:
            freqs: Vecteur de fréquences
            psd: Densité spectrale
            target_freq: Fréquence cible (défaut: f_Φ)
            bandwidth: Largeur de bande pour la recherche

        Returns:
            dict avec peak_freq, peak_power, z_score, is_significant
        """
        if target_freq is None:
            target_freq = self.f_phi

        # Trouver le pic dans la bande d'intérêt
        mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
        if not np.any(mask):
            return {'peak_freq': None, 'peak_power': 0, 'z_score': 0, 'is_significant': False}

        psd_band = psd[mask]
        freqs_band = freqs[mask]

        peak_idx = np.argmax(psd_band)
        peak_freq = freqs_band[peak_idx]
        peak_power = psd_band[peak_idx]

        # Calculer le z-score par rapport au bruit de fond
        # Utiliser les bandes adjacentes comme référence
        baseline_mask = ((freqs >= target_freq - 30) & (freqs < target_freq - bandwidth)) | \
                       ((freqs > target_freq + bandwidth) & (freqs <= target_freq + 30))

        if np.any(baseline_mask):
            baseline = psd[baseline_mask]
            z_score = (peak_power - np.mean(baseline)) / (np.std(baseline) + 1e-10)
        else:
            z_score = 0

        is_significant = z_score > 2.5  # Seuil de significativité

        return {
            'peak_freq': peak_freq,
            'peak_power': peak_power,
            'z_score': z_score,
            'is_significant': is_significant
        }

    def analyze_trial(self, data, label=None):
        """
        Analyse un essai complet.

        Args:
            data: Signal de l'essai
            label: 'conscious' ou 'unconscious' (optionnel)

        Returns:
            dict avec les résultats
        """
        # Prétraitement
        processed = self.preprocess(data)

        # PSD
        freqs, psd = self.compute_psd(processed)

        # Détection de pic
        peak_info = self.detect_peak(freqs, psd)

        # Puissance dans les bandes
        power_147 = self.extract_band_power(freqs, psd, self.band_of_interest)
        power_control = self.extract_band_power(freqs, psd, self.control_band)

        # Ratio normalisé
        ratio = power_147 / (power_control + 1e-10)

        return {
            'freqs': freqs,
            'psd': psd,
            'peak_freq': peak_info['peak_freq'],
            'peak_power': peak_info['peak_power'],
            'z_score': peak_info['z_score'],
            'is_significant': peak_info['is_significant'],
            'power_147': power_147,
            'power_control': power_control,
            'ratio': ratio,
            'label': label
        }

    def compare_conditions(self, results_conscious, results_unconscious):
        """
        Compare les conditions conscient vs inconscient.

        Args:
            results_conscious: Liste de résultats pour condition consciente
            results_unconscious: Liste de résultats pour condition inconsciente

        Returns:
            dict avec statistiques de comparaison
        """
        # Extraire les métriques
        power_conscious = [r['power_147'] for r in results_conscious]
        power_unconscious = [r['power_147'] for r in results_unconscious]

        ratio_conscious = [r['ratio'] for r in results_conscious]
        ratio_unconscious = [r['ratio'] for r in results_unconscious]

        z_conscious = [r['z_score'] for r in results_conscious]
        z_unconscious = [r['z_score'] for r in results_unconscious]

        # Tests statistiques
        t_power, p_power = ttest_ind(power_conscious, power_unconscious)
        t_ratio, p_ratio = ttest_ind(ratio_conscious, ratio_unconscious)
        t_z, p_z = ttest_ind(z_conscious, z_unconscious)

        # Pourcentage de pics significatifs
        pct_sig_conscious = np.mean([r['is_significant'] for r in results_conscious]) * 100
        pct_sig_unconscious = np.mean([r['is_significant'] for r in results_unconscious]) * 100

        return {
            'power_conscious_mean': np.mean(power_conscious),
            'power_unconscious_mean': np.mean(power_unconscious),
            'power_t': t_power,
            'power_p': p_power,
            'ratio_conscious_mean': np.mean(ratio_conscious),
            'ratio_unconscious_mean': np.mean(ratio_unconscious),
            'ratio_t': t_ratio,
            'ratio_p': p_ratio,
            'z_conscious_mean': np.mean(z_conscious),
            'z_unconscious_mean': np.mean(z_unconscious),
            'z_t': t_z,
            'z_p': p_z,
            'pct_significant_conscious': pct_sig_conscious,
            'pct_significant_unconscious': pct_sig_unconscious,
            'hypothesis_supported': p_power < 0.05 and np.mean(power_conscious) > np.mean(power_unconscious)
        }

    def plot_results(self, results, title="Analyse 147 Hz", save_path=None):
        """
        Visualise les résultats.

        Args:
            results: Résultats de analyze_trial
            title: Titre du graphique
            save_path: Chemin pour sauvegarder (optionnel)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # PSD complet
        ax1 = axes[0, 0]
        ax1.semilogy(results['freqs'], results['psd'], 'b-', linewidth=1)
        ax1.axvline(x=self.f_phi, color='r', linestyle='--', label=f'f_Φ = {self.f_phi:.1f} Hz')
        ax1.axvspan(self.band_of_interest[0], self.band_of_interest[1],
                    alpha=0.3, color='red', label='Bande d\'intérêt')
        ax1.set_xlabel('Fréquence (Hz)')
        ax1.set_ylabel('PSD (log)')
        ax1.set_title('Densité Spectrale de Puissance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 250)

        # Zoom sur 100-200 Hz
        ax2 = axes[0, 1]
        mask = (results['freqs'] >= 100) & (results['freqs'] <= 200)
        ax2.plot(results['freqs'][mask], results['psd'][mask], 'b-', linewidth=2)
        ax2.axvline(x=self.f_phi, color='r', linestyle='--', linewidth=2)
        ax2.axvspan(self.band_of_interest[0], self.band_of_interest[1],
                    alpha=0.3, color='red')
        if results['peak_freq']:
            ax2.plot(results['peak_freq'], results['peak_power'], 'ro', markersize=10)
            ax2.annotate(f"Pic: {results['peak_freq']:.1f} Hz\nz = {results['z_score']:.2f}",
                        xy=(results['peak_freq'], results['peak_power']),
                        xytext=(10, 10), textcoords='offset points')
        ax2.set_xlabel('Fréquence (Hz)')
        ax2.set_ylabel('PSD')
        ax2.set_title('Zoom High-Gamma (100-200 Hz)')
        ax2.grid(True, alpha=0.3)

        # Métriques
        ax3 = axes[1, 0]
        metrics = ['Power 147 Hz', 'Power Control', 'Ratio', 'Z-score']
        values = [results['power_147'], results['power_control'],
                  results['ratio'], results['z_score']]
        colors = ['#E94F37' if results['is_significant'] else '#2E86AB'] * 4
        colors[3] = '#28A745' if results['z_score'] > 2.5 else '#DC3545'

        bars = ax3.bar(metrics, values, color=colors)
        ax3.axhline(y=2.5, color='green', linestyle='--', label='Seuil z = 2.5')
        ax3.set_ylabel('Valeur')
        ax3.set_title('Métriques')
        ax3.legend()

        # Verdict
        ax4 = axes[1, 1]
        ax4.axis('off')
        verdict_text = f"""
        RÉSULTATS ANALYSE 147 Hz
        ========================

        Fréquence prédite: {self.f_phi:.2f} Hz
        Fréquence du pic:  {results['peak_freq']:.2f} Hz

        Z-score:           {results['z_score']:.2f}
        Significatif:      {'OUI ✓' if results['is_significant'] else 'NON ✗'}

        Ratio 147/control: {results['ratio']:.2f}

        VERDICT: {'HYPOTHÈSE SUPPORTÉE' if results['is_significant'] else 'PAS DE PIC SIGNIFICATIF'}
        """
        ax4.text(0.1, 0.5, verdict_text, fontsize=12, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure sauvegardée: {save_path}")

        plt.show()

    def plot_comparison(self, comparison_results, save_path=None):
        """
        Visualise la comparaison conscient vs inconscient.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Power comparison
        ax1 = axes[0]
        conditions = ['Conscient', 'Inconscient']
        powers = [comparison_results['power_conscious_mean'],
                  comparison_results['power_unconscious_mean']]
        colors = ['#28A745', '#DC3545']
        ax1.bar(conditions, powers, color=colors)
        ax1.set_ylabel('Puissance moyenne 140-155 Hz')
        ax1.set_title(f"Puissance à 147 Hz\np = {comparison_results['power_p']:.4f}")

        # Ratio comparison
        ax2 = axes[1]
        ratios = [comparison_results['ratio_conscious_mean'],
                  comparison_results['ratio_unconscious_mean']]
        ax2.bar(conditions, ratios, color=colors)
        ax2.set_ylabel('Ratio 147 Hz / Contrôle')
        ax2.set_title(f"Ratio normalisé\np = {comparison_results['ratio_p']:.4f}")

        # Percentage significant
        ax3 = axes[2]
        pcts = [comparison_results['pct_significant_conscious'],
                comparison_results['pct_significant_unconscious']]
        ax3.bar(conditions, pcts, color=colors)
        ax3.set_ylabel('% pics significatifs')
        ax3.set_title('Pics significatifs (z > 2.5)')
        ax3.set_ylim(0, 100)

        verdict = "✓ HYPOTHÈSE SUPPORTÉE" if comparison_results['hypothesis_supported'] else "✗ HYPOTHÈSE NON SUPPORTÉE"
        fig.suptitle(f"Comparaison Conscient vs Inconscient - {verdict}",
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


def demo_synthetic():
    """Démonstration avec données synthétiques."""
    print("\n" + "="*60)
    print("DÉMONSTRATION AVEC DONNÉES SYNTHÉTIQUES")
    print("="*60)

    # Paramètres
    fs = 1000  # Hz
    duration = 10  # secondes
    t = np.arange(0, duration, 1/fs)

    # Créer un signal avec un pic à 147 Hz (condition "consciente")
    np.random.seed(42)
    noise = np.random.randn(len(t)) * 0.5

    # Signal conscient: bruit + pic à 147 Hz
    signal_conscious = noise + 0.3 * np.sin(2 * np.pi * F_PHI * t)

    # Signal inconscient: juste du bruit
    signal_unconscious = noise.copy()

    # Analyser
    analyzer = AnalyseHolotheiaIEEG(sampling_rate=fs)

    print("\nAnalyse condition CONSCIENTE (avec pic 147 Hz):")
    results_conscious = analyzer.analyze_trial(signal_conscious, label='conscious')
    print(f"  Pic détecté à: {results_conscious['peak_freq']:.1f} Hz")
    print(f"  Z-score: {results_conscious['z_score']:.2f}")
    print(f"  Significatif: {results_conscious['is_significant']}")

    print("\nAnalyse condition INCONSCIENTE (sans pic):")
    results_unconscious = analyzer.analyze_trial(signal_unconscious, label='unconscious')
    print(f"  Pic détecté à: {results_unconscious['peak_freq']:.1f} Hz")
    print(f"  Z-score: {results_unconscious['z_score']:.2f}")
    print(f"  Significatif: {results_unconscious['is_significant']}")

    # Visualiser
    analyzer.plot_results(results_conscious, "Condition Consciente (Synthétique)")

    return analyzer, results_conscious, results_unconscious


def load_bids_ieeg(bids_path):
    """
    Charge des données iEEG au format BIDS.

    Args:
        bids_path: Chemin vers le dossier BIDS

    Returns:
        Dictionnaire avec les données
    """
    try:
        import mne
        import mne_bids

        # Cette fonction sera à adapter selon la structure exacte du dataset
        print(f"Chargement des données BIDS depuis: {bids_path}")

        # Exemple de chargement (à adapter)
        # raw = mne_bids.read_raw_bids(bids_path)
        # return raw

        print("Note: Adapter cette fonction selon la structure du dataset")
        return None

    except ImportError:
        print("Installer mne-bids: pip install mne-bids")
        return None


def main():
    """Point d'entrée principal."""
    print("="*60)
    print("HOLOTHEIA - Analyse iEEG pour prédiction 147 Hz")
    print("="*60)
    print(f"\nPrédiction théorique:")
    print(f"  D* = {D_STAR}")
    print(f"  φ = {PHI}")
    print(f"  f_Φ = 432 / φ^D* = {F_PHI:.2f} Hz")
    print(f"\nBande d'intérêt: 140-155 Hz")
    print(f"Seuil de significativité: z > 2.5")

    # Démonstration
    demo_synthetic()

    print("\n" + "="*60)
    print("POUR ANALYSER DES DONNÉES RÉELLES:")
    print("="*60)
    print("""
1. Télécharger le dataset iEEG:
   https://openneuro.org/datasets/ds004XXX

2. Charger les données:
   analyzer = AnalyseHolotheiaIEEG(sampling_rate=1000)
   results = analyzer.analyze_trial(your_data)

3. Comparer conscient vs inconscient:
   comparison = analyzer.compare_conditions(
       results_conscious_list,
       results_unconscious_list
   )
   analyzer.plot_comparison(comparison)
""")


if __name__ == '__main__':
    main()
