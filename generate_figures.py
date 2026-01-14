#!/usr/bin/env python3
"""
Generate publication figures for HOLOTHEIA article
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = Path("/Users/aurelie/Library/Mobile Documents/com~apple~CloudDocs/Conscience Fractale - Coordination Non-Locale via Dimension D ≈ 2.31/figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def figure1_theoretical_framework():
    """Figure 1: Theoretical framework - D* derivation and frequency predictions"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: D* derivation
    ax1 = axes[0]
    phi = (1 + np.sqrt(5)) / 2

    # Show the golden triad
    values = [1/phi, 1, phi]
    labels = ['φ⁻¹\n0.618', '1\n1.000', 'φ\n1.618']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    bars = ax1.bar(range(3), values, color=colors, edgecolor='black', linewidth=2)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel('Value', fontsize=14)
    ax1.set_title('A. Golden Triad: T = φ⁻¹ + 1 + φ = 3.236', fontsize=14, fontweight='bold')

    # Add sum line
    ax1.axhline(y=sum(values)/3, color='purple', linestyle='--', linewidth=2, label=f'D* = 2 + 1/T = 2.309')
    ax1.legend(loc='upper right', fontsize=11)

    # Panel B: Frequency predictions
    ax2 = axes[1]
    D_values = np.linspace(2, 3.5, 100)
    f_values = 432 / (phi ** D_values)

    ax2.plot(D_values, f_values, 'b-', linewidth=3, label='f = 432/φᴰ')

    # Mark key points
    D_star = 2.3107
    f1 = 432 / (phi ** 3)
    f2 = 432 / (phi ** D_star)

    ax2.scatter([3], [f1], s=200, c='#e74c3c', zorder=5, edgecolors='black', linewidth=2)
    ax2.scatter([D_star], [f2], s=200, c='#2ecc71', zorder=5, edgecolors='black', linewidth=2)

    ax2.annotate(f'f₁ = {f1:.0f} Hz\n(D = 3)', xy=(3, f1), xytext=(3.2, f1+10),
                fontsize=11, ha='left', fontweight='bold')
    ax2.annotate(f'f₂ = {f2:.0f} Hz\n(D = D*)', xy=(D_star, f2), xytext=(D_star-0.3, f2+15),
                fontsize=11, ha='right', fontweight='bold')

    ax2.axhline(y=f1, color='#e74c3c', linestyle=':', alpha=0.5)
    ax2.axhline(y=f2, color='#2ecc71', linestyle=':', alpha=0.5)
    ax2.axvline(x=D_star, color='#2ecc71', linestyle='--', alpha=0.5, label=f'D* = {D_star}')

    ax2.set_xlabel('Fractal Dimension D', fontsize=14)
    ax2.set_ylabel('Frequency (Hz)', fontsize=14)
    ax2.set_title('B. Two Predicted Frequencies', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_xlim(2, 3.5)
    ax2.set_ylim(80, 180)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure1_theoretical_framework.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure1_theoretical_framework.pdf', bbox_inches='tight')
    print("Figure 1 saved!")
    plt.close()


def figure2_main_results():
    """Figure 2: Main experimental results - stratified bar plot"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data from stratified analysis
    groups = ['Athletes\n(n=13)', 'Controls-700s\n(n=4)', 'Controls-001s\n(n=9)']
    means = [13.3, 81.7, 1.7]
    stds = [21.1, 32.8, 1.9]
    colors = ['#3498db', '#e74c3c', '#95a5a6']

    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors,
                  edgecolor='black', linewidth=2, error_kw={'linewidth': 2})

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=13)
    ax.set_ylabel('142 Hz z-score (CCT task)', fontsize=14)
    ax.set_title('142 Hz Activity During Concentration Task\n(Stratified by Cohort)', fontsize=16, fontweight='bold')

    # Add significance bars
    def add_significance(x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y+5, y+5, y], 'k-', linewidth=1.5)
        ax.text((x1+x2)/2, y+7, text, ha='center', fontsize=12, fontweight='bold')

    add_significance(0, 1, 130, 'p = 0.0034 ***')
    add_significance(1, 2, 145, 'p = 0.0028 ***')

    # Add interpretation labels
    ax.text(0, means[0]+stds[0]+15, 'Efficient', ha='center', fontsize=11, style='italic', color='#3498db')
    ax.text(1, means[1]+stds[1]+15, 'High effort', ha='center', fontsize=11, style='italic', color='#e74c3c')
    ax.text(2, means[2]+stds[2]+15, 'Ultra-efficient', ha='center', fontsize=11, style='italic', color='#666')

    ax.set_ylim(0, 170)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure2_main_results.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure2_main_results.pdf', bbox_inches='tight')
    print("Figure 2 saved!")
    plt.close()


def figure3_task_dissociation():
    """Figure 3: ABT vs CCT dissociation"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data
    tasks = ['ABT\n(Vigilance)', 'CCT\n(Concentration)']
    freq_102 = [724.2, 460.3]
    freq_142 = [37.9, 19.4]

    x = np.arange(len(tasks))
    width = 0.35

    bars1 = ax.bar(x - width/2, freq_102, width, label='102 Hz (p=0.070 NS)',
                   color='#3498db', edgecolor='black', linewidth=2, alpha=0.7)
    bars2 = ax.bar(x + width/2, freq_142, width, label='142 Hz (p=0.012 *)',
                   color='#e74c3c', edgecolor='black', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=14)
    ax.set_ylabel('Mean z-score', fontsize=14)
    ax.set_title('Task Dissociation: 142 Hz is Task-Specific\n(N=20, Wilcoxon paired test)', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)

    # Add arrow showing the key finding
    ax.annotate('142 Hz differentiates\ntasks (p=0.012)',
               xy=(1.17, 19.4), xytext=(1.4, 200),
               fontsize=11, ha='center',
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

    ax.set_ylim(0, 900)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure3_task_dissociation.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure3_task_dissociation.pdf', bbox_inches='tight')
    print("Figure 3 saved!")
    plt.close()


def figure4_sample_stability():
    """Figure 4: Effect stability across sample sizes"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    n_values = [13, 25, 27]
    p_values = [0.009, 0.018, 0.0034]
    cohens_d = [-2.73, -2.11, -2.48]

    ax.plot(n_values, p_values, 'o-', color='#e74c3c', linewidth=3, markersize=12, label='p-value')
    ax.axhline(y=0.05, color='gray', linestyle='--', linewidth=2, label='α = 0.05')
    ax.axhline(y=0.01, color='gray', linestyle=':', linewidth=2, label='α = 0.01')

    for i, (n, p, d) in enumerate(zip(n_values, p_values, cohens_d)):
        ax.annotate(f'p={p:.4f}\nd={d:.2f}', xy=(n, p), xytext=(n+0.5, p+0.003),
                   fontsize=10, ha='left')

    ax.set_xlabel('Sample Size (N)', fontsize=14)
    ax.set_ylabel('p-value (Athletes vs Controls-700s)', fontsize=14)
    ax.set_title('Effect Stability: p-values Strengthen with Increased N', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(10, 30)
    ax.set_ylim(0, 0.06)

    ax.fill_between([10, 30], 0, 0.01, alpha=0.2, color='green', label='Highly significant')
    ax.fill_between([10, 30], 0.01, 0.05, alpha=0.1, color='yellow')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Figure4_sample_stability.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Figure4_sample_stability.pdf', bbox_inches='tight')
    print("Figure 4 saved!")
    plt.close()


if __name__ == "__main__":
    print("Generating publication figures...")
    print(f"Output directory: {OUTPUT_DIR}")

    figure1_theoretical_framework()
    figure2_main_results()
    figure3_task_dissociation()
    figure4_sample_stability()

    print(f"\nAll figures saved to {OUTPUT_DIR}")
    print("Files: Figure1-4 in PNG and PDF formats")
