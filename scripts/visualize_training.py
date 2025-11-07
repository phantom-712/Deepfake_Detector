"""
visualize_training.py

Visualize training metrics from CSV log files.
Plots loss curves and metrics (AUC, F1, Accuracy) over epochs.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_training_log(log_path):
    """Load training log CSV."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    df = pd.read_csv(log_path)
    return df


def plot_metrics(df, model_name, save_dir):
    """Plot training metrics and save figures."""
    epochs = df['epoch'].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Metrics - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, df['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, df['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. AUC
    ax2 = axes[0, 1]
    ax2.plot(epochs, df['auc'], 'g-o', label='AUC', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Validation AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. F1 Score
    ax3 = axes[1, 0]
    ax3.plot(epochs, df['f1'], 'm-o', label='F1 Score', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Validation F1 Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. Accuracy
    ax4 = axes[1, 1]
    ax4.plot(epochs, df['acc'], 'c-o', label='Accuracy', linewidth=2, markersize=4)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / f"training_metrics_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close()


def print_summary(df, model_name):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print(f"Training Summary - {model_name}")
    print(f"{'='*60}")
    
    print(f"\nTotal Epochs: {len(df)}")
    print(f"\nBest Metrics:")
    print(f"  Best AUC:  {df['auc'].max():.4f} (Epoch {df.loc[df['auc'].idxmax(), 'epoch']})")
    print(f"  Best F1:   {df['f1'].max():.4f} (Epoch {df.loc[df['f1'].idxmax(), 'epoch']})")
    print(f"  Best Acc:  {df['acc'].max():.4f} (Epoch {df.loc[df['acc'].idxmax(), 'epoch']})")
    print(f"  Min Val Loss: {df['val_loss'].min():.4f} (Epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})")
    
    print(f"\nFinal Metrics (Epoch {df['epoch'].iloc[-1]}):")
    print(f"  Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Val Loss:   {df['val_loss'].iloc[-1]:.4f}")
    print(f"  AUC:        {df['auc'].iloc[-1]:.4f}")
    print(f"  F1:         {df['f1'].iloc[-1]:.4f}")
    print(f"  Accuracy:   {df['acc'].iloc[-1]:.4f}")
    
    print(f"\nAverage Metrics:")
    print(f"  Avg Train Loss: {df['train_loss'].mean():.4f}")
    print(f"  Avg Val Loss:   {df['val_loss'].mean():.4f}")
    print(f"  Avg AUC:        {df['auc'].mean():.4f}")
    print(f"  Avg F1:         {df['f1'].mean():.4f}")
    print(f"  Avg Accuracy:   {df['acc'].mean():.4f}")
    print(f"{'='*60}\n")


def main(args):
    """Load log, plot metrics, and print summary."""
    log_file = RESULTS_DIR / f"training_log_{args.model_name}.csv"
    
    # Load data
    df = load_training_log(log_file)
    
    # Print summary
    print_summary(df, args.model_name)
    
    # Plot metrics
    plot_metrics(df, args.model_name, PLOTS_DIR)
    
    print(f"Visualization complete! Check plots in: {PLOTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="Model name to match log file")
    args = parser.parse_args()
    main(args)

