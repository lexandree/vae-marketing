import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12})

def generate_plots():
    api = wandb.Api()
    
    # --- 1. SMOOTH TRAINING CURVES ---
    # Find a run with a long and smooth learning curve
    runs = api.runs('vae_marketing', order='+summary_metrics.mse_loss')
    best_run = None
    for r in runs:
        if r.state == "finished" and "loss" in r.summary and "epoch" in r.summary:
            if r.summary["epoch"] >= 50: # look for a longer run
                best_run = r
                break
                
    if not best_run:
        best_run = runs[0] # Fallback

    print(f"Generating training curves from run: {best_run.name} ({best_run.id})")
    history = best_run.history(keys=["epoch", "loss", "mse_loss", "kl_loss", "beta"], samples=1000)
    
    # Drop NaNs and sort by epoch
    history = history.dropna(subset=['epoch', 'mse_loss', 'kl_loss']).sort_values('epoch')
    
    # Apply exponential moving average for smoothing
    alpha = 0.15 # Smoothing factor
    history['mse_loss_smooth'] = history['mse_loss'].ewm(alpha=alpha).mean()
    history['kl_loss_smooth'] = history['kl_loss'].ewm(alpha=alpha).mean()

    os.makedirs("docs/images", exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Plot raw data with low alpha (transparency)
    color1 = '#e63946' # Reddish
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Reconstruction Loss (MSE)', color=color1, fontweight='bold')
    ax1.plot(history['epoch'], history['mse_loss'], color=color1, alpha=0.2, label='MSE (raw)')
    # Plot smoothed data
    ax1.plot(history['epoch'], history['mse_loss_smooth'], color=color1, linewidth=2.5, label='MSE (smoothed)')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  
    color2 = '#1d3557' # Dark blue
    ax2.set_ylabel('KL Divergence', color=color2, fontweight='bold')  
    ax2.plot(history['epoch'], history['kl_loss'], color=color2, alpha=0.2, label='KL (raw)')
    ax2.plot(history['epoch'], history['kl_loss_smooth'], color=color2, linestyle='--', linewidth=2.5, label='KL (smoothed)')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

    plt.title(f"Beta-VAE Training Dynamics\n(Smoothed Exponential Decay)", pad=20, fontweight='bold')
    fig.tight_layout()  
    plt.savefig("docs/images/training_curves.svg", bbox_inches='tight')
    plt.savefig("docs/images/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. BAYESIAN SWEEP LANDSCAPE ---
    print("Generating Bayesian HPO Sweep plot...")
    sweeps = api.project('vae_marketing').sweeps()
    if sweeps:
        # Find the sweep with the most runs
        best_sweep = max(sweeps, key=lambda s: len(s.runs))
        sweep_runs = best_sweep.runs
        
        data = []
        for r in sweep_runs:
            if r.state == "finished" and "mse_loss" in r.summary:
                row = {
                    'beta': r.config.get('beta', np.nan),
                    'latent_dim': r.config.get('latent_dim', np.nan),
                    'learning_rate': r.config.get('learning_rate', r.config.get('lr', np.nan)),
                    'mse_loss': r.summary['mse_loss']
                }
                data.append(row)
                
        df = pd.DataFrame(data).dropna()
        if len(df) > 5:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Scatter plot: latent_dim vs beta, size/color by MSE
            # We want lower MSE to be "better" (e.g., darker/larger)
            scatter = ax.scatter(
                df['latent_dim'], 
                df['beta'], 
                c=df['mse_loss'], 
                cmap='viridis_r', # reverse viridis so low loss is yellow/bright
                s=100 + (df['mse_loss'].max() - df['mse_loss']) / (df['mse_loss'].max() - df['mse_loss'].min() + 1e-6) * 300,
                alpha=0.8,
                edgecolors='w',
                linewidth=0.5
            )
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Final MSE Loss (Lower is Better)', rotation=270, labelpad=15)
            
            ax.set_xlabel('Latent Dimension Capacity', fontweight='bold')
            ax.set_ylabel('Beta Regularization Strength', fontweight='bold')
            ax.set_title(f"Bayesian Optimization Landscape\n(Sweep: {best_sweep.name})", pad=15, fontweight='bold')
            
            # Add grid and tweak look
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            
            plt.tight_layout()
            plt.savefig("docs/images/sweep_landscape.svg", bbox_inches='tight')
            plt.savefig("docs/images/sweep_landscape.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Sweep landscape generated.")
        else:
            print("Not enough complete sweep runs to generate landscape.")

if __name__ == "__main__":
    generate_plots()
