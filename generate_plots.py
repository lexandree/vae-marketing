import os
import wandb
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12})

def generate_plots():
    api = wandb.Api()
    os.makedirs("docs/images", exist_ok=True)
    
    # --- 1. PERFECTLY SMOOTH TRAINING CURVES ---
    print("Fetching training curves...")
    runs = api.runs('vae_marketing', order='+summary_metrics.mse_loss')
    best_run = next((r for r in runs if r.state == "finished" and r.summary.get("epoch", 0) >= 50), runs[0] if runs else None)

    if best_run:
        history = best_run.history(keys=["epoch", "loss", "mse_loss", "kl_loss"], samples=1000)
        history = history.dropna(subset=['epoch', 'mse_loss', 'kl_loss']).sort_values('epoch')
        
        # Apply strict EMA to remove noise, then Spline for visual curvature
        alpha = 0.1 # Strong smoothing
        history['mse_loss_ema'] = history['mse_loss'].ewm(alpha=alpha).mean()
        history['kl_loss_ema'] = history['kl_loss'].ewm(alpha=alpha).mean()
        
        # Spline Interpolation for "Perfectly Smooth" non-angular look
        epochs_smooth = np.linspace(history['epoch'].min(), history['epoch'].max(), 500)
        spline_mse = make_interp_spline(history['epoch'], history['mse_loss_ema'], k=3)
        spline_kl = make_interp_spline(history['epoch'], history['kl_loss_ema'], k=3)
        
        fig, ax1 = plt.subplots(figsize=(9, 5))
        color1 = '#e63946'
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Reconstruction Loss (MSE)', color=color1, fontweight='bold')
        
        # Plot raw as very transparent
        ax1.plot(history['epoch'], history['mse_loss'], color=color1, alpha=0.1, label='MSE (Raw)')
        # Plot buttery smooth Spline
        ax1.plot(epochs_smooth, spline_mse(epochs_smooth), color=color1, linewidth=3, label='MSE (Smoothed)')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  
        color2 = '#1d3557'
        ax2.set_ylabel('KL Divergence', color=color2, fontweight='bold')  
        ax2.plot(history['epoch'], history['kl_loss'], color=color2, alpha=0.1, label='KL (Raw)')
        ax2.plot(epochs_smooth, spline_kl(epochs_smooth), color=color2, linestyle='-', linewidth=3, label='KL (Smoothed)')
        ax2.tick_params(axis='y', labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

        plt.title(f"Beta-VAE Training Dynamics\n(Perfectly Smooth Curves)", pad=20, fontweight='bold')
        fig.tight_layout()  
        plt.savefig("docs/images/training_curves.svg", bbox_inches='tight')
        plt.savefig("docs/images/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Smooth curves generated.")

    # --- 2. PARALLEL COORDINATES PLOT (ALL PARAMETERS) ---
    print("Generating HPO Sweep plot...")
    sweeps = api.project('vae_marketing').sweeps()
    if sweeps:
        best_sweep = max(sweeps, key=lambda s: len(s.runs))
        data = []
        for r in best_sweep.runs:
            if r.state == "finished" and "mse_loss" in r.summary:
                data.append({
                    'Latent Dim': r.config.get('latent_dim', np.nan),
                    'Beta': r.config.get('beta', np.nan),
                    'LR': r.config.get('learning_rate', r.config.get('lr', np.nan)),
                    'MSE Loss': r.summary['mse_loss']
                })
        
        df = pd.DataFrame(data).dropna()
        if len(df) > 5:
            # 2.A: Generate Interactive HTML using Plotly
            try:
                import plotly.express as px
                fig = px.parallel_coordinates(
                    df, color="MSE Loss", 
                    dimensions=['Latent Dim', 'Beta', 'LR', 'MSE Loss'],
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    title="Hyperparameter Search (Parallel Coordinates)"
                )
                fig.write_html("docs/images/sweep_parallel_coords.html")
                print("Interactive HTML Parallel Coords saved.")
            except ImportError:
                print("Plotly not available for HTML export.")
            
            # 2.B: Generate Static Custom Matplotlib Parallel Coordinates WITH SMOOTH CURVES
            fig, axes = plt.subplots(1, 3, sharey=False, figsize=(10, 5))
            cols = ['Latent Dim', 'Beta', 'LR', 'MSE Loss']
            
            # Normalize for plotting
            df_norm = df.copy()
            min_max = {}
            for c in cols:
                c_min, c_max = df[c].min(), df[c].max()
                if c_max == c_min: c_max = c_min + 1e-5
                min_max[c] = (c_min, c_max)
                df_norm[c] = (df[c] - c_min) / (c_max - c_min)
                
            norm = plt.Normalize(df['MSE Loss'].min(), df['MSE Loss'].max())
            cmap = plt.cm.viridis_r
            
            Path = mpath.Path
            
            for i, ax in enumerate(axes):
                for idx, row in df_norm.iterrows():
                    y0 = row[cols[i]]
                    y1 = row[cols[i+1]]
                    x0 = i
                    x1 = i+1
                    
                    # Smooth S-Curve using Bezier Control Points
                    verts = [(x0, y0), (x0 + 0.5, y0), (x1 - 0.5, y1), (x1, y1)]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                    path = Path(verts, codes)
                    
                    color = cmap(norm(df.loc[idx, 'MSE Loss']))
                    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, alpha=0.4, lw=1.5)
                    ax.add_patch(patch)
                
                ax.set_xlim(i, i+1)
                # Left y-axis ticks
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels([f"{min_max[cols[i]][0]:.3g}", f"{(min_max[cols[i]][0]+min_max[cols[i]][1])/2:.3g}", f"{min_max[cols[i]][1]:.3g}"])
                ax.set_xticks([i])
                ax.set_xticklabels([cols[i]], fontweight='bold')
                
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                
                if i == 2: # Last axis
                    ax2 = ax.twinx()
                    ax2.set_ylim(0, 1)
                    ax2.set_yticks([0, 0.5, 1])
                    ax2.set_yticklabels([f"{min_max[cols[-1]][0]:.3g}", f"{(min_max[cols[-1]][0]+min_max[cols[-1]][1])/2:.3g}", f"{min_max[cols[-1]][1]:.3g}"])
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['bottom'].set_visible(False)
                    ax2.spines['left'].set_visible(False)
                    # Add title for last column as x-tick text is tricky
                    ax.set_xticks([i, i+1])
                    ax.set_xticklabels([cols[i], cols[i+1]], fontweight='bold')
            
            fig.subplots_adjust(wspace=0)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), pad=0.1)
            cbar.set_label('Final MSE Loss (Lower is Better)', rotation=270, labelpad=15)
            
            plt.suptitle(f"Parallel Coordinates Parameter Search\n(Sweep: {best_sweep.name})", fontweight='bold', y=1.05)
            plt.savefig("docs/images/sweep_parallel_coords.svg", bbox_inches='tight')
            plt.savefig("docs/images/sweep_parallel_coords.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("Static Smooth Parallel Coords generated.")

if __name__ == "__main__":
    generate_plots()
