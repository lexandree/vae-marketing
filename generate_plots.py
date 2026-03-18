import os
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'font.size': 12})

def generate_training_curves():
    api = wandb.Api()
    
    # Try to find a good run
    runs = api.runs('vae_marketing')
    best_run = None
    for r in runs:
        if r.state == "finished" and "loss" in r.summary and "epoch" in r.summary:
            if r.summary["epoch"] > 10:
                best_run = r
                break
                
    if not best_run:
        print("No suitable runs found for plotting.")
        return

    print(f"Generating plots from run: {best_run.name} ({best_run.id})")
    
    history = best_run.history(keys=["epoch", "loss", "mse_loss", "kl_loss", "beta"])
    
    # Ensure docs/images exists
    os.makedirs("docs/images", exist_ok=True)
    
    # Plot 1: Loss Components
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss (MSE)', color=color)
    ax1.plot(history['epoch'], history['mse_loss'], color=color, label='MSE Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('KL Divergence', color=color)  
    ax2.plot(history['epoch'], history['kl_loss'], color=color, linestyle='--', label='KL Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title(f"Beta-VAE Training Curves (Run: {best_run.name})")
    plt.savefig("docs/images/training_curves.svg", bbox_inches='tight')
    plt.savefig("docs/images/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Beta Annealing
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history['epoch'], history['beta'], color='tab:green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Beta Capacity')
    ax.set_title("Beta Annealing Schedule")
    plt.savefig("docs/images/beta_annealing.svg", bbox_inches='tight')
    plt.savefig("docs/images/beta_annealing.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plots generated successfully in docs/images/")

if __name__ == "__main__":
    generate_training_curves()
