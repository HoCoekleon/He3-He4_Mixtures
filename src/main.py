import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import argparse
from simulation import MCSimulation

# Set global plotting style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'

def plot_config(lattice, title, filename, N):
    """
    Plot the lattice configuration filling the grid with proper labels and style.
    """
    cmap = ListedColormap(['green', 'white', 'red'])
    norm_lattice = lattice + 1 # map -1,0,1 to 0,1,2
    
    # Square figure
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Use origin='lower' to match (x,y) from (0,0) to (N,N)
    im = ax.imshow(norm_lattice, cmap=cmap, interpolation='nearest', 
                   extent=[0, N, 0, N], origin='lower')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'$x$-coordinate', fontsize=12)
    ax.set_ylabel(r'$y$-coordinate', fontsize=12)
    
    # Explicit axis limits and box
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    
    # Custom legend with LaTeX superscripts
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label=r'$\mathrm{He}^4$ Spin Up (+1)'),
        Patch(facecolor='green', label=r'$\mathrm{He}^4$ Spin Down (-1)'),
        Patch(facecolor='white', edgecolor='gray', label=r'$\mathrm{He}^3$ (0)')
    ]
    # Position legend outside to avoid overlap
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='2D Monte-Carlo simulation in canonical ensemble')
    parser.add_argument('--N', type=int, default=40, help='Edge Size of the square lattice')
    parser.add_argument('--x', type=float, default=0.5, help='Concentration of Type A (He4)')
    parser.add_argument('--j', type=float, default=1.0, help='j=J/kBT')
    parser.add_argument('--niters', type=int, default=1000, help='Total number of Monte Carlo steps')
    parser.add_argument('--nndiag', action='store_true', help='Consider diagonal neighbors in Hamiltonian')
    parser.add_argument('--w_diag', type=float, default=1.0, help='Weight for diagonal neighbors (if nndiag is True)')
    parser.add_argument('--output_dir', type=str, default='output', help='Folder to save results')
    
    args = parser.parse_args()
    
    # Dynamic output directory name
    diag_str = f"T_w{args.w_diag}" if args.nndiag else "F"
    folder_name = f"sim_N{args.N}_x{args.x}_j{args.j}_iters{args.niters}_diag{diag_str}"
    full_output_dir = os.path.join(args.output_dir, folder_name)
    
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
    sim = MCSimulation(args.N, args.x, args.j, args.nndiag, args.w_diag)
    
    energies, magnetizations, init_conf, final_conf = sim.run(args.niters)
    
    # 1. Energy Plot (Square)
    plt.figure(figsize=(6, 6))
    plt.plot(energies, color='blue', linewidth=1)
    plt.xlabel(r'Monte-Carlo Steps $i$', fontsize=12)
    plt.ylabel(r'Total Energy $E$', fontsize=12)
    plt.title(r'Energy $E$ vs Steps $i$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(full_output_dir, 'energy_plot.png'))
    plt.close()
    
    # 2. Magnetization Plot (Square)
    plt.figure(figsize=(6, 6))
    plt.plot(magnetizations, color='red', linewidth=1)
    plt.xlabel(r'Monte-Carlo Steps $i$', fontsize=12)
    plt.ylabel(r'Magnetization $m$', fontsize=12)
    plt.title(r'Magnetization $m$ vs Steps $i$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(full_output_dir, 'magnetization_plot.png'))
    plt.close()
    
    # 3. Configuration Plots
    plot_config(init_conf, 'Initial Configuration', os.path.join(full_output_dir, 'initial_config.png'), args.N)
    plot_config(final_conf, 'Final Configuration', os.path.join(full_output_dir, 'final_config.png'), args.N)
    
    print(f"Simulation completed. Results saved in {full_output_dir}")

if __name__ == "__main__":
    main()
