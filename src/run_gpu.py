import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Add parent dir to path to import simulation
sys.path.append(os.path.join(os.path.dirname(__file__), 'gpu'))
from simulation_gpu import MCSimulationGPU

# Import plotting from main (reusing the logic)
sys.path.append(os.path.dirname(__file__))
from main import plot_config

def main():
    parser = argparse.ArgumentParser(description='2D Monte-Carlo simulation (GPU)')
    parser.add_argument('--N', type=int, default=40, help='Edge Size')
    parser.add_argument('--x', type=float, default=0.5, help='Concentration')
    parser.add_argument('--j', type=float, default=1.0, help='j=J/kBT')
    parser.add_argument('--niters', type=int, default=1000, help='Steps')
    parser.add_argument('--nndiag', action='store_true', help='Diagonal neighbors')
    parser.add_argument('--w_diag', type=float, default=1.0, help='Weight for diagonal neighbors')
    parser.add_argument('--output_dir', type=str, default='output', help='Base output folder')
    
    args = parser.parse_args()
    
    diag_str = f"T_w{args.w_diag}" if args.nndiag else "F"
    folder_name = f"gpu_sim_N{args.N}_x{args.x}_j{args.j}_iters{args.niters}_diag{diag_str}"
    full_output_dir = os.path.join(args.output_dir, folder_name)
    
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
    sim = MCSimulationGPU(args.N, args.x, args.j, args.nndiag, args.w_diag)
    energies, magnetizations, init_conf, final_conf = sim.run(args.niters)
    
    # Plot results (Square)
    plt.figure(figsize=(6, 6))
    plt.plot(energies)
    plt.xlabel('Steps i')
    plt.ylabel('Energy E')
    plt.savefig(os.path.join(full_output_dir, 'energy_plot.png'))
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.plot(magnetizations)
    plt.xlabel('Steps i')
    plt.ylabel('Magnetization m')
    plt.savefig(os.path.join(full_output_dir, 'magnetization_plot.png'))
    plt.close()
    
    plot_config(init_conf, 'Initial (GPU)', os.path.join(full_output_dir, 'initial_config.png'), args.N)
    plot_config(final_conf, 'Final (GPU)', os.path.join(full_output_dir, 'final_config.png'), args.N)
    
    print(f"GPU Simulation completed. Results in {full_output_dir}")

if __name__ == "__main__":
    main()
