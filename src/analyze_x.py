import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from simulation import MCSimulation
from multiprocessing import Pool
from tqdm import tqdm

# Set global plotting style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'

def run_single_x(params):
    """
    Worker function for a single concentration simulation.
    """
    N, x, j, niters, nndiag, eq_start, w_diag = params
    sim = MCSimulation(N, x, j, nndiag, w_diag)
    # Disable internal tqdm to prevent console flickering
    _, magnetizations, _, _ = sim.run(niters, disable_tqdm=True)
    
    # Calculate average magnetization from eq_start to end
    start_idx = int(niters * eq_start)
    if start_idx >= niters:
        start_idx = niters - 1
    
    m_avg = np.mean(np.abs(magnetizations[start_idx:]))
    return m_avg

def main():
    parser = argparse.ArgumentParser(description='Analyze magnetization vs x')
    parser.add_argument('--N', type=int, default=40, help='Edge Size of the square lattice')
    parser.add_argument('--j', type=float, default=1.0, help='j=J/kBT')
    parser.add_argument('--niters', type=int, default=10000, help='Steps per simulation')
    parser.add_argument('--x_steps', type=int, default=11, help='Concentration steps')
    parser.add_argument('--nndiag', action='store_true', help='Consider diagonal neighbors')
    parser.add_argument('--w_diag', type=float, default=0.5, help='Weight for diagonal neighbors')
    parser.add_argument('--output_dir', type=str, default='output', help='Base output folder')
    parser.add_argument('--nprocs', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--eq_start', type=float, default=0.5, help='Starting fraction for averaging (0.0 to 1.0, default: 0.5)')
    
    args = parser.parse_args()
    
    diag_str = f"T_w{args.w_diag}" if args.nndiag else "F"
    folder_name = f"scan_N{args.N}_j{args.j}_iters{args.niters}_diag{diag_str}"
    full_output_dir = os.path.join(args.output_dir, folder_name)
    
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)
        
    x_vals = np.linspace(0.1, 1.0, args.x_steps)
    
    # Prepare parameters for parallel execution
    task_params = [(args.N, x, args.j, args.niters, args.nndiag, args.eq_start, args.w_diag) for x in x_vals]
    
    print(f"Starting parallel analysis with {args.nprocs} processes (averaging from {args.eq_start*100:.1f}% steps)...")
    
    # Use tqdm on the pool results to show a single master progress bar
    with Pool(processes=args.nprocs) as pool:
        m_vals = list(tqdm(pool.imap(run_single_x, task_params), total=len(x_vals), desc="Scanning x"))
        
    plt.figure(figsize=(6, 6))
    plt.plot(x_vals, m_vals, 'o-', color='black', markerfacecolor='red', markersize=8)
    plt.xlabel(r'Concentration $x$', fontsize=12)
    plt.ylabel(r'Average Magnetization $<m>$', fontsize=12)
    plt.ylim(0, 1.02)
    # plt.title(f'Magnetization $|m|$ vs Concentration $x$ (j={args.j})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add info text to plot
    # info_text = f"N={args.N}, j={args.j}\niters={args.niters}\neq_start={args.eq_start}\nw_diag={args.w_diag}"
    # plt.text(0.95, 0.05, info_text, transform=plt.gca().transAxes, 
    #          verticalalignment='bottom', horizontalalignment='right',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(full_output_dir, 'm_vs_x_plot.png'))
    plt.close()
    
    print(f"Analysis completed. Results saved in {full_output_dir}")

    # save raw data
    data_file = os.path.join(full_output_dir, 'm_vs_x_data.txt')
    with open(data_file, 'w') as f:
        f.write("# x\t<m>\n")
        for x, m in zip(x_vals, m_vals):
            f.write(f"{x:.4f}\t{m:.6f}\n") 
    

if __name__ == "__main__":
    main()
