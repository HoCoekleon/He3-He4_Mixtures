import cupy as cp
import numpy as np
from tqdm import tqdm

class MCSimulationGPU:
    def __init__(self, N, x, j, nndiag=False, w_diag=0.5):
        self.N = N
        self.x = x
        self.j = j
        self.nndiag = nndiag
        self.w_diag = w_diag
        self.num_particles = N * N
        self.num_A = int(round(x * self.num_particles))
        
        # Initialize on CPU and transfer to GPU
        lattice_cpu = np.zeros(self.num_particles, dtype=np.int32)
        indices = np.random.choice(self.num_particles, self.num_A, replace=False)
        lattice_cpu[indices] = np.random.choice(np.array([1, -1], dtype=np.int32), self.num_A)
        
        self.lattice = cp.array(lattice_cpu.reshape((N, N)))
        
        # Precompute masks
        x_coord, y_coord = cp.indices((N, N))
        # 2-color for 4-neighbor (ortho)
        self.mask_even = (x_coord + y_coord) % 2 == 0
        self.mask_odd = ~self.mask_even
        
        # 4-color for 8-neighbor (ortho + diag)
        self.mask0 = (x_coord % 2 == 0) & (y_coord % 2 == 0)
        self.mask1 = (x_coord % 2 == 1) & (y_coord % 2 == 0)
        self.mask2 = (x_coord % 2 == 0) & (y_coord % 2 == 1)
        self.mask3 = (x_coord % 2 == 1) & (y_coord % 2 == 1)

    def get_neighbor_sum(self):
        # Use roll for fast PBC neighbor sum
        s_sum = cp.roll(self.lattice, 1, axis=0) + cp.roll(self.lattice, -1, axis=0) + \
                cp.roll(self.lattice, 1, axis=1) + cp.roll(self.lattice, -1, axis=1)
        
        if self.nndiag:
            diag_sum = cp.roll(cp.roll(self.lattice, 1, axis=0), 1, axis=1) + \
                       cp.roll(cp.roll(self.lattice, 1, axis=0), -1, axis=1) + \
                       cp.roll(cp.roll(self.lattice, -1, axis=0), 1, axis=1) + \
                       cp.roll(cp.roll(self.lattice, -1, axis=0), -1, axis=1)
            s_sum = s_sum + self.w_diag * diag_sum
        return s_sum

    def flip_step(self):
        # Determine masks based on connectivity
        masks = [self.mask0, self.mask1, self.mask2, self.mask3] if self.nndiag else [self.mask_even, self.mask_odd]
        
        for mask in masks:
            # Must recalculate sum after each color pass if they depend on each other
            s_sum = self.get_neighbor_sum()
            s_curr = self.lattice
            
            dE_red = 2.0 * self.j * s_curr * s_sum
            prob = cp.exp(-dE_red)
            rand = cp.random.rand(self.N, self.N)
            
            change_mask = (s_curr != 0) & mask & ((dE_red <= 0) | (rand < prob))
            self.lattice[change_mask] *= -1

    def swap_step(self):
        dr_list = [0, 0, 1, -1, 1, 1, -1, -1]
        dc_list = [1, -1, 0, 0, 1, -1, 1, -1]
        
        for _ in range(2):
            d_idx = np.random.randint(8)
            dr, dc = dr_list[d_idx], dc_list[d_idx]
            
            # For swaps, we still use 2-color as long as we pick a direction
            # and ensure partners are independent. Roll ensures 1-to-1 mapping.
            # However, partner of an even site might be even for diagonal.
            # We use a simple strategy: swap all sites in mask with their neighbors.
            # To be safe, we use the 4-color mask to ensure no site is both r1 and r2.
            masks = [self.mask0, self.mask1, self.mask2, self.mask3] if self.nndiag else [self.mask_even, self.mask_odd]
            
            for mask in masks:
                s1 = self.lattice
                s2 = cp.roll(cp.roll(self.lattice, -dr, axis=0), -dc, axis=1)
                
                sum1 = self.get_neighbor_sum()
                sum2 = cp.roll(cp.roll(sum1, -dr, axis=0), -dc, axis=1)
                
                dist_r, dist_c = abs(dr), abs(dc)
                w12 = 0.0
                if (dist_r + dist_c) == 1: w12 = 1.0
                elif (dist_r + dist_c) == 2: 
                    if self.nndiag: w12 = self.w_diag
                
                dE_red = -self.j * ((s2 - s1) * (sum1 - w12 * s2) + (s1 - s2) * (sum2 - w12 * s1))
                
                prob = cp.exp(-dE_red)
                rand = cp.random.rand(self.N, self.N)
                
                do_swap = (s1 != s2) & mask & ((dE_red <= 0) | (rand < prob))
                
                # Correct Swap: Use temporary storage for values
                s1_vals = self.lattice[do_swap].copy()
                s2_vals = s2[do_swap].copy()
                
                # Update r1
                self.lattice[do_swap] = s2_vals
                # Update r2
                s2_back_mask = cp.roll(cp.roll(do_swap, dr, axis=0), dc, axis=1)
                self.lattice[s2_back_mask] = s1_vals

    def calculate_total_energy(self):
        s_sum = self.get_neighbor_sum()
        total_e = -0.5 * self.j * cp.sum(self.lattice * s_sum)
        return float(total_e)

    def calculate_magnetization(self):
        if self.num_A == 0: return 0.0
        return float(cp.sum(self.lattice) / self.num_A)

    def run(self, niters):
        energies = []
        magnetizations = []
        
        initial_config = cp.asnumpy(self.lattice.copy())
        
        for _ in tqdm(range(niters), desc="GPU Simulating"):
            self.flip_step()
            self.swap_step()
            energies.append(self.calculate_total_energy())
            magnetizations.append(self.calculate_magnetization())
            
        return energies, magnetizations, initial_config, cp.asnumpy(self.lattice)
