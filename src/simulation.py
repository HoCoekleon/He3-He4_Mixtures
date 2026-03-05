import numpy as np
from tqdm import tqdm
from numba import njit

@njit
def get_local_energy_sum_fast(lattice, N, r, c, nndiag, w_diag=0.5):
    s_sum = 0.0
    # Orthogonal
    s_sum += lattice[(r + 1) % N, c]
    s_sum += lattice[(r - 1) % N, c]
    s_sum += lattice[r, (c + 1) % N]
    s_sum += lattice[r, (c - 1) % N]
    
    if nndiag:
        # Diagonal with weight w_diag
        s_sum += w_diag * lattice[(r + 1) % N, (c + 1) % N]
        s_sum += w_diag * lattice[(r + 1) % N, (c - 1) % N]
        s_sum += w_diag * lattice[(r - 1) % N, (c + 1) % N]
        s_sum += w_diag * lattice[(r - 1) % N, (c - 1) % N]
    return s_sum

@njit
def flip_loop_fast(lattice, N, j, nndiag, w_diag=0.5):
    rows, cols = np.where(lattice != 0)
    indices = np.arange(len(rows))
    np.random.shuffle(indices)
    
    for idx in indices:
        r, c = rows[idx], cols[idx]
        s_old = lattice[r, c]
        s_sum = get_local_energy_sum_fast(lattice, N, r, c, nndiag, w_diag)
        
        dE_red = 2.0 * j * s_old * s_sum
        
        if dE_red <= 0 or np.random.rand() < np.exp(-dE_red):
            lattice[r, c] = -s_old

@njit
def swap_loop_fast(lattice, N, j, nndiag, w_diag=0.5):
    # Iterate through all sites to attempt swaps
    num_particles = N * N
    indices = np.arange(num_particles)
    np.random.shuffle(indices)
    
    # Pre-defined offsets: 4 ortho + 4 diag
    dr_offsets = np.array([0, 0, 1, -1, 1, 1, -1, -1])
    dc_offsets = np.array([1, -1, 0, 0, 1, -1, 1, -1])
    
    for idx in indices:
        r1, c1 = idx // N, idx % N
        
        # Choose random neighbor
        neighbor_idx = np.random.randint(8)
        dr, dc = dr_offsets[neighbor_idx], dc_offsets[neighbor_idx]
        r2, c2 = (r1 + dr) % N, (c1 + dc) % N
        
        s1 = lattice[r1, c1]
        s2 = lattice[r2, c2]
        
        if s1 == s2:
            continue
            
        sum_nn1 = get_local_energy_sum_fast(lattice, N, r1, c1, nndiag, w_diag)
        sum_nn2 = get_local_energy_sum_fast(lattice, N, r2, c2, nndiag, w_diag)
        
        # Calculate interaction weight w12
        w12 = 0.0
        dist_r = abs(r1 - r2)
        dist_c = abs(c1 - c2)
        dist_r = min(dist_r, N - dist_r)
        dist_c = min(dist_c, N - dist_c)
        
        if dist_r <= 1 and dist_c <= 1:
            if (dist_r + dist_c) == 1:
                w12 = 1.0
            elif (dist_r + dist_c) == 2:
                if nndiag:
                    w12 = w_diag
        
        dE_red = -j * ((s2 - s1) * (sum_nn1 - w12 * s2) + (s1 - s2) * (sum_nn2 - w12 * s1))
        
        if dE_red <= 0 or np.random.rand() < np.exp(-dE_red):
            lattice[r1, c1], lattice[r2, c2] = s2, s1

class MCSimulation:
    def __init__(self, N, x, j, nndiag=False, w_diag=0.5):
        self.N = N
        self.x = x
        self.j = j
        self.nndiag = nndiag
        self.w_diag = w_diag
        self.num_particles = N * N
        self.num_A = int(round(x * self.num_particles))
        
        self.lattice = np.zeros(self.num_particles, dtype=np.int32)
        indices = np.random.choice(self.num_particles, self.num_A, replace=False)
        self.lattice[indices] = np.random.choice(np.array([1, -1], dtype=np.int32), self.num_A)
        self.lattice = self.lattice.reshape((N, N))

    def calculate_total_energy(self):
        # We can use a faster way here too, but it's called once per step
        energy_sum = 0.0
        rows, cols = np.where(self.lattice != 0)
        for r, c in zip(rows, cols):
            energy_sum += self.lattice[r, c] * get_local_energy_sum_fast(self.lattice, self.N, r, c, self.nndiag, self.w_diag)
        return -0.5 * self.j * energy_sum

    def calculate_magnetization(self):
        if self.num_A == 0: return 0.0
        return np.sum(self.lattice) / self.num_A

    def run(self, niters, disable_tqdm=False):
        energies = np.zeros(niters + 1)
        magnetizations = np.zeros(niters + 1)
        
        energies[0] = self.calculate_total_energy()
        magnetizations[0] = self.calculate_magnetization()
        
        initial_config = self.lattice.copy()
        
        for i in tqdm(range(1, niters + 1), desc="Simulating", disable=disable_tqdm):
            flip_loop_fast(self.lattice, self.N, self.j, self.nndiag, self.w_diag)
            swap_loop_fast(self.lattice, self.N, self.j, self.nndiag, self.w_diag)
            energies[i] = self.calculate_total_energy()
            magnetizations[i] = self.calculate_magnetization()
            
        return energies, magnetizations, initial_config, self.lattice.copy()
