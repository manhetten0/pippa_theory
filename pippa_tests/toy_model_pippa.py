# toy_model_pippa.py (updated for universe-like visualization)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftfreq
import imageio  # For GIF animation
import io  # For buffer

# Toy Model Parameters
N = 128
mu = 1.26
D_A = 0.1
D_B = 0.05
kappa_AB = 0.02
kappa_BA = 0.01
alpha_A = 0.6
alpha_B = 0.3
gamma_A = gamma_B = 0.01
lambda_val = 0.5
xi = 1.0
c_I = 0.5
dt = 0.05  # Smaller for stability with particles
num_steps = 200
sigma_k = 0.1

# Grid
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
kx = 2 * np.pi * fftfreq(N, d=1/N)
ky = 2 * np.pi * fftfreq(N, d=1/N)
KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)

# Initial A/B with multiple peaks (galaxy seeds)
centers = [(0.4, 0.4), (0.6, 0.6), (0.5, 0.3)]  # Multiple blobs
A = np.zeros((N, N))
B = np.zeros((N, N))
for cx, cy in centers:
    A += np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 0.03**2))
B = 0.5 * A
sigma = np.zeros_like(A)  # Add sources if needed

# N-body particles (simulate matter)
num_particles = 100
particles = np.random.uniform(0, 1, (num_particles, 2))  # Positions [x,y]
velocities = np.zeros((num_particles, 2))  # Initial velocities

# Functions
def fractional_laplacian(field, mu):
    field_fft = fft2(field)
    lap_fft = - (K ** mu) * field_fft
    return np.real(ifft2(lap_fft))

def mirror_operator(field, sigma=sigma_k):
    ref_x = np.fliplr(field)
    ref_y = np.flipud(field)
    ref_xy = np.fliplr(np.flipud(field))
    avg_ref = (ref_x + ref_y + ref_xy) / 3
    return gaussian_filter(avg_ref, sigma=sigma)

def compute_potential(rho_total):
    rho_fft = fft2(rho_total)
    phi_fft = - rho_fft / (K**2 + 1e-10)
    return np.real(ifft2(phi_fft))

# Simulation loop with animation frames
frames = []
S_history = []
a = 1.0
a_history = [a]
H_I = 0.0  # Initialize H_I to 0 for step=0

for step in range(num_steps):
    lap_A = fractional_laplacian(A, mu)
    lap_B = fractional_laplacian(B, mu)
    M_A = alpha_A * mirror_operator(A)
    M_B = alpha_B * mirror_operator(B)
    grad_A_x, grad_A_y = np.gradient(A)
    grad_A_sq = grad_A_x**2 + grad_A_y**2
    S_A = 0.1 * sigma
    S_B = 0.05 * sigma
    
    dA = D_A * lap_A + kappa_AB * B + M_A - gamma_A * A + S_A
    dB = D_B * lap_B + kappa_BA * A - lambda_val * grad_A_sq + M_B - gamma_B * B + S_B
    
    A += dt * dA
    B += dt * dB
    A = np.maximum(A, 0)
    B = np.maximum(B, 0)
    
    total = A + B + 1e-10
    P = total / np.sum(total)
    S = -np.sum(P * np.log(P))
    S_history.append(S)
    
    # Scale factor update
    if step > 0:
        dS_dt = (S - S_history[-2]) / dt
        H_I = c_I * dS_dt
        a += dt * H_I * a
    a_history.append(a)
    
    # DM and potential
    rho_DM = xi * mirror_operator(A)
    rho_total = rho_DM  # Simplified
    Phi = compute_potential(rho_total)
    
    # N-body update: Particles move under ∇Φ + Hubble drag
    grad_Phi_x, grad_Phi_y = np.gradient(-Phi)  # Force ~ -∇Φ
    for i in range(num_particles):
        px, py = particles[i]
        ix = min(max(int(px * (N-1)), 0), N-1)
        iy = min(max(int(py * (N-1)), 0), N-1)
        fx = grad_Phi_x[iy, ix]
        fy = grad_Phi_y[iy, ix]
        velocities[i] += dt * np.array([fx, fy])  # Acceleration
        particles[i] += dt * velocities[i]  # Position update
        particles[i] += dt * H_I * (particles[i] - 0.5)  # Hubble expansion (relative to center)
    particles = np.clip(particles, 0, 1)  # Keep in bounds
    
    # Save frame every 10 steps
    if step % 10 == 0:
        fig = plt.figure(figsize=(6, 6))  # Fixed size for consistency
        ax = fig.add_subplot(111)
        ax.imshow(rho_total, cmap='viridis', extent=[0,1,0,1])
        ax.scatter(particles[:,0], particles[:,1], c='red', s=5, label='Particles')
        ax.set_title(f'Step {step}, a={a:.2f}')
        ax.legend()
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frame = plt.imread(buf)[:, :, :3]  # Read as array, slice to RGB (ignore alpha if present)
        frames.append((frame * 255).astype(np.uint8))  # Convert to uint8 for imageio
        plt.close(fig)

# Save animation as GIF
imageio.mimsave('pippa_universe_evolution.gif', frames, fps=10)

# Final plots for S and a(t)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(S_history)
plt.title('Entropy S(t)')
plt.xlabel('Step')
plt.ylabel('S')

plt.subplot(1, 2, 2)
plt.plot(a_history)
plt.title('Scale Factor a(t)')
plt.xlabel('Step')
plt.ylabel('a')
plt.tight_layout()
plt.show()

# Final fields visualization
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(A, cmap='viridis')
plt.title('Final Data Field A')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(B, cmap='viridis')
plt.title('Final Coherence Field B')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(rho_DM, cmap='viridis')
plt.title('Final DM Density ρ_DM')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(Phi, cmap='viridis')
plt.title('Final Gravitational Potential Φ')
plt.colorbar()
plt.tight_layout()
plt.show()

print("Simulation complete. Check 'pippa_universe_evolution.gif' for evolution.")