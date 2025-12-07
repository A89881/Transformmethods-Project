# """
# Step Response Verification for Filter #3 (RL Parallel + Series C)
# Uppsala University - Transform Methods Project
# Author: Abhay Mishra
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal

# # Filter parameters
# R = 250.0        # Resistance [Ohm]
# L = 37.5e-3      # Inductance [H]
# C = 40e-6        # Capacitance [F]

# # Derived system parameters
# omega_n = 1.0 / np.sqrt(L * C)                      # Natural frequency
# gamma =  (1.0 / (2.0 * R)) * np.sqrt(L / C)         # Damping ratio (gamma)
# underdamped = gamma < 1.0

# omega_d = omega_n * np.sqrt(max(1.0 - gamma**2, 0.0))  # Damped natural frequency (real if gamma < 1)

# print(f"omega_n = {omega_n:.3f} rad/s")
# print(f"gamma     = {gamma:.5f} (≡ gamma)")
# print(f"omega_d  = {omega_d:.3f} rad/s")

# # Time axis and step input
# t_end = .25
# t = np.linspace(0.0, t_end, 3000)                  # Time vector
# u = np.heaviside(t, 1.0)                           # Unit step u(t)


# # Analytical step response from the derivation

# if underdamped and omega_d > 0:
#     v_analytical = 1.0 - np.exp(-gamma * omega_n * t) * (
#         np.cos(omega_d * t) - (gamma * omega_n / omega_d) * np.sin(omega_d * t)
#     )
# else:
#     # Fallbacks for critical/over-damped (not expected with given R,L,C, but kept for robustness)
#     if np.isclose(gamma, 1.0):
#         # Critical damping: limit of the underdamped formula as omega_d -> 0
#         v_analytical = 1.0 - np.exp(-omega_n * t) * (1.0 + omega_n * t)
#     else:
#         # Overdamped: compute from real poles
#         a = gamma * omega_n
#         rad = omega_n**2 * (gamma**2 - 1.0)
#         s1 = -a + np.sqrt(rad)
#         s2 = -a - np.sqrt(rad)
#         # Standard unit-step response for distinct real poles
#         K1 = s2 / (s2 - s1)
#         K2 = -s1 / (s2 - s1)
#         v_analytical = 1.0 - (K1 * np.exp(s1 * t) + K2 * np.exp(s2 * t))
# # Apply step explicitly (no-op for t>=0)
# v_analytical = v_analytical * u

# # Numerical step response using scipy.signal
# # Transfer function as in your base script

# num = [L, R]                           # Numerator coefficients
# den = [R * L * C, L, R]                # Denominator coefficients
# system = signal.TransferFunction(num, den)

# # Use the same time vector for apples-to-apples comparison
# t_num, v_numerical = signal.step(system, T=t)


# # Plot: Analytical vs Numerical
# plt.figure(figsize=(8, 5))
# plt.plot(t_num, v_numerical, label='Numerical (scipy.signal.step)', linewidth=2)
# plt.plot(t, v_analytical, '--', label='Analytical (boxed formula)', linewidth=2)
# plt.xlabel('Time [s]')
# plt.ylabel('Output voltage $v_{out}(t)$ [V]')
# plt.title('Step Response Comparison  Filter #3')
# plt.grid(True, alpha=0.4)
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.show()

# # Quantitative verification
# err_rms = np.sqrt(np.mean((v_analytical - v_numerical)**2))
# err_max = np.max(np.abs(v_analytical - v_numerical))
# print(f"RMS error between analytical and numerical responses: {err_rms:.3e}")
# print(f"Max abs error between analytical and numerical responses: {err_max:.3e}")

# Self-contained script to compute and PLOT Bode magnitude/phase for the RL‖ + Series C filter,
# and AUTO-CLASSIFY the filter as low-pass / high-pass / band-pass / notch using simple heuristics.
#
# Feel free to tweak R, L, C and the frequency range. The script prints key metrics and a classification.

# """
# Bode Plot for Filter #3 (RL Parallel + Series C)
# Uppsala University - Transform Methods Project
# Author: Abhay Mishra
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal


# # Filter parameters
# R = 250.0       # Resistance [Ohm]
# L = 37.5e-3     # Inductance [H]
# C = 40e-6       # Capacitance [F]

# # Transfer Function: H(s) = (L s + R) / (R L C s^2 + L s + R)
# num = [L, R]
# den = [R * L * C, L, R]
# system = signal.lti(num, den)

# # Frequency response (Bode)
# f = np.logspace(0, 5, 2000)   # Frequency range: 1 Hz – 100 kHz
# w = 2 * np.pi * f
# w, mag_db, phase_deg = signal.bode(system, w=w)

# # Combined Bode plot (Magnitude + Phase)
# fig, ax1 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# # Magnitude plot
# ax1[0].semilogx(f, mag_db, linewidth=2)
# ax1[0].set_ylabel('Magnitude [dB]')
# ax1[0].set_title('Bode Plot – Filter #3 (RL‖C)')
# ax1[0].grid(which='both', linestyle='--', alpha=0.6)

# # Phase plot
# ax1[1].semilogx(f, phase_deg, linewidth=2, color='C1')
# ax1[1].set_xlabel('Frequency [Hz]')
# ax1[1].set_ylabel('Phase [°]')
# ax1[1].grid(which='both', linestyle='--', alpha=0.6)

# plt.tight_layout()
# plt.show()

"""
Q5 - Square Wave Response via Fourier Series (first 5 nonzero harmonics)
Filter #3 (RL Parallel + Series C) — Frequency Response Method
Period: T0 = 10 ms  (f0 = 100 Hz)
"""

import numpy as np
import matplotlib.pyplot as plt

# Filter parameters
R = 250.0       # Ohm
L = 37.5e-3     # H
C = 40e-6       # F

# Square wave (0..1 V), T0=10 ms
T0 = 10e-3                     # s
f0 = 1.0 / T0                  # 100 Hz
w0 = 2.0 * np.pi * f0

# 0..1V, square wave Fourier series:
# x(t) = 1/2 + (2/pi) * sum_{k odd} (1/k) sin(k*w0*t)
# First five nonzero harmonics: k = 1, 3, 5, 7, 9
harmonics = [1, 3, 5, 7, 9]

def H_jw(w):
    """Frequency response H(jw) = (L*j*w + R) / (R*L*C*(j*w)^2 + L*j*w + R)."""
    jw = 1j * w
    num = L * jw + R
    den = R * L * C * (jw**2) + L * jw + R
    return num / den

# Collect rows: xi(t)=Ai sin(wi t + phii), fi, |H(jwi)|, arg(H(jwi)), yi(t)=Bi sin(wi t + thetai)
rows = []
for k in harmonics:
    fi = k * f0
    wi = 2.0 * np.pi * fi
    Ai = (2.0 / np.pi) * (1.0 / k)      # amplitude of the k-th sine term [V]
    phii = 0.0                          # radians (series uses sin with zero phase)
    H_i = H_jw(wi)
    gain = float(np.abs(H_i))
    phase_rad = float(np.angle(H_i))    # in (-pi, pi]
    phase_deg = float(np.degrees(phase_rad))
    Bi = Ai * gain
    thetai = phii + phase_rad           # radians

    rows.append({
        "xi(t)": f"{Ai:.6f}*sin({wi:.3f}*t + {phii:.3f})",
        "fi_Hz": fi,
        "H_abs": gain,
        "H_arg_deg": phase_deg,
        "yi(t)": f"{Bi:.6f}*sin({wi:.3f}*t + {thetai:.3f})"
    })

header = (
    "i  |  Input xi(t) = Ai sin(wi t + phii)         |  fi [Hz]   |  |H(jwi)|   "
    "|  arg(H) [deg]  |  Output yi(t) = Bi sin(wi t + thetai)"
)

# Save table output to a text file
output_lines = []
output_lines.append(header)
output_lines.append("-" * len(header))
for i, r in enumerate(rows, start=1):
    line = (f"{i:<2} | {r['xi(t)']:<42} | {r['fi_Hz']:<10.2f} | "
            f"{r['H_abs']:<10.6f} | {r['H_arg_deg']:<12.3f} | {r['yi(t)']}")
    output_lines.append(line)

log_filename = "square_wave_output_table.txt"
with open(log_filename, "w") as f:
    f.write("\n".join(output_lines))

print(f"\nTable saved to '{log_filename}'")

# Optional: reconstruct steady-state output with first 5 harmonics
# (DC passes with H(0)=1, so DC_out = 0.5 V)
reconstruct = True
if reconstruct:
    t = np.linspace(0, 3*T0, 4000)  # show 3 periods
    y = 0.5 * np.ones_like(t)       # DC term (H(0)=1)
    for k in harmonics:
        fi = k * f0
        wi = 2*np.pi*fi
        Ai = (2/np.pi) * (1/k)
        H_i = H_jw(wi)
        Bi = Ai * np.abs(H_i)
        theta = np.angle(H_i)
        y += Bi * np.sin(wi*t + theta)

    # Also: the original 5-term input approximation (for reference)
    x5 = 0.5 * np.ones_like(t)
    for k in harmonics:
        x5 += (2/np.pi) * (1/k) * np.sin(k*w0*t)

    # Plot
    plt.figure(figsize=(9, 4))
    plt.plot(t*1e3, x5, label='Input (5-term approx.)')
    plt.plot(t*1e3, y,  label='Output (steady-state, 5 harmonics)')
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [V]")
    plt.title("Square Wave Response (First 5 Nonzero Harmonics, T0 = 10 ms)")
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
