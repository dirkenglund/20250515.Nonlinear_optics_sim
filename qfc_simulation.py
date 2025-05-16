"""
Quantum Frequency Conversion (QFC) Simulation via Three-Wave Mixing

This simulation models the phase matching spectrum and conversion efficiency
for sum-frequency generation in a PPLN waveguide, similar to the setup
described in the paper "Efficient quantum frequency conversion for networking
on the telecom E-band".
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

# Waveguide and experimental parameters from the paper
class PPLNWaveguide:
    def __init__(self, 
                 length=0.015,           # Waveguide length in meters (1.5 cm)
                 poling_period=18.5e-6,  # Poling period in meters (estimated)
                 temp=55,                # Temperature in Celsius
                 n_e=2.2,                # Refractive index at emitter wavelength (estimated)
                 n_n=2.1,                # Refractive index at networking wavelength (estimated)
                 n_p=2.1,                # Refractive index at pump wavelength (estimated)
                 kappa=5.0,              # Nonlinear coupling coefficient (arbitrary units)
                 T_in_n=0.558,           # Input transmission for networking wavelength
                 T_out_e=0.899):         # Output transmission for emitter wavelength
        
        self.length = length
        self.poling_period = poling_period
        self.temp = temp
        self.n_e = n_e
        self.n_n = n_n
        self.n_p = n_p
        self.kappa = kappa
        self.T_in_n = T_in_n
        self.T_out_e = T_out_e
        
        # Temperature dependence of phase matching (simplified model)
        self.temp_coefficient = 1.0e-6  # Temperature coefficient for phase matching
        
    def phase_mismatch(self, lambda_n, lambda_p, lambda_e):
        """Calculate phase mismatch for given wavelengths"""
        k_e = 2 * pi * self.n_e / lambda_e
        k_n = 2 * pi * self.n_n / lambda_n
        k_p = 2 * pi * self.n_p / lambda_p
        
        # QPM grating vector
        k_g = 2 * pi / self.poling_period
        
        # Temperature effect on phase matching (simplified)
        temp_effect = (self.temp - 20) * self.temp_coefficient
        
        return k_e - k_n - k_p - k_g + temp_effect
    
    def conversion_efficiency_vs_wavelength(self, lambda_n_range, lambda_p, lambda_e, pump_power):
        """Calculate conversion efficiency vs networking wavelength"""
        delta_k_values = [self.phase_mismatch(lambda_n, lambda_p, lambda_e) for lambda_n in lambda_n_range]
        
        # Normalized conversion efficiency based on sinc^2 phase matching
        norm_efficiency = [np.sinc(delta_k * self.length / (2 * pi))**2 for delta_k in delta_k_values]
        
        # Scale by pump power (simplified model)
        # In reality, this would follow the sin^2(sqrt(P/P_max)) relationship
        power_factor = np.sin(np.pi/2 * np.sqrt(pump_power/1.5))**2 if pump_power <= 1.5 else 1.0
        
        # Apply external efficiency factors (coupling losses)
        external_efficiency = [power_factor * self.T_in_n * self.T_out_e * eff for eff in norm_efficiency]
        
        return external_efficiency
    
    def conversion_efficiency_vs_power(self, pump_powers, max_efficiency=0.43):
        """Calculate conversion efficiency vs pump power"""
        # Model based on the equation in the paper
        # η_ext = η_max * sin^2(π/2 * sqrt(P/P_max))
        P_max = 1.51  # From the paper
        efficiencies = [max_efficiency * np.sin(np.pi/2 * np.sqrt(P/P_max))**2 if P <= 2*P_max else max_efficiency for P in pump_powers]
        return efficiencies

def simulate_phase_matching_spectrum():
    """Simulate phase matching spectrum similar to Figure 3b in the paper"""
    # Create waveguide with parameters similar to the paper
    waveguide = PPLNWaveguide()
    
    # Wavelength parameters
    lambda_e = 737e-9  # Emitter wavelength (737 nm)
    lambda_p = 1561e-9  # Pump wavelength (1561 nm)
    
    # Create wavelength range for networking wavelength
    lambda_n_center = 1397.5e-9  # Center wavelength from paper
    lambda_n_range = np.linspace(1390e-9, 1405e-9, 1000)  # Range to plot
    
    # Pump powers from the paper (Figure 3b)
    pump_powers = [0.123, 0.390, 0.647, 0.918, 1.390, 1.434]  # In Watts
    
    # Plot phase matching curves for different pump powers
    plt.figure(figsize=(10, 6))
    
    for power in pump_powers:
        efficiencies = waveguide.conversion_efficiency_vs_wavelength(
            lambda_n_range, lambda_p, lambda_e, power)
        
        # Normalize to the maximum power at each pump power for better comparison
        max_eff = max(efficiencies)
        if max_eff > 0:
            normalized_eff = [e / max_eff * power * 8 for e in efficiencies]  # Scale for visibility
        else:
            normalized_eff = efficiencies
            
        # Convert wavelengths to nm for plotting
        lambda_n_nm = [l * 1e9 for l in lambda_n_range]
        
        plt.plot(lambda_n_nm, normalized_eff, label=f'P = {power:.3f} W')
    
    plt.xlabel('Networking λ (nm)')
    plt.ylabel('Power at 737 ± 7 nm (mW)')
    plt.title('Phase Matching Spectrum vs Networking Wavelength')
    plt.legend()
    plt.grid(True)
    plt.savefig('phase_matching_spectrum.png', dpi=300)
    
    return 'phase_matching_spectrum.png'

def simulate_conversion_efficiency():
    """Simulate conversion efficiency vs pump power similar to Figure 3c in the paper"""
    # Create waveguide with parameters similar to the paper
    waveguide = PPLNWaveguide()
    
    # Pump power range
    pump_powers = np.linspace(0, 2.0, 100)  # 0 to 2 W
    
    # Calculate conversion efficiency
    efficiencies = waveguide.conversion_efficiency_vs_power(pump_powers)
    
    # Plot conversion efficiency vs pump power
    plt.figure(figsize=(10, 6))
    plt.plot(pump_powers * 1000, efficiencies, 'b-', linewidth=2)  # Convert to mW for plotting
    
    # Add data points similar to those in the paper
    data_powers = [0, 250, 500, 750, 1000, 1250, 1500, 1750]  # mW
    data_efficiencies = [0, 0.1, 0.2, 0.3, 0.38, 0.42, 0.43, 0.41]  # Approximate values from the paper
    plt.plot(data_powers, data_efficiencies, 'go', markersize=8)
    
    # Add annotations
    plt.annotate(f'P_max = 1.51±0.01 W\nη_max = 43±1.8 %', 
                 xy=(1510, 0.43), xytext=(1200, 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.xlabel('External pump power (mW)')
    plt.ylabel('External photon conversion efficiency')
    plt.title('Conversion Efficiency vs Pump Power')
    plt.grid(True)
    plt.savefig('conversion_efficiency.png', dpi=300)
    
    return 'conversion_efficiency.png'

if __name__ == "__main__":
    # Run simulations
    phase_matching_file = simulate_phase_matching_spectrum()
    efficiency_file = simulate_conversion_efficiency()
    
    print(f"Simulations complete. Output files: {phase_matching_file}, {efficiency_file}")
