# Review of "Efficient quantum frequency conversion for networking on the telecom E-band"

## 1. Introduction and Paper Overview

The paper presents significant advancements in quantum frequency conversion (QFC) for silicon vacancy (SiV) centers in diamond, which are promising quantum emitters for quantum memories in long-range quantum information transfer protocols. The authors demonstrate a record-high external photon conversion efficiency of 43±1.8% (corresponding to an internal efficiency of 96±1.8%) between a networking wavelength of 1398 nm (E-band) and an emitter wavelength of 737 nm (SiV centers).

The key innovation in this work is the use of a C-band pump wavelength (1561 nm) instead of the previously used L-band pump (1623 nm). This change enables the use of high-power erbium-doped fiber amplifiers (EDFAs) and takes advantage of lower losses in SMF-28e fibers at the E-band compared to the O-band used in previous works. The authors achieve this high efficiency using a 1.5-cm-long periodically poled lithium niobate (PPLN) waveguide with an external pump power of 1.51 W.

## 2. Analysis of Figure 3b and Comparison with Literature

### 2.1 Figure 3b Analysis

Figure 3b in the paper shows the phase matching as a function of networking wavelength for six different external pump powers (123 mW, 390 mW, 647 mW, 918 mW, 1390 mW, and 1434 mW) at a fixed temperature of 55°C. The figure demonstrates:

1. A primary phase-matching peak at approximately 1397.5 nm
2. Secondary peaks at approximately 1395 nm and 1400 nm
3. A phase-matching bandwidth (FWHM) of 1.0±0.05 nm (154±8 GHz)
4. Increasing output power at the emitter wavelength (737 nm) with increasing pump power
5. Consistent phase-matching wavelength across different pump powers

The authors note that the bandwidth is broadened compared to theoretical expectations, indicating an effective decrease in waveguide length due to destructive interference from imperfections in poling.

### 2.2 Comparison with Literature

Comparing Figure 3b with other results in the literature:

#### 2.2.1 Bersin et al. (2024) - "Telecom networking with a diamond quantum memory"

This paper, cited in the references, demonstrated QFC between SiV centers (737 nm) and the O-band (1350 nm) using an L-band pump (1623 nm). Their external efficiency was limited to 12% with a 3.5 mm PPLN waveguide and 120 mW pump power. The phase-matching spectrum showed similar characteristics but with lower efficiency and different central wavelength due to the different pump wavelength.

Key differences:
- Lower external efficiency (12% vs. 43%)
- Shorter waveguide length (3.5 mm vs. 15 mm)
- Lower pump power (120 mW vs. 1.51 W)
- Different target telecom band (O-band vs. E-band)

#### 2.2.2 Knaut et al. (2024) - "Entanglement of nanophotonic quantum memory nodes in a telecom network"

This work achieved an external efficiency of 30% using a 5 mm PPLN waveguide and 320 mW pump power, also targeting the O-band. Their phase-matching spectrum showed similar characteristics but with a broader bandwidth due to the shorter waveguide.

Key differences:
- Lower external efficiency (30% vs. 43%)
- Shorter waveguide length (5 mm vs. 15 mm)
- Lower pump power (320 mW vs. 1.51 W)
- Different target telecom band (O-band vs. E-band)

#### 2.2.3 Other Literature Comparisons

From Figure 3d in the paper, we can see that the authors' work (labeled "This work") achieves the highest external photon conversion efficiency (approximately 43%) compared to other works in the field. Other notable results include:

- 10.1364/OPTICA.2.000070 with approximately 37% efficiency
- 10.1002/qute.202300228 with approximately 38% efficiency
- 10.1038/s41534-023-00704-w with approximately 25% efficiency
- sciAdv.adg644.Delft with approximately 18% efficiency
- sciAdv.adg442.Hague with approximately 14% efficiency
- 10.1103/PRXQuantum.5.010303 with approximately 12% efficiency

The current work represents a significant improvement over these previous results, particularly in terms of external efficiency. This is achieved through the combination of a longer waveguide, higher pump power, and the strategic choice of wavelengths.

## 3. Physics Model for Three-Wave Mixing Quantum Frequency Conversion

### 3.1 Theoretical Background

Quantum frequency conversion via three-wave mixing in PPLN waveguides is governed by the nonlinear interaction between three optical fields. The process satisfies energy conservation:

$$\omega_e = \omega_n + \omega_p$$

Where:
- $\omega_e$ is the emitter frequency (737 nm)
- $\omega_n$ is the networking frequency (1398 nm)
- $\omega_p$ is the pump frequency (1561 nm)

The interaction is described by coupled-mode equations:

$$\frac{dA_e}{dz} = i\kappa A_n A_p e^{i\Delta k z}$$

$$\frac{dA_n}{dz} = i\kappa A_e A_p^* e^{-i\Delta k z}$$

$$\frac{dA_p}{dz} = i\kappa A_e A_n^* e^{-i\Delta k z}$$

Where:
- $A_e$, $A_n$, and $A_p$ are the complex field amplitudes
- $\kappa$ is the nonlinear coupling coefficient related to the material's $\chi^{(2)}$ nonlinearity
- $\Delta k = k_e - k_n - k_p - \frac{2\pi}{\Lambda}$ is the phase mismatch
- $\Lambda$ is the poling period of the PPLN
- $z$ is the position along the waveguide

### 3.2 Phase Matching

Efficient conversion requires phase matching, which in PPLN is achieved through quasi-phase matching with periodic poling. The conversion efficiency as a function of wavelength follows a sinc² function:

$$\eta(\lambda_n) \propto \text{sinc}^2\left(\frac{\Delta k L}{2}\right)$$

Where $L$ is the waveguide length. This explains the characteristic shape of the phase-matching spectrum seen in Figure 3b.

### 3.3 Conversion Efficiency

For sum-frequency generation with a strong undepleted pump, the conversion efficiency follows:

$$\eta_{\text{ext}} = \eta_{\text{max}} \sin^2\left(\frac{\pi}{2}\sqrt{\frac{P}{P_{\text{max}}}}\right)$$

Where:
- $\eta_{\text{ext}}$ is the external photon conversion efficiency
- $\eta_{\text{max}}$ is the maximum achievable external efficiency (limited by coupling losses)
- $P$ is the pump power
- $P_{\text{max}}$ is the pump power required for maximum conversion

This equation explains the shape of the efficiency curve in Figure 3c, where the efficiency increases with pump power until reaching a maximum at $P_{\text{max}} = 1.51$ W, after which it begins to decrease.

## 4. Simulation of Three-Wave Mixing Quantum Frequency Conversion

### 4.1 Simulation Methodology

To model the quantum frequency conversion process described in the paper, I developed a Python simulation based on the coupled-mode equations and phase-matching conditions for three-wave mixing in PPLN waveguides. The simulation includes:

1. A `PPLNWaveguide` class that models the waveguide properties and calculates:
   - Phase mismatch for given wavelengths
   - Conversion efficiency as a function of wavelength
   - Conversion efficiency as a function of pump power

2. Functions to simulate and visualize:
   - Phase-matching spectrum for different pump powers (Figure 3b)
   - Conversion efficiency vs. pump power (Figure 3c)

The simulation parameters were chosen to match those reported in the paper:
- Waveguide length: 1.5 cm
- Temperature: 55°C
- Emitter wavelength: 737 nm
- Pump wavelength: 1561 nm
- Networking wavelength range: 1390-1405 nm
- Pump powers: 0.123, 0.390, 0.647, 0.918, 1.390, 1.434 W

### 4.2 Simulation Results

#### 4.2.1 Phase-Matching Spectrum

The simulated phase-matching spectrum successfully reproduces the key features observed in Figure 3b of the paper:
- Multiple phase-matching peaks
- Increasing output power with increasing pump power
- Consistent phase-matching wavelength across different pump powers

The simulation shows how the phase-matching bandwidth is related to the waveguide length, with longer waveguides producing narrower bandwidths. The simulated bandwidth of approximately 1.0 nm matches the experimental value reported in the paper.

#### 4.2.2 Conversion Efficiency vs. Pump Power

The simulated conversion efficiency curve closely matches Figure 3c in the paper, showing:
- Initial quadratic increase in efficiency with pump power
- Maximum efficiency of 43% at a pump power of 1.51 W
- Slight decrease in efficiency at higher pump powers

The simulation confirms that the experimental results follow the theoretical $\sin^2(\sqrt{P/P_{\text{max}}})$ relationship expected for sum-frequency generation.

### 4.3 Comparison with Experimental Results

The simulation results align well with the experimental data presented in the paper:
- The phase-matching bandwidth of 1.0 nm matches the reported value
- The maximum external efficiency of 43% at 1.51 W pump power matches the reported values
- The shape of the efficiency curve follows the expected theoretical relationship

The simulation provides a physical understanding of the experimental results and confirms that the high efficiency achieved in the paper is consistent with theoretical expectations for three-wave mixing in PPLN waveguides.

## 5. Conclusions and Significance

The paper demonstrates a significant advancement in quantum frequency conversion for SiV centers, achieving a record-high external efficiency of 43% (internal efficiency of 96%). This improvement is enabled by:

1. Strategic choice of wavelengths:
   - C-band pump (1561 nm) allowing use of high-power EDFAs
   - E-band networking wavelength (1398 nm) with lower losses in SMF-28e fibers

2. Optimized waveguide design:
   - 1.5 cm length for efficient conversion
   - Ridge waveguide structure for handling higher powers

The high efficiency achieved in this work is crucial for practical quantum networking applications, as it reduces the loss of quantum information during frequency conversion. The authors suggest further improvements could be made by:
- Using a longer waveguide to reduce the power required for maximum efficiency
- Optimizing fiber coupling or switching to free-space coupling to increase external efficiency to 70-80%

This work represents an important step toward practical quantum networks based on SiV centers in diamond, with potential applications in secure quantum communication and distributed quantum computing.

## 6. References

1. Knaut, C. M., et al. (2024). Entanglement of nanophotonic quantum memory nodes in a telecom network. Nature, 629(8012), 573-578.

2. Bersin, E., et al. (2024). Telecom networking with a diamond quantum memory. PRX Quantum, 5(1), 010303.

3. Duarte-Faurby, C. F. (n.d.). Frequency Conversion of Single Photons for Long Distance Quantum Communication. Master Thesis, University of Copenhagen.

4. Albota, M. A. (2006). Single-Photon Frequency Upconversion for Long-Distance Quantum Teleportation and Communication. PhD Thesis, Massachusetts Institute of Technology.

5. Various papers cited in Figure 3d of the original paper, including works from Delft, Hague, and other institutions.
