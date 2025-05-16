# Three-Wave Mixing Quantum Frequency Conversion Model

## Physics Background

Quantum frequency conversion (QFC) via three-wave mixing in periodically poled lithium niobate (PPLN) waveguides is a process where two photons combine in a nonlinear medium to produce a third photon. The frequencies satisfy energy conservation:

$$\omega_e = \omega_n + \omega_p$$

Where:
- $\omega_e$ is the emitter frequency (higher frequency, shorter wavelength)
- $\omega_n$ is the networking frequency (lower frequency, longer wavelength)
- $\omega_p$ is the pump frequency (lower frequency, longer wavelength)

In the paper, the process converts between:
- Emitter wavelength: $\lambda_e = 737$ nm (SiV centers in diamond)
- Networking wavelength: $\lambda_n = 1398$ nm (E-band telecom)
- Pump wavelength: $\lambda_p = 1561$ nm (C-band telecom)

## Coupled-Mode Equations

The three-wave mixing process in a PPLN waveguide can be described by the following coupled-mode equations:

$$\frac{dA_e}{dz} = i\kappa A_n A_p e^{i\Delta k z}$$

$$\frac{dA_n}{dz} = i\kappa A_e A_p^* e^{-i\Delta k z}$$

$$\frac{dA_p}{dz} = i\kappa A_e A_n^* e^{-i\Delta k z}$$

Where:
- $A_e$, $A_n$, and $A_p$ are the complex field amplitudes of the emitter, networking, and pump waves
- $\kappa$ is the nonlinear coupling coefficient related to the material's $\chi^{(2)}$ nonlinearity
- $\Delta k = k_e - k_n - k_p - \frac{2\pi}{\Lambda}$ is the phase mismatch
- $\Lambda$ is the poling period of the PPLN
- $z$ is the position along the waveguide

## Phase Matching

For efficient conversion, phase matching is crucial. In PPLN, quasi-phase matching is achieved through periodic poling with period $\Lambda$ chosen to satisfy:

$$\Delta k = k_e - k_n - k_p - \frac{2\pi}{\Lambda} \approx 0$$

The phase matching condition determines the spectral response of the conversion process. The conversion efficiency as a function of wavelength follows a sinc² function:

$$\eta(\lambda_n) \propto \text{sinc}^2\left(\frac{\Delta k L}{2}\right)$$

Where $L$ is the waveguide length.

## Conversion Efficiency

For sum-frequency generation (SFG) with a strong undepleted pump, the conversion efficiency can be expressed as:

$$\eta_{\text{ext}} = \eta_{\text{max}} \sin^2\left(\frac{\pi}{2}\sqrt{\frac{P}{P_{\text{max}}}}\right)$$

Where:
- $\eta_{\text{ext}}$ is the external photon conversion efficiency
- $\eta_{\text{max}}$ is the maximum achievable external efficiency (limited by coupling losses)
- $P$ is the pump power
- $P_{\text{max}}$ is the pump power required for maximum conversion

The internal efficiency (ignoring coupling losses) approaches 100% when the pump power reaches $P_{\text{max}}$.

## Spectral Bandwidth

The spectral bandwidth of the conversion process is inversely proportional to the waveguide length:

$$\Delta \lambda_{\text{FWHM}} \propto \frac{1}{L}$$

This explains why longer waveguides have narrower phase-matching bandwidths.

## Noise Sources

The main noise sources in the QFC process are:
1. Spontaneous parametric down-conversion (SPDC)
2. Raman scattering (Stokes and anti-Stokes)

These noise mechanisms limit the signal-to-noise ratio of the converted photons.
