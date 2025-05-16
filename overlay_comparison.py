"""
Create an overlay comparison between simulated spectrum and experimental data
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

def extract_data_from_image(image_path):
    """Extract data points from the experimental image"""
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to isolate the curves
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract points from the contours
    points = []
    for contour in contours:
        for point in contour:
            points.append(point[0])
    
    # Sort points by x-coordinate
    points.sort(key=lambda p: p[0])
    
    # Convert to numpy arrays
    x_points = np.array([p[0] for p in points])
    y_points = np.array([img.shape[0] - p[1] for p in points])  # Invert y-axis
    
    return x_points, y_points

def run_simulation_for_overlay():
    """Run the asymmetric simulation and return the data"""
    import numpy as np
    
    # Define the wavelength range
    lambda_n_range = np.linspace(1390, 1405, 1000)  # in nm
    
    # Define the peaks and their parameters
    peaks = [
        {"position": 1397.5, "weight": 1.0, "asymmetry": 0.3},
        {"position": 1395.0, "weight": 0.6, "asymmetry": 0.25},
        {"position": 1400.0, "weight": 0.6, "asymmetry": 0.35},
        {"position": 1393.0, "weight": 0.3, "asymmetry": 0.2}
    ]
    
    # Define the asymmetric sinc function
    def asymmetric_sinc(x, asymmetry=0.3):
        if np.isscalar(x):
            if x == 0:
                return 1.0
            else:
                sinc_val = np.sin(x) / x
                if x > 0:
                    return sinc_val * (1 - asymmetry * np.sin(x/2))
                else:
                    return sinc_val * (1 + asymmetry * np.sin(x/2))
        else:
            result = np.zeros_like(x, dtype=float)
            for i, val in enumerate(x):
                if val == 0:
                    result[i] = 1.0
                else:
                    sinc_val = np.sin(val) / val
                    if val > 0:
                        result[i] = sinc_val * (1 - asymmetry * np.sin(val/2))
                    else:
                        result[i] = sinc_val * (1 + asymmetry * np.sin(val/2))
            return result
    
    # Calculate the response for each peak
    total_response = np.zeros_like(lambda_n_range, dtype=float)
    for peak in peaks:
        delta_k = 2 * np.pi * (lambda_n_range - peak["position"]) / (0.5)
        response = peak["weight"] * asymmetric_sinc(delta_k, peak["asymmetry"])**2
        total_response += response
    
    # Add background
    background = 0.02 * np.exp(-(lambda_n_range - 1397)**2 / (10)**2) * (1 + 0.5 * np.sin(2*np.pi*(lambda_n_range - 1390)/(20)))
    total_response += background
    
    # Scale for highest pump power
    scaled_response = total_response * 4.0  # Scale to match experimental max
    
    return lambda_n_range, scaled_response

def create_overlay_comparison():
    """Create an overlay comparison between simulation and experimental data"""
    # Path to the experimental data image
    exp_image_path = "/home/ubuntu/upload/image.png"
    
    # Run simulation
    sim_x, sim_y = run_simulation_for_overlay()
    
    # Create the overlay plot
    plt.figure(figsize=(10, 6))
    
    # Plot the experimental image as background
    img = plt.imread(exp_image_path)
    plt.imshow(img, extent=[1390, 1405, 0, 8], aspect='auto', alpha=0.7)
    
    # Plot the simulation result
    plt.plot(sim_x, sim_y, 'r-', linewidth=2, label='Simulation')
    
    # Add labels and title
    plt.xlabel('Networking λ (nm)')
    plt.ylabel('Power at 737 ± 7 nm (mW)')
    plt.title('Overlay: Simulation vs Experimental Data')
    plt.xlim(1390, 1405)
    plt.ylim(0, 8)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the overlay comparison
    plt.savefig('/home/ubuntu/simulation_vs_experiment_overlay.png', dpi=300, bbox_inches='tight')
    
    return '/home/ubuntu/simulation_vs_experiment_overlay.png'

if __name__ == "__main__":
    overlay_file = create_overlay_comparison()
    print(f"Overlay comparison created: {overlay_file}")
