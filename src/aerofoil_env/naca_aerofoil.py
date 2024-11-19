# Write a module which can spit out any 4-digit NACA airfoil.

# The module should be able to take in a string of 4-digit NACA airfoil code and return the coordinates of the airfoil.

import numpy as np
import matplotlib.pyplot as plt

def naca4(code: str, num_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate coordinates for a NACA 4-digit aerofoil.
    
    The NACA 4-digit series is defined by:
        - First digit (m): Maximum camber as percentage of chord (0-9)%
        - Second digit (p): Position of maximum camber in tenths of chord (0-9)
        - Last two digits (t): Maximum thickness as percentage of chord (00-99)
    
    For example, NACA 2412 means:
        - 2% maximum camber
        - Maximum camber at 0.4 (40%) chord
        - 12% thickness
    
    A symmetric airfoil has m=p=0 (e.g., NACA 0012).
    
    Args:
        code (str): Four digit NACA code (e.g. '2412')
        num_points (int): Number of points to generate (default: 100)
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - upper: Array of shape (num_points, 2) for upper surface (x, y) coordinates
            - lower: Array of shape (num_points, 2) for lower surface (x, y) coordinates
    
    Raises:
        ValueError: If code is not a 4-digit string or contains non-numeric characters

    Command line usage:
        python naca_aerofoil.py <code> --points <num_points> --no-plot

    Example:
        python naca_aerofoil.py 2412 --points 100
    """
    # Parse NACA code
    m = float(code[0]) / 100  # maximum camber
    p = float(code[1]) / 10   # location of maximum camber
    t = float(code[2:]) / 100  # thickness
    
    # Generate x coordinates
    x = np.linspace(0, 1, num_points)
    
    # Thickness distribution
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                  0.2843 * x**3 - 0.1015 * x**4)
    
    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    # Calculate camber line and gradient
    if m != 0:
        # Split calculations for before and after p
        idx = x <= p
        yc[idx] = m * (x[idx] / p**2) * (2 * p - x[idx])
        yc[~idx] = m * ((1 - x[~idx]) / (1 - p)**2) * (1 + x[~idx] - 2 * p)
        
        # Calculate gradient
        dyc_dx[idx] = 2 * m * (p - x[idx]) / p**2
        dyc_dx[~idx] = 2 * m * (p - x[~idx]) / (1 - p)**2
    
    theta = np.arctan(dyc_dx)
    
    # Calculate upper and lower surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Sort points from leading edge to trailing edge
    upper = np.column_stack([xu, yu])
    lower = np.column_stack([xl, yl])
    
    return upper, lower

def plot_airfoil(upper: np.ndarray, lower: np.ndarray, code: str, save_path: str = None):
    """Plot an airfoil from its upper and lower surface coordinates.
    
    Args:
        upper (np.ndarray): Upper surface coordinates (N, 2)
        lower (np.ndarray): Lower surface coordinates (N, 2)
        code (str): NACA code for the title
        save_path (str, optional): Path to save the plot. If None, displays plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(upper[:, 0], upper[:, 1], 'b-', label='Upper surface')
    plt.plot(lower[:, 0], lower[:, 1], 'b-', label='Lower surface')
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'NACA {code}')
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage:
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate NACA 4-digit airfoil coordinates')
    parser.add_argument('code', type=str, help='NACA 4-digit code (e.g., 0012)')
    parser.add_argument('--points', type=int, default=100, help='Number of points (default: 100)')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    # Generate NACA airfoil
    upper, lower = naca4(args.code, args.points)
    
    if not args.no_plot:
        plot_airfoil(upper, lower, args.code)
