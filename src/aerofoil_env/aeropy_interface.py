# src/aerofoil_env/aeropy_interface.py

from aeropy.airfoils import shapes
from aeropy.airfoils import analysis
import numpy as np

def compute_aero_coefficients(airfoil: str, num_points: int, Reynolds: float, alpha: float | list[float] = 0.0) -> dict:
    """
    Compute aerodynamic coefficients for a specified aerofoil using AeroPy.

    Args:
        airfoil (str): Aerofoil designation (e.g., '0012').
        num_points (int): Number of points to generate the aerofoil geometry.
        Reynolds (float): Reynolds number.
        alpha (float or list[float], optional): Angle of attack in degrees. Can be either:
            - A single value (e.g., 0.0)
            - A list of values (e.g., [0.0, 5.0, 10.0]) to compute coefficients at multiple angles
            Defaults to 0.0.

    Returns:
        dict: Dictionary containing aerodynamic coefficients. For single alpha:
            - 'CL': float - Lift coefficient
            - 'CD': float - Drag coefficient
            - 'CM': float - Moment coefficient
            - 'CP': float - Pressure coefficient
            - 'Alpha': float - Angle of attack used
            For multiple alphas, each value will be a list corresponding to each input angle.
    """
    # Get coordinates
    x, y = shapes.naca(airfoil, num_points)
    
    # Convert alpha to radians
    alpha_rad = np.array([alpha]) * np.pi / 180. if isinstance(alpha, (int, float)) else np.array(alpha) * np.pi / 180.
    
    try:
        # Call xfoil with viscous analysis enabled
        alphas, c_p, control_x, control_y, c_l, c_d = analysis.xfoil(
            alpha_rad, x, y,
            Re=Reynolds,
            Mach=0.0,
            Ncrit=9,
            xtr=None,
            iterlim=100
        )
        
        return {
            'CL': c_l[0] if isinstance(alpha, (int, float)) else c_l,
            'CD': c_d[0] if isinstance(alpha, (int, float)) else c_d,
            'Alpha': alpha,
            'converged': True
        }
        
    except ValueError as e:
        print(f"XFOIL failed to converge for NACA {airfoil}")
        # Return default values that discourage the agent from this shape
        return {
            'CL': 0.0,
            'CD': 1.0,  # High drag to discourage non-convergent shapes
            'Alpha': alpha,
            'converged': False
        }
