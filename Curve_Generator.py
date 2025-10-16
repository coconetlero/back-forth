import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial import distance
import warnings

class CurveGenerator:
    def __init__(self):
        self.curves = {}
    
    def generate_bezier_curve(self, control_points, num_points=100, curvature_factor=1.0):
        """
        Generate Bézier curve with adjustable curvature.
        
        Parameters:
        - control_points: array of control points (n, 2)
        - num_points: number of points in the final curve
        - curvature_factor: >1 for more curvature, <1 for less curvature
        """
        n = len(control_points) - 1
        
        # Adjust control points based on curvature factor
        if curvature_factor != 1.0:
            center = np.mean(control_points, axis=0)
            adjusted_points = center + curvature_factor * (control_points - center)
        else:
            adjusted_points = control_points
        
        # Generate Bézier curve
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        
        for i in range(num_points):
            curve[i] = self._bezier_point(adjusted_points, t[i])
        
        return curve
    
    def _bezier_point(self, control_points, t):
        """Calculate point on Bézier curve at parameter t."""
        n = len(control_points) - 1
        point = np.zeros(2)
        
        for i in range(n + 1):
            binomial = math.comb(n, i)
            point += binomial * (1 - t)**(n - i) * t**i * control_points[i]
        
        return point
    
    def generate_cubic_spline(self, points, num_points=100, smoothing=0, curvature_weights=None):
        """
        Generate cubic spline with adjustable smoothing and curvature.
        
        Parameters:
        - points: array of points (n, 2)
        - num_points: number of points in the final curve
        - smoothing: smoothing factor (0 for interpolation, >0 for smoothing)
        - curvature_weights: weights for curvature control at each point
        """
        x = points[:, 0]
        y = points[:, 1]
        
        # Calculate cumulative distance for parameterization
        dist = np.zeros(len(points))
        for i in range(1, len(points)):
            dist[i] = dist[i-1] + np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
        dist = dist / dist[-1]
        
        # Generate parameter values for interpolation
        t_new = np.linspace(0, 1, num_points)
        
        if curvature_weights is not None and len(curvature_weights) == len(points):
            # Use weighted spline for curvature control
            spline_x = interpolate.UnivariateSpline(dist, x, w=curvature_weights, s=smoothing)
            spline_y = interpolate.UnivariateSpline(dist, y, w=curvature_weights, s=smoothing)
        else:
            spline_x = interpolate.UnivariateSpline(dist, x, s=smoothing)
            spline_y = interpolate.UnivariateSpline(dist, y, s=smoothing)
        
        curve_x = spline_x(t_new)
        curve_y = spline_y(t_new)
        
        return np.column_stack((curve_x, curve_y))
    
    def generate_bspline(self, points, num_points=100, degree=3, curvature_control=0.5):
        """
        Generate B-spline with curvature control.
        
        Parameters:
        - points: array of points (n, 2)
        - num_points: number of points in the final curve
        - degree: degree of B-spline
        - curvature_control: 0-1, lower values = tighter curves
        """
        # Adjust knot vector based on curvature control
        n = len(points)
        
        # Create parameter values with curvature control
        t = np.zeros(n)
        for i in range(1, n):
            t[i] = t[i-1] + curvature_control ** i
        
        t = t / t[-1]
        
        # Fit B-spline
        try:
            tck, u = interpolate.splprep([points[:, 0], points[:, 1]], u=t, k=degree, s=0)
            u_new = np.linspace(0, 1, num_points)
            curve = interpolate.splev(u_new, tck)
            return np.column_stack(curve)
        except:
            # Fallback to simpler interpolation if B-spline fails
            return self.generate_cubic_spline(points, num_points)
    
    def generate_variable_curvature_curve(self, base_points, curvature_profile, num_points=100):
        """
        Generate curve with variable curvature along its length.
        
        Parameters:
        - base_points: array of base points (n, 2)
        - curvature_profile: array of curvature values along the curve
        - num_points: number of points in the final curve
        """
        # First generate a base spline
        base_curve = self.generate_cubic_spline(base_points, num_points * 2)
        
        # Calculate normals to the curve
        tangents = np.gradient(base_curve, axis=0)
        tangents_norm = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        normals = np.column_stack((-tangents_norm[:, 1], tangents_norm[:, 0]))
        
        # Interpolate curvature profile to match curve length
        curvature_interp = np.interp(
            np.linspace(0, 1, len(base_curve)),
            np.linspace(0, 1, len(curvature_profile)),
            curvature_profile
        )
        
        # Apply curvature as offset along normals
        max_curvature = np.max(np.abs(curvature_profile))
        if max_curvature > 0:
            curvature_interp = curvature_interp / max_curvature * 0.1  # Scale factor
        
        curved_points = base_curve + normals * curvature_interp[:, np.newaxis]
        
        # Resample to desired number of points
        final_curve = self.resample_curve(curved_points, num_points)
        
        return final_curve
    
    def resample_curve(self, curve, num_points):
        """Resample curve to have exactly num_points with equal spacing."""
        # Calculate cumulative distance
        distances = np.zeros(len(curve))
        for i in range(1, len(curve)):
            distances[i] = distances[i-1] + np.linalg.norm(curve[i] - curve[i-1])
        
        # Create new parameterization
        t_new = np.linspace(0, distances[-1], num_points)
        
        # Interpolate
        curve_x = np.interp(t_new, distances, curve[:, 0])
        curve_y = np.interp(t_new, distances, curve[:, 1])
        
        return np.column_stack((curve_x, curve_y))
    

    def calculate_curvature(self, curve):
        """Calculate curvature along the curve."""
        dx = np.gradient(curve[:, 0])
        dy = np.gradient(curve[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
        return curvature


# Example usage and demonstration
def demonstrate_curve_generator():
    # Initialize generator
    generator = CurveGenerator()
    
    # Create some sample control points
    np.random.seed(42)
    base_points = np.array([
        [0, 0],
        [1, 2],
        [3, 1],
        [5, 3],
        [7, 0],
        [9, 2]
    ])
    
    # Generate different curves with varying curvature
    plt.figure(figsize=(15, 10))
    
    # 1. Bézier curves with different curvature factors
    plt.subplot(2, 3, 1)
    curvature_factors = [0.5, 1.0, 1.5, 2.0]
    colors = ['red', 'blue', 'green', 'orange']
    
    for factor, color in zip(curvature_factors, colors):
        bezier_curve = generator.generate_bezier_curve(base_points, curvature_factor=factor)
        plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], color=color, 
                label=f'Factor: {factor}', linewidth=2)
    
    plt.plot(base_points[:, 0], base_points[:, 1], 'ko--', alpha=0.5, label='Control Points')
    plt.title('Bézier Curves - Varying Curvature Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 2. Cubic splines with different smoothing
    plt.subplot(2, 3, 2)
    smoothing_factors = [0, 0.1, 1.0, 10.0]
    
    for smoothing, color in zip(smoothing_factors, colors):
        spline_curve = generator.generate_cubic_spline(base_points, smoothing=smoothing)
        plt.plot(spline_curve[:, 0], spline_curve[:, 1], color=color, 
                label=f'Smoothing: {smoothing}', linewidth=2)
    
    plt.plot(base_points[:, 0], base_points[:, 1], 'ko--', alpha=0.5, label='Control Points')
    plt.title('Cubic Splines - Varying Smoothing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 3. B-splines with different curvature control
    plt.subplot(2, 3, 3)
    curvature_controls = [0.2, 0.5, 0.8, 1.0]
    
    for control, color in zip(curvature_controls, colors):
        bspline_curve = generator.generate_bspline(base_points, curvature_control=control)
        plt.plot(bspline_curve[:, 0], bspline_curve[:, 1], color=color, 
                label=f'Control: {control}', linewidth=2)
    
    plt.plot(base_points[:, 0], base_points[:, 1], 'ko--', alpha=0.5, label='Control Points')
    plt.title('B-Splines - Varying Curvature Control')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 4. Variable curvature curve
    plt.subplot(2, 3, 4)
    curvature_profile = np.sin(np.linspace(0, 4*np.pi, len(base_points))) * 2
    variable_curve = generator.generate_variable_curvature_curve(base_points, curvature_profile)
    
    plt.plot(variable_curve[:, 0], variable_curve[:, 1], 'purple', linewidth=2, label='Variable Curvature')
    plt.plot(base_points[:, 0], base_points[:, 1], 'ko--', alpha=0.5, label='Base Points')
    plt.title('Variable Curvature Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 5. Curvature analysis
    plt.subplot(2, 3, 5)
    curves = []
    labels = []
    
    # Generate curves for analysis
    bezier_low = generator.generate_bezier_curve(base_points, curvature_factor=0.5)
    bezier_high = generator.generate_bezier_curve(base_points, curvature_factor=2.0)
    spline_smooth = generator.generate_cubic_spline(base_points, smoothing=1.0)
    
    curves.extend([bezier_low, bezier_high, spline_smooth])
    labels.extend(['Bézier (low curv)', 'Bézier (high curv)', 'Spline (smooth)'])
    
    for curve, label in zip(curves, labels):
        curvature = generator.calculate_curvature(curve)
        arc_length = np.arange(len(curvature)) / len(curvature)
        plt.plot(arc_length, curvature, label=label, linewidth=2)
    
    plt.title('Curvature Analysis')
    plt.xlabel('Normalized Arc Length')
    plt.ylabel('Curvature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Complex example with custom curvature
    plt.subplot(2, 3, 6)
    
    # Create a spiral-like curvature profile
    t_custom = np.linspace(0, 1, 50)
    custom_curvature = np.sin(8 * np.pi * t_custom) * np.exp(-3 * t_custom) * 5
    
    custom_curve = generator.generate_variable_curvature_curve(base_points, custom_curvature)
    
    plt.plot(custom_curve[:, 0], custom_curve[:, 1], 'red', linewidth=2, label='Custom Curvature')
    plt.plot(base_points[:, 0], base_points[:, 1], 'ko--', alpha=0.5, label='Base Points')
    plt.title('Custom Curvature Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Interactive function to experiment with parameters
def interactive_curve_exploration():
    """Interactive function to explore different curvature parameters."""
    generator = CurveGenerator()
    
    # Simple example points
    points = np.array([[0, 0], [1, 3], [3, 1], [5, 4], [7, 0]])
    
    print("Curve Generator Demo")
    print("===================")
    
    # Test different curvature factors for Bézier
    print("\n1. Bézier Curves with different curvature factors:")
    for factor in [0.3, 0.7, 1.0, 1.5, 2.0]:
        curve = generator.generate_bezier_curve(points, curvature_factor=factor)
        curvature = generator.calculate_curvature(curve)
        print(f"   Factor {factor}: Max curvature = {np.max(curvature):.3f}")
    
    # Test different smoothing for splines
    print("\n2. Cubic Splines with different smoothing:")
    for smoothing in [0, 0.5, 2.0, 10.0]:
        curve = generator.generate_cubic_spline(points, smoothing=smoothing)
        curvature = generator.calculate_curvature(curve)
        print(f"   Smoothing {smoothing}: Max curvature = {np.max(curvature):.3f}")

if __name__ == "__main__":
    # Run demonstration
    demonstrate_curve_generator()
    
    # Run interactive exploration
    interactive_curve_exploration()