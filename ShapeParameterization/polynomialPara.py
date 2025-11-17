import numpy as np
import matplotlib.pyplot as plt

# Control points for the central curve Cs in the x-z plane
P0 = np.array([0.0, 0.0])
P1 = np.array([1.2, 0.4])
P2 = np.array([3.0, 0.4])
P3 = np.array([5.0, 0.0])

# Shape control parameter for central curve
omega_c = np.pi

def central_curve(x, P0, P1, P2, P3, omega_c):
    # Parameterize x between 0 and 1
    t = (x - P0[0]) / (P3[0] - P0[0])
    return (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1] + 0.02 * np.sin(omega_c * t)

# Generate grid for the surface
x_vals = np.linspace(0, 5, 100)
y_vals = np.linspace(-0.2, 0.2, 50)
x, y = np.meshgrid(x_vals, y_vals)

# Define cross-sectional curve shape control
omega = np.pi

def cross_section(y, height, omega):
    return height * (1 - (y / 0.2)**2) + 0.005 * np.sin(omega * y / 0.2)

# Generate z values for the surface
z = np.zeros_like(x)
for i in range(len(x_vals)):
    z_center = central_curve(x_vals[i], P0, P1, P2, P3, omega_c)
    z[:, i] = cross_section(y[:, i], z_center, omega)

# Plot the generated surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z (Bump Height)')
plt.show()
