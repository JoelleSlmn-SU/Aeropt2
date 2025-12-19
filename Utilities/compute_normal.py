import numpy as np
import pyvista as pv

mesh = pv.read(r"C:\Users\joell\OneDrive - Swansea University\Desktop\PhD Documents\01-Codes\Aeropt2\examples\Corner Bump Surface Opt\surfaces\n_0\output.vtk")
P = np.asarray(mesh.points, float)

# central CN (index 0 from earlier)
c = np.array([6407.1123, 888.3950, 265.0482], float)

# pick the 4 nearest points to the center (assumes those are your edge nodes)
d = np.linalg.norm(P - c[None,:], axis=1)
idx = np.argsort(d)[:5]          # include center + 4 nearest
Q = P[idx]

# PCA plane normal: smallest-variance eigenvector
Q0 = Q - Q.mean(axis=0)
_, _, Vt = np.linalg.svd(Q0, full_matrices=False)
n_hat = Vt[-1]
n_hat = n_hat / np.linalg.norm(n_hat)

print("indices used:", idx)
print("unit normal:", n_hat)