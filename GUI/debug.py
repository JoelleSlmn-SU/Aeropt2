import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

output_dir = os.path.join(os.getcwd(), "Outputs", "pleaseeeee")

def load_and_show(fig_path):
    with open(fig_path, "rb") as f:
        old_fig = pickle.load(f)

    new_fig = plt.figure()
    new_ax = new_fig.add_subplot(111, projection='3d')

    for old_ax in old_fig.axes:
        for col in old_ax.collections:
            try:
                segments = col._segments3d
                xs = [seg[0][0] for seg in segments]
                ys = [seg[0][1] for seg in segments]
                zs = [seg[0][2] for seg in segments]
                new_ax.scatter(xs, ys, zs, s=5)  # basic re-plot
            except AttributeError:
                pass  # skip if not a scatter/line

    new_ax.set_title(old_ax.get_title())
    new_ax.set_xlabel(old_ax.get_xlabel())
    new_ax.set_ylabel(old_ax.get_ylabel())
    new_ax.set_zlabel(old_ax.get_zlabel())

    plt.tight_layout()
    plt.show()

# Show the figures
load_and_show(os.path.join(output_dir, "step_2_t_nodes_morphed.fig.pkl"))
load_and_show(os.path.join(output_dir, "step_3_boundary_motion_recovered.fig.pkl"))
load_and_show(os.path.join(output_dir, "step_4_propagate_into_u_nodes.fig.pkl"))
load_and_show(os.path.join(output_dir, "step_5_final_morphed_mesh.fig.pkl"))
