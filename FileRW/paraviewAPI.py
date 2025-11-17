import pyvista as pv
import panel as pn
import numpy as np
import os

pn.extension('vtk')

# === Upload/Load VTM ===
# Define the path to your VTM file
# Example: your_file = "Outputs/Mesh Data/wing.vtm"
your_file = os.path.join(os.getcwd(), "Outputs", "Mesh Data", "wing.vtm")

# Read the multiblock dataset
reader = pv.get_reader(your_file)
multiblock = reader.read()

# === Assign surface IDs and combine ===
tagged_blocks = []
surface_labels = []

for i, block in enumerate(multiblock):
    if block is None:
        continue
    block = block.copy()
    block.cell_data["surface_id"] = np.full(block.n_cells, i)
    tagged_blocks.append(block)
    surface_labels.append(f"Surface {i+1}")

if not tagged_blocks:
    raise ValueError("No surfaces found in the VTM file.")

combined = tagged_blocks[0]
for block in tagged_blocks[1:]:
    combined = combined.merge(block)

# === Visibility Dictionary ===
visible = {i: True for i in range(len(surface_labels))}

# === Plotter Setup ===
plotter = pv.Plotter(notebook=True)
plot_pane = plotter.show(return_viewer=True)

def update_plot():
    plotter.clear()
    visible_ids = [i for i, show in visible.items() if show]
    mask = np.isin(combined.cell_data["surface_id"], visible_ids)
    visible_mesh = combined.extract_cells(mask)
    plotter.add_mesh(visible_mesh, scalars="surface_id", cmap="tab20", show_edges=True)
    plotter.reset_camera()
    plotter.render()

# === GUI Button Setup ===
buttons = []
for i, label in enumerate(surface_labels):
    toggle = pn.widgets.Toggle(name=label, value=True, button_type='primary')

    def make_callback(index):
        def cb(event):
            visible[index] = event.new
            update_plot()
        return cb

    toggle.param.watch(make_callback(i), "value")
    buttons.append(toggle)

# === Build Layout ===
controls = pn.Column(*buttons, width=200, sizing_mode="stretch_height")
layout = pn.Row(controls, plot_pane, sizing_mode="stretch_both")

# === Initial Display ===
update_plot()
layout.servable()
