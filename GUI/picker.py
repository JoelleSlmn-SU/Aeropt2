import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.backend_bases import MouseEvent

class Picker3D:
    def __init__(self, ax, ctrlpts, threshold=10):
        self.ax = ax
        self.ctrlpts = np.array(ctrlpts)
        self.threshold = threshold
        self.selected = []

        self.scatter = ax.scatter([], [], [], c='red', label='Selected Points')
        self.fig = ax.figure
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x_proj, y_proj, _ = proj3d.proj_transform(
            self.ctrlpts[:, 0],
            self.ctrlpts[:, 1],
            self.ctrlpts[:, 2],
            self.ax.get_proj()
        )

        # Use event.x and event.y in pixels
        click = np.array([event.x, event.y])
        coords = np.column_stack((x_proj, y_proj))
        coords_display = self.ax.transData.transform(coords)
        dist = np.linalg.norm(coords_display - click, axis=1)
        idx = np.argmin(dist)

        if dist[idx] < self.threshold:
            if idx not in self.selected:
                self.selected.append(idx)
                self.sel_pts = self.ctrlpts[self.selected]
                self.scatter._offsets3d = (self.sel_pts[:, 0], self.sel_pts[:, 1], self.sel_pts[:, 2])
                self.fig.canvas.draw_idle()


