from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from views.mpl_canvas import MplCanvas
from matplotlib.path import Path as Polypath


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)

        self.canvas = MplCanvas(self, width=width, height=height, dpi=dpi)
        self.ax = self.canvas.ax
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.segmenter = None
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.setup_connections()

    def plot_image(self, img):
        # Clear the previous plot
        self.ax.clear()
        self.opacity = 0.7
        self.ax.imshow(img, alpha=self.opacity)
        self.ax.axis("off")  # Turn off axis labels
        self.canvas.draw()

    def handle_mouse_wheel(self, event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata

            zoom_factor = 1.1 if event.button == "up" else 1 / 1.1

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            new_xlim = [
                x - (x - xlim[0]) * zoom_factor,
                x + (xlim[1] - x) * zoom_factor,
            ]
            new_ylim = [
                y - (y - ylim[0]) * zoom_factor,
                y + (ylim[1] - y) * zoom_factor,
            ]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw()

    def setup_connections(self):
        self.canvas.mpl_connect("scroll_event", self.handle_mouse_wheel)

 