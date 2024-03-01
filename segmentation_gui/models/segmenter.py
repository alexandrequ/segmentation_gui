import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
from matplotlib.lines import Line2D
import pickle
import cv2
from matplotlib.widgets import RectangleSelector, PolygonSelector
from matplotlib.widgets import RectangleSelector, PolygonSelector
from matplotlib.path import Path as Polypath
import numpy as np
from PySide6.QtWidgets import QFileDialog


class Segmenter:
    def __init__(
        self, img, filePath, matplotlib_widget, auto_mask_generator, predictor
    ):
        self.matplotlib_widget = matplotlib_widget
        # self.sam_checkbox.setChecked(True)
        self.img = img  
        self.filePath = filePath
        self.min_mask_region_area = 100
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
        self.rect = (0, 0, self.img.shape[1], self.img.shape[0])
        self.auto_mask_generator = auto_mask_generator
        self.predictor = predictor

        # load image
        self.predictor.set_image(self.img)
        file_root, _ = os.path.splitext(filePath)
        file_path_pkl = file_root + ".pkl"
        if os.path.exists(file_path_pkl):
            with open(file_path_pkl, "rb") as f:
                masks = pickle.load(f)
            print("load embedding")
        else:
            print("Generating automatic masks ... ", end="")
            masks = self.auto_mask_generator.generate(self.img)
            with open(file_path_pkl, "wb") as f:
                pickle.dump(masks, f)
                print(file_path_pkl)
            print("Done")

        max_n_masks = 10000
        self.auto_masks = np.zeros(
            (self.img.shape[0], self.img.shape[1], min(len(masks), max_n_masks)),
            dtype=bool,
        )
        for i in range(self.auto_masks.shape[2]):
            self.auto_masks[:, :, i] = masks[i]["segmentation"]

        # Initialize the Matplotlib plot within the MatplotlibWidget
        self.matplotlib_widget.ax.clear()
        self.matplotlib_widget.plot_image(self.img)
        self.matplotlib_widget.ax.set_title(
            "Press 'h' to show/hide commands.", fontsize=10
        )
        self.matplotlib_widget.ax.autoscale(False)

        self.label = 1
        (self.add_plot,) = self.matplotlib_widget.ax.plot(
            [], [], "o", markerfacecolor="green", markeredgecolor="k", markersize=5
        )
        (self.rem_plot,) = self.matplotlib_widget.ax.plot(
            [], [], "x", markerfacecolor="red", markeredgecolor="red", markersize=5
        )
        self.color_set = set()
        self.current_color = self.pick_color()
        self.legend_elements = [
            Line2D(
                [0],
                [0],
                color=np.array(self.current_color) / 255,
                lw=2,
                label=f"Mask {self.label}\n",
            )
        ]

        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = (
            [],
            [],
            [],
            [],
            [],
        )
        self.mask_data = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8
        )
        self.draw_mask = np.zeros(
            (self.img.shape[0], self.img.shape[1]), dtype=np.uint8
        )

        self.drawing_case = 1

        line_properties = {"color": "blue", "linewidth": 2}
        vertex_properties = {
            "color": "red",
            "markersize": 3,
        }  # 'size' controls the size of the dots
        self.ps = PolygonSelector(
            self.matplotlib_widget.ax,
            self.polygon_select_callback,
            useblit=True,
            props=line_properties,
            handle_props=vertex_properties,
        )
        self.ps.set_active(False)

        # Create a rectangle selector for ROI
        self.rs = RectangleSelector(
            self.matplotlib_widget.ax,
            self.roi_select_callback,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="none", edgecolor="green", linewidth=1.5),
        )
        self.rs.set_active(False)

        for i in range(3):
            self.mask_data[:, :, i] = self.current_color[i]

        self.mask_plot = self.matplotlib_widget.ax.imshow(self.mask_data)

        self.prev_mask_data = np.zeros(
            (self.img.shape[0], self.img.shape[1], 4), dtype=np.uint8
        )
        self.prev_mask_plot = self.matplotlib_widget.ax.imshow(
            self.prev_mask_data, label=r"$y={}x$".format(self.label)
        )

        # Connect Matplotlib events to handling methods

        self.show_help_text = False
        self.help_text = self.matplotlib_widget.ax.text(2, 0, "", fontsize=10)
        self.full_legend = []
        self.opacity = 100  # out of 255
        self.global_masks = np.zeros(
            (self.img.shape[0], self.img.shape[1]), dtype=np.uint8
        )

    def pick_color(self):
        while True:
            color = tuple(np.random.randint(low=0, high=255, size=3).tolist())
            if color not in self.color_set:
                self.color_set.add(color)
                return color

    def _on_key(self, event):
        if event.key == "z":
            self.undo()

        elif event.key == "enter":
            print("enter")
            self.new_mask()

        elif event.key == "h":
            if not self.show_help_text:
                self.help_text.set_text(
                    "• 'left click': select a point inside object to label\n"
                    "• 'right click': select a point to exclude from label\n"
                    "• 'enter': confirm current label and create new\n"
                    "• 'z': undo point\n"
                    "• 'esc': close and save"
                )
                self.help_text.set_bbox(
                    dict(facecolor="white", alpha=1, edgecolor="black")
                )
                self.show_help_text = True
            else:
                self.help_text.set_text("")
                self.show_help_text = False
            self.matplotlib_widget.canvas.draw()

    def show_points(self, plot, xs, ys):
        plot.set_data(xs, ys)
        self.matplotlib_widget.canvas.draw()

    def clear_mask(self):
        self.mask_data.fill(0)
        self.draw_mask.fill(0)
        self.mask_plot.set_data(self.mask_data)
        self.matplotlib_widget.canvas.draw()

    def get_mask(self):
        add_compare = np.sum(self.auto_masks[self.add_ys, self.add_xs], axis=0)
        add_compare += np.sum(~self.auto_masks[self.rem_ys, self.rem_xs], axis=0)
        matches = np.where(add_compare == len(self.add_xs) + len(self.rem_xs))[0]

        found_match = False
        for match in matches:

            mask = self.auto_masks[:, :, match]
            if not np.any(np.logical_and(mask, self.global_masks)):
                found_match = True
                break

        if not found_match:
            # print("No matches in auto masks, calling single predictor")
            mask = self.generate_mask()
            mask[self.global_masks > 0] = 0
            mask = self.remove_small_regions(mask, self.min_mask_region_area, "holes")
            mask = self.remove_small_regions(mask, self.min_mask_region_area, "islands")

        self.mask_data[:, :, 3] = mask * self.opacity
        self.mask_plot.set_data(self.mask_data)
        self.matplotlib_widget.canvas.draw()

    def load_embedding(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Load Embedding",
            "",
            "All Files (*);;Text Files (*.txt)",
            options=options,
        )
        if fileName:
            file_root, _ = os.path.splitext(fileName)
            file_path_pkl = file_root + ".pkl"
            if os.path.exists(file_path_pkl):
                with open(file_path_pkl, "rb") as f:
                    masks = pickle.load(f)
        max_n_masks = 10000
        self.auto_masks = np.zeros(
            (self.img.shape[0], self.img.shape[1], min(len(masks), max_n_masks)),
            dtype=bool,
        )
        for i in range(self.auto_masks.shape[2]):
            self.auto_masks[:, :, i] = masks[i]["segmentation"]

    def gen_embedding(self):
        masks = self.auto_mask_generator.generate(self.img)
        file_root, _ = os.path.splitext(filePath)
        file_path_pkl = file_root + ".pkl"
        with open(file_path_pkl, "wb") as f:
            pickle.dump(masks, f)
        max_n_masks = 10000
        self.auto_masks = np.zeros(
            (self.img.shape[0], self.img.shape[1], min(len(masks), max_n_masks)),
            dtype=bool,
        )
        for i in range(self.auto_masks.shape[2]):
            self.auto_masks[:, :, i] = masks[i]["segmentation"]

    def generate_mask(self):
        mask, _, _ = self.predictor.predict(
            point_coords=np.array(
                list(zip(self.add_xs, self.add_ys))
                + list(zip(self.rem_xs, self.rem_ys))
            ),
            point_labels=np.array([1] * len(self.add_xs) + [0] * len(self.rem_xs)),
            multimask_output=False,
        )
        return mask[0].astype(np.uint8)

    def undo(self):
        if len(self.trace) == 0:
            return

        if self.trace[-1]:
            self.add_xs = self.add_xs[:-1]
            self.add_ys = self.add_ys[:-1]
            self.show_points(self.add_plot, self.add_xs, self.add_ys)
        else:
            self.rem_xs = self.rem_xs[:-1]
            self.rem_ys = self.rem_ys[:-1]
            self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

        self.trace.pop()

        if len(self.trace) != 0:
            self.get_mask()
        else:
            self.clear_mask()

    def CLAHE_normalize(self, bgr, clahe):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return bgr

    def grabcut(self):

        if self.draw_mask.sum() == 0:  # initialize first grabcut with rect
            initparam = cv2.GC_INIT_WITH_RECT
        else:
            initparam = cv2.GC_INIT_WITH_MASK

        grabcut_mask = np.zeros(self.img.shape[:2], np.uint8)
        grabcut_mask[self.draw_mask > 0] = cv2.GC_FGD
        grabcut_mask[self.draw_mask == 0] = cv2.GC_BGD
        grabcut_mask[self.mask_data[:, :, 3] > 0] = cv2.GC_FGD
        grabcut_mask[self.mask_data[:, :, 3] == 0] = cv2.GC_BGD
        grabcut_mask[self.global_masks > 0] = cv2.GC_BGD
        for x, y in zip(self.add_xs, self.add_ys):
            # Mark points added by left-click as foreground
            grabcut_mask[y, x] = cv2.GC_FGD

        for x, y in zip(self.rem_xs, self.rem_ys):

            # Mark points added by right-click as background
            grabcut_mask[y, x] = cv2.GC_BGD

        # Dummy initialization of GrabCut internal models, required by the algorithm
        self.bgdmodel = np.zeros((1, 65), np.float64)
        self.fgdmodel = np.zeros((1, 65), np.float64)

        self.grabcut_image = self.CLAHE_normalize(self.img, self.clahe)

        # Apply GrabCut
        self.mask, self.fgdmodel, self.bgdmodel = cv2.grabCut(
            self.grabcut_image,
            self.draw_mask,
            self.rect,
            self.bgdmodel,
            self.fgdmodel,
            5,
            initparam,
        )
        self.mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype("uint8")
        self.update_visualization()

    def polygon_select_callback(self, verts):
        # Convert the polygon vertices into a mask
        img_shape = self.img.shape[:2]  # Only need the spatial dimensions
        poly_verts = np.array([verts], dtype=np.int32)  # Format for cv2.fillPoly

        # Create a new mask for each polygon
        mask = np.zeros(img_shape, dtype=np.uint8)

        # Use cv2.fillPoly to fill the polygon
        cv2.fillPoly(mask, poly_verts, 1)  # Fill with 1

        # Update the class attribute
        self.draw_mask = mask
        # self.update_mask()
        # self.matplotlib_widget.canvas.draw()

    def roi_select_callback(self, eclick, erelease):
        a, b = int(self.rs.extents[0]), int(self.rs.extents[2])
        c, d = int(self.rs.extents[1]) - a, int(self.rs.extents[3]) - b
        self.rect = (a, b, c, d)

    def update_visualization(self):
        # Convert self.mask to RGBA for visualization, applying current_color for the mask
        self.mask_data = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
        foreground = self.mask > 0
        self.mask_data[foreground] = [
            *self.current_color,
            self.opacity,
        ]  # Apply current_color and full opacity
        self.mask_data[~foreground, 3] = 0  # Fully transparent for background
        print("update visualization")
        # Assuming self.mask_plot is the matplotlib image object for the mask
        self.mask_plot.set_data(self.mask_data)
        self.matplotlib_widget.canvas.draw()

    def select_roi_mode(self):
        if not self.main_window.image_loaded:
            return
        self.roi_mode = not self.roi_mode
        if self.roi_mroi_checkboxode and self.toolbar.mode == "pan/zoom":
            # If the pan tool is active, deactivate it
            self.toolbar.pan()
        if self.roi_checkbox and self.toolbar.mode == "zoom rect":
            # If the zoom tool is active, deactivate it
            self.toolbar.zoom()
        if self.roi_checkbox is not None:
            self.roi_checkbox.setChecked(self.roi_mode)
        self.rs.set_active(self.roi_mode)
        if not self.roi_mode:
            self.matplotlib_widget.canvas.draw()

    def new_mask(self):
        # clear points
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = (
            [],
            [],
            [],
            [],
            [],
        )
        self.show_points(self.add_plot, self.add_xs, self.add_ys)
        self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

        mask = self.mask_data[:, :, 3] > 0
        self.global_masks[mask] = self.label
        self.label += 1

        self.prev_mask_data[:, :, :3][mask] = self.current_color
        self.prev_mask_data[:, :, 3][mask] = 255
        self.prev_mask_plot.set_data(self.prev_mask_data)

        self.legend_elements = [
            Line2D(
                [0],
                [0],
                color=np.array(self.current_color) / 255,
                lw=2,
                label=f"Mask {self.label-1}\n",
            )
        ]
        self.full_legend += self.legend_elements
        self.matplotlib_widget.ax.legend(handles=self.full_legend)
        self.matplotlib_widget.canvas.draw()

        self.current_color = self.pick_color()
        for i in range(3):
            self.mask_data[:, :, i] = self.current_color[i]
        self.clear_mask()

    def confirm_label(self):
        # clear points
        self.add_xs, self.add_ys, self.rem_xs, self.rem_ys, self.trace = (
            [],
            [],
            [],
            [],
            [],
        )
        self.show_points(self.add_plot, self.add_xs, self.add_ys)
        self.show_points(self.rem_plot, self.rem_xs, self.rem_ys)

        mask = self.mask_data[:, :, 3] > 0
        self.global_masks[mask] = self.label

        self.prev_mask_data[:, :, :3][mask] = self.current_color
        self.prev_mask_data[:, :, 3][mask] = 255
        self.prev_mask_plot.set_data(self.prev_mask_data)
        self.matplotlib_widget.canvas.draw()

    @staticmethod
    def remove_small_regions(mask, area_thresh, mode):
        """Function from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py"""
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
        sizes = stats[:, -1][1:]  # Row 0 is background label
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
        if len(small_regions) == 0:
            return mask
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask

    def save_mask(self, filePath):
        options = QFileDialog.Options()

        # Suggest a default file name based on the frame name
        suggested_name = os.path.splitext(os.path.basename(filePath))[0] + "_mask.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save Mask",
            suggested_name,
            "Images (*.png *.jpg);;All Files (*)",
            options=options,
        )

        if save_path:
            dir_path = os.path.split(save_path)[0]
            if dir_path != "" and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            cv2.imwrite(save_path, self.global_masks)
        else:
            pass
