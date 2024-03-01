import numpy as np
import cv2
import os
from sklearn.neighbors import LocalOutlierFactor
from PySide6.QtCore import Signal, QThread, Slot, QTimer
from PySide6.QtWidgets import QFileDialog
from models.segmenter import Segmenter
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import pickle


class ImageController:
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_loaded = False
        self.video_loaded = False
        self.video = None
        self.filePath = None
        self.image0 = None
        self.FIRST_LOAD_FILE = True
        self.auto_mask_generator, self.predictor = load_sam(
            "models/sam_vit_h_4b8939.pth"
        )

    def load_image(self):
        options = QFileDialog.Options()
        self.filePath, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Load Image/Video",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;Videos (*.mp4 *.avi);;All Files (*)",
            options=options,
        )
        if self.filePath:
            if self.filePath.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                # Load the selected image file
                self.image_loaded = True
                image0 = cv2.imread(self.filePath)

            elif self.filePath.endswith((".mp4", ".avi")):
                # Load the selected video file
                self.video_loaded = True
                self.main_window.frame_spinbox.setEnabled(True)
                self.video = cv2.VideoCapture(self.filePath)
                ret, image0 = self.video.read()
                total_frames = int(
                    self.video.get(cv2.CAP_PROP_FRAME_COUNT)
                )  # Get the total number of frames
                self.main_window.frame_spinbox.setRange(
                    1, total_frames
                )  # Set the range of the QSpinBox

            self.segmenter = Segmenter(
                image0,
                self.filePath,
                self.main_window.matplotlib_widget,
                self.auto_mask_generator,
                self.predictor,
            )
            if self.FIRST_LOAD_FILE:
                self.main_window.undo_button.clicked.connect(self.segmenter.undo)
                self.main_window.create_new.clicked.connect(self.segmenter.new_mask)
                self.main_window.grabcut_button.clicked.connect(self.segmenter.grabcut)
                self.main_window.save_button.clicked.connect(self.save_mask)
                self.FIRST_LOAD_FILE = False

            # Enable or disable buttons based on whether an image or video is loaded
            self.main_window.sam_checkbox.setEnabled(True)
            self.main_window.grabcut_checkbox.setEnabled(True)
            self.main_window.confirm_button.setEnabled(True)
            self.main_window.confirm_button.clicked.connect(
                self.segmenter.confirm_label
            )
            self.main_window.create_new.setEnabled(True)
            self.main_window.undo_button.setEnabled(True)
            self.main_window.save_button.setEnabled(True)
            self.main_window.grabcut_button.setEnabled(True)

            # Connect Matplotlib events to handling methods
            self.main_window.matplotlib_widget.canvas.mpl_connect(
                "button_press_event", self._on_click
            )
            self.main_window.matplotlib_widget.canvas.mpl_connect(
                "key_press_event", self._on_key
            )

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

    def gen_embedding(self):
        masks = self.auto_mask_generator.generate(self.img)
        file_root, _ = os.path.splitext(filePath)
        file_path_pkl = file_root + ".pkl"
        with open(file_path_pkl, "wb") as f:
            pickle.dump(masks, f)

    def apply_sam(self):
        self.segmenter.get_mask()

    def apply_grabcut(self):
        self.segmenter.grabcut()

    def update_video_frame(self, frame_number):
        if self.video_loaded:
            frame_number -= 1  # Adjust for 0-based indexing
            self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.video.read()
            if ret:
                self.image0 = frame
                self.main_window.matplotlib_widget.setup_segmenter(
                    self.image0, self.filePath, self.main_window.matplotlib_widget
                )

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

    def _on_click(self, event):
        if self.main_window.sam_checkbox.isChecked():
            if event.inaxes != self.main_window.matplotlib_widget.ax and (
                event.button in [1, 3]
            ):
                return
            x = int(np.round(event.xdata))
            y = int(np.round(event.ydata))

            if event.button == 1:  # left click
                self.segmenter.trace.append(True)
                self.segmenter.add_xs.append(x)
                self.segmenter.add_ys.append(y)
                self.segmenter.show_points(
                    self.segmenter.add_plot,
                    self.segmenter.add_xs,
                    self.segmenter.add_ys,
                )

            else:  # right click
                self.segmenter.trace.append(False)
                self.segmenter.rem_xs.append(x)
                self.segmenter.rem_ys.append(y)
                self.segmenter.show_points(
                    self.segmenter.rem_plot,
                    self.segmenter.rem_xs,
                    self.segmenter.rem_ys,
                )

            self.segmenter.get_mask()

        elif (
            self.main_window.grabcut_checkbox.isChecked()
            and self.main_window.draw_checkbox.isChecked()
        ):
            self.segmenter.ps.set_active(self.main_window.draw_checkbox)
            self.segmenter.ps.set_visible(self.main_window.draw_checkbox)
            self.segmenter.rs.set_active(False)
            self.segmenter.rs.set_visible(False)
        elif (
            self.main_window.grabcut_checkbox.isChecked()
            and self.main_window.roi_checkbox.isChecked()
        ):
            self.segmenter.rs.set_active(self.main_window.roi_checkbox)
            self.segmenter.rs.set_visible(self.main_window.roi_checkbox)
            self.segmenter.ps.set_active(False)
            self.segmenter.ps.set_visible(False)

    def draw_checkbox_changed(self):
        # If the draw checkbox is checked, uncheck the ROI checkbox
        if self.main_window.draw_checkbox.isChecked():
            self.main_window.roi_checkbox.setChecked(False)
            self.segmenter.ps.set_active(True)
            self.segmenter.ps.set_visible(True)
        if self.main_window.draw_checkbox.isChecked() == False:
            self.segmenter.ps.set_active(False)
            self.segmenter.ps.set_visible(False)

    def roi_checkbox_changed(self):
        # If the draw checkbox is checked, uncheck the ROI checkbox
        if self.main_window.roi_checkbox.isChecked():
            self.main_window.draw_checkbox.setChecked(False)
            self.segmenter.rs.set_active(True)
            self.segmenter.rs.set_visible(True)
        if self.main_window.roi_checkbox.isChecked() == False:
            self.segmenter.rs.set_active(False)
            self.segmenter.rs.set_visible(False)

    def grabcut_checkbox_changed(self):
        if self.main_window.grabcut_checkbox.isChecked():
            self.main_window.sam_group_box.setDisabled(True)
            self.main_window.sam_checkbox.setChecked(False)
        else:
            self.main_window.sam_group_box.setEnabled(True)
            self.main_window.draw_checkbox.setChecked(False)
            self.main_window.roi_checkbox.setChecked(False)

    def sam_checkbox_changed(self):
        if self.main_window.sam_checkbox.isChecked():
            self.main_window.grabcut_group_box.setDisabled(True)
            self.main_window.draw_checkbox.setChecked(False)
            self.main_window.roi_checkbox.setChecked(False)
            self.main_window.grabcut_checkbox.setChecked(False)
        else:
            self.main_window.grabcut_group_box.setEnabled(True)

    def save_mask(self):
        options = QFileDialog.Options()

        # Suggest a default file name based on the frame name

        suggested_name = (
            os.path.splitext(os.path.basename(self.filePath))[0] + "_mask.png"
        )
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
            cv2.imwrite(save_path, self.segmenter.global_masks)


def load_sam(sam_path):
    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    if torch.cuda.is_available():
        sam.to(device="cuda")
    else:
        sam.to(device="cpu")
    min_mask_region_area = 100
    auto_mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.5,
        stability_score_thresh=0.5,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_region_area,
    )
    predictor = SamPredictor(sam)
    return auto_mask_generator, predictor


def splitfn(fn: str):
    fn = os.path.abspath(fn)
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def convert_mask_gray_to_BGR(img):
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # Red -> 0, Green->1, Blue -> 2
    mask[:, :, 0] = (img == 2) * 255
    mask[:, :, 1] = (img == 1) * 255
    mask[:, :, 2] = (img == 0) * 255

    return mask


def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]

    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[int(window_len / 2) - 1 : -int(window_len / 2) - 1]


def CLAHE_normalize(bgr, clahe):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def annotateImage(orig, flags, top=True, left=True):

    try:
        y, x, c = np.shape(orig)

        if top:
            yp = int(y * 0.035)
        else:
            yp = int(y * 0.85)
        if left:
            xp = int(x * 0.035)
        else:
            xp = int(y * 0.85)

        offset = 0
        for key in ["OVEREXPOSED", "UNDEREXPOSED"]:
            if flags[key]:
                cv2.putText(
                    orig,  # numpy array on which text is written
                    "{0}: {1}".format(key, True),  # text
                    (xp, yp + offset),  # position at which writing has to start
                    cv2.FONT_HERSHEY_SIMPLEX,  # font family
                    1,  # font size
                    (0, 255, 255, 255),  # font color
                    3,
                )  # font stroke
            offset += int(y * 0.035)
    except:
        print("Annotation error")


def getOutlierMask(metrics):
    X = np.nan_to_num(np.array(metrics).T)
    clf = LocalOutlierFactor(n_neighbors=11, contamination="auto")
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    return clf.fit_predict(X)


def annotate_image_with_frame_number(image, frame_number):
    # Convert frame number to string
    frame_text = f"Frame: {frame_number}"

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Set the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    # Calculate the size of the frame text
    (text_width, text_height), _ = cv2.getTextSize(
        frame_text, font, font_scale, font_thickness
    )

    # Calculate the position to place the frame text in the top right corner
    text_position = (width - text_width - 10, 30)

    # Draw the frame text on the image
    cv2.putText(
        image, frame_text, text_position, font, font_scale, (0, 255, 0), font_thickness
    )
