import sys
from PySide6.QtCore import Qt
from views.matplotlib_widget import MatplotlibWidget
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QFrame,
    QSpinBox,
    QCheckBox,
    QHBoxLayout,
    QGroupBox,
)

from controllers.utils import ImageController


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_controller = ImageController(self)
        self.connectUI()

    def initUI(self):
        self.setWindowTitle("Segmentation GUI")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Create a horizontal layout to organize buttons and the MatplotlibWidget
        layout = QHBoxLayout(main_widget)

        # Create a vertical layout for the buttons on the left
        button_layout = QVBoxLayout()
        # Set size policy for the button_layout to Fixed

        self.load_button = QPushButton("Load Image/Video", main_widget)
        button_layout.addWidget(self.load_button)

        # Create a frame to hold the Confirm and Undo buttons
        button_frame = QFrame(main_widget)
        button_frame_layout = QVBoxLayout(button_frame)
        # Create a QHBoxLayout for the label and spinbox
        video_index_layout = QHBoxLayout()

        # Create a QLabel for 'Video frame index:'
        video_index_label = QLabel("Video frame index:", main_widget)
        video_index_layout.addWidget(video_index_label)

        # Create a QSpinBox for selecting video frames
        self.frame_spinbox = QSpinBox(main_widget)
        self.frame_spinbox.setRange(
            1, 1
        )  # Initial range, update it when a video is loaded
        self.frame_spinbox.setEnabled(False)  # Initially disabled

        video_index_layout.addWidget(self.frame_spinbox)

        # Add the QHBoxLayout with label and spinbox to the button_frame_layout
        button_frame_layout.addLayout(video_index_layout)

        # Assuming 'button_frame_layout' is a QVBoxLayout or similar that is already defined
        grabcut_layout = QVBoxLayout()  # New layout to group SAM checkbox and group box
        grabcut_layout.setSpacing(1)
        # Create a group box with the title 'SAM'
        self.grabcut_checkbox = QCheckBox("GrabCut", self)

        self.grabcut_checkbox.setChecked(True)
        self.grabcut_checkbox.setEnabled(False)
        # Add the checkbox to the layout
        grabcut_layout.addWidget(self.grabcut_checkbox)
        # Create a group box with the title 'GrabCut'
        self.grabcut_group_box = QGroupBox()
        self.grabcut_group_box.setEnabled(False)
        grabcut_group_box_layout = (
            QVBoxLayout()
        )  # Layout for widgets inside the group box
        self.grabcut_group_box.setLayout(grabcut_group_box_layout)

        # Set a line border style for the group box using a stylesheet
        self.grabcut_group_box.setStyleSheet(
            """
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 0.5em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """
        )

        # Add the QHBoxLayout with label and spinbox to the grabcut_group_box_layout
        grabcut_group_box_layout.addLayout(video_index_layout)

        # Since 'video_index_layout' is not directly defined here, ensure it is a QHBoxLayout that contains your label and spinbox

        self.draw_checkbox = QCheckBox("Draw")
        self.draw_checkbox.setChecked(True)
        grabcut_group_box_layout.addWidget(self.draw_checkbox)

        # Create and add the checkbox for 'GrabCut'
        self.roi_checkbox = QCheckBox("Region of interest")

        grabcut_group_box_layout.addWidget(self.roi_checkbox)

        # Create and add the button for 'GrabCut Segmentation'
        self.grabcut_button = QPushButton("GrabCut Segmentation")
        grabcut_group_box_layout.addWidget(self.grabcut_button)

        # Finally, add the 'grabcut_group_box' to the original layout where you want it displayed
        grabcut_layout.addWidget(self.grabcut_group_box)
        button_frame_layout.addLayout(grabcut_layout)

        # Assuming 'button_frame_layout' is a QVBoxLayout or similar that is already defined
        sam_layout = QVBoxLayout()  # New layout to group SAM checkbox and group box
        sam_layout.setSpacing(1)
        # Create a group box with the title 'SAM'
        self.sam_checkbox = QCheckBox("SAM", self)

        self.sam_checkbox.setEnabled(False)
        # Add the checkbox to the layout
        sam_layout.addWidget(self.sam_checkbox)
        self.sam_group_box = QGroupBox()
        self.sam_group_box.setEnabled(False)
        sam_group_box_layout = QVBoxLayout()  # Layout for widgets inside the group box
        self.sam_group_box.setLayout(sam_group_box_layout)

        # Set a line border style for the group box using a stylesheet
        self.sam_group_box.setStyleSheet(
            """
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 0.5em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
        """
        )

        # Now, add your buttons to the 'sam_group_box_layout' instead of 'button_frame_layout'

        self.load_embedding_button = QPushButton("Load Embedding")
        sam_group_box_layout.addWidget(self.load_embedding_button)

        self.embedding_button = QPushButton("Generate Embedding")
        sam_group_box_layout.addWidget(self.embedding_button)

        self.sam_button = QPushButton("SAM Segmentation")
        sam_group_box_layout.addWidget(self.sam_button)

        sam_layout.addWidget(self.sam_group_box)
        button_frame_layout.addLayout(sam_layout)

        self.confirm_button = QPushButton("Confirm Label", button_frame)
        button_frame_layout.addWidget(self.confirm_button)
        self.confirm_button.setEnabled(False)

        # Add the button frame to the button layout
        button_layout.addWidget(button_frame)

        # Create a button to undo a point
        self.undo_button = QPushButton("Undo", button_frame)
        button_frame_layout.addWidget(self.undo_button)
        self.undo_button.setEnabled(False)

        # Create a button to confirm the current label and create a new one
        self.create_new = QPushButton("Confirm and Create New Mask", button_frame)
        button_frame_layout.addWidget(self.create_new)
        self.create_new.setEnabled(False)

        self.save_button = QPushButton("Save Mask", main_widget)
        button_layout.addWidget(self.save_button)
        self.save_button.setEnabled(False)

        # Add a stretchable space at the bottom of button_layout to align buttons to the top
        button_layout.addStretch()

        # Create a new vertical layout for aligning the button_layout at the top
        vertical_layout = QVBoxLayout()
        vertical_layout.addLayout(button_layout)

        # Create a container widget for the vertical_layout
        vertical_container = QWidget()
        vertical_container.setLayout(vertical_layout)
        vertical_container.setFixedWidth(300)  # Adjust the width as needed

        # Add the button_layout with alignment and spacing to the main layout
        layout.addWidget(vertical_container)
        layout.addSpacing(20)  # Add some spacing between buttons and MatplotlibWidget
        self.matplotlib_widget = MatplotlibWidget(main_widget)
        self.matplotlib_widget.setFocusPolicy(Qt.StrongFocus)
        layout.addWidget(self.matplotlib_widget)

    def connectUI(self):
        self.frame_spinbox.valueChanged.connect(
            self.image_controller.update_video_frame
        )
        self.load_button.clicked.connect(self.image_controller.load_image)
        self.sam_button.clicked.connect(self.image_controller.apply_sam)
        self.grabcut_checkbox.stateChanged.connect(
            self.image_controller.grabcut_checkbox_changed
        )
        self.draw_checkbox.stateChanged.connect(
            self.image_controller.draw_checkbox_changed
        )
        self.roi_checkbox.stateChanged.connect(
            self.image_controller.roi_checkbox_changed
        )
        self.roi_checkbox.stateChanged.connect(
            self.image_controller.roi_checkbox_changed
        )
        self.grabcut_button.clicked.connect(self.image_controller.apply_grabcut)
        self.sam_checkbox.stateChanged.connect(
            self.image_controller.sam_checkbox_changed
        )
        self.load_embedding_button.clicked.connect(self.image_controller.load_embedding)
        self.embedding_button.clicked.connect(self.image_controller.gen_embedding)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
