import os
import cv2
import callable
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QGraphicsView, QGraphicsScene, QMenuBar, QMenu, QAction
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import GraphicsView

from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QLabel, QListWidget, QGraphicsScene, QWidget
from GraphicsView import GraphicsView

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Image Viewer and Graphic Generator")
        self.setGeometry(0, 0, 1920, 1080)

        self.create_menu()

        self.folder_path = "D:/XRay/Images/chest_xray/test/NORMAL"

        self.main_layout = QHBoxLayout()
        self.layout = QVBoxLayout()
        self.leftVerticalLayout = QVBoxLayout()

        self.path_label = QLabel("Current Path: ")
        self.layout.addWidget(self.path_label)

        self.file_listbox = QListWidget()
        self.file_listbox.itemClicked.connect(self.load_selected_image)
        # self.file_listbox.setStyleSheet("background-color: black; color: white;")
        self.file_listbox.setMaximumWidth(250)
        self.layout.addWidget(self.file_listbox)

        self.main_layout.addLayout(self.layout)

        self.image_view = GraphicsView(self)
        self.image_view.setMinimumWidth(1000)
        self.main_layout.addWidget(self.image_view)

        self.fig = Figure(figsize=(3, 3), facecolor=(30/255, 30/255, 30/255))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMaximumSize(350,350)
        self.leftVerticalLayout.addWidget(self.canvas)

        # Create a QListWidget for the photo details
        self.photo_details_listbox = QListWidget()
        self.photo_details_listbox.setMaximumWidth(350)

        self.leftVerticalLayout.addWidget(self.photo_details_listbox)

        self.main_layout.addLayout(self.leftVerticalLayout)

        # final doings
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)
        self.setStyleSheet("background-color: rgb(30, 30, 30); color: white;")

        # Set the layout for the main window
        # central_widget = QWidget()
        # central_widget.setLayout(self.main_layout)
        # self.setCentralWidget(central_widget)


        # # Create a QLabel to display the image
        # self.image_label = QLabel()
        # self.main_layout.addWidget(self.image_label)

        # # Create a QPushButton to load files
        # self.load_files_button = QPushButton("Load Files")
        # self.load_files_button.clicked.connect(self.load_files)
        # self.layout.addWidget(self.load_files_button)

        # # Create a new matplotlib figure and draw a graph
        # self.fig, self.ax = plt.subplots(figsize=(14, 7))

        # # Create a new FigureCanvas and embed the matplotlib figure into it
        # self.graph_canvas = FigureCanvas(self.fig)
        # self.layout.addWidget(self.graph_canvas)

        # # Set the layout on the application's window
        # self.container = QWidget()
        # self.setCentralWidget(self.container)

    def photoClicked(self, pos):
        if self.viewer.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def create_menu(self):
        # Create a QMenuBar
        menuBar = self.menuBar()

        # Create a QMenu
        fileMenu = QMenu("File", self)

        # Create a QAction
        openAction = QAction("Open", self)
        saveAction = QAction("Save", self)

        # Add QAction to QMenu
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)

        # Add QMenu to QMenuBar
        menuBar.addMenu(fileMenu)

        # Connect the actions to their respective functions
        openAction.triggered.connect(self.load_files)
        saveAction.triggered.connect(self.save_file)

    def save_file(self):
        # Implement your function to save a file
        pass

    def load_files(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.path_label.setText(f"Current Path: {self.folder_path}")
        if self.folder_path:
            self.file_listbox.clear()
            for file_name in os.listdir(self.folder_path):
                self.file_listbox.addItem(file_name)

    def load_selected_image(self):
        selected_file = self.file_listbox.currentItem().text()
        file_path = os.path.join(self.folder_path, selected_file)
        self.display_image(file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)

        self.image_view._photo.setPixmap(pixmap)

        self.image_view.fitInView()

    def update_graph(self, file_path):
        # Clear previous plot
        print("Aici")
        classification_result = callable.classify_image(file_path)
        print("2")
        self.ax.clear()

        # Extract values from the classification result tuple
        print("3")
        value1, value2 = classification_result

        print("4")
        # Create a list of probabilities and class labels
        probabilities = [value1, value2]
        print("5")
        class_labels = ["NORMAL", "PNEUMONIA"]

        print("6")
        # Visualize predictions using provided function
        fig = callable.visualize_predictions(file_path, probabilities, class_labels)
        # Update the FigureCanvas
        print("7")
        self.graph_canvas.draw()



if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
