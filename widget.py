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

        self.folder_path = None

        self.main_layout = QHBoxLayout()
        self.layout = QVBoxLayout()
        self.leftVerticalLayout = QVBoxLayout()

        self.path_label = QLabel()
        self.layout.addWidget(self.path_label)

        self.file_listbox = QListWidget()
        self.file_listbox.itemClicked.connect(self.load_selected_image)
        self.file_listbox.setMaximumWidth(250)
        self.layout.addWidget(self.file_listbox)

        self.main_layout.addLayout(self.layout)

        self.image_view = GraphicsView(self)
        self.image_view.setMinimumWidth(1000)
        self.main_layout.addWidget(self.image_view)

        self.fig = Figure(figsize=(6, 3), facecolor="none")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMaximumSize(400,400)
        # Create an axes instance in your figure

        self.leftVerticalLayout.addWidget(self.canvas)

        # Create a QListWidget for the photo details
        self.photo_details_listbox = QListWidget()
        self.photo_details_listbox.setMaximumWidth(400)

        self.leftVerticalLayout.addWidget(self.photo_details_listbox)

        self.main_layout.addLayout(self.leftVerticalLayout)

        # final doings
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)
        # self.setStyleSheet("background-color: rgb(30, 30, 30); color: white;")

    def create_bar_graph(self, data):
        # Clear the current axes, necessary if you want to update the graph
        self.fig.clear()

        ax = self.fig.add_subplot(111, facecolor='none')
        self.fig.subplots_adjust(0.25)

        # Create a horizontal bar graph with your data
        bars = ax.barh(range(len(data)), data, color='orange')

        # Set the limits of y-axis
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, len(data)-0.5])  # Adjust the limits of y-axis

        # Set the color of the labels and ticks to white for visibility on dark background
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')

        # Label the bars
        labels = ['NORMAL', 'PNEUMONIA']
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(labels)

        # Add value labels to each bar
        for bar, value in zip(bars, data):
            ax.text(value, bar.get_y(), ' {:.2f}'.format(value), va='center', color='white')

        self.canvas.draw()

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
        self.path_label.setText(f"{self.folder_path}")
        if self.folder_path:
            self.file_listbox.clear()
            for file_name in os.listdir(self.folder_path):
                self.file_listbox.addItem(file_name)

    def load_selected_image(self):
        selected_file = self.file_listbox.currentItem().text()
        file_path = os.path.join(self.folder_path, selected_file)
        self.display_image(file_path)
        self.call_classify_image(file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.image_view._photo.setPixmap(pixmap)
        self.image_view.fitInView()

    def call_classify_image(self, file_path):
        var = callable.classify_image(file_path)
        print(var)
        self.create_bar_graph(var)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
