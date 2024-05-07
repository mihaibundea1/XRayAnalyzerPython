import os
import sys

import resource_rc
import callable

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QListWidgetItem, QWidget, QGraphicsView, QGraphicsScene, QMenuBar, QMenu, QAction
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QLabel, QListWidget, QGraphicsScene, QWidget
from GraphicsView import GraphicsView
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt

from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils

class Property:
    def __init__(self, name):
        self.name = name

    def change_name(self, new_name):
        self.name = new_name

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Image Viewer and Graphic Generator")
        self.setGeometry(0, 0, 1920, 1080)
        self.showMaximized()
        self.create_menu()

        self.folder_path = None

        self.main_layout = QHBoxLayout()
        self.layout = QVBoxLayout()
        self.centerVerticalLayout = QVBoxLayout()
        self.rightVerticalLayout = QVBoxLayout()

        self.path_label = QLabel()
        self.path_label.setStyleSheet("border: 1px solid rgb(60, 60, 60);")
        self.layout.addWidget(self.path_label)

        self.file_listbox = QListWidget()
        self.file_listbox.itemClicked.connect(self.load_selected_image)
        self.file_listbox.setStyleSheet("border: 1px solid rgb(60, 60, 60);")
        self.file_listbox.setMaximumWidth(250)
        self.layout.addWidget(self.file_listbox)

        self.main_layout.addLayout(self.layout)

        self.image_view = GraphicsView(self)
        self.image_view.setMinimumWidth(1000)
        self.image_view.photoClicked.connect(self.photoClicked)
        self.image_view.setStyleSheet("border: 1px solid rgb(60, 60, 60);")

        self.buttonMove = QPushButton()
        self.buttonMove.setIcon(QIcon(":/moveIcon.png"))  # replace with your image path
        self.buttonMove.setIconSize(QSize(24, 24))  # set the icon size to 32x32 pixels
        self.buttonMove.setCheckable(True)

        # Set the QPushButton size to 32x32 pixels
        self.buttonMove.setFixedSize(QSize(32, 32))
        self.buttonMove.setStyleSheet("border: 1px solid rgb(60, 60, 60);")


        # Make the QPushButton toggleable
        self.buttonMove.setCheckable(True)
        self.buttonMove.clicked.connect(self.pixInfo)
        self.centerVerticalLayout.addWidget(self.buttonMove)

        self.centerVerticalLayout.addWidget(self.image_view)
        self.main_layout.addLayout(self.centerVerticalLayout)

        self.fig = Figure(figsize=(6, 3), facecolor="none")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMaximumSize(400,400)
        self.canvas.setStyleSheet("border: 1px solid rgb(60, 60, 60);")

        # Create an axes instance in your figure

        self.rightVerticalLayout.addWidget(self.canvas)

        self.saveButton = QPushButton("Save Investigation Details", self)
        self.saveButton.setStyleSheet("border: 1px solid rgb(60, 60, 60);font-size: 20px;")
        self.saveButton.setMinimumHeight(100)
        self.saveButton.clicked.connect(self.save_to_pdf)

        self.rightVerticalLayout.addWidget(self.saveButton)

        # Create a QListWidget for the photo details
        self.photo_details_listbox = QListWidget()
        self.photo_details_listbox.setMaximumWidth(400)
        self.photo_details_listbox.setStyleSheet("border: 1px solid rgb(60, 60, 60); font-size: 20px;")

        properties = [Property("Result: "), Property("Image size: "), Property("Position of cursor: ")]

        # Add items to the QListWidget
        for prop in properties:
            item = QListWidgetItem(prop.name)
            item.setData(Qt.UserRole, prop)
            self.photo_details_listbox.addItem(item)

        self.rightVerticalLayout.addWidget(self.photo_details_listbox)

        self.main_layout.addLayout(self.rightVerticalLayout)

        # final doings
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)
        self.setStyleSheet("background-color: rgb(30, 30, 30); color: white;")

    def save_to_pdf(self):
        # Prompt user to select file location
        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)")
        if not file_path:  # User cancelled
            return

        # Create a PDF document
        c = canvas.Canvas(file_path, pagesize=letter)
        c.setFillColor("black")  # Set text color to black

        # Draw "Investigation Details" centered horizontally
        c.drawString((letter[0] - c.stringWidth("Investigation Details", "Helvetica", 12)) / 2, 750, "Investigation Details")

        # Draw text from self.photo_details_listbox centered horizontally
        text = "\n".join(self.photo_details_listbox.item(i).text() for i in range(self.photo_details_listbox.count()))
        lines = text.split("\n")
        line_height = 12  # Assuming font size is 12
        y_position = 600
        for line in lines:
            c.drawString(100, y_position, line)
            y_position -= line_height

        # Add graph from self.canvas on the first page
        graph_path = "temp_graph.png"
        ax = self.fig.add_subplot(111, facecolor='none')
        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        self.fig.savefig(graph_path)
        c.drawImage(graph_path, (letter[0] - 400) / 2, 100, width=400, height=300)  # Center the graph

        c.showPage()  # Start a new page

        # Add image from self.image_view on the second page
        pixmap = self.image_view.grab()
        image_path = "temp_image.png"
        pixmap.save(image_path)
        c.drawImage(image_path, (letter[0] - pixmap.width() / 3) / 2, 200, width=pixmap.width() / 3, height=pixmap.height() / 3)  # Center the image

        c.save()



    def pixInfo(self):
        self.image_view.toggleDragMode()

    def photoClicked(self, pos):
        if self.image_view.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
            for i in range(self.photo_details_listbox.count()):
                item = self.photo_details_listbox.item(i)
                prop = item.data(Qt.UserRole)
                if prop.name == "Position of cursor: ":
                    prop.value = '%d, %d' % (pos.x(), pos.y())
                    item.setText(prop.name + prop.value)

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

        # ax.tick_params(colors='black')
        # ax.xaxis.label.set_color('black')
        # ax.yaxis.label.set_color('black')

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
        if os.path.splitext(file_path)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
            try:
                file_path = os.path.join(self.folder_path, selected_file)
                self.display_image(file_path)
                self.pixInfo()
                self.call_classify_image(file_path)
            except IOError:
                print(f"{file_path} is not a valid image file.")
                return

    def display_image(self, file_path):
        pix = QPixmap(file_path)
        self.image_view.setPhoto(pix)
        for i in range(self.photo_details_listbox.count()):
            item = self.photo_details_listbox.item(i)
            prop = item.data(Qt.UserRole)
            if prop.name == "Image size: ":
                prop.value = '%d, %d' % (pix.width(), pix.height())
                item.setText(prop.name + prop.value)

    def call_classify_image(self, file_path):
        var = callable.classify_image(file_path)
        print(var)
        self.create_bar_graph(var)
        for i in range(self.photo_details_listbox.count()):
            item = self.photo_details_listbox.item(i)
            prop = item.data(Qt.UserRole)
            if prop.name == "Result: ":
                if var[0] > var[1]:
                    prop.value = '%s' % ("Normal")
                else:
                    prop.value = '%s' % ("Pneumonia")
                item.setText(prop.name + prop.value)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
