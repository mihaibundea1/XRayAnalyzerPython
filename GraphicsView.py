from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGraphicsView, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter

class GraphicsView(QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(GraphicsView, self).__init__(parent)
        # # self.setRenderHint(QPainter.Antialiasing)
        # # self.setRenderHint(QPainter.SmoothPixmapTransform)
        # # self.setDragMode(QGraphicsView.ScrollHandDrag) # zooms out incontrollably even if i cant drag and drop
        # self.setOptimizationFlags(QGraphicsView.DontSavePainterState)
        # self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        # self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        # to make the background black
        # self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))

    def hasPhoto(self):
        return not self._empty


    # trebuie gasita  o alta metoda pentru ca functia asta nu e nicioadata apelata pentru a face 'resize'
    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                print(unity.width())
                print(unity.height())
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def resizeEvent(self, event):
        self.fitInView()
        super(GraphicsView, self).resizeEvent(event)

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom <= 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(GraphicsView, self).mousePressEvent(event)
