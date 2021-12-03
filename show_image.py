import sys
import cv2
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow



def show_image(image_path='****'):
    app = QtWidgets.QApplication(sys.argv)
    pixmap = QtGui.QPixmap(image_path)
    screen = QtWidgets.QLabel()
    screen.setPixmap(pixmap)
    screen.showFullScreen()
    sys.exit(app.exec_())

show_image()
cv2.destroyAllWindows()