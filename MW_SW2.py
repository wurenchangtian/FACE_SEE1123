# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MW_SW2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog2(object):
    def setupUi(self, Dialog2):
        Dialog2.setObjectName("Dialog2")
        Dialog2.resize(1110, 618)
        Dialog2.setStyleSheet("QDialog{\n"
"background-color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"}")
        self.stackedWidget = QtWidgets.QStackedWidget(Dialog2)
        self.stackedWidget.setGeometry(QtCore.QRect(130, 40, 961, 571))
        self.stackedWidget.setStyleSheet("QStackedWidget{\n"
"background-color: rgb(255, 255, 255);\n"
"border: 5px solid rgb(194, 194, 194);\n"
"border-radius:20px;\n"
"}")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.tableWidget = QtWidgets.QTableWidget(self.page_2)
        self.tableWidget.setGeometry(QtCore.QRect(310, 220, 641, 261))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.tableWidget.setFont(font)
        self.tableWidget.setStyleSheet("")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        self.lineEdit = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit.setGeometry(QtCore.QRect(30, 110, 231, 41))
        self.lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_7.setGeometry(QtCore.QRect(30, 160, 231, 41))
        self.lineEdit_7.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_7.setText("")
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.pushButton = QtWidgets.QPushButton(self.page_2)
        self.pushButton.setGeometry(QtCore.QRect(320, 110, 191, 91))
        self.pushButton.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(122, 122, 122);\n"
"color: rgb(255, 255, 255);\n"
"border:None;\n"
"border-radius:10px;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.label_5 = QtWidgets.QLabel(self.page_2)
        self.label_5.setGeometry(QtCore.QRect(350, 30, 151, 21))
        self.label_5.setStyleSheet("font: 14pt \"华文琥珀\";")
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.page_2)
        self.label_7.setGeometry(QtCore.QRect(30, 220, 231, 221))
        self.label_7.setStyleSheet("background-color: rgb(222, 222, 222);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.lineEdit_16 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_16.setGeometry(QtCore.QRect(30, 450, 231, 41))
        self.lineEdit_16.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_16.setText("")
        self.lineEdit_16.setReadOnly(True)
        self.lineEdit_16.setPlaceholderText("")
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.pushButton_2 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_2.setGeometry(QtCore.QRect(750, 110, 191, 91))
        self.pushButton_2.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(122, 122, 122);\n"
"color: rgb(255, 255, 255);\n"
"border:None;\n"
"border-radius:10px;\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.stackedWidget.addWidget(self.page_2)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.pushButton_3 = QtWidgets.QPushButton(self.page)
        self.pushButton_3.setGeometry(QtCore.QRect(150, 440, 211, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setAutoFillBackground(False)
        self.pushButton_3.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(122, 122, 122);\n"
"color: rgb(255, 255, 255);\n"
"border:None;\n"
"border-radius:10px;\n"
"}")
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_6.setGeometry(QtCore.QRect(140, 360, 231, 51))
        self.lineEdit_6.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_6.setText("")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setGeometry(QtCore.QRect(490, 440, 331, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setAutoFillBackground(False)
        self.pushButton_4.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(122, 122, 122);\n"
"color: rgb(255, 255, 255);\n"
"border:None;\n"
"border-radius:10px;\n"
"}")
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_6 = QtWidgets.QLabel(self.page)
        self.label_6.setGeometry(QtCore.QRect(140, 130, 231, 221))
        self.label_6.setStyleSheet("background-color: rgb(222, 222, 222);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_2 = QtWidgets.QLabel(self.page)
        self.label_2.setGeometry(QtCore.QRect(340, 0, 161, 41))
        self.label_2.setStyleSheet("font: 14pt \"华文琥珀\";")
        self.label_2.setObjectName("label_2")
        self.lineEdit_11 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_11.setGeometry(QtCore.QRect(90, 70, 341, 41))
        self.lineEdit_11.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"padding: 10 10 10 20;")
        self.lineEdit_11.setReadOnly(True)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_21 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_21.setGeometry(QtCore.QRect(480, 250, 351, 51))
        self.lineEdit_21.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_21.setText("")
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.lineEdit_22 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_22.setGeometry(QtCore.QRect(480, 310, 351, 51))
        self.lineEdit_22.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_22.setText("")
        self.lineEdit_22.setObjectName("lineEdit_22")
        self.lineEdit_23 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_23.setGeometry(QtCore.QRect(480, 110, 351, 51))
        self.lineEdit_23.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_23.setText("")
        self.lineEdit_23.setObjectName("lineEdit_23")
        self.lineEdit_24 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_24.setGeometry(QtCore.QRect(480, 170, 351, 51))
        self.lineEdit_24.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_24.setText("")
        self.lineEdit_24.setObjectName("lineEdit_24")
        self.stackedWidget.addWidget(self.page)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.pushButton_12 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_12.setGeometry(QtCore.QRect(530, 430, 201, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_12.setFont(font)
        self.pushButton_12.setAutoFillBackground(False)
        self.pushButton_12.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(122, 122, 122);\n"
"color: rgb(255, 255, 255);\n"
"border:None;\n"
"border-radius:10px;\n"
"}")
        self.pushButton_12.setObjectName("pushButton_12")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_8.setGeometry(QtCore.QRect(530, 110, 201, 51))
        self.lineEdit_8.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_8.setText("")
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_9.setGeometry(QtCore.QRect(530, 230, 201, 51))
        self.lineEdit_9.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_9.setText("")
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_12.setGeometry(QtCore.QRect(530, 170, 201, 51))
        self.lineEdit_12.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_12.setText("")
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.lineEdit_13 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_13.setGeometry(QtCore.QRect(530, 290, 201, 51))
        self.lineEdit_13.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_13.setText("")
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.label_3 = QtWidgets.QLabel(self.page_4)
        self.label_3.setGeometry(QtCore.QRect(560, 70, 151, 21))
        self.label_3.setStyleSheet("font: 14pt \"华文琥珀\";")
        self.label_3.setObjectName("label_3")
        self.lineEdit_17 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_17.setGeometry(QtCore.QRect(50, 90, 231, 51))
        self.lineEdit_17.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_17.setText("")
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.lineEdit_18 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_18.setGeometry(QtCore.QRect(50, 150, 231, 51))
        self.lineEdit_18.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_18.setText("")
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.label_8 = QtWidgets.QLabel(self.page_4)
        self.label_8.setGeometry(QtCore.QRect(90, 60, 151, 21))
        self.label_8.setStyleSheet("font: 14pt \"华文琥珀\";")
        self.label_8.setObjectName("label_8")
        self.pushButton_14 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_14.setGeometry(QtCore.QRect(50, 430, 231, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_14.setFont(font)
        self.pushButton_14.setAutoFillBackground(False)
        self.pushButton_14.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(122, 122, 122);\n"
"color: rgb(255, 255, 255);\n"
"border:None;\n"
"border-radius:10px;\n"
"}")
        self.pushButton_14.setObjectName("pushButton_14")
        self.label_9 = QtWidgets.QLabel(self.page_4)
        self.label_9.setGeometry(QtCore.QRect(50, 210, 231, 201))
        self.label_9.setStyleSheet("border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"background-color: rgb(220, 220, 220);")
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.page_4)
        self.label_10.setGeometry(QtCore.QRect(400, 50, 451, 451))
        self.label_10.setStyleSheet("border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"background-color:rgb(255, 255, 255);")
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.page_4)
        self.label_11.setGeometry(QtCore.QRect(20, 50, 301, 451))
        self.label_11.setStyleSheet("border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"background-color:rgb(255, 255, 255);")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.lineEdit_20 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_20.setGeometry(QtCore.QRect(460, 510, 341, 41))
        self.lineEdit_20.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"padding: 10 10 10 50;")
        self.lineEdit_20.setReadOnly(True)
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.lineEdit_25 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_25.setGeometry(QtCore.QRect(530, 350, 201, 51))
        self.lineEdit_25.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_25.setText("")
        self.lineEdit_25.setObjectName("lineEdit_25")
        self.label_11.raise_()
        self.label_10.raise_()
        self.pushButton_12.raise_()
        self.lineEdit_8.raise_()
        self.lineEdit_9.raise_()
        self.lineEdit_12.raise_()
        self.lineEdit_13.raise_()
        self.label_3.raise_()
        self.lineEdit_17.raise_()
        self.lineEdit_18.raise_()
        self.label_8.raise_()
        self.pushButton_14.raise_()
        self.label_9.raise_()
        self.lineEdit_20.raise_()
        self.lineEdit_25.raise_()
        self.stackedWidget.addWidget(self.page_4)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_10.setGeometry(QtCore.QRect(140, 330, 231, 51))
        self.lineEdit_10.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_10.setText("")
        self.lineEdit_10.setReadOnly(True)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.pushButton_13 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_13.setGeometry(QtCore.QRect(330, 390, 231, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_13.setFont(font)
        self.pushButton_13.setAutoFillBackground(False)
        self.pushButton_13.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(189, 189, 189);\n"
"}\n"
"QPushButton{\n"
"background-color: rgb(122, 122, 122);\n"
"color: rgb(255, 255, 255);\n"
"border:None;\n"
"border-radius:10px;\n"
"}")
        self.pushButton_13.setObjectName("pushButton_13")
        self.lineEdit_14 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_14.setGeometry(QtCore.QRect(510, 140, 231, 51))
        self.lineEdit_14.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_14.setText("")
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.lineEdit_15 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_15.setGeometry(QtCore.QRect(510, 210, 231, 51))
        self.lineEdit_15.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_15.setText("")
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.label_4 = QtWidgets.QLabel(self.page_3)
        self.label_4.setGeometry(QtCore.QRect(360, 40, 151, 21))
        self.label_4.setStyleSheet("font: 14pt \"华文琥珀\";")
        self.label_4.setObjectName("label_4")
        self.lineEdit_19 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_19.setGeometry(QtCore.QRect(450, 310, 341, 41))
        self.lineEdit_19.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"padding: 10 10 10 70;")
        self.lineEdit_19.setReadOnly(True)
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.label_12 = QtWidgets.QLabel(self.page_3)
        self.label_12.setGeometry(QtCore.QRect(140, 110, 231, 201))
        self.label_12.setStyleSheet("border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"background-color: rgb(220, 220, 220);")
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.stackedWidget.addWidget(self.page_3)
        self.listWidget = QtWidgets.QListWidget(Dialog2)
        self.listWidget.setGeometry(QtCore.QRect(0, 30, 121, 631))
        self.listWidget.setStyleSheet("QListView::item {\n"
"    height: 80px;\n"
"}\n"
"QListView::item:hover {\n"
"    background-color: transparent;\n"
"    padding: 10px;\n"
"    border-left: 3px solid rgb(255, 210, 29);\n"
"}\n"
"QListView::item:selected {\n"
"    background-color: transparent;\n"
"    color: black;\n"
"    padding: 10px;\n"
"    border-left: 3px solid rgb(230, 168, 23);\n"
"}\n"
"QListView{\n"
"    font: 10pt \"微软雅黑\";\n"
"    border: 1px solid rgb(194, 194, 194); /* 设置边框的大小，样式，颜色 */\n"
"    \n"
"}")
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.graphicsView = QtWidgets.QGraphicsView(Dialog2)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 1091, 31))
        self.graphicsView.setStyleSheet("    background-color: rgb(222, 222, 222);\n"
"    font: 10pt \"微软雅黑\";\n"
"    border: 1px solid rgb(194, 194, 194); /* 设置边框的大小，样式，颜色 */\n"
"\n"
"")
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_3 = QtWidgets.QGraphicsView(Dialog2)
        self.graphicsView_3.setGeometry(QtCore.QRect(0, 0, 121, 31))
        self.graphicsView_3.setStyleSheet("    background-color: rgb(222, 222, 222);\n"
"    font: 10pt \"微软雅黑\";\n"
"    border: 1px solid rgb(194, 194, 194); /* 设置边框的大小，样式，颜色 */\n"
"")
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.label = QtWidgets.QLabel(Dialog2)
        self.label.setGeometry(QtCore.QRect(320, 0, 211, 31))
        self.label.setStyleSheet("font: 10pt \"微软雅黑\";")
        self.label.setObjectName("label")
        self.pushButton_5 = QtWidgets.QPushButton(Dialog2)
        self.pushButton_5.setGeometry(QtCore.QRect(1060, 0, 32, 32))
        self.pushButton_5.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(218, 218, 218);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(218, 218, 218);\n"
"}\n"
"QPushButton{\n"
"border:None\n"
"}")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(Dialog2)
        self.pushButton_6.setGeometry(QtCore.QRect(1020, 0, 32, 32))
        self.pushButton_6.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(218, 218, 218);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(218, 218, 218);\n"
"}\n"
"QPushButton{\n"
"border:None\n"
"}")
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_13 = QtWidgets.QLabel(Dialog2)
        self.label_13.setGeometry(QtCore.QRect(550, 0, 161, 31))
        self.label_13.setStyleSheet("font: 10pt \"微软雅黑\";")
        self.label_13.setObjectName("label_13")
        self.graphicsView.raise_()
        self.listWidget.raise_()
        self.stackedWidget.raise_()
        self.graphicsView_3.raise_()
        self.label.raise_()
        self.pushButton_5.raise_()
        self.pushButton_6.raise_()
        self.label_13.raise_()

        self.retranslateUi(Dialog2)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog2)

    def retranslateUi(self, Dialog2):
        _translate = QtCore.QCoreApplication.translate
        Dialog2.setWindowTitle(_translate("Dialog2", "Dialog"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("Dialog2", "0"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog2", "姓名"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog2", "年龄"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Dialog2", "学院"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Dialog2", "学号"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("Dialog2", "总成绩"))
        self.lineEdit.setPlaceholderText(_translate("Dialog2", "姓名"))
        self.lineEdit_7.setPlaceholderText(_translate("Dialog2", "学号"))
        self.pushButton.setText(_translate("Dialog2", "查询学生信息"))
        self.label_5.setText(_translate("Dialog2", "学生信息查询"))
        self.pushButton_2.setText(_translate("Dialog2", "输出所有学生信息"))
        self.pushButton_3.setText(_translate("Dialog2", "人脸采集"))
        self.lineEdit_6.setPlaceholderText(_translate("Dialog2", "请输入拼音/英文"))
        self.pushButton_4.setText(_translate("Dialog2", "信息修改"))
        self.label_2.setText(_translate("Dialog2", "管理员信息修改"))
        self.lineEdit_11.setPlaceholderText(_translate("Dialog2", "识别时请正对摄像头,在光照好的环境下采集"))
        self.lineEdit_21.setPlaceholderText(_translate("Dialog2", "新的管理员账号"))
        self.lineEdit_22.setPlaceholderText(_translate("Dialog2", "新的管理员密码"))
        self.lineEdit_23.setPlaceholderText(_translate("Dialog2", "旧的管理员账号"))
        self.lineEdit_24.setPlaceholderText(_translate("Dialog2", "旧的管理员密码"))
        self.pushButton_12.setText(_translate("Dialog2", "信息修改"))
        self.lineEdit_8.setPlaceholderText(_translate("Dialog2", "姓名"))
        self.lineEdit_9.setPlaceholderText(_translate("Dialog2", "学院"))
        self.lineEdit_12.setPlaceholderText(_translate("Dialog2", "年龄"))
        self.lineEdit_13.setPlaceholderText(_translate("Dialog2", "学号"))
        self.label_3.setText(_translate("Dialog2", "对象信息修改"))
        self.lineEdit_17.setPlaceholderText(_translate("Dialog2", "请输入修改对象的姓名"))
        self.lineEdit_18.setPlaceholderText(_translate("Dialog2", "请输入修改对象的学号"))
        self.label_8.setText(_translate("Dialog2", "修改对象查询"))
        self.pushButton_14.setText(_translate("Dialog2", "查询对象"))
        self.lineEdit_20.setPlaceholderText(_translate("Dialog2", "先查询对象信息之后，再进行修改"))
        self.lineEdit_25.setPlaceholderText(_translate("Dialog2", "总成绩"))
        self.pushButton_13.setText(_translate("Dialog2", "信息删除"))
        self.lineEdit_14.setPlaceholderText(_translate("Dialog2", "姓名"))
        self.lineEdit_15.setPlaceholderText(_translate("Dialog2", "学号"))
        self.label_4.setText(_translate("Dialog2", "学生信息删除"))
        self.lineEdit_19.setPlaceholderText(_translate("Dialog2", "请输入删除对象的姓名和学号"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("Dialog2", "学生信息查询"))
        item = self.listWidget.item(1)
        item.setText(_translate("Dialog2", "管理员信息修改"))
        item = self.listWidget.item(2)
        item.setText(_translate("Dialog2", "学生信息修改"))
        item = self.listWidget.item(3)
        item.setText(_translate("Dialog2", "学生信息删除"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.label.setText(_translate("Dialog2", "信息管理系统(管理员模式)  |"))
        self.pushButton_5.setText(_translate("Dialog2", "X"))
        self.pushButton_6.setText(_translate("Dialog2", "-"))
        self.label_13.setText(_translate("Dialog2", "学生信息查询"))
