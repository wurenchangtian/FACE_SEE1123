# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MW_SW1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1097, 615)
        Dialog.setStyleSheet("QDialog{\n"
"background-color: rgb(255, 255, 255);\n"
"border-radius:10px;\n"
"}")
        self.stackedWidget = QtWidgets.QStackedWidget(Dialog)
        self.stackedWidget.setGeometry(QtCore.QRect(140, 40, 961, 571))
        self.stackedWidget.setStyleSheet("QStackedWidget{\n"
"background-color: rgb(255, 255, 255);\n"
"border: 5px solid rgb(194, 194, 194);\n"
"border-radius:20px;\n"
"}")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.tableWidget = QtWidgets.QTableWidget(self.page_2)
        self.tableWidget.setGeometry(QtCore.QRect(300, 110, 641, 381))
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
        self.label_5 = QtWidgets.QLabel(self.page_2)
        self.label_5.setGeometry(QtCore.QRect(350, 30, 151, 21))
        self.label_5.setStyleSheet("font: 14pt \"????????????\";")
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.page_2)
        self.label_7.setGeometry(QtCore.QRect(30, 110, 231, 221))
        self.label_7.setStyleSheet("background-color: rgb(222, 222, 222);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.lineEdit_16 = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_16.setGeometry(QtCore.QRect(30, 360, 231, 41))
        self.lineEdit_16.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_16.setText("")
        self.lineEdit_16.setReadOnly(True)
        self.lineEdit_16.setPlaceholderText("")
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.stackedWidget.addWidget(self.page_2)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.pushButton_12 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_12.setGeometry(QtCore.QRect(530, 430, 201, 51))
        font = QtGui.QFont()
        font.setFamily("????????????")
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
        self.lineEdit_8.setGeometry(QtCore.QRect(470, 150, 321, 51))
        self.lineEdit_8.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_8.setText("")
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_12.setGeometry(QtCore.QRect(470, 240, 321, 51))
        self.lineEdit_12.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_12.setText("")
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.label_3 = QtWidgets.QLabel(self.page_4)
        self.label_3.setGeometry(QtCore.QRect(560, 70, 151, 21))
        self.label_3.setStyleSheet("font: 14pt \"????????????\";")
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
        self.label_8.setStyleSheet("font: 14pt \"????????????\";")
        self.label_8.setObjectName("label_8")
        self.pushButton_14 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_14.setGeometry(QtCore.QRect(50, 430, 231, 51))
        font = QtGui.QFont()
        font.setFamily("????????????")
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
        self.label_11.raise_()
        self.label_10.raise_()
        self.pushButton_12.raise_()
        self.lineEdit_8.raise_()
        self.lineEdit_12.raise_()
        self.label_3.raise_()
        self.lineEdit_17.raise_()
        self.lineEdit_18.raise_()
        self.label_8.raise_()
        self.pushButton_14.raise_()
        self.label_9.raise_()
        self.lineEdit_20.raise_()
        self.stackedWidget.addWidget(self.page_4)
        self.listWidget = QtWidgets.QListWidget(Dialog)
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
"    font: 10pt \"????????????\";\n"
"    border: 1px solid rgb(194, 194, 194); /* ??????????????????????????????????????? */\n"
"    \n"
"}")
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(0, 0, 1091, 31))
        self.graphicsView.setStyleSheet("    background-color: rgb(222, 222, 222);\n"
"    font: 10pt \"????????????\";\n"
"    border: 1px solid rgb(194, 194, 194); /* ??????????????????????????????????????? */\n"
"\n"
"")
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_3 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView_3.setGeometry(QtCore.QRect(0, 0, 121, 31))
        self.graphicsView_3.setStyleSheet("    background-color: rgb(222, 222, 222);\n"
"    font: 10pt \"????????????\";\n"
"    border: 1px solid rgb(194, 194, 194); /* ??????????????????????????????????????? */\n"
"")
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(420, 0, 121, 31))
        self.label.setStyleSheet("font: 10pt \"????????????\";")
        self.label.setObjectName("label")
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setGeometry(QtCore.QRect(1050, 0, 32, 32))
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
        self.pushButton_6 = QtWidgets.QPushButton(Dialog)
        self.pushButton_6.setGeometry(QtCore.QRect(1010, 0, 32, 32))
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
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setGeometry(QtCore.QRect(550, 0, 161, 31))
        self.label_13.setStyleSheet("font: 10pt \"????????????\";")
        self.label_13.setObjectName("label_13")
        self.graphicsView.raise_()
        self.listWidget.raise_()
        self.stackedWidget.raise_()
        self.graphicsView_3.raise_()
        self.label.raise_()
        self.pushButton_5.raise_()
        self.pushButton_6.raise_()
        self.label_13.raise_()

        self.retranslateUi(Dialog)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("Dialog", "0"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "??????"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "??????"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "??????"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Dialog", "??????"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("Dialog", "?????????"))
        self.label_5.setText(_translate("Dialog", "??????????????????"))
        self.pushButton_12.setText(_translate("Dialog", "????????????"))
        self.lineEdit_8.setPlaceholderText(_translate("Dialog", "??????"))
        self.lineEdit_12.setPlaceholderText(_translate("Dialog", "??????"))
        self.label_3.setText(_translate("Dialog", "??????????????????"))
        self.lineEdit_17.setPlaceholderText(_translate("Dialog", "??????????????????????????????"))
        self.lineEdit_18.setPlaceholderText(_translate("Dialog", "??????????????????????????????"))
        self.label_8.setText(_translate("Dialog", "??????????????????"))
        self.pushButton_14.setText(_translate("Dialog", "????????????"))
        self.lineEdit_20.setPlaceholderText(_translate("Dialog", "?????????????????????????????????????????????"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("Dialog", "??????????????????"))
        item = self.listWidget.item(1)
        item.setText(_translate("Dialog", "??????????????????"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.label.setText(_translate("Dialog", "??????????????????  |"))
        self.pushButton_5.setText(_translate("Dialog", "X"))
        self.pushButton_6.setText(_translate("Dialog", "-"))
        self.label_13.setText(_translate("Dialog", "??????????????????"))
