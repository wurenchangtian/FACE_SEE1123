# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MW_SW3.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog3(object):
    def setupUi(self, Dialog3):
        Dialog3.setObjectName("Dialog3")
        Dialog3.resize(916, 618)
        self.stackedWidget = QtWidgets.QStackedWidget(Dialog3)
        self.stackedWidget.setGeometry(QtCore.QRect(10, 10, 901, 601))
        self.stackedWidget.setStyleSheet("QStackedWidget{\n"
"background-color: rgb(255, 255, 255);\n"
"border: 5px solid rgb(194, 194, 194);\n"
"border-radius:20px;\n"
"}")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.pushButton_3 = QtWidgets.QPushButton(self.page)
        self.pushButton_3.setGeometry(QtCore.QRect(150, 450, 211, 51))
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
        self.lineEdit_2 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_2.setGeometry(QtCore.QRect(550, 100, 211, 51))
        self.lineEdit_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_2.setText("")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_3.setGeometry(QtCore.QRect(550, 180, 211, 51))
        self.lineEdit_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_3.setText("")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_4.setGeometry(QtCore.QRect(550, 260, 211, 51))
        self.lineEdit_4.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_4.setText("")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_5.setGeometry(QtCore.QRect(550, 340, 211, 51))
        self.lineEdit_5.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_5.setText("")
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.page)
        self.lineEdit_6.setGeometry(QtCore.QRect(140, 370, 231, 51))
        self.lineEdit_6.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.lineEdit_6.setText("")
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_4.setGeometry(QtCore.QRect(550, 450, 201, 51))
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
        self.label_6.setGeometry(QtCore.QRect(140, 140, 231, 221))
        self.label_6.setStyleSheet("background-color: rgb(222, 222, 222);\n"
"border: 1px solid rgb(194, 194, 194);\n"
"border-radius:10px;\n"
"")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_2 = QtWidgets.QLabel(self.page)
        self.label_2.setGeometry(QtCore.QRect(380, 0, 161, 41))
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
        self.pushButton_5 = QtWidgets.QPushButton(self.page)
        self.pushButton_5.setGeometry(QtCore.QRect(0, 520, 871, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setAutoFillBackground(False)
        self.pushButton_5.setStyleSheet("QPushButton::hover{\n"
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
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.page)
        self.pushButton_6.setGeometry(QtCore.QRect(850, 10, 32, 32))
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
        self.pushButton_7 = QtWidgets.QPushButton(self.page)
        self.pushButton_7.setGeometry(QtCore.QRect(810, 10, 32, 32))
        self.pushButton_7.setStyleSheet("QPushButton::hover{\n"
"background-color: rgb(218, 218, 218);\n"
"}\n"
"QPushButton::pressed{\n"
"background-color: rgb(218, 218, 218);\n"
"}\n"
"QPushButton{\n"
"border:None\n"
"}")
        self.pushButton_7.setObjectName("pushButton_7")
        self.stackedWidget.addWidget(self.page)

        self.retranslateUi(Dialog3)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog3)

    def retranslateUi(self, Dialog3):
        _translate = QtCore.QCoreApplication.translate
        Dialog3.setWindowTitle(_translate("Dialog3", "Dialog"))
        self.pushButton_3.setText(_translate("Dialog3", "人脸采集"))
        self.lineEdit_2.setPlaceholderText(_translate("Dialog3", "姓名"))
        self.lineEdit_3.setPlaceholderText(_translate("Dialog3", "年龄"))
        self.lineEdit_4.setPlaceholderText(_translate("Dialog3", "学院"))
        self.lineEdit_5.setPlaceholderText(_translate("Dialog3", "学号"))
        self.lineEdit_6.setPlaceholderText(_translate("Dialog3", "请输入拼音/英文"))
        self.pushButton_4.setText(_translate("Dialog3", "信息录入"))
        self.label_2.setText(_translate("Dialog3", "学生信息录入"))
        self.lineEdit_11.setPlaceholderText(_translate("Dialog3", "识别时请正对摄像头,在光照好的环境下采集"))
        self.pushButton_5.setText(_translate("Dialog3", "完成录入"))
        self.pushButton_6.setText(_translate("Dialog3", "X"))
        self.pushButton_7.setText(_translate("Dialog3", "-"))
