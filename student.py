import os
from threading import Thread

import cv2
from PyQt5 import Qt
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPixmap, QPalette, QMouseEvent, QCursor
from sqlalchemy import event
from MainWindows import *
from MW_SW1 import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox
import login_rc
import sys
import pymysql
import FACE_get
import use_face_modle

# 这个文件作为打开多个窗口的主要管理文件，并且实现多个功能

class parentWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)


class childWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child = Ui_Dialog()
        self.child.setupUi(self)


class Login_window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        # 这里需要重载一下Login_window，同时也包含了QtWidgets.QMainWindow的预加载项。
        super(Login_window, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.Qt.FramelessWindowHint)
        self.setAttribute(Qt.Qt.WA_TranslucentBackground)
        QPushButton:("hover{background_color: rgb(50, 170, 200)}")
        # 将点击事件与槽函数进行连接
        self.pushButton.clicked.connect(self.on_pushButton_enter_clicked)
        self.pushButton_2.clicked.connect(self.close)
        self.pushButton_3.clicked.connect(self.showMinimized)
    def on_pushButton_enter_clicked(self):
        # 账号判断
        if self.lineEdit_2.text() == "123456" and self.lineEdit.text() == "123456":
            child.show()
            window.close()
        else:
            reply = QMessageBox.warning(self, "警告", "请输入正确的用户名和密码")

    def mousePressEvent(self, event):
        if event.button() == Qt.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(Qt.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.Qt.ArrowCursor))


class MWSW1(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(MWSW1, self).__init__(parent)
        self.setupUi(self)
        self.listWidget.currentRowChanged.connect(self.stackedWidget.setCurrentIndex)  # list和右侧窗口的index对应绑定
        self.listWidget.itemClicked.connect(self.switch_stack)
        self.listWidget.setCurrentRow(0)
        self.switch_stack()
        # 窗口无边框
        self.setWindowFlags(Qt.Qt.FramelessWindowHint)

        # 将按钮和函数进行连接
        self.pushButton_3.clicked.connect(self.Face_get)
        self.pushButton_4.clicked.connect(self.input_db)
        self.pushButton.clicked.connect(self.char_find_db)
        self.pushButton_13.clicked.connect(self.delete_db)
        self.pushButton_12.clicked.connect(self.change_db)
        self.pushButton_2.clicked.connect(self.FACE_SEE)
        self.pushButton_14.clicked.connect(self.change_find_db)
        self.pushButton_5.clicked.connect(self.close)
        self.pushButton_6.clicked.connect(self.showMinimized)
    # 右侧边框控制页面切换
    def switch_stack(self):
        # 把list和stack控件连接起来，得以用list控制stack切换页面
        try:
            i = self.listWidget.currentIndex().row()
            self.stackedWidget.setCurrentIndex(i)
            self.label_13.setText(self.listWidget.item(i).text())
        except:
            pass
    # 人脸识别模块的调用
    def Face_get(self):
        # 调用人脸识别模块，并且把识别的人类储存进新建的文件夹中
        text = self.lineEdit_6.text()
        path_name = './face_data/' + text
        # 建立文件夹
        FACE_get.CreateFolder(path_name)
        th = Thread(FACE_get.CatchPICFromVideo("DP", 0, 100, path_name))
        image_path = path_name + "/0.jpg"
        self.label_6.setPixmap(QPixmap(image_path))

    # 插入相关数据进入数据库
    def input_db(self):
        # 连接数据库
        self.conn = pymysql.connect(
            host='localhost',
            user='root',
            port=3306,
            password='123456',
            db='test',
            charset='utf8',
        )
        # 建立连接
        self.cur = self.conn.cursor()

        # 用于转换字符类型
        name = self.lineEdit_2.text()
        age = self.lineEdit_3.text()
        cla = self.lineEdit_4.text()
        num = self.lineEdit_5.text()

        sql = "INSERT INTO 账号注册 (`姓名`, `年龄`, `学院`, `学号`, `人脸识别名`) VALUES (" + "'" + name + "'" \
              + "," + "'" + age + "'" + "," + "'" + cla + "'" + "," + "'" + num + "'" + "," + "'" + \
              self.lineEdit_6.text() + "'" + ");"

        # 调用sql语句
        self.cur.execute(sql)
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def char_find_db(self):
        self.conn = pymysql.connect(
            host='localhost',
            user='root',
            port=3306,
            password='123456',
            db='test',
            charset='utf8',
        )
        self.cur = self.conn.cursor()

        name = self.lineEdit.text()
        num = self.lineEdit_7.text()

        sql = "SELECT * FROM 账号注册 WHERE 姓名 = " + "'" + name + "'" + " AND 学号 = " + "'" + num + "'"

        self.cur.execute(sql)
        if not self.cur.rowcount:
            reply = QMessageBox.warning(self, "警告", "没有查询到相关学生的信息")
            self.conn.commit()
            self.cur.close()
            self.conn.close()
        else:
            self.conn.commit()
            # 获取遍历得到的数据，并且储存进data中，读取后的data格式为data[][]
            data = self.cur.fetchall()
            # 向tableWidget插入数据
            self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem(str(data[0][0])))
            self.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem(str(data[0][1])))
            self.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem(str(data[0][2])))
            self.tableWidget.setItem(0, 3, QtWidgets.QTableWidgetItem(str(data[0][3])))
            # 在label中插入图片,并且显示查询信息
            image_path = "./face_data/" + str(data[0][4]) + "/0.jpg"
            self.label_7.setPixmap(QPixmap(image_path))
            self.lineEdit_16.setText(str(data[0][4]))
            self.cur.close()
            self.conn.close()

    def change_find_db(self):
        self.conn = pymysql.connect(
            host='localhost',
            user='root',
            port=3306,
            password='123456',
            db='test',
            charset='utf8',
        )
        self.cur = self.conn.cursor()

        old_name = "'" + self.lineEdit_17.text() + "'"
        old_num = "'" + self.lineEdit_18.text() + "'"

        sql = "SELECT * FROM 账号注册 WHERE 姓名 = " + old_name + " AND 学号 = " + old_num

        self.cur.execute(sql)
        self.conn.commit()
        # 获取查询的数据
        data = self.cur.fetchall()
        image_path = "./face_data/" + str(data[0][4]) + "/0.jpg"
        self.label_9.setPixmap(QPixmap(image_path))

        reply = QMessageBox.warning(self, "警告", "查询成功")

        self.cur.close()
        self.conn.close()

    def change_db(self):
        self.conn = pymysql.connect(
            host='localhost',
            user='root',
            port=3306,
            password='123456',
            db='test',
            charset='utf8',
        )
        self.cur = self.conn.cursor()

        old_name = "'" + self.lineEdit_17.text() + "'"
        old_num = "'" + self.lineEdit_18.text() + "'"
        name = "'"+self.lineEdit_8.text()+"'"
        age = "'"+self.lineEdit_12.text()+"'"
        cla = "'"+self.lineEdit_9.text()+"'"
        num = "'"+self.lineEdit_13.text()+"'"

        sql = "UPDATE 账号注册 SET 姓名 = " + str(name) +" , 年龄=" + str(age) +" , 学院="+\
             str(cla)+" , 学号= " +str(num)+ " WHERE 姓名 = " + str(old_name) + " AND 学号 = " + str(old_num)


        self.cur.execute(sql)
        self.conn.commit()

        reply = QMessageBox.warning(self, "警告", "修改完成！")

        self.cur.close()
        self.conn.close()

    def delete_db(self):
        self.conn = pymysql.connect(
            host='localhost',
            user='root',
            port=3306,
            password='123456',
            db='test',
            charset='utf8',
        )
        self.cur = self.conn.cursor()

        name = self.lineEdit_14.text()
        num = self.lineEdit_15.text()

        sql = "SELECT * FROM 账号注册 WHERE 姓名 = " + "'" + name + "'" + " AND " + "'" + num + "'"

        self.cur.execute(sql)

        if not self.cur.rowcount:
            reply = QMessageBox.warning(self, "警告", "未能查找到相关信息")
            self.cur.close()
            self.conn.commit()
        # 获取查询的数据
        else:
            data = self.cur.fetchall()
            image_path = "./face_data/" + str(data[0][4]) + "/0.jpg"
            self.label_12.setPixmap(QPixmap(image_path))
            self.lineEdit_10.setText(str(data[0][4]))
            self.cur.close()
            self.conn.commit()

            self.cur1 = self.conn.cursor()

            sql1 = "DELETE FROM 账号注册 WHERE 姓名 = " + "'" + name + "'" + " AND " + "'" + num + "'"

            self.cur1.execute(sql1)
            if not self.cur1.rowcount:
                reply = QMessageBox.warning(self, "警告", "未能查找到相关信息")
                self.conn.commit()
                self.cur1.close()
                self.conn.close()
            else:
                reply = QMessageBox.warning(self, "警告", "删除成功！")
                self.conn.commit()
                self.cur1.close()
                self.conn.close()

    def FACE_SEE(self):
        ID=use_face_modle.use_face()
        reply = QMessageBox.warning(self, "警告", str(ID))

        for i in range(len(os.listdir('./face_data/'))):
            if i == int(ID):
                face_name = os.listdir('./face_data/')[i]
                reply = QMessageBox.warning(self, "警告", face_name)
                image_path = "./face_data/" + os.listdir('./face_data/')[i] + "/0.jpg"
                self.label_7.setPixmap(QPixmap(image_path))
                self.lineEdit_16.setText(face_name)

            self.conn = pymysql.connect(
                host='localhost',
                user='root',
                port=3306,
                password='123456',
                db='test',
                charset='utf8',
            )
            self.cur = self.conn.cursor()
            sql = "SELECT * FROM 账号注册 WHERE 人脸识别名 = " + "'" + face_name + "'"

            self.cur.execute(sql)
            if not self.cur.rowcount:
                reply = QMessageBox.warning(self, "警告", "学生识别失败")
                self.conn.commit()
                self.cur.close()
                self.conn.close()
            # 获取遍历得到的数据，并且储存进data中，读取后的data格式为data[][]
            else:
                data = self.cur.fetchall()
                # 向tableWidget插入数据
                self.tableWidget.setItem(0, 0, QtWidgets.QTableWidgetItem(str(data[0][0])))
                self.tableWidget.setItem(0, 1, QtWidgets.QTableWidgetItem(str(data[0][1])))
                self.tableWidget.setItem(0, 2, QtWidgets.QTableWidgetItem(str(data[0][2])))
                self.tableWidget.setItem(0, 3, QtWidgets.QTableWidgetItem(str(data[0][3])))
                self.conn.commit()
                self.cur.close()
                self.conn.close()





    def mousePressEvent(self, event):
        if event.button() == Qt.Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(Qt.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.Qt.ArrowCursor))





if __name__ == '__main__':
    app = QApplication(sys.argv)
    child = MWSW1()
    window = Login_window()
    # 显示
    window.show()
    sys.exit(app.exec_())
