# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'designerZlIkHV.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1195, 787)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.feed = QLabel(self.centralwidget)
        self.feed.setObjectName(u"feed")
        self.feed.setGeometry(QRect(0, 0, 871, 761))
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(910, 0, 281, 751))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.EmpPhoto = QLabel(self.frame)
        self.EmpPhoto.setObjectName(u"EmpPhoto")
        self.EmpPhoto.setGeometry(QRect(0, 80, 81, 91))
        self.EmpPhoto.setAutoFillBackground(False)
        self.EmpName = QLabel(self.frame)
        self.EmpName.setObjectName(u"EmpName")
        self.EmpName.setGeometry(QRect(100, 100, 47, 13))
        self.EmpID = QLabel(self.frame)
        self.EmpID.setObjectName(u"EmpID")
        self.EmpID.setGeometry(QRect(100, 130, 47, 13))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1195, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.feed.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.EmpPhoto.setText(QCoreApplication.translate("MainWindow", u"EmpPhoto", None))
        self.EmpName.setText(QCoreApplication.translate("MainWindow", u"EmpName", None))
        self.EmpID.setText(QCoreApplication.translate("MainWindow", u"EmpID", None))
    # retranslateUi

