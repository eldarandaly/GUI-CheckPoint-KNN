# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'untitledLYUxpx.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1216, 803)
        self.feed = QLabel(Form)
        self.feed.setObjectName(u"feed")
        self.feed.setGeometry(QRect(10, 0, 870, 760))
        self.EmpID = QLabel(Form)
        self.EmpID.setObjectName(u"EmpID")
        self.EmpID.setGeometry(QRect(1050, 120, 47, 13))
        self.EmpPhoto = QLabel(Form)
        self.EmpPhoto.setObjectName(u"EmpPhoto")
        self.EmpPhoto.setGeometry(QRect(940, 60, 81, 91))
        self.EmpPhoto.setAutoFillBackground(False)
        self.EmpName = QLabel(Form)
        self.EmpName.setObjectName(u"EmpName")
        self.EmpName.setGeometry(QRect(1050, 90, 47, 13))

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.feed.setText(QCoreApplication.translate("Form", u"TextLabel", None))
        self.EmpID.setText(QCoreApplication.translate("Form", u"EmpID", None))
        self.EmpPhoto.setText(QCoreApplication.translate("Form", u"EmpPhoto", None))
        self.EmpName.setText(QCoreApplication.translate("Form", u"EmpName", None))
    # retranslateUi

