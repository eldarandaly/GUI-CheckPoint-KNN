# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'notfi.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(500, 501)
        self.EmpPhoto = QLabel(Dialog)
        self.EmpPhoto.setObjectName(u"EmpPhoto")
        self.EmpPhoto.setGeometry(QRect(180, 40, 121, 141))
        self.EmpPhoto.setFrameShape(QFrame.Box)
        self.EmpPhoto.setFrameShadow(QFrame.Plain)
        self.EmpPhoto.setLineWidth(2)
        self.EmpName = QLabel(Dialog)
        self.EmpName.setObjectName(u"EmpName")
        self.EmpName.setGeometry(QRect(9, 254, 481, 31))
        self.EmpName.setFrameShape(QFrame.Box)
        self.EmpID = QLabel(Dialog)
        self.EmpID.setObjectName(u"EmpID")
        self.EmpID.setGeometry(QRect(10, 290, 481, 31))
        self.EmpID.setFrameShape(QFrame.Box)
        self.EmpDep = QLabel(Dialog)
        self.EmpDep.setObjectName(u"EmpDep")
        self.EmpDep.setGeometry(QRect(10, 330, 481, 31))
        self.EmpDep.setFrameShape(QFrame.Box)
        self.EmpJobTitle = QLabel(Dialog)
        self.EmpJobTitle.setObjectName(u"EmpJobTitle")
        self.EmpJobTitle.setGeometry(QRect(10, 370, 481, 31))
        self.EmpJobTitle.setFrameShape(QFrame.Box)

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.EmpPhoto.setText(QCoreApplication.translate("Dialog", u"EmpPhoto", None))
        self.EmpName.setText(QCoreApplication.translate("Dialog", u"EmpName", None))
        self.EmpID.setText(QCoreApplication.translate("Dialog", u"EmpID", None))
        self.EmpDep.setText(QCoreApplication.translate("Dialog", u"EmpDep", None))
        self.EmpJobTitle.setText(QCoreApplication.translate("Dialog", u"EmpJobTitle", None))
    # retranslateUi

