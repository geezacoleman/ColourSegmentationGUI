from PyQt5 import QtGui, uic, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from segmentation import exG, hsv_segmentation, crop_to_contour
from pascal_voc_writer import Writer
from imutils import paths
import shutil
import json
import cv2
import sys
import os

def crop_to_rect(image, mask, startX, startY, width, height):
    rect = QRect(startX, startY, width, height)

    imageROI = image.copy(rect)
    maskROI = mask.copy(rect)

    return imageROI, maskROI


class ImageSignalWorker(QObject):
    finished = pyqtSignal()


class ThreadWorker(QRunnable):
    def __init__(self, func, *args, **kwargs):
        super(ThreadWorker, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = ImageSignalWorker()

    @pyqtSlot()
    def run(self):
        self.func(*self.args, **self.kwargs)
        self.signals.finished.emit()


class SegmentationApp(QMainWindow):
    def __init__(self, parent=None):
        super(SegmentationApp, self).__init__(parent)
        uic.loadUi("qtdesigner/gui.ui", self)

        self.setWindowTitle("Colour Annotation Tool v0.1")
        self.saveDirlineEdit.setText(r'C:\Users\gcol4791\PycharmProjects\Tools\Synthetic\weeds')

        self.imageCounter = 0
        self.imagePath = None
        self.completedImages = []
        self.thresholdDict = {}
        self.contourList = []
        self.maskedImageROI = None

        self.classDict = {0: [self.class1lineEdit, self.class1checkBox, self.class1pushButton],
                          1: [self.class2lineEdit, self.class2checkBox, self.class2pushButton],
                          2: [self.class3lineEdit, self.class3checkBox, self.class3pushButton],
                          3: [self.class4lineEdit, self.class4checkBox, self.class4pushButton],
                          4: [self.class5lineEdit, self.class5checkBox, self.class5pushButton]}

        self.checkBoxList = [self.class1checkBox, self.class2checkBox, self.class3checkBox, self.class4checkBox, self.class5checkBox]

        self.class1checkBox.clicked.connect(lambda: self.class_enabled_check(classNumber=0))
        self.class2checkBox.clicked.connect(lambda: self.class_enabled_check(classNumber=1))
        self.class3checkBox.clicked.connect(lambda: self.class_enabled_check(classNumber=2))
        self.class4checkBox.clicked.connect(lambda: self.class_enabled_check(classNumber=3))
        self.class5checkBox.clicked.connect(lambda: self.class_enabled_check(classNumber=4))

        self.setpushButton.clicked.connect(self.set_classes)
        self.resetpushButton.clicked.connect(self.reset_classes)
        self.saveParameterspushButton.clicked.connect(self.save_params)
        self.loadpushButton.clicked.connect(self.load_params)

        self.nextimagepushButton.clicked.connect(self.segment_images)
        self.nextcontourpushButton.clicked.connect(self.filter_contours)
        self.backpushButton.clicked.connect(self.back)
        self.skippushButton.clicked.connect(self.filter_contours)

        self.class1pushButton.clicked.connect(lambda: self.save_roi(classNumber=0))
        self.class2pushButton.clicked.connect(lambda: self.save_roi(classNumber=1))
        self.class3pushButton.clicked.connect(lambda: self.save_roi(classNumber=2))
        self.class4pushButton.clicked.connect(lambda: self.save_roi(classNumber=3))
        self.class5pushButton.clicked.connect(lambda: self.save_roi(classNumber=4))

    def segment_images(self):
        try:
            self.annWriter.save(self.xmlfileName)
            self.infolineEdit.setText('Saved annwriter {}'.format(self.xmlfileName))
            cv2.imwrite(self.imagefileName, self.image)

        except:
            self.infolineEdit.setText('Did not save annwriter')

        try:
            if self.imagePath and self.movecheckBox.isChecked():
                shutil.move(src=self.imagefullPath, dst=os.path.join(self.completedDir, os.path.basename(self.imagePath)))
                print('Moved')
            self.imagePath = self.imageList[0]
            self.imagefullPath = os.path.join(self.imageDir, os.path.basename(self.imagePath))
            self.image = cv2.imread(self.imagefullPath)

            self.displayimage = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], QImage.Format_RGB888).rgbSwapped()
            self.displayimageResized = self.displayimage.scaled(self.displayLabel.width(), self.displayLabel.height(), Qt.KeepAspectRatio)
            self.displayLabel.setPixmap(QPixmap.fromImage(self.displayimageResized))

            self.runpushButton.clicked.connect(lambda: self.segmentation(image=self.image))

            self.imageList.pop(0)
            self.completedImages.append(self.imagefullPath)
            if len(self.imageList) < 1:
                self.nextimagepushButton.setEnabled(False)
                self.infolineEdit.setText('No images left')
                self.imageDirlineEdit.setEnabled(True)

            if self.xmlradioButton.isChecked():
                xmlName = self.imagePrefix + "_{}".format(self.imageCounter) + ".xml"
                imageName = self.imagePrefix + "_{}".format(self.imageCounter) + ".jpg"
                self.xmlfileName = os.path.join(self.vocSaveDir, xmlName)
                self.imagefileName = os.path.join(self.imageSaveDir, imageName)

                self.annWriter = Writer(self.imagefullPath, width=self.image.shape[1], height=self.image.shape[0])
                self.imageCounter += 1
        except Exception as e:
            print(e)

    def segmentation(self, image):
        self.read_sliders()
        saturation = self.satdoubleSpinBox.value()
        brightness = self.brightdoubleSpinBox.value()
        blur = self.blurspinBox.value()
        extOnly = self.extOnlycheckBox.isChecked()

        if self.exGradioButton.isChecked():
            self.infolineEdit.setText('[INFO] Using excess green index for segmentation.')

            extOnly = self.extOnlycheckBox.isChecked()
            self.contours, self.maskedImage, self.mask, display = exG(image,
                                                                      thresholdMin=self.thresholdDict['t1min'],
                                                                      thresholdMax=self.thresholdDict['t1max'],
                                                                      saturation=saturation,
                                                                      brightness=brightness,
                                                                      blur=blur,
                                                                      externalOnly=extOnly)

        else:
            self.infolineEdit.setText('[INFO] Using HSV thresholding index for segmentation.')
            self.contours, self.maskedImage, self.mask, display = hsv_segmentation(image,
                                                                                   hmin=self.thresholdDict['t1min'],
                                                                                   hmax=self.thresholdDict['t1max'],
                                                                                   smin=self.thresholdDict['t2min'],
                                                                                   smax=self.thresholdDict['t2max'],
                                                                                   vmin=self.thresholdDict['t3min'],
                                                                                   vmax=self.thresholdDict['t3max'],
                                                                                   saturation=saturation,
                                                                                   brightness=brightness,
                                                                                   blur=blur,
                                                                                   externalOnly=extOnly)

        self.displayimage = QImage(display.data, display.shape[1], display.shape[0], display.strides[0], QImage.Format_RGB888)
        self.displayimageResized = self.displayimage.scaled(self.maskdisplayLabel.width(), self.maskdisplayLabel.height(), Qt.KeepAspectRatio)
        self.maskdisplayLabel.setPixmap(QPixmap.fromImage(self.displayimageResized))
        self.nextcontourpushButton.setEnabled(True)
        self.backpushButton.setEnabled(True)
        self.skippushButton.setEnabled(True)

    def filter_contours(self):
        contour = self.contours[0]
        image, mask, self.startX, self.startY, self.boxW, self.boxH = crop_to_contour(self.mask, self.maskedImage, contour)

        gray_color_table = [qRgb(i, i, i) for i in range(256)]
        imageQ = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        maskQ = QImage(mask.data, mask.shape[1], mask.shape[0], mask.strides[0], QImage.Format_Indexed8)
        maskQ.setColorTable(gray_color_table)
        self.maskedImageROI, self.maskROI = crop_to_rect(imageQ, maskQ, self.startX, self.startY, self.boxW, self.boxH)
        self.maskedImageROIResized = self.maskedImageROI.scaled(self.outDisplay.width(), self.outDisplay.height(), Qt.KeepAspectRatio)
        self.outDisplay.setPixmap(QPixmap.fromImage(self.maskedImageROIResized))
        self.contourList.append(contour)
        # if len(self.contourList) > 10:
        #     self.contourList = []
        self.contours.pop(0)

        if len(self.contours) < 1:
            self.nextcontourpushButton.setEnabled(False)
            self.skippushButton.setEnabled(False)
            self.infolineEdit.setText('Contours complete')

    def save_roi(self, classNumber=0):
        if self.maskedImageROI:
            classList = self.classDict[classNumber]
            lineEdit = classList[0]
            classDir = lineEdit.text()

            if self.masksradioButton.isChecked():
                self.imageCounter += 1
                imageName = os.path.join(classDir, os.path.basename(self.saveDir) + "_{}".format(self.imageCounter) + ".png")
                maskName = os.path.join(classDir, os.path.basename(self.saveDir)+ "_{}".format(self.imageCounter) + ".png")
                imagefileName = os.path.join(self.saveDir, os.path.join('image', imageName))
                maskfileName = os.path.join(self.saveDir, os.path.join('ann', maskName))
                self.maskedImageROI.save(imagefileName)
                self.maskROI.save(maskfileName)

                self.infolineEdit.setText('{} saved'.format(imagefileName))
                self.filter_contours()

            elif self.xmlradioButton.isChecked():
                endX = self.startX + self.boxW
                endY = self.startY + self.boxH

                self.annWriter.addObject(classDir, self.startX, self.startY, endX, endY)
                self.display = self.image
                cv2.rectangle(self.display, (self.startX, self.startY), (endX, endY), (255, 20, 20), 5)
                self.displayimage = QImage(self.display.data, self.display.shape[1],
                                           self.display.shape[0], self.display.strides[0], QImage.Format_RGB888)
                self.displayimageResized = self.displayimage.scaled(self.displayLabel.width(),
                                                                    self.displayLabel.height(), Qt.KeepAspectRatio)
                self.displayLabel.setPixmap(QPixmap.fromImage(self.displayimageResized))
                self.filter_contours()

        else:
            self.infolineEdit.setText('ERROR No contours to save')

    def read_sliders(self):
        self.thresholdDict['t1min'] = self.t1minSlider.value()
        self.thresholdDict['t2min'] = self.t2minSlider.value()
        self.thresholdDict['t3min'] = self.t3minSlider.value()

        self.thresholdDict['t1max'] = self.t1maxSlider.value()
        self.thresholdDict['t2max'] = self.t2maxSlider.value()
        self.thresholdDict['t3max'] = self.t3maxSlider.value()

    def set_classes(self):
        print('setting')
        self.imageDir = self.imageDirlineEdit.text()
        self.completedDir = os.path.join(self.imageDir, 'completed')

        if not os.path.exists(self.completedDir) and self.movecheckBox.isChecked():
            os.mkdir(self.completedDir)

        self.saveDir = self.saveDirlineEdit.text()
        print(os.path.basename(self.saveDir))
        if self.prefixlineEdit.text():
            self.imagePrefix = self.prefixlineEdit.text()
        else:
            self.imagePrefix = self.prefixlineEdit.text()

        if self.masksradioButton.isChecked():
            self.infolineEdit.setText('Starting SEGMENTATION labelling')
            for i, checkBox in enumerate(self.checkBoxList):
                if checkBox.isChecked():
                    classNumberList = self.classDict[i]
                    lineEdit = classNumberList[0]

                    imageclassSaveDir = os.path.join(self.saveDir, os.path.join('image', lineEdit.text()))
                    maskclassSaveDir = os.path.join(self.saveDir, os.path.join('ann', lineEdit.text()))
                    if not os.path.exists(imageclassSaveDir):
                        os.makedirs(imageclassSaveDir)

                    if not os.path.exists(maskclassSaveDir):
                        os.makedirs(maskclassSaveDir)

        elif self.xmlradioButton.isChecked():
            self.infolineEdit.setText('Starting PASCAL VOC labelling')
            self.imageSaveDir = os.path.join(self.saveDir, 'image')
            self.vocSaveDir = os.path.join(self.saveDir, 'voc')
            if not os.path.exists(self.imageSaveDir):
                os.makedirs(self.imageSaveDir)

            if not os.path.exists(self.vocSaveDir):
                os.makedirs(self.vocSaveDir)

        try:
            self.imageList = list(paths.list_images(self.imageDir))
            if len(self.imageList) < 1:
                self.nextimagepushButton.setEnabled(False)
                self.infolineEdit.setText('No images left')
                self.imageDirlineEdit.setEnabled(True)
            else:
                self.infolineEdit.setText('Success')
                self.nextimagepushButton.setEnabled(True)
                self.runpushButton.setEnabled(True)
                self.segment_images()

            self.imageDirlineEdit.setEnabled(False)
        except FileNotFoundError:
            self.nextimagepushButton.setEnabled(False)
            self.infolineEdit.setText('ERROR: directory not found')

        self.masksradioButton.setEnabled(False)
        self.xmlradioButton.setEnabled(False)
        self.saveDirlineEdit.setEnabled(False)
        self.prefixlineEdit.setEnabled(False)
        self.movecheckBox.setEnabled(False)

    def reset_classes(self):
        self.masksradioButton.setEnabled(True)
        self.xmlradioButton.setEnabled(True)
        self.imageDirlineEdit.setEnabled(True)
        self.saveDirlineEdit.setEnabled(True)
        self.prefixlineEdit.setEnabled(True)
        self.movecheckBox.setEnabled(True)
        self.imageCounter = 0
        for i, checkBox in enumerate(self.checkBoxList):
            if checkBox.isChecked():
                classNumberList = self.classDict[i]
                lineEdit = classNumberList[0]
                lineEdit.clear()

    def class_enabled_check(self, classNumber):
        classNumberList = self.classDict[classNumber]
        lineEdit = classNumberList[0]
        checkBox = classNumberList[1]
        pushButton = classNumberList[2]

        lineEdit.setEnabled(checkBox.isChecked())
        pushButton.setEnabled(checkBox.isChecked())

    def back(self):
        contour = self.contourList[-2]
        image, mask, startX, startY, boxW, boxH = crop_to_contour(self.mask, self.maskedImage, contour)

        gray_color_table = [qRgb(i, i, i) for i in range(256)]

        step = image.shape[2] * image.shape[1]
        imageQ = QImage(image.data, image.shape[1], image.shape[0], step, QImage.Format_RGB888)
        maskQ = QImage(mask.data, mask.shape[1], mask.shape[0], mask.strides[0], QImage.Format_Indexed8)
        maskQ.setColorTable(gray_color_table)
        self.maskedImageROI, self.maskROI = crop_to_rect(imageQ, maskQ, startX, startY, boxW, boxH)
        self.maskedImageROIResized = self.maskedImageROI.scaled(self.outDisplay.width(), self.outDisplay.height(), Qt.KeepAspectRatio)
        self.outDisplay.setPixmap(QPixmap.fromImage(self.maskedImageROIResized))
        self.contourList.append(contour)

    def save_params(self):
        self.thresholdDict['t1min'] = self.t1minSlider.value()
        self.thresholdDict['t2min'] = self.t2minSlider.value()
        self.thresholdDict['t3min'] = self.t3minSlider.value()

        self.thresholdDict['t1max'] = self.t1maxSlider.value()
        self.thresholdDict['t2max'] = self.t2maxSlider.value()
        self.thresholdDict['t3max'] = self.t3maxSlider.value()

        self.thresholdDict['saturation'] = self.satdoubleSpinBox.value()
        self.thresholdDict['brightness'] = self.brightdoubleSpinBox.value()
        self.thresholdDict['blur'] = self.blurspinBox.value()
        self.thresholdDict['extOnly'] = self.extOnlycheckBox.isChecked()

        with open('parameters.json', 'w') as log:
            json.dump(self.thresholdDict, log)

    def load_params(self):
        print('here')
        try:
            with open('parameters.json') as log:
                self.thresholdDict = json.load(log)

                self.satdoubleSpinBox.setValue(self.thresholdDict['saturation'])
                self.brightdoubleSpinBox.setValue(self.thresholdDict['brightness'])
                self.blurspinBox.setValue(self.thresholdDict['blur'])
                self.extOnlycheckBox.setChecked(self.thresholdDict['extOnly'])

            try:
                self.t1minSlider.setValue(self.thresholdDict['t1min'])
                self.t2minSlider.setValue(self.thresholdDict['t2min'])
                self.t3minSlider.setValue(self.thresholdDict['t3min'])

                self.t1maxSlider.setValue(self.thresholdDict['t1max'])
                self.t2maxSlider.setValue(self.thresholdDict['t2max'])
                self.t3maxSlider.setValue(self.thresholdDict['t3max'])

            except KeyError:
                self.infolineEdit.setText('ERROR: key not found')
        except FileExistsError:
            self.infolineEdit.setText('ERROR: parameters file not found')

        except KeyError:
            self.infolineEdit.setText('ERROR: key not found')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            self.save_roi(classNumber=0)

        elif event.key() == Qt.Key_2:
            self.save_roi(classNumber=1)

        elif event.key() == Qt.Key_3:
            self.save_roi(classNumber=2)

        elif event.key() == Qt.Key_4:
            self.save_roi(classNumber=3)

        elif event.key() == Qt.Key_5:
            self.save_roi(classNumber=4)

        elif event.key() == Qt.Key_6:
            self.save_roi(classNumber=5)

        elif event.key() == Qt.Key_S:
            self.filter_contours()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    AppWindow = SegmentationApp()
    AppWindow.show()
    app.exec()