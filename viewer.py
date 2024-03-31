import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QGridLayout, QPushButton, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt
from data import cifar10
from retriever import simple_retriever
import numpy as np

class CIFAR10Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset = cifar10()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('CIFAR-10 Viewer and Retriever')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #333333; color: #FFA500;")

        self.layout = QVBoxLayout()
        self.layout.setSpacing(10)

        self.gridLayout = QGridLayout()
        self.gridLayout.setSpacing(10)
        self.displayImagesWithLabels()

        self.instructionLabel = QLabel("Pick a model, preprocessing method, and click an image to retrieve similar images.")
        self.instructionLabel.setFont(QFont('Arial', 10)) 
        self.layout.addWidget(self.instructionLabel)

        self.modelSelector = self.createStyledComboBox(["AdaBoost", "SVM"])

        self.preprocessingSelector = self.createStyledComboBox(["PCA"])

        self.retrievedImagesLayout = QGridLayout()
        self.retrievedImagesLayout.setSpacing(10)  
        self.layout.addLayout(self.retrievedImagesLayout)

        self.setLayout(self.layout)


    def createStyledComboBox(self, items):
        comboBox = QComboBox(self)
        comboBox.addItems(items)  
        comboBox.setFont(QFont('Arial', 10)) 
        comboBox.setStyleSheet(
            "QComboBox { background-color: #555555; color: #FFA500; padding: 5px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox::down-arrow { image: url(down-arrow.png); width: 14px; height: 14px; }"
            "QComboBox:hover { background-color: #666666; }"
            "QComboBox::drop-down:hover { background-color: #777777; }"
        )  
        self.layout.addWidget(comboBox)
        return comboBox


    def displayImagesWithLabels(self):
        for i, class_name in enumerate(self.dataset.classes):
            idx = np.where(self.dataset.train_labels == i)[0][0]
            image = self.dataset.train_images[idx]
            pixmap = self.convertToPixmap(image)

            imageFrame = QFrame(self)  
            imageFrame.setFrameShape(QFrame.StyledPanel)
            imageFrame.setStyleSheet("background-color: #444444; padding: 5px;") 
            imageLayout = QVBoxLayout(imageFrame)
            imageLabel = QLabel(imageFrame)
            imageLabel.setPixmap(pixmap)
            imageLabel.mousePressEvent = lambda event, idx=idx: self.imageClicked(idx)
            imageLayout.addWidget(imageLabel)
            self.gridLayout.addWidget(imageFrame, i // 5, i % 5)

            label = QLabel(class_name, self)
            label.setFont(QFont('Arial', 10))  
            self.gridLayout.addWidget(label, i // 5, i % 5, Qt.AlignCenter | Qt.AlignBottom)


        self.layout.addLayout(self.gridLayout)

    def imageClicked(self, idx):
        self.selectedImageIdx = idx
        image = self.dataset.train_images[idx]
        model_name = self.modelSelector.currentText()
        preprocessing_method = self.preprocessingSelector.currentText()

        retriever = simple_retriever(model_name, image, preprocessing=preprocessing_method)
        retriever.load_preprocessing()
        retriever.load_model()
        label = retriever.inference(image.flatten())  
        retrievedImagePaths = retriever.retrieve_images(self.dataset.classes[label])

        self.displayRetrievedImages(retrievedImagePaths[:10])


    def displayRetrievedImages(self, images):

        for i in reversed(range(self.retrievedImagesLayout.count())): 
            widgetToRemove = self.retrievedImagesLayout.itemAt(i).widget()
            self.retrievedImagesLayout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)

        for i, img_array in enumerate(images):
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)

            bytesPerLine = 3 * img_array.shape[1] 
            qimg = QImage(img_array.data, img_array.shape[1], img_array.shape[0], bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(128, 128, Qt.KeepAspectRatio)
            label = QLabel(self)
            label.setPixmap(pixmap)
            self.retrievedImagesLayout.addWidget(label, i // 5, i % 5)


    def convertToPixmap(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        bytesPerLine = 3 * 32
        qimg = QImage(image.data, image.shape[1], image.shape[0], bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(128, 128, Qt.KeepAspectRatio)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CIFAR10Viewer()
    ex.show()
    sys.exit(app.exec_())
