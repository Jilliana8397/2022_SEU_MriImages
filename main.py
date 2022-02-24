import os
import SimpleITK as sitk
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from tvtk.api import tvtk
from tvtk.pyface.api import Scene

from MainWidget import Ui_MainWindow

os.environ['QT_API'] = 'pyqt5'
matplotlib.use("Qt5Agg")  # 声明使用QT5

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.cwd = os.getcwd()  # 获取当前程序文件位置

        self.visualization = Visualization()
        self.mayaviWidget = self.visualization.edit_traits(
            parent=self, kind='subpanel').control
        self.mayaviWidget.setParent(self.Image)
        self.verticalLayout.addWidget(self.mayaviWidget)
        self.visualization.scene.mlab.clf()

        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.figCanvas = FigureCanvas(self.figure)  # 创建FigureCanvas对象
        self.figCanvas.axes = self.figure.add_subplot(111)
        self.figCanvas.axes.axis('off')
        self.verticalLayout_3.addWidget(self.figCanvas)
        colors = ['white', 'red', 'lime', 'yellow']
        self.cmap = matplotlib.colors.ListedColormap(colors)

        self.figure2 = Figure(figsize=(3, 3), dpi=100)
        self.figCanvas2 = FigureCanvas(self.figure2)  # 创建FigureCanvas对象
        self.figCanvas2.axes = self.figure2.add_subplot(111)
        self.figCanvas2.axes.axis('off')
        self.verticalLayout_4.addWidget(self.figCanvas2)

        # flag
        self.isLabelDrawn = False
        self.isPredictDrawn = False
        self.isContourDrawn = False
        self.isSliceDrawn = False

        # data
        self.image = None
        self.imageNum = 0  # 当前显示层数
        self.file_name = None
        self.shape = None
        self.label = None

        self.LabelObj = None
        self.PredictObj = None
        # self.Unet = UNet(n_channels=1, n_classes=1)
        # self.Unet.load_state_dict(torch.load(os.path.join("Data/Evaluation.dat")))
        self.PredictResult = None
        self.StateLabelDotCounter = 1

        # Btn connect
        self.LoadImage.clicked.connect(self.slotChooseFile)
        self.View3D.clicked.connect(self.slotView3D)
        self.View2D.clicked.connect(self.slotView2D)
        self.ShowLabel.clicked.connect(self.slotShowLabel)
        self.HideLabel.clicked.connect(self.slotHideLabel)
        self.ShowPredict.clicked.connect(self.slotShowPredict)
        self.HidePredict.clicked.connect(self.slotHidePredict)
        self.SaveFig.clicked.connect(self.slotSaveFig)

        self.__cid = self.figCanvas.mpl_connect("scroll_event", self.mouse_scroll)  # View2D支持鼠标滚轮切换层数
        self.__cid2 = self.figCanvas2.mpl_connect("scroll_event", self.mouse_scroll)

    def slotChooseFile(self):
        # 选择文件
        fileName_image, fileType = QFileDialog.getOpenFileName(self,
                                                               "Choose File",
                                                               self.cwd + "/Data/image",  # 起始路径
                                                               "nii(*.nii.gz)")  # 设置文件扩展名过滤,用双分号间隔
        # read into numpy
        if fileName_image != "":
            # load
            self.file_name = fileName_image.split("/")[-1].rstrip(".nii.gz")
            thisImg = sitk.ReadImage("./Data/image/" + self.file_name + ".nii.gz")
            thisImage = sitk.GetArrayFromImage(thisImg).astype(float)
            thisLab = sitk.ReadImage("./Data/label/" + self.file_name + "_gt.nii.gz")
            thisLabel = sitk.GetArrayFromImage(thisLab).astype(float)

            self.image = thisImage[:, :, :]
            self.label = thisLabel[:, :, :]

            # self.ImagePath.setText("ImagePath:" + "Data/image/" + self.file_name + ".nii.gz")
            # self.LabelPath.setText("LabelPath: " + "Data/label/" + self.file_name + ".nii.gz")
            self.groupBox_3.setEnabled(True)
            self.View2D.setEnabled(True)
            self.View3D.setEnabled(True)
        else:
            QMessageBox.about(self, "LoadImage", "No Nii File Selected!")

    def slotView3D(self):
        QApplication.processEvents()
        self.visualization.scene.mlab.clf()
        # fig = mlab.gcf()
        # fig.scene.interactor.interactor_style = tvtk.InteractorStyleSwitch()
        contour = self.visualization.scene.mlab.contour3d(self.image, color=(1, 1, 0), opacity=0.5)
        self.visualization.scene.mlab.axes(xlabel='x', ylabel='y', zlabel='z', line_width=4)
        # update flags
        self.isLabelDrawn = False
        self.isPredictDrawn = False
        self.isContourDrawn = True
        self.isSliceDrawn = False

    def slotView2D(self):
        self.figCanvas.axes.cla()
        self.figCanvas.axes.axis('off')  # 去掉坐标轴
        self.figCanvas2.axes.cla()
        self.figCanvas2.axes.axis('off')  # 去掉坐标轴

        theFigure = self.image[0, :, :]
        self.figCanvas.axes.imshow(theFigure, cmap='gray')
        self.figCanvas.draw()
        theLabel = self.label[0, :, :]
        newLabel = theLabel.copy()
        newLabel[newLabel != 0] = 1
        figure2 = theFigure * newLabel
        self.figCanvas2.axes.imshow(figure2, cmap='Greens')
        self.figCanvas2.draw()

        # update flags
        self.isSliceDrawn = True
        self.isContourDrawn = False

    def mouse_scroll(self, event):  # 通过鼠标滚轮切换层数
        if event.button == 'up':
            # 鼠标向上滚
            if self.imageNum != 0:
                self.imageNum -= 1
                self.figCanvas.axes.cla()
                self.figCanvas.axes.axis('off')  # 去掉坐标轴
                self.figCanvas2.axes.cla()
                self.figCanvas2.axes.axis('off')  # 去掉坐标轴

                theFigure = self.image[self.imageNum, :, :]
                self.figCanvas.axes.imshow(theFigure, cmap='gray')
                self.figCanvas.draw()
                theLabel = self.label[self.imageNum, :, :]
                newLabel = theLabel.copy()
                newLabel[newLabel != 0] = 1
                figure2 = theFigure * newLabel
                self.figCanvas2.axes.imshow(figure2, cmap='Greens')
                self.figCanvas2.draw()

                if self.isLabelDrawn:
                    self.figCanvas.axes.imshow(theLabel, cmap=self.cmap, alpha=0.15)
                    self.figCanvas.draw()

        elif event.button == 'down':
            # 鼠标向下滚
            if self.imageNum != 9:
                self.imageNum += 1
                self.figCanvas.axes.cla()
                self.figCanvas.axes.axis('off')  # 去掉坐标轴
                self.figCanvas2.axes.cla()
                self.figCanvas2.axes.axis('off')  # 去掉坐标轴

                theFigure = self.image[self.imageNum, :, :]
                self.figCanvas.axes.imshow(theFigure, cmap='gray')
                self.figCanvas.draw()
                theLabel = self.label[self.imageNum, :, :]
                newLabel = theLabel.copy()
                newLabel[newLabel != 0] = 1
                figure2 = theFigure * newLabel
                self.figCanvas2.axes.imshow(figure2, cmap='Greens')
                self.figCanvas2.draw()

                if self.isLabelDrawn:
                    self.figCanvas.axes.imshow(theLabel, cmap=self.cmap, alpha=0.15)
                    self.figCanvas.draw()

    def slotShowLabel(self):
        if not self.isLabelDrawn:
            QApplication.processEvents()
            self.LabelObj = self.visualization.scene.mlab.contour3d(self.label, colormap='Blues', opacity=0.9)

            theLabel = self.label[self.imageNum, :, :]
            self.figCanvas.axes.imshow(theLabel, cmap=self.cmap, alpha=0.15)
            self.figCanvas.draw()

            # update flags
            self.isLabelDrawn = True
        else:
            self.LabelObj.visible = True

    def slotHideLabel(self):
        self.figCanvas.axes.cla()
        self.figCanvas.axes.axis('off')  # 去掉坐标轴
        theFigure = self.image[self.imageNum, :, :]
        self.figCanvas.axes.imshow(theFigure, cmap='gray')
        self.figCanvas.draw()

        self.isLabelDrawn = False
        self.LabelObj.visible = False

    def slotShowPredict(self):
        if not self.isPredictDrawn:
            QApplication.processEvents()
            # self.PredictResult = predict(self.Unet, self.shape, self.image, 1)
            QApplication.processEvents()
            # self.PredictObj = self.visualization.scene.mlab.contour3d(self.PredictResult,
            #                                                           color=(1, 0, 0),
            #                                                           opacity=1)
            # update flags
            self.isPredictDrawn = True
        else:
            self.PredictObj.visible = True

    def slotHidePredict(self):
        self.PredictObj.visible = False

    def slotSaveFig(self):
        if self.PredictResult is not None:
            self.PredictResult.tofile("Data/predict/" + self.file_name + ".nii")
            QMessageBox.about(self, "SaveFig",
                              "The nii has been saved at /Data/predict/" + self.file_name + ".nii successfully")
        else:
            QMessageBox.about(self, "SaveFig", "No result to save!")


class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(Item('scene', editor=SceneEditor(scene_class=Scene),
                     show_label=False),
                resizable=True  # We need this to resize with the parent widget
                )


if __name__ == "__main__":
    app = QApplication.instance()
    main_window = MainWindow()
    main_window.show()
    app.exec_()

# 我改一下这里
# 我再改一下这里 然后按crtl+S