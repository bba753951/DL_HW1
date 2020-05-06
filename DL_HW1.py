import sys
from PyQt5.QtGui import  QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QGroupBox, QPushButton, QLabel, QHBoxLayout,  QVBoxLayout, QGridLayout, QFormLayout, QLineEdit, QTextEdit,  QComboBox
from hw1_4 import hw1_4
from hw1_2 import hw1_2
from hw1_1_1 import hw1_1_1
from hw1_1_2 import hw1_1_2
from hw1_1_3 import hw1_1_3
from hw1_1_4 import hw1_1_4
from show_pic import show_pic,show_para,show_result
from show_train_loss import show_epoch1_loss
from load_mod import show_predict

class Cv2019_Hw1(QWidget):
    def __init__(self):
        super(Cv2019_Hw1,self).__init__()
        self.initUi()

    def initUi(self):
        self.createGridGroupBox()
        self.creatVboxGroupBox()
        self.Num1_3=1

        qbtn_ok = QPushButton('OK', self)
        qbtn_cancel = QPushButton('Cancel', self)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.gridGroupBox1,0,0,2,2)
        mainLayout.addWidget(self.vboxGroupBox3,0,2,3,2)
        mainLayout.addWidget(self.vboxGroupBox2,2,0)
        mainLayout.addWidget(self.vboxGroupBox4,2,1)
        mainLayout.addWidget(self.vboxGroupBox5,0,4,3,2)
        mainLayout.addWidget(qbtn_ok,3,4)
        mainLayout.addWidget(qbtn_cancel,3,5)
        self.setLayout(mainLayout)

    def createGridGroupBox(self):
        self.gridGroupBox1 = QGroupBox("1. Calibration")
        self.vboxGroupBox2=QGroupBox("2. Augmented Reality")
        self.vboxGroupBox4=QGroupBox("4. Find Contour")
        self.vboxGroupBox5=QGroupBox("5. Cifar10 Lenet")
        GB1_3=QGroupBox("1.3 Extrinsic")

        layout = QGridLayout()
        layout1_3 = QVBoxLayout() 
        layout2 = QVBoxLayout() 
        layout4 = QVBoxLayout() 
        layout5 = QVBoxLayout() 
        layout5_5 = QHBoxLayout() 
        layout_all = QGridLayout()

        qbtn1_1 = QPushButton('1.1 Find Corners', self)
        qbtn1_1.clicked.connect(hw1_1_1)


        qbtn1_2 = QPushButton('1.2 Intrinsic', self)
        qbtn1_2.clicked.connect(hw1_1_2)

        qbtn1_4 = QPushButton('1.4 Distortion', self)
        qbtn1_4.clicked.connect(hw1_1_4)

        qbtn1_3 = QPushButton('1.3 Extrinsic', self)

        qbtn2_1 = QPushButton('2.1 Augmented Reality', self)
        qbtn2_1.clicked.connect(hw1_2)



        qbtn4_1 = QPushButton('4.1 Find Contour', self)
        qbtn4_1.clicked.connect(hw1_4)

        qbtn5_1 = QPushButton('5.1 Show Train Images', self)
        qbtn5_1.clicked.connect(show_pic)
        qbtn5_2 = QPushButton('5.2 Show Hyperparameters', self)
        qbtn5_2.clicked.connect(show_para)
        qbtn5_3 = QPushButton('5.3 Train 1 Epoch', self)
        qbtn5_3.clicked.connect(show_epoch1_loss)
        qbtn5_4 = QPushButton('5.4 Show Traning Result', self)
        qbtn5_4.clicked.connect(show_result)        
        qbtn5_5 = QPushButton('Inference', self)
        qbtn5_5.clicked.connect(self.onclick5)
        self.textbox = QLineEdit('(0-9999)',self)
        combo = QComboBox(self)
        combo.addItem("1")
        combo.addItem("2")
        combo.addItem("3")
        combo.addItem("4")
        combo.addItem("5")
        combo.addItem("6")
        combo.addItem("7")
        combo.addItem("8")
        combo.addItem("9")
        combo.addItem("10")
        combo.addItem("11")
        combo.addItem("12")
        combo.addItem("13")
        combo.addItem("14")
        combo.addItem("15")
        combo.activated[str].connect(self.selectNum)
        qbtn1_3.clicked.connect(lambda:hw1_1_3(self.Num1_3))

        nameLabel = QLabel("Select image")
        nameLabel5_5 = QLabel("Test Image Index:")
        layout1_3.setSpacing(10) 
        layout1_3.addWidget(nameLabel)
        layout1_3.addWidget(combo)
        layout1_3.addWidget(qbtn1_3)

        layout2.addWidget(qbtn2_1)

        layout4.addWidget(qbtn4_1)

        layout5.addWidget(qbtn5_1)
        layout5.addWidget(qbtn5_2)
        layout5.addWidget(qbtn5_3)
        layout5.addWidget(qbtn5_4)
        layout5_5.addWidget(nameLabel5_5)
        layout5_5.addWidget(self.textbox)
        layout5.addLayout(layout5_5)
        layout5.addWidget(qbtn5_5)

        layout.setSpacing(10) 
        layout.addWidget(qbtn1_1,0,0)
        layout.addWidget(qbtn1_2,1,0)
        layout.addWidget(qbtn1_4,2,0)
        layout.addWidget(GB1_3,0,1,3,1)

        GB1_3.setLayout(layout1_3)
        self.vboxGroupBox2.setLayout(layout2)
        self.vboxGroupBox4.setLayout(layout4)
        self.vboxGroupBox5.setLayout(layout5)
        self.gridGroupBox1.setLayout(layout)
        self.setWindowTitle('Cv2019_Hw1')
    def selectNum(self,text):
    	self.Num1_3=int(text)
    def onclick5(self):
    	textboxValue = int(self.textbox.text())
    	show_predict(textboxValue)

    def creatVboxGroupBox(self):
        self.vboxGroupBox3 = QGroupBox("3. Image Transformation")
        GB3_1=QGroupBox("3.1 Rot, scale, Translate")
        GB_p=QGroupBox("Parameters")
        layout_p = QGridLayout()
        layout3_1 = QVBoxLayout()
        layout3 = QVBoxLayout()

        Angle = QLabel("Angle：")
        Scale = QLabel("Scale：")
        Tx = QLabel("Tx：")
        Ty = QLabel("Ty：")
        deg = QLabel("deg")
        pixel = QLabel("pixel")
        pixel1 = QLabel("pixel")

        angleEditor = QLineEdit()
        scaleEditor = QLineEdit()
        txEditor = QLineEdit()
        tyEditor = QLineEdit()

          
        layout_p.addWidget(Angle,0,0)
        layout_p.addWidget(Scale,1,0)
        layout_p.addWidget(Tx,2,0)
        layout_p.addWidget(Ty,3,0)
        layout_p.addWidget(angleEditor,0,1)
        layout_p.addWidget(scaleEditor,1,1)
        layout_p.addWidget(txEditor,2,1)
        layout_p.addWidget(tyEditor,3,1)
        layout_p.addWidget(deg,0,2)
        layout_p.addWidget(pixel,2,2)
        layout_p.addWidget(pixel1,3,2)
        GB_p.setLayout(layout_p)

        qbtn3_1 = QPushButton('3.1 Rotation, scaling, translation', self)

        layout3_1.addWidget(GB_p)
        layout3_1.addWidget(qbtn3_1)
        GB3_1.setLayout(layout3_1)

        qbtn3_2 = QPushButton('3.2 Perspective Transform', self)


        layout3.addWidget(GB3_1)
        layout3.addWidget(qbtn3_2)
        self.vboxGroupBox3.setLayout(layout3)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Cv2019_Hw1()
    ex.show()
    sys.exit(app.exec_())
