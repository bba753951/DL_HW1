import tensorflow as tf
import numpy as np
import cv2

def show_pic():
	cifar10 = tf.keras.datasets.cifar10
	(train_data, train_label), (test_data, test_label) = cifar10.load_data()
	train_label = train_label.astype(np.int32) 
	index = np.random.randint(0, np.shape(train_data)[0], 10)

	label_dict=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
	for i in index:
		print(label_dict[int(train_label[i])])
		cv2.namedWindow(label_dict[int(train_label[i])],0)
		cv2.imshow(label_dict[int(train_label[i])],train_data[i])
		cv2.waitKey(500)
		cv2.destroyAllWindows()

def show_para():
	print("hyperparameters:")
	print("batch size: 50")
	print("learning rate: 0.001")
	print("optimizer: Adam")


def show_result():
	pic_arr=['accurate.png','loss.png']
	for i in pic_arr:
		img = cv2.imread(i)
		cv2.namedWindow('image',0)
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
# show_result()
