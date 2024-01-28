from src.model.model import MNISTCNNClassifier

classifier = MNISTCNNClassifier()
classifier.train(epochs=5)
classifier.evaluate()
weights_path = "C:/Users/Administrator/PycharmProjects/MNIST-CNN-demo/weights/mnist_cnn_weights_3.h5"
classifier.save_weights(weights_path)