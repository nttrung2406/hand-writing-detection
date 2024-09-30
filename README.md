# Handwriting Recognition(OCR)

**Dataset**: https://www.kaggle.com/datasets/ssarkar445/handwriting-recognitionocr/data

This dataset consists of more than four hundred thousand handwritten names collected through charity projects to *support disadvantaged children* around the world.

*Optical Character Recognition (OCR)* utilizes image processing technologies to convert characters on scanned documents into digital forms. It typically performs well in machine printed fonts. However, it still poses a difficult challenges for machines to recognize handwritten characters, because of the huge variation in individual writing styles. Although deep learning is pretty robust to such changes.

**Data Structure**:

This data set is a comprehensive collection of handwritten images and their corresponding text. The data set is divided into 4 distinct sections to facilitate *training, testing, and validation* of machine learning models.

The first section, named *CSV, contains CSV files* with the Image file names and text for training, testing, and validation purposes. These files can be used to build, train, and evaluate machine learning models.

The second section, named *test_v2, is a directory containing 41.4K testing images*. These images can be used to evaluate the performance of the machine learning models on unseen data.

The third section, named *train_v2, is a directory containing 331K training images*. These images can be used to train the machine learning models on a large and diverse set of handwriting samples.

The fourth section, named *validation_v2, is a directory containing 41.4K validation images*. These images can be used to tune the machine learning models' hyperparameters and to prevent overfitting.

Overall, this data set provides a rich resource for training and testing machine learning models on handwritten OCR tasks.

**Model**:

The model consists of three main components:

CNN Layer: Extracts features from the input image.

LSTM Layer: Processes the sequence of features to understand context and dependencies between characters.

Fully Connected (FC) Layer: Produces the final output, predicting the class labels (characters).

**Visulize process**:
![image](https://github.com/user-attachments/assets/5abd87a6-1dd0-499d-b436-0201deb8f109)


**Training**:

Training monitoring with Prometheus and Grafana.

Prometheus is used to monitor the system through pre-installed daemons on nodes, thereby collecting necessary information. Prometheus communicates with the node via http/https protocol and stores data in a time-series database.

Grafana is a vizualizer that displays metrics as charts or graphs, assembled into a highly customizable dashboard, making it easy to monitor node health. Simply for you to understand, after getting the metrics from the devices, grafana will use that metric to analyze and create a dashboard that visually depicts the metrics necessary for monitoring such as CPU, RAM, disks, IO. operations...

**Result**:

