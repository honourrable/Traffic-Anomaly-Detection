from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import winsound
import paramiko
import pickle
import random
import time
import cv2
import os


DATADIR = "C:/Users/Onur/PycharmProjects/TrafficAnomalyDetection/dataset"
CATEGORIES = ["Animal", "Normal", "Object", "Pothole"]


TEST_SIZE = 0.1
WIDTH = 200
HEIGHT = 150
AUGMENTATION_NUMBER = 8


# Images' pixels only
def read_data_pixels():
    train_data = []
    counter = {'Animal': 0, 'Normal': 0, 'Object': 0, 'Pothole': 0}

    # Creating new images with augmentation
    # data_augmentation()

    # Reading all images
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_number = CATEGORIES.index(category)
        for imgg in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, imgg))
                new_image = cv2.resize(image, (WIDTH, HEIGHT))

                train_data.append([new_image.flatten(), class_number])

                counter[category] += 1

            except Exception as e:
                print(e)

    random.shuffle(train_data)

    plt.figure(figsize=(5, 5))
    plt.bar(x=counter.keys(), height=counter.values())
    plt.show()

    # Separating features and label from data
    X = []
    y = []
    for features, labels in train_data:
        X.append(features)
        y.append(labels)

    # Saving train data not to read image files over and over again
    np.save('features.npy', X, allow_pickle=True)
    np.save('labels.npy', y, allow_pickle=True)

    return


def prepare_data_pixels():
    # Loading train data and splitting into train, test and validation
    X = np.load('features.npy', allow_pickle=True)
    y = np.load('labels.npy', allow_pickle=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    # Data preprocessing
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Printing some information of data
    print("\nTraining data shape     :", X_train.shape, y_train.shape)
    print("Testing data shape      :", X_test.shape, y_test.shape)
    print('Output classes          :', np.unique(y_train))

    data = [X_train, X_test, y_train, y_test]

    return data


# Images' features extracted by SIFT
def read_data_features():
    train_data = []
    min_feature_number = 999999
    counter = {'Animal': 0, 'Normal': 0, 'Object': 0, 'Pothole': 0}

    # Creating new images with augmentation
    # data_augmentation()

    # Reading all images
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_number = CATEGORIES.index(category)
        for imgg in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, imgg))
                new_image = cv2.resize(image, (WIDTH, HEIGHT))

                sift = cv2.xfeatures2d.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(new_image, None)

                if (np.array(descriptors)).flatten().shape[0] < min_feature_number:
                    min_feature_number = (np.array(descriptors)).flatten().shape[0]

                train_data.append([(np.array(descriptors)).flatten(), class_number])

                counter[category] += 1

            except Exception as e:
                print(e)

    random.shuffle(train_data)

    plt.figure(figsize=(5, 5))
    plt.bar(x=counter.keys(), height=counter.values())
    plt.show()

    # Separating features and label from data
    X = []
    y = []
    for features, labels in train_data:
        X.append(features)
        y.append(labels)

    print("\nMin feature number:", min_feature_number)
    X = [value[:min_feature_number] for value in X]

    # Saving train data not to read image files over and over again
    np.save('features.npy', X, allow_pickle=True)
    np.save('labels.npy', y, allow_pickle=True)

    return


def prepare_data_features():
    # Loading train data and splitting into train, test and validation
    X = np.load('features.npy', allow_pickle=True)
    y = np.load('labels.npy', allow_pickle=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Printing some information of data
    print("\nTraining data shape     :", X_train.shape, y_train.shape)
    print("Testing data shape      :", X_test.shape, y_test.shape)
    print('Output classes          :', np.unique(y_train))

    data = [X_train, X_test, y_train, y_test]

    return data


def prepare_frame(image, limit):
    image = cv2.resize(image, (WIDTH, HEIGHT))

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    image = np.array(descriptors).flatten()

    # image = image.flatten()

    return image[:limit]


def data_augmentation():
    augmentation_file_number = 0

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for imgg in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, imgg))
                new_image = img_to_array(image)
                new_image = new_image.reshape((1,) + new_image.shape)

                datagen = ImageDataGenerator(rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                             brightness_range=(0.5, 1.5))

                augmentation_limit = 1
                for _ in datagen.flow(new_image, batch_size=1, save_prefix='augmented', save_to_dir=path,
                                      save_format='jpeg'):

                    augmentation_limit += 1
                    augmentation_file_number += 1

                    if augmentation_limit > AUGMENTATION_NUMBER:
                        break

            except Exception as e:
                print(e)

    print("\nNumber of augmented images:", augmentation_file_number)

    return


def logistic_reg(X_train, X_test, y_train, y_test):
    lr_classifier = LogisticRegression(random_state=0, max_iter=1000)
    lr_classifier.fit(X_train, y_train)
    y_pred = lr_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    pickle.dump(lr_classifier, open('saved_models/logistic_regression.sav', 'wb'))

    return


def svm(X_train, X_test, y_train, y_test):
    svm_classifier = SVC(kernel='linear', random_state=0)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    pickle.dump(svm_classifier, open('saved_models/svm.sav', 'wb'))

    return


def random_forest(X_train, X_test, y_train, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=0)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    print("\nSuccess metrics:\n\n", classification_report(y_test, y_pred, zero_division=1))

    pickle.dump(rf_classifier, open('saved_models/random_forest.sav', 'wb'))

    return


def send_to_rpi(file):
    host = "raspberrypi.mshome.net"
    port = 22
    username = "pi"
    password = "raspberry"

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)

    sftp = ssh.open_sftp()

    sftp.put(file, os.path.join("/home/pi/Desktop/Anomaly Detection/", "message.txt"))
    sftp.close()
    ssh.close()

    return


def anomaly_prediction(prediction):
    if prediction == 1:
        print("There is no anomaly detected")

    else:
        traffic_warning = ''

        if prediction == 3:
            traffic_warning = "Pothole on the road"
            winsound.Beep(440, 800)

        elif prediction == 0:
            traffic_warning = "Animal on the road"
            winsound.Beep(540, 800)

        elif prediction == 2:
            traffic_warning = "Dangerous object on the road"
            winsound.Beep(640, 800)

        file_name = 'message ' + datetime.now().strftime('%H-%M-%S %d-%m-%Y')
        file_path = 'C:/Users/Onur/PycharmProjects/TrafficAnomalyDetection/rpi_messages/' + file_name + '.txt'

        with open(file_path, 'w') as file:
            file.write(traffic_warning)

        send_to_rpi(file_path)

        print("\nInfo: Message was sent to Raspberry Pi")

        # Wait time for investigation of detected anomaly
        time.sleep(6)

    return


if __name__ == '__main__':
    # Measurement of execution time
    start_time = time.monotonic()

    # Preparing data with images' pixels
    # read_data_pixels()
    # dataset = prepare_data_pixels()

    # Preparing data with images' features extracted by SIFT
    read_data_features()
    dataset = prepare_data_features()

    train_X = dataset[0]
    test_X = dataset[1]
    train_y = dataset[2]
    test_y = dataset[3]

    print("\nLogistic Regression")
    logistic_reg(train_X, test_X, train_y, test_y)
    # print("\nSVM")
    # svm(train_X, test_X, train_y, test_y)
    # print("\nRandom Forest")
    # random_forest(train_X, test_X, train_y, test_y)

    # Loading saved Logistic Regression model
    loaded_model = pickle.load(open("saved_models/logistic_regression.sav", 'rb'))
    min_feature = loaded_model.n_features_in_
    print("\nFeature number in image dataset:", min_feature)

    cap_near = cv2.VideoCapture(0)
    cap_far = cv2.VideoCapture(1)

    if not cap_near.isOpened():
        print("No signal from Traffic Camera 1")
        exit()
    elif not cap_far.isOpened():
        print("No signal from Traffic Camera 2")
        exit()

    while True:
        ret, frame_near = cap_near.read()
        ret2, frame_far = cap_far.read()

        cv2.imshow('Live Traffic - Camera 1 (Near)', frame_near)
        cv2.imshow('Live Traffic - Camera 2 (Far)', frame_far)

        frame_near = prepare_frame(frame_near, min_feature)
        traffic_status_near = loaded_model.predict([frame_near])

        frame_far = prepare_frame(frame_far, min_feature)
        traffic_status_far = loaded_model.predict([frame_far])

        print("\nPrediction of Camera 1 (near):", CATEGORIES[traffic_status_near[0]])
        print("Prediction of Camera 2 (far) :", CATEGORIES[traffic_status_far[0]])

        anomaly_prediction(traffic_status_near[0])
        anomaly_prediction(traffic_status_far[0])

        # Wait time for the next frame capturing
        time.sleep(3)

        # Exit with ESC
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap_near.release()
    cap_far.release()
    cv2.destroyAllWindows()

    end_time = time.monotonic()
    execution_time = timedelta(seconds=end_time - start_time)
    print("\nProgram execution time :", execution_time, "seconds")
