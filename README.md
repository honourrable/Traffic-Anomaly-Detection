# Traffic-Anomaly-Detection

Detection of anomaly studies are carried out to allow human systems work well. In this study, I aimed to find out whether the system finds an anomaly case on road which is a 3D
model. To classify real time images that are captured by USB webcams, first the images were preproccessed and prepared to perform the task. To achieve this, SIFT algorithm were
used to extract features of images. Then, the following machine learning algorithms were implemented to predict the real time images:

- Support Vector Machines (SVM)
- Logistic Regression
- Random Forest

The system includes dataset creation by taking picture of the 3D design with webcams, data augmentation to create new data samples, a machine learning algorithm for training and
real time system execution by taking frames in per 3 seconds. After detecting any anomaly in the system, the relevant signals were sent to Raspberry Pi devices from laptop. These
R. Pi devices played role in printing messages to LCD displays and sending signals to led & buzzer components to simulate an emergent situation in anomaly case.

The connection between laptop and R. Pi is set by using SSH connection via MobaXTerm software. It also allows to view and control R. Pi from laptop with a graphical interface. The
wired connection was set with a ethernet cable and relevant username & password information were provided by a Python library which is called Paramiko. This was quite efficient
to make the system components communicate with each other in program execution, in code instead of running an external program or sub process.

### System Requirements 

An IDE with Python support can be used to open the project, specifically PyCharm were used to build the project. 

### Author

- [honourrable](https://github.com/honourrable)
