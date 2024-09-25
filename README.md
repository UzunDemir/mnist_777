# Handwritten Digit Recognition

1. Initially, I trained a handwritten digit recognition model based on the MNIST (Modified National Institute of Standards and Technology) database. The accuracy on the test dataset needed to be at least 68%. I tried various models and finally chose a Convolutional Neural Network (CNN), which achieved a test accuracy of 0.99. You can view the notebook with the research [here]([link-to-notebook](https://github.com/UzunDemir/mnist_777/blob/main/RESEARCH%26MODEL/prepare_model.ipynb)).

2. The second step was to wrap the trained model into a service and deploy it as part of a web application for recognizing handwritten digits. After that, I needed to create a Docker image and run the application in a Docker container.

3. I decided to build a full-fledged application that uploads an image of a digit and predicts it. Being a perfectionist, I thought: what if I could draw the digit myself and let the model predict it? After some research, I decided to use Streamlit to implement this idea. [Here's](https://mnist777.streamlit.app/) what I came up with!

## Эта [приложка](https://mnist777.streamlit.app/) выполнена в рамках практической работы по модулю Computer Vision курса Machine Learning Advanced от Skillbox.

![200w](https://github.com/UzunDemir/mnist_777/assets/94790150/09956e06-04b2-43fb-9eac-993f1201db74)


1. Вначале была обучена модель распознавания рукописных цифр на базе MNIST (Modified National Institute of Standards and Technology database).
Точность на тестовой выборке датасета должна быть не ниже 68%. Я использовал много разных моделей и остановил свой выбор на сверточной нейронной сети (Convolutional Neural Network, CNN) которая показала точность на тестовом наборе данных: 0.99.
Ноутбук с исследованиями можно посмотреть здесь.
2. Вторым шагом необходимо было обернуть готовую модель в сервис и запустить её как часть веб-приложения для распознавания самостоятельно написанных символов. После этого нужно было создать docker-образ и запустить приложение в docker-контейнере.
3. Я решил сделать полноценное приложение, которое загружает изображение цифры и предсказывает ее. Но как злостный перфекционист, я подумал: а что если самому рисовать цифру и пусть модель ее предсказывает! Немного поискал как реализовать эту идею и остановил свой выбор на Streamlit. И вот что [получилось!](https://mnist777.streamlit.app/) 

 absl-py==2.1.0
 altair==5.4.1
 astunparse==1.6.3
 attrs==24.2.0
 blinker==1.8.2
 cachetools==5.5.0
 certifi==2024.8.30
 charset-normalizer==3.3.2
 click==8.1.7
 contourpy==1.3.0
 cycler==0.12.1
 flatbuffers==24.3.25
 fonttools==4.54.1
 gast==0.6.0
 gitdb==4.0.11
 gitpython==3.1.43
 google-pasta==0.2.0
 grpcio==1.66.1
 h5py==3.11.0
 idna==3.10
 jinja2==3.1.4
 joblib==1.4.2
 jsonschema==4.23.0
 jsonschema-specifications==2023.12.1
 keras==3.5.0
 kiwisolver==1.4.7
 libclang==18.1.1
 markdown==3.7
 markdown-it-py==3.0.0
 markupsafe==2.1.5
 matplotlib==3.9.2
 mdurl==0.1.2
 ml-dtypes==0.4.1
 namex==0.0.8
 narwhals==1.8.3
 numpy==1.26.4
 opencv-python-headless==4.10.0.84
 opt-einsum==3.3.0
 optree==0.12.1
 packaging==24.1
 pandas==2.2.3
 pillow==10.4.0
 protobuf==4.25.5
 pyarrow==17.0.0
 pydeck==0.9.1
 pygments==2.18.0
 pyparsing==3.1.4
 python-dateutil==2.9.0.post0
 pytz==2024.2
 referencing==0.35.1
 requests==2.32.3
 rich==13.8.1
 rpds-py==0.20.0
 scikit-learn==1.5.2
 scipy==1.14.1
 six==1.16.0
 smmap==5.0.1
 streamlit==1.36.0
 streamlit-drawable-canvas==0.9.3
 tenacity==8.5.0
 tensorboard==2.17.1
 tensorboard-data-server==0.7.2
 tensorflow==2.17.0
 tensorflow-io-gcs-filesystem==0.37.1
 termcolor==2.4.0
 threadpoolctl==3.5.0
 toml==0.10.2
 tornado==6.4.1
 typing-extensions==4.12.2
 tzdata==2024.2
 urllib3==2.2.3
 watchdog==4.0.2
 werkzeug==3.0.4
 wrapt==1.16.0
