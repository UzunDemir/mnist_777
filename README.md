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

