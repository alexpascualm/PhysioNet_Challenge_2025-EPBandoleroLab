FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r ./requirements.txt

# RUN pip install numpy
# RUN pip install scipy
# RUN pip install wfdb
# RUN pip install scikit-learn
# RUN pip install torch
# RUN pip install joblib