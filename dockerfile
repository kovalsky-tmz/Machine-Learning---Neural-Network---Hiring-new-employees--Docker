FROM python:3
WORKDIR app/
COPY . app/
RUN pip install numpy

CMD [ "python", "app/Neural_Network.py" ]