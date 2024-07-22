FROM ultralytics/ultralytics:latest-python

WORKDIR /youlldetect

COPY ./requirements.txt /youlldetect/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /youlldetect/requirements.txt

RUN wget https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.0-1/libedgetpu1-std_16.0tf2.17.0-1.bookworm_amd64.deb
RUN dpkg -i libedgetpu1-std_16.0tf2.17.0-1.bookworm_amd64.deb

COPY ./app /youlldetect/app
CMD ["fastapi", "run", "app/main.py", "--port", "80"]