FROM python:3.7

# Create working directory
WORKDIR /app

COPY requirements.txt ./requirements.txt

COPY . /app

RUN mkdir -p /app/checkpoint/UGATIT_sample_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing

# Copy requirements. From source, to destination.

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8080

# copying all files over. From source, to destination.


#Run app
CMD streamlit run --server.port 8080 --server.address=0.0.0.0 --server.enableCORS false app.py
