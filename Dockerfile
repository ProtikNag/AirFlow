FROM apache/airflow:2.4.1
RUN pip install --user --upgrade pip
RUN pip install tensorflow
RUN pip install numpy 
RUN pip install pandas 