from datetime import (
    timedelta,
    datetime,
)

from airflow import DAG
from airflow.operators.python import PythonOperator 

default_args = {
    'owner': 'protik.nag',
    'retry': 5,
    'retry_delay': timedelta(seconds=5)
}


def get_tensorflow():
    import tensorflow as tf
    print("Tensorflow Version: {}".format(tf.__version__))
    

def get_numpy():
    import numpy as np
    print("Numpy Version: {}".format(np.__version__))


with DAG(
    default_args = default_args,
    dag_id = 'testing_dag',
    start_date = datetime(2022, 1, 1),
    schedule_interval = '@daily'
) as dag:
    
    get_tensorflow_executor = PythonOperator(
        task_id = 'get_tensorflow',
        python_callable = get_tensorflow
    )
    
    get_numpy_executor = PythonOperator(
        task_id = 'get_numpy',
        python_callable = get_numpy
    )
    
get_tensorflow_executor >> get_numpy_executor