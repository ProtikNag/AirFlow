from datetime import (
    timedelta,
    datetime
)
from airflow import DAG
from airflow.operators.python import PythonOperator

from preprocess import preprocess as pp
from train import train_model as tm
from dataset import intel_dataset

default_args = {
    'owner': 'protik.nag',
    'retry': 5,
    'retry_delay': timedelta(seconds=5)
}

with DAG(
    default_args = default_args,
    dag_id = 'testing_pipeline',
    description = "Testing Pipeline of Machine Learning System",
    start_date = datetime(2022, 1, 1),
    schedule_interval = '@daily'
) as dag:
    
    preprocess_executor = PythonOperator(
        task_id = 'Preprocess',
        python_callable = pp
    )
    
    train_executor = PythonOperator(
        task_id = 'train',
        python_callable = tm
    )
    
preprocess_executor >> train_executor