from __future__ import annotations
import pickle
from datetime import datetime, timedelta
import json
from airflow import DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
}

def _prepare(**context):
    from src.model_deployment import load_data, split_preprocess, train_and_save, evaluate
    from src.success_email import send_success_email
    """Load data, split, build pipeline; stash in XCom."""
    df = load_data()
    pipe, X_train, X_test, y_train, y_test = split_preprocess(df)
    ti = context["ti"]
    ti.xcom_push(key="pipe", value=pickle_dumps(pipe))
    ti.xcom_push(key="X_train", value=pickle_dumps(X_train))
    ti.xcom_push(key="X_test", value=pickle_dumps(X_test))
    ti.xcom_push(key="y_train", value=pickle_dumps(y_train))
    ti.xcom_push(key="y_test", value=pickle_dumps(y_test))

def _train(**context):
    from airflow.operators.python import PythonOperator
    from src.model_deployment import load_data, split_preprocess, train_and_save, evaluate
    from src.success_email import send_success_email
    ti = context["ti"]
    pipe = pickle_loads(ti.xcom_pull(key="pipe"))
    X_train = pickle_loads(ti.xcom_pull(key="X_train"))
    y_train = pickle_loads(ti.xcom_pull(key="y_train"))
    model_path = train_and_save(pipe, X_train, y_train)
    ti.xcom_push(key="model_path", value=model_path)

def _evaluate(**context):
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from src.model_deployment import load_data, split_preprocess, train_and_save, evaluate
    from src.success_email import send_success_email
    ti = context["ti"]
    X_test = pickle_loads(ti.xcom_pull(key="X_test"))
    y_test = pickle_loads(ti.xcom_pull(key="y_test"))
    result = evaluate(X_test, y_test)   # {'accuracy':..., 'artifact':...}
    ti.xcom_push(key="metrics", value=json.dumps(result))

def _notify(**context):
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from src.model_deployment import load_data, split_preprocess, train_and_save, evaluate
    from src.success_email import send_success_email
    ti = context["ti"]
    conf = (context.get("dag_run") or {}).conf or {}
    email_to = conf.get("email_to")  
    metrics = json.loads(ti.xcom_pull(key="metrics"))
    acc = metrics.get("accuracy")
    artifact = metrics.get("artifact")

    subject = "Airflow Lab-3: Model pipeline succeeded ✅"
    body = f"Accuracy: {acc:.4f}\nMetrics artifact: {artifact}"
    send_success_email(subject=subject, body=body, to=email_to)

# small helpers to move numpy arrays via XCom (pickle bytes)
def pickle_dumps(obj): return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_loads(b):   return pickle.loads(b)

with DAG(
    dag_id="sample_dag",
    description="Lab-3: pipeline with params + email",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["lab3", "airflow"],
) as dag:

    prepare = PythonOperator(
        task_id="prepare",
        python_callable=_prepare,
        provide_context=True,
    )

    train = PythonOperator(
        task_id="train",
        python_callable=_train,
        provide_context=True,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate,
        provide_context=True,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify,
        provide_context=True,
    )
    prepare >> train >> evaluate_task >> notify