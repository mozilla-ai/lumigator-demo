"""Common definitions and methods for the lumigator demo notebook."""

import io
import os
from pathlib import Path
from typing import Any, Dict  # noqa: UP035
from uuid import UUID
import json

import pandas as pd
import requests

# APP URL
API_HOST = os.environ["LUMIGATOR_SERVICE_HOST"]
API_URL = f"https://{API_HOST}/api/v1"

# Ray URL
RAY_HEAD_HOST = os.environ["RAYCLUSTER_KUBERAY_HEAD_SVC_PORT_8265_TCP_ADDR"]
RAY_SERVER_URL = f"http://{RAY_HEAD_HOST}:8265"

# base S3 path
S3_BASE_PATH = "lumigator-storage/experiments/results/"

# - BASE --------------------------------------------------------------


def make_request(
    url: str,
    method: str = "GET",
    params: Dict[str, Any] = None,  # noqa: UP006
    data: Dict[str, Any] = None,  # noqa: UP006
    files: Dict[str, Any] = None,  # noqa: UP006
    headers: Dict[str, str] = None,  # noqa: UP006
    json_: Dict[str, str] = None,  # noqa: UP006
    timeout: int = 10,
    verbose: bool = True,
    *args,
    **kwargs,
) -> requests.Response:
    """Make an HTTP request using the requests library.

    Args:
        url (str)
        method (str, optional): The HTTP method to use. Defaults to "GET".
        params (Dict[str, Any], optional): URL parameters to include in the request.
        data (Dict[str, Any], optional): Data to send in the request body.
        files (Dict[str, Any], optional): Files to send in the request body.
        headers (Dict[str, str], optional): Headers to include in the request.
        timeout (int, optional): Timeout for the request in seconds. Defaults to 10.

    Returns:
        requests.Response: The response object from the request.

    Raises:
        requests.RequestException
    """
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            params=params,
            data=data,
            files=files,
            headers=headers,
            timeout=timeout,
            json=json_,
            *args,
            **kwargs,  # noqa: B026
        )
        response.raise_for_status()
        if verbose:
            print(f"{json.dumps(response.json(), indent = 2)}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        raise
    return response


def get_ray_link(job_id: UUID, RAY_SERVER_URL: str) -> str:
    return f"{RAY_SERVER_URL}/#/jobs/{job_id}"


def get_resource_id(response: requests.Response) -> UUID:
    if response.status_code == requests.codes.created:
        return json.loads(response.text).get("id")


def get_job_status(response: requests.Response) -> UUID:
    if response.status_code == requests.codes.ok:
        return json.loads(response.text).get("status")


def download_text_file(response: requests.Response) -> str:
    """Downloads a text file from the API.

    Given a response from an API `/download` URL, returns the
    corresponding text file.
    Can be used both for textual datasets and evaluation results/logs.
    """
    download_url = json.loads(response.text)["download_url"]  
    download_response = make_request(download_url, verbose=False)
    return download_response.text


# - DATASETS ----------------------------------------------------------


def dataset_upload(filename: str) -> requests.Response:
    with Path(filename).open("rb") as f:
        files = {"dataset": f}
        payload = {"format": "experiment"}
        r = make_request(f"{API_URL}/datasets", method="POST", data=payload, files=files)
    return r


def dataset_info(dataset_id: UUID) -> requests.Response:
    r = make_request(f"{API_URL}/datasets/{dataset_id}")
    return r


def dataset_download(dataset_id: UUID) -> pd.DataFrame:
    """Downloads a CSV dataset from the backend and returns a pandas df.

    NOTE: currently limited to CSV (single-file) datasets, to be extended
          with more general dataset types (e.g. HF datasets as we already
          support their upload).
    """
    r = make_request(f"{API_URL}/datasets/{dataset_id}/download", verbose=False)
    csv_dataset = download_text_file(r)
    return pd.read_csv(io.StringIO(csv_dataset))


# - EXPERIMENTS -------------------------------------------------------


def experiments_submit(
    model_name: str,
    name: str,
    description: str | None,
    dataset_id: UUID,
    max_samples: int | None = None,
    system_prompt: str | None = None,
) -> requests.Response:
    if system_prompt is None and (
        model_name.startswith("oai://") or model_name.startswith("http://")
    ):
        system_prompt = "You are a helpful assistant, expert in text summarization. For every prompt you receive, provide a summary of its contents in at most two sentences."  # noqa: E501

    payload = {
        "name": name,
        "description": description,
        "model": model_name,
        "dataset": dataset_id,
        "max_samples": max_samples,
        "system_prompt": system_prompt,
    }

    r = make_request(f"{API_URL}/experiments", method="POST", data=json.dumps(payload))
    return r


def experiments_info(experiment_id: UUID) -> requests.Response:
    r = make_request(f"{API_URL}/experiments/{experiment_id}")
    return r


def experiments_status(experiment_id: UUID) -> str:
    r = make_request(f"{API_URL}/health/jobs/{experiment_id}", verbose=False)
    return get_job_status(r)


def show_experiment_statuses(job_ids):
    still_running = False
    for job_id in job_ids:
        job_status = experiments_status(job_id)
        print(f"{job_id}: {job_status}")
        if job_status == "PENDING" or job_status == "RUNNING":
            still_running = True
    return still_running


# - RESULTS -----------------------------------------------------------


def experiments_result_download(experiment_id: UUID) -> str:
    r = make_request(f"{API_URL}/experiments/{experiment_id}/result/download", verbose=False)
    exp_results = json.loads(download_text_file(r))
    return exp_results


def eval_results_to_table(models, eval_results):
    """Format evaluation results jsons into one pandas dataframe."""
    # metrics dict format is
    # "column name": [list of keys to get val in nested results dict]
    metrics = {
        "Meteor": ["meteor", "meteor_mean"],
        "BERT Precision": ["bertscore", "precision_mean"],
        "BERT Recall": ["bertscore", "recall_mean"],
        "BERT F1": ["bertscore", "f1_mean"],
        "ROUGE-1": ["rouge", "rouge1_mean"],
        "ROUGE-2": ["rouge", "rouge2_mean"],
        "ROUGE-L": ["rouge", "rougeL_mean"],
    }

    def parse_model_results(model, results):
        row = {}

        # remove prefix from model name
        model_name = model.split("://")
        if len(model_name) > 0:
            model_name = model_name[1]

        row["Model"] = model_name

        for column, metric in metrics.items():
            temp_results = results
            for key in metric:
                value = temp_results.get(key)
                if value is None:
                    break
                temp_results = value

            row[column] = value
        return row

    eval_table = []
    for model, results in zip(models, eval_results, strict=True):
        eval_table.append(parse_model_results(model, results))

    return pd.DataFrame(eval_table)

# - GROUND TRUTH -----------------------------------------------------------

# Mistral Ground Truth
def get_mistral_ground_truth(prompt: str) -> str:
    response = make_request(f"{API_URL}/completions/mistral",method="POST", data=json.dumps({"text": prompt}))
    return json.loads(response.text).get("text")

def create_deployment(gpus:float, replicas: float) -> str:
    data = {
    "num_gpus": gpus,
    "num_replicas": replicas
    }
    headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
    response = make_request(f"{API_URL}/ground-truth/deployments", headers=headers,data=json.dumps(data), method="POST")
    return json.loads(response.text).get("id")

def get_deployments() -> requests.Response:
    response = make_request(f"{API_URL}/ground-truth/deployments" )
    return response

def get_bart_ground_truth(deployment_id: UUID, prompt:str) -> str:
    response = make_request(f"{API_URL}/ground-truth/deployments/{deployment_id}", method="POST", data=json.dumps({"text": prompt}))
    return json.loads(response.text).get("text")


