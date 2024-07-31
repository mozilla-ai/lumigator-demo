import lumigator_demo as ld
import pandas as pd
import multiprocessing
from loguru import logger
from functools import partial

# Global Vars

TEAM_NAME = "stress_testers"
DATASET_NAME = "thunderbird.csv"
DATASET_ID = "db7ff8c2-a255-4d75-915d-77ba73affc53" # thunderbird dataset 

## Run Ground Truth Generation 
def process_ground_truth(index, deployment_id):
    responses = []
    # Load dataset of 100 rows into dataframe for processing
    df = ld.dataset_download(DATASET_ID)
    logger.info("Loaded dataset...")
    for sample in df['examples']:
        response = ld.get_bart_ground_truth(deployment_id, sample)
        responses.append((sample, response.text))
    logger.info("Processed samples")

## Run Evaluation
def run_eval(max_samples=10, experiment_multiplier=3):
    """
    max_samples: change the following to 0 to use all samples in the dataset (probably best for load testing)
    experiment_multiplier: how many times you want to run each experiment
    """

    models = [
        'hf://facebook/bart-large-cnn',
        'hf://mikeadimech/longformer-qmsum-meeting-summarization',
        'hf://mrm8488/t5-base-finetuned-summarize-news',
        'hf://Falconsai/text_summarization',
        "oai://gpt-3.5-turbo-0125", # cheapest for load-test
        "mistral://open-mistral-7b"
    ]

    responses = []
    for i in range(experiment_multiplier):
        for model in models:
            descr = f"Testing {model} summarization model on {DATASET_NAME}"
            responses.append(ld.experiments_submit(model, TEAM_NAME, descr, DATASET_ID, max_samples))

if __name__ == '__main__':
    logger.info("Running ground truth generation...")

    
    #---Ground Truth Generation---
    # Create a single ground truth deployment with 1 GPU and 3 Instances
    logger.info("Creating deployment...")
    deployment_id = ld.create_deployment(1.0, 3.0)
    partial_process = partial(process_ground_truth, deployment_id=deployment_id)
    logger.info("Processing n times...")
    # create 10 concurrent instances of processing 100 samples from dataset
    with multiprocessing.Pool(processes=10) as pool:
        results = pool.map(partial_process, range(10))
    logger.info("Deleting deployment...")
    ld.delete_deployment( deployment_id)
    
    #---Model Evaluation---
    logger.info("Running model evaluation....")
    run_eval()