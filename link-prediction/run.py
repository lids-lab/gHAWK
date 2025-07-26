#!/usr/bin/env python3
"""
Entry-point script for training and evaluation.
Parses command-line args, sets up logging, and kicks off run_training.
"""
import sys

from config import get_args
from logger import get_logger
from training import run_training
from evaluation import run_evaluation

def main():
    # Parse all shared flags
    args = get_args()
    # Initialize logger
    logger = get_logger(__name__)
    logger.info("Starting experiment")

    # Run training (includes validation & final test per config)
    run_training(args)

    # If you ever want to manually trigger evaluation only:
    # logger.info("Running standalone validation")
    # run_evaluation(model, args, split='valid')
    # logger.info("Running standalone test")
    # run_evaluation(model, args, split='test')

if __name__ == "__main__":
    main()