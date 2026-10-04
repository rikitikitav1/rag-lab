import argparse
import os

import config
import job_queue
import logging_setup
from engines import ollama

import db

log = logging_setup.get_logger(__name__)


def main():
    logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="RAG over the corpus")
    parser.add_argument(
        "--index", action="store_true", help="queue an index_data job for the served variant"
    )
    parser.add_argument(
        "--ensure-index", action="store_true", help="build index only if empty"
    )
    parser.add_argument("--console", action="store_true", help="run console")
    parser.add_argument(
        "--pull-models", action="store_true", help="pull default models for llm"
    )

    args = parser.parse_args()

    # through the job, so the embedder check and the card handover are the ones every index passes
    if args.index:
        print(job_queue.enqueue("index_data", {}))

    if args.ensure_index:
        if db.is_empty(variant=config.settings.corpus.variant):
            print(job_queue.enqueue("index_data", {}))
        else:
            log.info("index.skip", reason="already_indexed")

    if args.console:
        import console

        console.start()

    if args.pull_models:
        ollama.ensure_models()


if __name__ == "__main__":
    main()
