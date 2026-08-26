import argparse
import os

import config
import llm
import logging_setup
import sources.factory
import use_cases.index

import db

log = logging_setup.get_logger(__name__)


def main():
    logging_setup.configure(os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="RAG over notes")
    parser.add_argument(
        "--index", action="store_true", help="reindex vault (resets the configured corpus variant, then builds it)"
    )
    parser.add_argument(
        "--ensure-index", action="store_true", help="build index only if empty"
    )
    parser.add_argument("--console", action="store_true", help="run console")
    parser.add_argument(
        "--pull-models", action="store_true", help="pull default models for llm"
    )

    args = parser.parse_args()

    if args.index:
        db.cleanup(variant=config.settings.corpus.variant)
        print(use_cases.index.collect_data(list(sources.factory.all_sources())))

    if args.ensure_index:
        if db.is_empty(variant=config.settings.corpus.variant):
            print(use_cases.index.collect_data(list(sources.factory.all_sources())))
        else:
            log.info("index.skip", reason="already_indexed")

    if args.console:
        import console

        console.start()

    if args.pull_models:
        llm.ensure_models()


if __name__ == "__main__":
    main()
