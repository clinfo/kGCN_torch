#!/usr/bin/env python
# coding: utf-8

from ..logger import get_logger
import click


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--config", type=str, help="config json file")
@click.option("--save-config", default=None, help="save config json file")
@click.option("--no-config", is_flag=True, help="use default setting")
@click.option("--model", type=str, default="", help="model")
@click.option("--dataset", type=str, default="", help="dataset")
@click.option("--cv_path", type=str, default="cv", help="dataset")
@click.option("--fold", type=int, default=5, help="#fold")
@click.option("--seed", type=int, default=1234, help="seed")
@click.option("--prohibit_shuffle", is_flag=True, help="without shuffle")
@click.option("--without_config", is_flag=True, help="without config output")
@click.option("--without_train", is_flag=True, help="without train data output")
@click.option("--without_test", is_flag=True, help="without test data output")
@click.option("--use_info", is_flag=True, help="using cv_info to split data")
def main(**kwargs):
    logger = get_logger("tkgcn-cv-splitter")
