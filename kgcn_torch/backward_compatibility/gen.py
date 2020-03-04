#!/usr/bin/env python
# coding: utf-8
import click

from ..utils.logger import get_logger

@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("mode", type=str, nargs=1)
@click.option("--config", type=click.Path(exists=True), help="config json file")
@click.option("--save-config", type=click.Path(exists=True), help="config json file")
@click.option("--no-config", is_flag=True, help="config json file")
@click.option("--dataset", type=str, default=None, help="dataset")
@click.option(
    "--gpu", type=str, default=None, help="constraint gpus (default: all) (e.g. --gpu 0,2)",
)
@click.option("--cpu", is_flag=True, help="cpu mode (calcuration only with cpu)")
def main(**kwargs):
    """ kGCN gen entry point
    """
    logger = get_logger("tkgcn-gen")
    logger.debug(f"arguments: {kwargs}")
