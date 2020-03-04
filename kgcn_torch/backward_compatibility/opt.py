#!/usr/bin/env python
# coding: utf-8
import click

from ..utils.logger import get_logger


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--config", type=str, help="config json file",
)
@click.option("--max_itr", type=int, default=3, help="maximum iteration")
@click.option("--opt_path", type=click.Path(), default="opt/", help="path")
@click.option("--domain", type=str, default=None, help="domain file")
@click.option("--gpu", type=str, default=None, help="[kgcn arg]")
@click.option("--cpu", is_flag=True, help="[kgcn arg]")
def main(**kwargs):
    """ kGCN opt entry point
    """
    logger = get_logger("tkgcn-opt")
    logger.debug(f"arguments {kwargs}")
