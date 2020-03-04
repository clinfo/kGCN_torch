#!/usr/bin/env python
# coding: utf-8
import click

from ..utils.logger import get_logger



@click.command(context_settings={"ignore_unknown_options": True})
@click.option("-l", "--label", default=None, type=str, help="help")
@click.option("--input_sparse_label", is_flag=True, default=False, help="help")
@click.option("--label_dim", default=None, type=int, help="help")
@click.option("-s", "--smarts", default=None, type=str, help="help")
@click.option(
    "--smiles", default=None, type=str,
)
@click.option("--sdf", default=None, type=str, help="help")
@click.option("--sdf_dir", default=None, type=str, help="help")
@click.option("--assay_dir", default=None, type=str, help="help")
@click.option("--assay_num_limit", default=None, type=int, help="help")
@click.option("--assay_pos_num_limit", default=None, type=int, help="help")
@click.option("--assay_neg_num_limit", default=None, type=int, help="help")
@click.option("--output_sparse_label", is_flag=True, default=False, help="help")
@click.option("-a", "--atom_num_limit", type=int, help="help")
@click.option("--no_header", is_flag=True, default=False, help="no header line in the label file")
@click.option("--without_mask", is_flag=True, default=False, help="without label mask")
@click.option("-o", "--output", default="dataset.jbl", type=str, help="help")
@click.option("--vector_modal", default=None, type=str, help="vector modal csv")
@click.option("--sdf_label", default=None, type=str, help="property name used as labels")
@click.option(
    "--sdf_label_active", default="Active", type=str, help="property name used as labels",
)
@click.option(
    "--sdf_label_inactive", default="Inactive", type=str, help="property name used as labels",
)
@click.option("--solubility", is_flag=True, default=False, help="solubilites in SDF as labels")
@click.option("--csv_reaxys", default=None, type=str, help="path to a csv containing reaxys data.")
@click.option("--multimodal", is_flag=True, default=False, help="help")
@click.option("--no_pseudo_negative", is_flag=True, default=False, help="help")
@click.option("--max_len_seq", type=int, default=None, help="help")
@click.option(
    "--generate_mfp", is_flag=True, default=False, help="generate Morgan Fingerprint using RDkit",
)
@click.option(
    "--use_sybyl", is_flag=True, default=False, help="[Additional features] SYBYL atom types",
)
@click.option(
    "--use_electronegativity",
    is_flag=True,
    default=False,
    help="[Additional features] electronegativity",
)
@click.option(
    "--use_gasteiger", is_flag=True, default=False, help="[Additional features] gasteiger charge",
)
@click.option(
    "--degree_dim", type=int, default=17, help="[Additional features] maximum number of degree",
)
@click.option(
    "--use_deepchem_feature", is_flag=True, default=False, help="75dim used in deepchem default",
)
@click.option(
    "--tfrecords", is_flag=True, default=False, help="output .tfrecords files instead of joblib.",
)
@click.option("--regression", is_flag=True, default=False, help="regression")
def main(**kwargs):
    """ GCN(chem entry point)
    """
    logger = get_logger("tkgcn-chem")
    logger.debug(kwargs)
