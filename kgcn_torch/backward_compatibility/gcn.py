import click

from ..utils.logger import get_logger


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("mode", type=str, nargs=1)
@click.option("--config", type=click.Path(exists=True), help="config json file")
@click.option("--save-config", type=click.Path(exists=True), help="save config json file")
@click.option("--retrain", type=str, default=None, help="retrain from checkpoint")
@click.option("--no-config", is_flag=True, help="config json file")
@click.option("--dataset", type=str, default=None, help="dataset")
@click.option(
    "--gpu", type=str, default=None, help="constraint gpus (default: all) (e.g. --gpu 0,2)",
)
@click.option("--cpu", type=bool, is_flag=True, help="cpu mode (calcuration only with cpu)")
@click.option("--bspmm", is_flag=True, help="bspmm")
@click.option("--bconv", is_flag=True, help="bconv")
@click.option("--batched", is_flag=True, help="batched")
@click.option("--profile", is_flag=True, help="")
@click.option("--skfold", is_flag=True, help="stratified k-fold")
@click.option("--param", type=str, default=None, help="parameter")
@click.option(
    "--ig_targets",
    default="all",
    type=click.Choice(["all", "profeat", "features", "adjs", "dragon", "embedded_layer"]),
    help="[deplicated (use ig_modal_target)]set scaling targets for Integrated Gradients",
)
@click.option(
    "--ig_modal_target",
    default="all",
    type=click.Choice(["all", "profeat", "features", "adjs", "dragon", "embedded_layer"]),
    help="set scaling targets for Integrated Gradients",
)
@click.option(
    "--ig_label_target",
    type=str,
    default="max",
    help="[visualization mode only] max/all/(label index)",
)
@click.option(
    "--visualize_type",
    default="graph",
    type=click.Choice(["graph", "node", "edge_loss", "edge_score"]),
    help="graph: visualize graph's property. node: create an integrated gradients map"
    " using target node. edge_loss: create an integrated gradients map"
    " using target edge and loss function. edge_score: create an integrated gradients map"
    " using target edge and score function.",
)
@click.option(
    "--visualize_target",
    type=int,
    default=None,
    help="set the target's number you want to visualize. from: [0, ~)",
)
@click.option(
    "--visualize_resample_num",
    type=int,
    default=None,
    help="resampling for visualization: [0, ~v)",
)
@click.option(
    "--visualize_method",
    default="ig",
    type=click.Choice(["ig", "grad", "grad_prod", "smooth_grad", "smooth_ig"]),
    help="visualization methods",
)
@click.option(
    "--graph_distance",
    type=int,
    default=1,
    help=(
        "set the distance from target node. An output graph is created within "
        "the distance from target node. :[1, ~)"
    ),
)
@click.option(
    "--verbose",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]),
    help="set log level. [default: DEBUG]",
    default="DEBUG",
)
@click.option(
    "--visualization_header", type=str, default=None, help="filename header of visualization",
)
def main(**kwargs):
    """GCN ::
    positional arguments:

    mode [train/infer/train_cv/visualize]
    """
    logger = get_logger("tkgcn")
    logger.debug(kwargs)
