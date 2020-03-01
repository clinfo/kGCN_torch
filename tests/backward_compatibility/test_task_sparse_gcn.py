import pytest
from click.testing import CliRunner

from kgcn_torch.backward_compatibility.task_sparse_gcn import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, ["train"])
    assert result.exit_code == 0
