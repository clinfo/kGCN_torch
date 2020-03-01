import pytest
from click.testing import CliRunner

from kgcn_torch.backward_compatibility.chem import main


def test_main():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
