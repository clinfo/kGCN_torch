import pytest
from click.testing import CliRunner

import kgcn_torch.backward_compatibility.gcn as gcn


def test_main():
    runner = CliRunner()
    result = runner.invoke(gcn.main, ['train'])
    assert result.exit_code == 0


@pytest.mark.xfail()
def test_failed_case_no_positinal_arguments():
    runner = CliRunner()
    result = runner.invoke(gcn.main, [])
    assert result.exit_code == 0
