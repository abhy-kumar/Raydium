"""Unit tests for Typer CLI commands."""

from typer.testing import CliRunner
from raydium.cli import app

runner = CliRunner()


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Raydium" in result.output
    assert "Bhadla" in result.output
    assert "Pavagada" in result.output


def test_cli_collect_simulate():
    result = runner.invoke(app, ["collect", "--resolution", "1.0", "--simulate", "--output", "test_sim_out.csv"])
    assert result.exit_code == 0
    assert "Successfully collected" in result.output
