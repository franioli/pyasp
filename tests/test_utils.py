import subprocess

import pytest

from pyasp.utils.utils import (
    Command,
    cmd_list_to_string,
    cmd_string_to_list,
    run_command,
)


def test_cmd_list_to_string():
    assert cmd_list_to_string(["ls", "-l"]) == "ls -l"
    assert cmd_list_to_string(["echo", "hello", "world"]) == "echo hello world"
    with pytest.raises(ValueError):
        cmd_list_to_string([])
    with pytest.raises(TypeError):
        cmd_list_to_string("not a list")


def test_cmd_string_to_list():
    assert cmd_string_to_list("ls -l") == ["ls", "-l"]
    assert cmd_string_to_list("echo hello world") == ["echo", "hello", "world"]
    with pytest.raises(TypeError):
        cmd_string_to_list(123)


def test_run_command(mocker):
    mocker.patch("subprocess.run")
    subprocess.run.return_value = subprocess.CompletedProcess(
        args=["ls"], returncode=0, stdout="output", stderr=""
    )

    result = run_command(["ls"], verbose=True)
    assert result.stdout == "output"

    result = run_command("ls", silent=True)
    assert result is None

    with pytest.raises(TypeError):
        run_command(123)


def test_command_class(mocker):
    mocker.patch("subprocess.run")
    subprocess.run.return_value = subprocess.CompletedProcess(
        args=["ls"], returncode=0, stdout="output", stderr=""
    )

    cmd = Command("ls -l")
    assert str(cmd) == "Command ls -l"
    assert repr(cmd) == "Command(Command: ls -l)"

    cmd()
    cmd.run()

    cmd.extend("new_arg")
    assert str(cmd) == "Command ls -l new_arg"

    cmd.extend(["another_arg"], key="value")
    assert str(cmd) == "Command ls -l new_arg another_arg --key value"
