"""The OpenAI key is read from a local, git-ignored .env file."""
import os
import subprocess

import pytest

from llm_investors import env
from llm_investors.paths import ROOT


def test_parse_formats():
    text = '''# comment
OPENAI_API_KEY="sk-quoted"
export OTHER='single'
PLAIN=abc # trailing comment
EMPTY=

not a line
'''
    assert env.parse(text) == {"OPENAI_API_KEY": "sk-quoted", "OTHER": "single",
                               "PLAIN": "abc", "EMPTY": ""}


def test_load_env_sets_missing_and_never_overrides(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("OPENAI_API_KEY=sk-from-file\nX_TEST_VAR=1\n")
    f.chmod(0o600)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("X_TEST_VAR", "from-shell")
    loaded = env.load_env(f)
    assert loaded == ["OPENAI_API_KEY"]                    # names only, never values
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
    assert os.environ["X_TEST_VAR"] == "from-shell"         # an exported variable wins


def test_world_readable_env_file_warns(tmp_path, capsys, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("SOMETHING=1\n")
    f.chmod(0o644)
    monkeypatch.delenv("SOMETHING", raising=False)
    env.load_env(f)
    assert "chmod 600" in capsys.readouterr().err


def test_missing_key_fails_fast_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "ENV_FILE", tmp_path / ".env")          # no file there
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set.*--replay"):
        env.require_api_key()


def test_env_is_git_ignored():
    def ignored(p):
        return subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", p]).returncode == 0
    assert ignored(".env") and ignored(".env.local")
