"""published/logs/ mirrors logs/ (gzipped, deterministic) so a deployed viewer has the logs."""
import gzip

from llm_investors.sim import publish_logs as P


def test_publish_is_deterministic_and_mirrors(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "PUBLISHED", tmp_path / "published" / "logs")
    src = tmp_path / "logs" / "study2_v2_pilot"
    src.mkdir(parents=True)
    (src / "run_a.jsonl").write_text('{"type": "run_start"}\n')
    (src / "run_b.jsonl").write_text('{"type": "run_end"}\n')
    r = P.publish("study2_v2_pilot")
    out = tmp_path / "published" / "logs" / "study2_v2_pilot"
    assert r["written"] == 2 and gzip.decompress((out / "run_a.jsonl.gz").read_bytes()) \
        == b'{"type": "run_start"}\n'
    before = (out / "run_a.jsonl.gz").read_bytes()
    assert P.publish("study2_v2_pilot")["unchanged"] == 2              # no git diff
    assert (out / "run_a.jsonl.gz").read_bytes() == before
    (src / "run_b.jsonl").unlink()
    assert P.publish("study2_v2_pilot")["removed"] == 1 and not (out / "run_b.jsonl.gz").exists()
    assert not P.publishable("study1_dummy") and P.publishable("study1_v2")
