"""Tests for the live multi-model blend.

The averaging is trivial; the guards are not. Filenames on disk are stable and
overwritten in place, so a failed fetch leaves the previous cycle's file looking
perfectly valid — a stale member would be folded into the mean and nothing
downstream would show it. These pin that it raises instead.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "blend_live", Path(__file__).resolve().parents[1] / "scripts" / "blend_live.py")
bl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bl)

REF = "2026-08-10T12:00:00Z"


def header(ref=REF, lead=72, nx=4, ny=2, param=2):
    return {"parameterNumber": param, "nx": nx, "ny": ny,
            "la1": 89.0, "lo1": 0.0, "la2": -89.0, "lo2": 358.0,
            "dx": 2.0, "dy": 2.0, "refTime": ref, "forecastTime": lead}


def record(u_vals, v_vals, ref=REF, lead=72, **kw):
    return [{"header": header(ref, lead, param=2, **kw), "data": list(u_vals)},
            {"header": header(ref, lead, param=3, **kw), "data": list(v_vals)}]


def write(tmp: Path, sid: str, lead: int, rec) -> None:
    (tmp / f"{sid}_latest_{lead:03d}.json").write_text(json.dumps(rec))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(bl, "OUT_DIR", tmp_path)
    return tmp_path


def test_mean_of_identical_members_is_that_member(workspace):
    rec = record([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [-1.0] * 8)
    write(workspace, "a_live", 72, rec)
    write(workspace, "b_live", 72, rec)
    assert bl.blend_lead(["a_live", "b_live"], 72) == REF
    out = json.loads((workspace / "live_blend_latest_072.json").read_text())
    assert out[0]["data"] == rec[0]["data"]
    assert out[1]["data"] == rec[1]["data"]


def test_mean_is_the_actual_mean(workspace):
    write(workspace, "a_live", 72, record([0.0, 10.0], [4.0, 4.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([2.0, 20.0], [0.0, 8.0], nx=2, ny=1))
    bl.blend_lead(["a_live", "b_live"], 72)
    out = json.loads((workspace / "live_blend_latest_072.json").read_text())
    assert out[0]["data"] == [1.0, 15.0]
    assert out[1]["data"] == [2.0, 6.0]


def test_stale_member_is_refused(workspace):
    """The failure this guard exists for: one fetch failed, its file survived."""
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([9.0, 9.0], [0.0, 0.0], nx=2, ny=1,
                                          ref="2026-08-09T00:00:00Z"))
    with pytest.raises(SystemExit, match="init time mismatch"):
        bl.blend_lead(["a_live", "b_live"], 72)


def test_grid_mismatch_is_refused(workspace):
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([1.0] * 4, [0.0] * 4, nx=4, ny=1))
    with pytest.raises(SystemExit, match="grid mismatch"):
        bl.blend_lead(["a_live", "b_live"], 72)


def test_single_member_does_not_produce_a_blend(workspace):
    """One model is not an ensemble; it must not be published as one."""
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    assert bl.blend_lead(["a_live", "b_live"], 72) is None
    assert not (workspace / "live_blend_latest_072.json").exists()


def test_blend_keeps_the_component_ordering(workspace):
    """u must stay u and v stay v -- swapping them would be silently plausible."""
    write(workspace, "a_live", 72, record([5.0, 5.0], [-5.0, -5.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([7.0, 7.0], [-7.0, -7.0], nx=2, ny=1))
    bl.blend_lead(["a_live", "b_live"], 72)
    out = json.loads((workspace / "live_blend_latest_072.json").read_text())
    assert out[0]["header"]["parameterNumber"] == 2 and out[0]["data"] == [6.0, 6.0]
    assert out[1]["header"]["parameterNumber"] == 3 and out[1]["data"] == [-6.0, -6.0]


# --- stale against the CLOCK, not against each other ------------------------
#
# Every guard above compares members to one another, and for twelve days that
# was not enough: the 6-hourly refresh workflow stopped firing, so each cycle
# re-blended the same untouched files. They agreed with each other perfectly and
# passed every check, because nothing compared them to the time of day.

def test_age_is_measured_from_the_init_time():
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    six_ago = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:00:00Z")
    age = bl.age_hours(six_ago)
    assert 5.0 < age < 7.0, age          # within the hour truncation


def test_a_future_init_is_negative_not_stale():
    """Clock skew must not read as staleness -- it is a different problem."""
    from datetime import datetime, timedelta, timezone
    ahead = (datetime.now(timezone.utc) + timedelta(hours=3)).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    assert bl.age_hours(ahead) < 0


# --- align_members: which members reach blend_lead at all -------------------
#
# blend_lead still refuses to average two cycles (above). What changed is who
# gets that far. The members do not share a refresh clock -- AIFS and GFS come
# from the 6-hourly workflow, WeatherNext from a credentialed export that only
# runs from a developer machine -- so treating "one member is behind" as fatal
# let the slowest member freeze the whole live blend. It is now excluded by
# name instead, and only the genuinely ambiguous cases still raise.

def test_a_member_on_an_older_init_is_excluded_not_fatal(workspace, capsys):
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([3.0, 3.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "c_live", 72, record([9.0, 9.0], [0.0, 0.0], nx=2, ny=1,
                                          ref="2026-08-09T00:00:00Z"))
    assert bl.align_members(["a_live", "b_live", "c_live"], [72]) == [
        "a_live", "b_live"]
    assert "EXCLUDED c_live" in capsys.readouterr().out


def test_the_survivors_still_blend(workspace):
    """Exclusion is only useful if the remaining members produce a blend."""
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([3.0, 3.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "c_live", 72, record([9.0, 9.0], [0.0, 0.0], nx=2, ny=1,
                                          ref="2026-08-09T00:00:00Z"))
    kept = bl.align_members(["a_live", "b_live", "c_live"], [72])
    assert bl.blend_lead(kept, 72) == REF
    out = json.loads((workspace / "live_blend_latest_072.json").read_text())
    assert out[0]["data"] == [2.0, 2.0]          # mean of 1 and 3, not of 1,3,9


def test_excluding_down_to_one_member_is_fatal(workspace):
    """Dropping laggards must not quietly turn the mean into a single model."""
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([9.0, 9.0], [0.0, 0.0], nx=2, ny=1,
                                          ref="2026-08-09T00:00:00Z"))
    with pytest.raises(SystemExit, match="only 1 member"):
        bl.align_members(["a_live", "b_live"], [72])


def test_a_member_disagreeing_with_itself_is_fatal(workspace):
    """Half-finished fetch: lead 72 moved to the new cycle, lead 96 did not.

    This is not staleness -- there is no single init to compare against -- so
    excluding it would be guessing which half is real.
    """
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "a_live", 96, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1,
                                          lead=96, ref="2026-08-09T00:00:00Z"))
    with pytest.raises(SystemExit, match="more than one init across its own"):
        bl.align_members(["a_live"], [72, 96])


def test_a_member_with_no_files_is_not_a_member(workspace, capsys):
    write(workspace, "a_live", 72, record([1.0, 1.0], [0.0, 0.0], nx=2, ny=1))
    write(workspace, "b_live", 72, record([3.0, 3.0], [0.0, 0.0], nx=2, ny=1))
    assert bl.align_members(["a_live", "b_live", "gone_live"], [72]) == [
        "a_live", "b_live"]
    assert "gone_live: no files on disk" in capsys.readouterr().out


def test_member_init_reads_the_shared_init(workspace):
    write(workspace, "a_live", 72, record([1.0], [0.0], nx=1, ny=1))
    write(workspace, "a_live", 96, record([1.0], [0.0], nx=1, ny=1, lead=96))
    assert bl.member_init("a_live", [72, 96, 120]) == REF
    assert bl.member_init("absent_live", [72]) is None
