"""``scripts/research_runs.py``: job construction, GPU queueing, process identity, completion checks.

No simulator, GPU or network: jobs in the queue test are trivial Python processes.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_runs.py"
spec = importlib.util.spec_from_file_location("research_runs", SCRIPT)
rr = importlib.util.module_from_spec(spec)
sys.modules["research_runs"] = rr
spec.loader.exec_module(rr)

OVERRIDES = [
    "agent.algorithm.learning_rate=0.0005",
    "agent.policy.actor.hidden_dims=[128,64]",
    "+env.x='a b'",
]


def _train(variant="temporal128", seed=1, **kw):
    return rr.train_job(task="go2w-train-dev-parkour", variant=variant, seed=seed, iterations=3000,
                        project="go2w_parkour", group="depth-ablation", **kw)  # fmt: skip


# -- job construction ---------------------------------------------------------------------------------


def test_train_job_naming_and_exact_overrides():
    job = _train(extra=OVERRIDES)
    assert job["name"] == "temporal128-seed1"
    argv = job["argv"]
    assert argv[-len(OVERRIDES) :] == OVERRIDES  # verbatim, in order
    pairs = dict(zip(argv, argv[1:], strict=False))
    assert pairs["--run_name"] == "temporal128-seed1" and pairs["--group"] == "depth-ablation"
    assert pairs["--job_type"] == "train" and pairs["--max_iterations"] == "3000" and pairs["--seed"] == "1"
    assert pairs["--log_project_name"] == "go2w_parkour" and pairs["--logger"] == "wandb"
    assert pairs["--device"] == "cuda:{gpu}" and "agent.device=cuda:{gpu}" in argv
    tags = argv[argv.index("--wandb_tags") + 1 : argv.index("agent.device=cuda:{gpu}")]
    assert tags == ["variant:temporal128", "seed:1", "task:go2w-train-dev-parkour"]
    # Substitution touches only the GPU token.
    placed = rr.substitute_gpu(argv, 3)
    assert placed[-len(OVERRIDES) :] == OVERRIDES and "--device" in placed and "cuda:3" in placed
    assert [a for a in placed if "{gpu}" in a] == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--max_iterations", "10"],
        ["--seed=4"],
        ["agent.max_iterations=10"],
        ["+agent.device=cuda:1"],
        ["--group", "x"],
    ],
)
def test_managed_settings_cannot_be_overridden_by_extras(extra):
    with pytest.raises(ValueError, match="set by this script"):
        _train(extra=extra)


def test_no_hardcoded_interpreter():
    assert _train()["argv"][0] == sys.executable
    assert _train(python="python")["argv"][0] == "python"
    source = SCRIPT.read_text()
    assert "/home/" not in source and "miniconda" not in source and "/envs/" not in source


def test_eval_jobs_matrix_cells_write_locally_only():
    jobs = rr.eval_jobs(task="go2w-eval-dev-parkour", checkpoint="logs/rsl_rl/x/r/model_2999.pt",
                        name="temporal128-seed1", seeds=[41, 42, 43], levels=[0, 4, 8],
                        output_root="logs/rsl_rl/x/r/eval/depth-ablation_model_2999",
                        num_envs=150, num_episodes=1)  # fmt: skip
    assert [j["name"] for j in jobs][:2] == ["temporal128-seed1__L0-seed41", "temporal128-seed1__L0-seed42"]
    assert len({j["name"] for j in jobs}) == 9
    pairs = dict(zip(jobs[4]["argv"], jobs[4]["argv"][1:], strict=False))
    assert pairs["--force_terrain_level"] == "4" and pairs["--note"] == "temporal128-seed1__L4-seed42"
    assert pairs["--eval_output_dir"].endswith("/L4-seed42") and pairs["--num_envs"] == "150"
    # One W&B run per cell is what flooded comparisons; cells never log to W&B.
    assert all("--logger" not in j["argv"] and "--group" not in j["argv"] for j in jobs)

    # Cells differ only in their seed, level and cell id: the protocol is identical by construction.
    def normalized(job):
        cell = job["name"].split("__")[1]
        out = []
        for previous, arg in zip([""] + job["argv"][:-1], job["argv"], strict=True):
            if previous in ("--seed", "--force_terrain_level"):
                out.append("<varies>")
            else:
                out.append(arg.replace(cell, "<cell>"))
        return out

    assert all(normalized(j) == normalized(jobs[0]) for j in jobs)
    mixed = rr.eval_jobs(task="t", checkpoint="c", name="n-seed1", seeds=[41, 42], output_root="o",
                         levels=[0, 9], difficulties=["hard"])  # fmt: skip
    assert [j["name"].split("__")[1] for j in mixed] == [
        "L0-seed41", "L0-seed42", "L9-seed41", "L9-seed42", "hard-seed41", "hard-seed42",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="<variant>-seed<seed>"):
        rr.eval_jobs(task="t", checkpoint="c", name="not-a-run", seeds=[41], output_root="o")


# -- scheduling -----------------------------------------------------------------------------------------


def _jobs(n, pins=None):
    pins = pins or {}
    return [{"name": f"j{i}", "argv": ["x", "{gpu}"], "gpu": pins.get(i)} for i in range(n)]


def test_more_jobs_than_gpus_queue_in_order():
    pending = _jobs(6)
    wave = rr.assign(pending, [0, 1, 2, 3])
    assert [(j["name"], g) for j, g in wave] == [("j0", 0), ("j1", 1), ("j2", 2), ("j3", 3)]
    rest = [j for j in pending if j not in [w for w, _ in wave]]
    assert [(j["name"], g) for j, g in rr.assign(rest, [2])] == [("j4", 2)]


def test_occupied_gpus_are_never_double_assigned():
    idle = rr.idle_gpus([0, 1, 2, 3], ours={0}, foreign={2}, allow_shared=False)
    assert idle == [1, 3]
    wave = rr.assign(_jobs(4), idle)
    assert sorted(g for _, g in wave) == [1, 3] and len({g for _, g in wave}) == len(wave)
    assert rr.idle_gpus([0, 1], ours={0}, foreign={1}, allow_shared=True) == [1]


def test_pinned_job_waits_for_its_gpu_while_others_proceed():
    wave = rr.assign(_jobs(3, pins={0: 2}), [0, 1])
    assert [(j["name"], g) for j, g in wave] == [("j1", 0), ("j2", 1)]


def test_foreign_gpu_use_parsing():
    outputs = {"--query-gpu=index,uuid": "0, GPU-a\n1, GPU-b\n", "--query-compute-apps=gpu_uuid": "GPU-b\n"}

    def fake_run(cmd, **_):
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs[cmd[-1]])

    assert rr.foreign_gpu_use(fake_run) == {1}

    def missing(cmd, **_):
        raise FileNotFoundError

    assert rr.foreign_gpu_use(missing) is None


def test_load_jobs_rejects_unsafe_input(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text(json.dumps({"name": "a", "argv": ["python", "train.py"]}) + "\n")
    with pytest.raises(ValueError, match="assigned GPU"):
        rr.load_jobs(path)
    line = json.dumps({"name": "a", "argv": ["x", "cuda:{gpu}"]})
    path.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        rr.load_jobs(path)


def test_queue_runs_all_jobs_with_one_job_per_gpu(tmp_path):
    jobs_file = tmp_path / "jobs.jsonl"
    code = "import sys,time; print('gpu', sys.argv[1]); time.sleep(0.3)"
    jobs = [{"name": f"j{i}", "argv": [sys.executable, "-c", code, "{gpu}"], "gpu": None} for i in range(3)]
    jobs_file.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    log_dir = tmp_path / "queue"
    assert rr.run_queue(rr.load_jobs(jobs_file), [5, 7], log_dir, poll=0.05, occupancy=lambda: set()) == 0
    events = rr.read_events(log_dir)
    starts = [e for e in events if e["event"] == "start"]
    assert [e["name"] for e in starts] == ["j0", "j1", "j2"] and {e["gpu"] for e in starts} == {5, 7}
    # No GPU ever ran two jobs: replay the event log.
    busy: set[int] = set()
    for e in events:
        if e["event"] == "start":
            assert e["gpu"] not in busy
            busy.add(e["gpu"])
        elif e["event"] == "exit":
            busy.discard(e["gpu"])
    assert (log_dir / "j2.log").read_text().startswith("gpu ")
    with pytest.raises(SystemExit, match="already launched"):
        rr.run_queue(rr.load_jobs(jobs_file), [5], log_dir, poll=0.05, occupancy=lambda: set())


# -- process identity -------------------------------------------------------------------------------------


def test_identity_gates_signals(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        recorded = rr.identity(child.pid)
        assert rr.matches(recorded)
        for key, value in (("start_ticks", recorded["start_ticks"] + 1), ("cmdline", ["other"]), ("pid", 1)):
            forged = {**recorded, key: value}
            assert not rr.matches(forged)
            assert "not signalled" in rr.stop_process(forged, timeout=1)
        assert child.poll() is None  # forged identities never touched it
        assert rr.stop_process(recorded, timeout=5) == "stopped"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
    assert not rr.matches(recorded)
    assert rr.identity(child.pid) is None or not rr.matches(recorded)


# -- completion checks ------------------------------------------------------------------------------------


def _run_dir(tmp_path, iterations=3, records=None, checkpoint=True):
    run = tmp_path / "logs" / "2026-09-30_10-00-00_temporal128-seed1"
    (run / "params").mkdir(parents=True)
    (run / "params" / "agent.yaml").write_text("{}")
    if checkpoint:
        import torch

        torch.save({"model_state_dict": {"weight": torch.ones(2)}}, run / f"model_{iterations}.pt")
    if records is not None:
        (run / "locomotion_metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run


def _records(n):
    return [{"iteration": i, "Policy/std_mean": 0.5, "Checks/finite_rewards_std": 1.0} for i in range(n)]


def test_check_train_complete_run(tmp_path):
    run = _run_dir(tmp_path, records=_records(3))
    verdict = rr.check_train(run, 3)
    assert verdict["complete"] and not verdict["unverified"]
    assert rr.find_run_dir(tmp_path / "logs", "temporal128-seed1") == (run, [])


def test_check_train_missing_checkpoint(tmp_path):
    verdict = rr.check_train(_run_dir(tmp_path, records=_records(3), checkpoint=False), 3)
    assert not verdict["complete"] and "model_3.pt missing" in verdict["problems"][0]


def test_check_train_nonfinite_or_short_metric_stream(tmp_path):
    records = _records(3)
    records[1]["Policy/std_mean"] = float("nan")
    verdict = rr.check_train(_run_dir(tmp_path, records=records), 3)
    assert not verdict["complete"] and any("non-finite" in p for p in verdict["problems"])
    short = rr.check_train(_run_dir(tmp_path / "b", records=_records(2)), 3)
    assert any("2 records, expected 3" in p for p in short["problems"])


def test_check_train_exit_zero_with_traceback_is_not_complete(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("...\nTraceback (most recent call last):\n  RuntimeError: boom\n")
    verdict = rr.check_train(_run_dir(tmp_path, records=_records(3)), 3, log=log)
    assert not verdict["complete"] and "traceback" in verdict["problems"][0]


def test_check_train_without_local_stream_is_unverified(tmp_path):
    verdict = rr.check_train(_run_dir(tmp_path), 3)
    assert verdict["complete"] and verdict["unverified"]


def _cell(root, seed, level, n=6, forward=False, **run):
    cell = root / rr.cell_id(seed, level)
    cell.mkdir(parents=True)
    rows = [{"episode_id": 0, "env_id": i, **({"completed": True} if forward else {})} for i in range(n)]
    if (run.get("research") or {}).get("protocol_contract"):
        for row in rows:
            row.update(
                lin_vel_tracking_error=0.2,
                ang_vel_tracking_error=0.3,
                success=True,
                fell=False,
                timed_out=True,
            )
    if (run.get("research") or {}).get("protocol_contract"):
        for row in rows:
            row.update({m: 0.2 for m in rr.required_eval_metrics(run["research"]["protocol_contract"])})
    (cell / "evaluation_results.json").write_text(json.dumps(rows))
    info = {"seed": seed, "forced_terrain_level": level, "checkpoint": "ckpt.pt", "num_envs": n,
            "num_episodes_requested_per_env": 1, "task": "t", "git": {"commit": "abc"}, **run}  # fmt: skip
    (cell / "evaluation_run.json").write_text(json.dumps(info))
    metrics = {"success_rate": 0.5}
    contract = (run.get("research") or {}).get("protocol_contract")
    if contract:
        metrics.update({m: 0.2 for m in rr.required_eval_metrics(contract)})
    (cell / "evaluation_summary.json").write_text(json.dumps({"run": info, "metrics": metrics}))


def test_check_eval_reports_missing_cells_and_counts(tmp_path):
    root = tmp_path / "a"
    _cell(root, 41, 0)
    _cell(root, 42, 0, n=5)
    verdict = rr.check_eval([root], seeds=[41, 42, 43], levels=[0], expected_episodes=6, metrics=["success_rate"])
    status = {c["cell"]: c["problems"] for c in verdict["cells"]}
    assert status["L0-seed41"] == [] and "5 episodes, expected 6" in status["L0-seed42"]
    assert status["L0-seed43"] == ["missing evaluation artifacts"] and not verdict["complete"]
    first = rr.check_eval([root], seeds=[41, 42], levels=[0], expected_first_episodes=6)
    assert [c["first_episodes"] for c in first["cells"]] == [6, 5] and not first["complete"]


def test_check_eval_flags_undefined_metrics_and_profile_mismatch(tmp_path):
    _cell(tmp_path / "a", 41, 0)
    _cell(tmp_path / "b", 41, 0, forward=True)
    verdict = rr.check_eval([tmp_path / "a", tmp_path / "b"], seeds=[41], levels=[0], metrics=["survival_rate"])
    assert not verdict["comparable"]  # general vs forward-course: survival_rate means different things
    assert {p["profile"] for p in verdict["protocols"]} == {"general", "forward_course"}
    assert all("metric survival_rate is None" in c["problems"] for c in verdict["cells"])


def test_cli_dry_run_and_job_lines(tmp_path, capsys):
    assert rr.main(["train-job", "--task", "t", "--variant", "v", "--seed", "1", "--iterations", "5",
                    "--project", "p", "--group", "g", "--", *OVERRIDES]) == 0  # fmt: skip
    job = json.loads(capsys.readouterr().out)
    assert job["argv"][-len(OVERRIDES) :] == OVERRIDES
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("".join(json.dumps({**job, "name": f"v-seed{i}"}) + "\n" for i in range(3)))
    queue = [
        "queue",
        "--jobs",
        str(jobs),
        "--gpus",
        "0",
        "1",
        "--log-dir",
        str(tmp_path / "q"),
        "--group",
        "g",
    ]
    assert rr.main([*queue, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "gpu 0: v-seed0" in out and "gpu 1: v-seed1" in out and "1 more job(s) queue" in out
    assert "launch checks:" in out and "dev/agent/g" in out  # this checkout is not the agent branch
    assert not (tmp_path / "q").exists()  # a dry run launches and records nothing


# -- W&B structure: projects, job types, tags, publication ------------------------------------------------


def test_fallback_project_is_refused_and_job_types():
    with pytest.raises(ValueError, match="experiment project"):
        rr.train_job(task="t", variant="v", seed=1, iterations=5, project="isaaclab", group="g")
    assert rr.check_project("isaaclab", legacy=True) == "isaaclab"  # publishing/tagging old comparisons only
    with pytest.raises(ValueError, match="experiment project"):
        rr.publish_eval({"group": "g", "name": "v-seed1__eval"}, "isaaclab", wandb_module=FakeWandb([]))
    smoke = _train(smoke=True, dirty=True)["argv"]
    assert smoke[smoke.index("--job_type") + 1] == "smoke" and "dirty" in smoke


@pytest.mark.parametrize("tag", ["variant:temporal128-p0.1", "seed:3", "ckpt:model_1499", "status:final",
                                 "decision:chosen", "dirty", "replay"])  # fmt: skip
def test_tag_vocabulary_accepts(tag):
    assert rr.check_tag(tag) == tag


@pytest.mark.parametrize("tag", ["temporal128", "colour:red", "status:done", "decision:winner", "variant:", "x" * 65,
                                 "protocol:a+b"])  # fmt: skip
def test_tag_vocabulary_rejects(tag):
    with pytest.raises(ValueError):
        rr.check_tag(tag)


def test_protocol_slug_and_run_name_parsing():
    assert rr.protocol_slug([41, 42, 43], [0, 5, 9], ["easy", "hard"], 260, 1) == "L0-5-9_easy-hard_s41-42-43_e260x1"
    assert rr.parse_run_name("temporal128-p0.1-seed12") == ("temporal128-p0.1", 12)
    assert rr.check_tag("protocol:" + rr.protocol_slug(list(range(40, 60)), list(range(10)), [], 4096, 10))


def _rows_for(n, failures=0):
    rows = [{"episode_id": 0, "env_id": i, "fell": i < failures} for i in range(n)]
    return rows + [{"episode_id": 1, "env_id": 0, "fell": True}]  # replacement episode: excluded by default


def _aggregate(rows):
    return {
        "num_episodes": len(rows),
        "fall_rate": sum(r["fell"] for r in rows) / len(rows),
        "undefined": float("nan"),
    }


def test_eval_publication_pools_first_episodes_with_denominators(tmp_path):
    run_dir = tmp_path / "2026-09-30_17-59-41_lstm-seed1"
    (run_dir / "videos" / "play").mkdir(parents=True)
    (run_dir / "videos" / "play" / "lstm-seed1__L0-seed41-step-0.mp4").write_bytes(b"")
    cells = [{"cell": rr.cell_id(s, lv), "level": lv, "difficulty": None, "seed": s, "num_envs": 4, "num_episodes": 1,
              "commit": "abc", "dirty": False} for lv in (0, 9) for s in (41, 42)]  # fmt: skip
    rows = {c["cell"]: _rows_for(4, failures=2 if c["level"] == 9 else 0) for c in cells}
    pub = rr.build_eval_publication(
        group="history-lstm", name="lstm-seed1", task="go1-train-x", eval_task="go1-eval-x",
        checkpoint=str(run_dir / "model_1499.pt"), output_root=str(run_dir / "eval"), cells=cells,
        rows_by_cell=rows, aggregate=_aggregate, levels=[0, 9],
    )  # fmt: skip
    assert pub["name"] == "lstm-seed1__eval" and pub["job_type"] == "eval"
    assert pub["tags"] == [
        "variant:lstm",
        "seed:1",
        "task:go1-train-x",
        "ckpt:model_1499",
        "protocol:L0-9_s41-42_e4x1",
    ]
    assert pub["summary"]["eval/episodes"] == 16 and pub["summary"]["eval/fall_rate"] == 0.25
    assert pub["summary"]["eval/L9/episodes"] == 8 and pub["summary"]["eval/L9/fall_rate"] == 0.5
    assert "eval/undefined" not in pub["summary"]  # non-finite metrics are not published as numbers
    assert len(pub["table"]) == 4 and pub["table"][0]["episodes"] == 4
    assert list(pub["videos"]) == ["video/L0-seed41"]
    research = pub["config"]["research"]
    assert research["protocol"]["population"] == "first" and research["variant"] == "lstm"
    every = rr.build_eval_publication(group="g", name="lstm-seed1", task="t", eval_task="e", checkpoint="c.pt",
                                      output_root="o", cells=cells, rows_by_cell=rows, aggregate=_aggregate,
                                      levels=[0, 9], population="all")  # fmt: skip
    assert every["summary"]["eval/episodes"] == 20
    dirty = [{**c, "dirty": True} for c in cells]
    assert "dirty" in rr.build_eval_publication(group="g", name="lstm-seed1", task="t", eval_task="e", checkpoint="c.pt",
                                                output_root="o", cells=dirty, rows_by_cell=rows,
                                                aggregate=_aggregate)["tags"]  # fmt: skip


class FakeRun:
    def __init__(self, name, job_type, tags=(), run_id=None, summary=None, config=None, **attrs):
        self.name, self.job_type, self.tags, self.id = name, job_type, list(tags), run_id or name
        self.summary, self.config, self.updated = dict(summary or {}), dict(config or {}), 0
        self.__dict__.update(attrs)

    def update(self):
        self.updated += 1


class FakeArtifact:
    def __init__(self, name, type):
        self.name, self.type, self.entries, self.files = name, type, {}, []

    def add(self, obj, name):
        self.entries[name] = obj

    def add_file(self, path, name):
        self.files.append((path, name))


class FakeWandb:
    def __init__(self, runs):
        self.runs, self.inits, self.logged, self.artifacts = runs, [], [], []
        self.logged_run = FakeRun("g__analysis", "analysis", group="g")
        outer = self

        class Api:
            default_entity = "team"

            def runs(self, path, filters):
                outer.query = (path, filters)
                return outer.runs

            def run(self, path):
                outer.run_path = path
                return outer.logged_run

        class Run:
            url, id = "https://wandb/run", "newrun"

            def __init__(self):
                self.summary = {}

            def log(self, data):
                outer.logged.append(data)

            def log_artifact(self, artifact):
                outer.artifacts.append(artifact)

            def finish(self):
                pass

        self.Api, self._run = Api, Run()

    def init(self, **kw):
        self.inits.append(kw)
        return self._run

    def Table(self, columns, data):
        return {"columns": columns, "data": data}

    def Video(self, path, format, caption):
        return {"video": path, "caption": caption}

    def Image(self, figure):
        return {"image": figure}

    def Artifact(self, name, type):
        return FakeArtifact(name, type)


def _publication():
    return {"name": "lstm-seed1__eval", "job_type": "eval", "group": "g", "tags": ["variant:lstm"],
            "config": {"research": {"train_run_id": None, "output_root": "o"}}, "summary": {"eval/episodes": 4},
            "table": [{"cell": "L0-seed41", "episodes": 4}], "videos": {"video/L0-seed41": "v.mp4"},
            "episodes": [{"cell": "L0-seed41", "episode_id": 0, "fell": False, "contacts": [1, 0]}]}  # fmt: skip


def test_publish_eval_links_training_run_uploads_episodes_and_refuses_duplicates():
    train = FakeRun("2026-09-30_17-59-41_lstm-seed1", "train", run_id="tr1")
    fake = FakeWandb([train, FakeRun("2026-09-30_17-59-41_lstm-seed2", "train")])
    rr.publish_eval(_publication(), "rwm-explore", wandb_module=fake)
    init = fake.inits[0]
    assert init["config"]["research"]["train_run_id"] == "tr1" and init["job_type"] == "eval"
    assert init["name"] == "lstm-seed1__eval" and init["group"] == "g" and init["project"] == "rwm-explore"
    assert fake.logged[0]["video/L0-seed41"]["caption"] == "scored"
    episodes = fake.artifacts[0]
    assert (episodes.name, episodes.type) == ("eval-episodes-lstm-seed1", "eval-episodes")
    table = episodes.entries["episodes"]
    assert table["columns"] == ["cell", "episode_id", "fell", "contacts"] and table["data"][0][3] == "[1, 0]"
    published = FakeRun("lstm-seed1__eval", "eval", tags=["variant:lstm"])
    again = FakeWandb([train, published])
    with pytest.raises(SystemExit, match="already published"):
        rr.publish_eval(_publication(), "rwm-explore", wandb_module=again)
    rr.publish_eval(_publication(), "rwm-explore", supersede=True, wandb_module=again)
    assert "status:superseded" in published.tags and published.updated == 1
    with pytest.raises(SystemExit, match="pass --train-run-id"):
        rr.publish_eval(_publication(), "rwm-explore", wandb_module=FakeWandb([]))


# -- W&B Reports ----------------------------------------------------------------------------------------------

GOOD_PLOTS = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

CWD = []


def make_figures(api, project, group, entity):
    CWD.append(os.getcwd())
    runs = api.runs(f"{entity}/{project}", filters={"group": group, "jobType": "eval"})
    fig, ax = plt.subplots()
    ax.bar(range(len(runs)), [1.0] * len(runs))
    return {"seed_spread": fig, "where": os.getcwd()}
"""


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import pandas as pd\ndef make_figures(api, project, group, entity):\n    return pd.read_parquet('x')",
         "read_parquet"),
        ("def make_figures(api, project, group, entity):\n    return open('a').read()", "open"),
        ("def make_figures(api, project, group, entity):\n    p = 'logs/rsl_rl/x'\n    return {}", "local data path"),
        ("import mbrl\ndef make_figures(api, project, group, entity):\n    return {}", "import mbrl"),
        ("def figures():\n    return {}", "make_figures"),
        ("def make_figures(api, project, group, entity):\n    return api.artifact('a').download()", "download"),
    ],
)  # fmt: skip
def test_plot_scripts_must_read_through_the_wandb_api(source, expected):
    assert any(expected in problem for problem in rr.plot_source_problems(source))


def test_plot_script_runs_in_an_empty_directory(tmp_path):
    script = tmp_path / "plots.py"
    script.write_text(GOOD_PLOTS)
    assert rr.plot_source_problems(GOOD_PLOTS) == []
    with pytest.raises(SystemExit, match="make_figures must return"):  # "where" is a string, not a figure name map?
        rr.run_plot_script(script, FakeWandb([FakeRun("a__eval", "eval")]).Api(), project="p", group="g", entity="e")
    script.write_text(GOOD_PLOTS.replace(', "where": os.getcwd()', ""))
    figures = rr.run_plot_script(script, FakeWandb([]).Api(), project="p", group="g", entity="e")
    assert list(figures) == ["seed_spread"] and hasattr(figures["seed_spread"], "savefig")
    bad = tmp_path / "bad.py"
    bad.write_text("def make_figures(api, project, group, entity):\n    return open('x')")
    with pytest.raises(SystemExit, match="plot script rejected"):
        rr.run_plot_script(bad, None, project="p", group="g", entity="e")


def _report_runs():
    research = {"train_cfg": {"policy": {"actor": {"class_name": "Actor", "lstm_hidden_dim": 64}}, "research": {"seed": 1, "task": "rl-fixture", "protocol_id": "p-fixture", "protocol_contract": {"evaluation": {"dataset": "rl-fixture", "levels": [0]}}, "variant": "lstm", "launch_tag": "agent/g/train-1", "commit": "c0ffee" * 6, "branch": "dev/agent/g"}}}  # fmt: skip
    return [
        FakeRun("2026_lstm-seed1", "train", ["variant:lstm"], config=research, state="finished", history=lambda **kw: [{"_step": 0, "Train/mean_reward": 1.0, "m": 2.0}]),
        FakeRun("2026_lstm-seed0", "smoke", ["variant:lstm"]),
        FakeRun("2026_old-seed1", "train", ["status:superseded"]),
        FakeRun("lstm-seed1__eval", "eval", ["variant:lstm"], config={"research": {"variant": "lstm", "seed": 1, "task": "rl-fixture", "protocol_id": "p-fixture", "train_run_id": "2026_lstm-seed1", "protocol_contract": {"domain": "rl", "evaluation": {"dataset": "rl-fixture", "levels": [0]}}, "commits": ["c0ffee" * 6]}}, state="finished",
                summary={"eval/success_rate": 0.9, "eval/L0/success_rate": 1.0, "eval/episodes": 10,
                         "video/L0-seed41": {"_type": "video-file"}, "eval/flag": True}),
        FakeRun("g__analysis", "analysis", run_id="an1", state="finished",
                summary={"analysis/fig/robust_vs_baseline": {}, "analysis/table/per_seed": {}, "analysis/draft_md": "x"}),
        FakeRun("g__analysis", "analysis", run_id="an0", state="crashed", summary={"analysis/fig/stale": {}}),
        FakeRun("g__analysis", "analysis", ["status:superseded"], run_id="an-1", summary={"analysis/fig/old": {}}),
    ]  # fmt: skip


def test_report_inventory_uses_current_train_and_eval_runs():
    inventory = rr.report_inventory(_report_runs())
    assert inventory["train"] == ["2026_lstm-seed1"] and inventory["eval"] == ["lstm-seed1__eval"]
    assert inventory["eval_keys"] == ["eval/L0/success_rate", "eval/episodes", "eval/success_rate"]
    assert inventory["videos"] == ["video/L0-seed41"] and inventory["launch_tags"] == ["agent/g/train-1"]
    assert inventory["branches"] == ["dev/agent/g"]
    assert inventory["analysis"] == {
        "id": "an1",
        "name": "g__analysis",
        "figures": ["analysis/fig/robust_vs_baseline"],
        "tables": ["analysis/table/per_seed"],
    }  # finished and current only


def test_report_blocks_sections_panels_and_preserved_human_text():
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    inventory = rr.report_inventory(_report_runs())
    kept = [wr.MarkdownBlock(text="LSTM wins at L9 but costs 2x."), wr.P(text="edited in W&B")]
    blocks = rr.build_report_blocks(
        wr, entity="team", project="rwm-explore", group="g", inventory=inventory, intro="Hypothesis: …",
        train_metrics=["Train/mean_reward"], eval_metrics=["success_rate"], report_run_id="rep1",
        custom_keys=["custom/seed_spread"], preserved={"Analysis": kept}, analysis="ignored draft",
        plot_artifact="report-code-g",
    )  # fmt: skip
    headings = [rr._heading(b) for b in blocks if type(b).__name__ == "H1"]
    assert headings == list(rr.SECTIONS)
    human = rr.human_sections(blocks)
    assert human["Analysis"] == kept  # the researcher's blocks, not the draft
    assert [b.text for b in human["Decision"]] == ["_Pending the researcher's decision._"]
    grids = [b for b in blocks if type(b).__name__ == "PanelGrid"]
    training = next(g for g in grids if isinstance(g.panels[0], wr.LinePlot))
    assert training.runsets[0].groupby == ["config.train_cfg.research.variant"]
    assert training.panels[0].groupby_rangefunc == "stderr"
    assert all(
        (p.point_visualization_method, p.smoothing_type, p.smoothing_factor)
        == ("sampling", "exponentialTimeWeighted", 0.95)
        for p in training.panels
    )
    evaluation = next(g for g in grids if isinstance(g.panels[0], wr.BarPlot))
    assert evaluation.runsets[0].groupby == ["config.research.variant"]
    assert [p.metrics for g in grids for p in g.panels if isinstance(p, wr.BarPlot)] == [
        ["eval/success_rate"],
        ["eval/L0/success_rate"],
    ]
    assert any(
        isinstance(p, wr.MediaBrowser) and p.media_keys == ["custom/seed_spread"] for g in grids for p in g.panels
    )
    kept = [g for g in grids if g.runsets[0].name == "Analysis run"]
    assert [type(p).__name__ for g in kept for p in g.panels] == ["MediaBrowser", "WeavePanelSummaryTable"]
    assert kept[0].panels[0].media_keys == ["analysis/fig/robust_vs_baseline"]
    assert kept[1].panels[0].table_name == "analysis/table/per_seed"
    assert "g__analysis" in [b for b in blocks if "Provenance" not in str(b) and "Analysis run" not in str(b)][-1].text
    decided = rr.build_report_blocks(
        wr, entity="team", project="p", group="g", inventory=inventory, intro="", train_metrics=[], eval_metrics=[],
        report_run_id=None, custom_keys=[], preserved={"Decision": [wr.P(text="old")]}, decision="Keep lstm.",
    )  # fmt: skip
    assert rr.human_sections(decided)["Decision"][0].text == "Keep lstm."
    # Grouping is resolved against run metadata before the pure renderer receives its plan.
    from research_workflow.metric_resolution import MetricResolver

    legacy_keys = [
        "config.train_cfg.policy.actor.class_name",
        "config.train_cfg.policy.actor.lstm_hidden_dim",
    ]
    legacy_runs = [r for r in _report_runs() if r.job_type == "train"]
    resolver = MetricResolver(legacy_runs, train_metrics=["m"], groupby=legacy_keys)
    assert resolver.resolve()["panels"][0]["variant_keys"] == legacy_keys
    assert "Custom plots" not in [rr._heading(b) for b in decided]
    bare = rr.build_report_blocks(
        wr, entity="team", project="p", group="g", inventory={**inventory, "analysis": None}, intro="",
        train_metrics=[], eval_metrics=[], report_run_id=None, custom_keys=[], preserved={},
    )  # fmt: skip
    assert "Analysis figures" not in [rr._heading(b) for b in bare]


class FakeReport:
    loaded = {}

    def __init__(self, project=None, entity=None, blocks=None, **kw):
        self.project, self.entity, self.blocks, self.saved, self.url = (
            project,
            entity,
            blocks or [],
            0,
            "https://wandb/report/1",
        )

    @classmethod
    def from_url(cls, url):
        return cls.loaded[url]

    def save(self):
        self.saved += 1


def test_publish_report_refreshes_in_place_and_keeps_human_sections(tmp_path):
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    fake_wr = type("WR", (), {name: getattr(wr, name) for name in dir(wr) if not name.startswith("_")})
    fake_wr.Report = FakeReport
    old = FakeReport(blocks=[wr.H1(text="Training"), wr.P(text="generated"), wr.H1(text="Analysis"),
                             wr.MarkdownBlock(text="researcher text")])  # fmt: skip
    FakeReport.loaded = {"https://wandb/report/1": old}
    previous = FakeRun("g__report", "report", summary={"report_url": "https://wandb/report/1"})
    fake = FakeWandb([*_report_runs(), previous])
    script = tmp_path / "plots.py"
    script.write_text(GOOD_PLOTS.replace(', "where": os.getcwd()', ""))
    url = rr.publish_report("rwm-explore", "g", intro="Hypothesis", analysis="draft", plots=script, evidence_only="Legacy regression fixture",
                            wandb_module=fake, wr_module=fake_wr)  # fmt: skip
    assert url == "https://wandb/report/1" and old.saved == 1  # same report, saved in place
    assert rr.human_sections(old.blocks)["Analysis"][0].text == "researcher text"
    assert fake.inits[0]["job_type"] == "report" and fake.inits[0]["name"] == "g__report"
    assert set(fake.logged[0]) == {"custom/seed_spread"}
    code = fake.artifacts[0]
    assert (code.name, code.type) == ("report-code-g", "code") and code.files == [(str(script), "plots.py")]
    assert fake._run.summary["report_url"] == url and "status:superseded" in previous.tags
    with pytest.raises(SystemExit, match="no training runs"):
        rr.publish_report("rwm-explore", "g", wandb_module=FakeWandb([]), wr_module=fake_wr)


# -- Git: one agent branch per comparison -----------------------------------------------------------------------


def _git_repo(tmp_path):
    repo = tmp_path / "isaac_wm"
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "README.md").write_text("x\n")
    subprocess.run([*git, "add", "README.md"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return repo, git


def test_launch_requires_clean_agent_branch():
    clean = {"branch": "dev/agent/history-lstm", "dirty": False, "commit": "abc"}
    assert rr.launch_problems("history-lstm", clean) == []
    assert "dev/agent/history-lstm" in rr.launch_problems("history-lstm", {**clean, "branch": "main"})[0]
    assert "not clean" in rr.launch_problems("history-lstm", {**clean, "dirty": True})[0]
    jobs = [_train(), rr.train_job(task="t", variant="v", seed=1, iterations=5, project="p", group="other")]
    assert rr.job_groups(jobs) == {"depth-ablation", "other"}


def test_worktree_branch_shared_logs_and_launch_tags(tmp_path):
    repo, git = _git_repo(tmp_path)
    tree = rr.create_worktree("history-lstm", root=repo)
    assert tree == tmp_path / "isaac_wm-agent-history-lstm"
    assert rr.git_source(tree) == {"commit": rr.git_source(repo)["commit"], "branch": "dev/agent/history-lstm",
                                   "dirty": False}  # fmt: skip
    assert (tree / "logs").is_symlink() and (tree / "logs").resolve() == (repo / "logs").resolve()
    with pytest.raises(SystemExit, match="already exists"):
        rr.create_worktree("history-lstm", root=repo)
    assert rr.pin_launch("history-lstm", "train", push=False, root=tree) == "agent/history-lstm/train-1"
    assert rr.pin_launch("history-lstm", "train", push=False, root=tree) == "agent/history-lstm/train-2"
    assert rr.pin_launch("history-lstm", "eval", push=False, root=tree) == "agent/history-lstm/eval-1"
    follow_up = rr.create_worktree("history-lstm-2", base="dev/agent/history-lstm", root=repo)
    assert rr.git_source(follow_up)["branch"] == "dev/agent/history-lstm-2"


def test_jobs_import_this_checkouts_source(tmp_path):
    env = rr.source_env(rr.ROOT, base={"PYTHONPATH": "/elsewhere"})
    assert env["PYTHONPATH"].split(":")[0] == str(rr.ROOT / "src") and env["PYTHONPATH"].endswith("/elsewhere")
    assert rr.check_imports([sys.executable], env) == []
    other = tmp_path / "checkout"
    (other / "src").mkdir(parents=True)
    problems = rr.check_imports([sys.executable], rr.source_env(other, base={}), root=other)
    assert problems and "not this checkout" in problems[0]


def test_retag_only_lifecycle_tags_on_selected_runs():
    runs = [FakeRun("2026_lstm-seed1", "train", ["variant:lstm"]), FakeRun("lstm-seed1__eval", "eval", ["variant:lstm"]),
            FakeRun("2026_mlp-seed1", "train", ["variant:mlp"]), FakeRun("g__report", "report")]  # fmt: skip
    chosen = rr.select_runs(runs, variants=["lstm"])
    assert [r.name for r in chosen] == ["2026_lstm-seed1", "lstm-seed1__eval"]
    assert [r.name for r in rr.select_runs(runs, names=["mlp-seed1"])] == ["2026_mlp-seed1"]
    preview = rr.retag(chosen, ["decision:chosen"], [], dry_run=True)
    assert len(preview) == 2 and all(r.updated == 0 for r in chosen)
    rr.retag(chosen, ["decision:chosen"], [])
    assert all("decision:chosen" in r.tags and r.updated == 1 for r in chosen)
    with pytest.raises(ValueError, match="only"):
        rr.retag(chosen, ["variant:other"], [])
    with pytest.raises(ValueError, match="only"):
        rr.retag(chosen, [], ["variant:lstm"])


# -- Analysis notebooks ---------------------------------------------------------------------------------------

nbformat = pytest.importorskip("nbformat")
TEMPLATE = Path(__file__).resolve().parents[1] / ".agents/skills/research-review/references/comparison_template.ipynb"
DRAFT_CELL = 'print("<<<ANALYSIS_DRAFT\\n### Comparison\\nbody\\nANALYSIS_DRAFT>>>")'


def _notebook(*sources, group="g", project="rwm-explore"):
    cells = [nbformat.v4.new_code_cell(f"GROUP = {group!r}\nPROJECT = {project!r}"),
             *(nbformat.v4.new_code_cell(src) for src in sources)]  # fmt: skip
    return nbformat.v4.new_notebook(cells=cells)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import pandas as pd\npd.read_parquet('x')", "read_parquet"),
        ("open('a').read()", "open"),
        ("x = 'outputs/run'", "local data path"),
        ("import mbrl", "import mbrl"),
        ("!ls", "'!ls'"),
        ("%%bash\necho hi", "'%%bash'"),
        ("%load_ext autoreload", "'%load_ext'"),
        ("api.artifact('a').download()", "download"),
        ("def broken(:", "syntax error"),
    ],
)  # fmt: skip
def test_analysis_notebooks_obey_the_plot_script_rules(source, expected):
    problems = rr.notebook_source_problems(_notebook(source), group="g", project="rwm-explore")
    assert any(expected in p for p in problems), problems
    assert rr.notebook_source_problems(_notebook("%matplotlib inline\nx = 1"), group="g", project="rwm-explore") == []


def test_analysis_notebook_must_target_the_requested_comparison():
    problems = rr.notebook_source_problems(_notebook(group="other"), group="g", project="rwm-explore")
    assert any("GROUP must be assigned exactly 'g'" in p for p in problems)
    problems = rr.notebook_source_problems(_notebook(project="isaaclab"), group="g", project="rwm-explore")
    assert any("PROJECT" in p for p in problems)
    assert rr.notebook_source_problems(nbformat.v4.new_notebook(), group="g", project="p")[0].startswith(
        "the notebook has no"
    )


def test_run_analysis_guards_logs_attaches_and_writes_the_draft(tmp_path):
    path = tmp_path / "analysis.ipynb"
    nbformat.write(_notebook(DRAFT_CELL), path)
    seen = {}

    def executor(nb, *, env, timeout):
        seen.update(env, timeout=timeout)
        nb.cells[1]["outputs"] = [nbformat.v4.new_output("stream", name="stdout",
                                                         text="<<<ANALYSIS_DRAFT\n### Comparison\nbody\nANALYSIS_DRAFT>>>\n")]  # fmt: skip
        return nb

    stale = FakeRun("g__analysis", "analysis", run_id="an0")
    fake = FakeWandb([stale, FakeRun("2026_lstm-seed1", "train")])
    out = tmp_path / "draft.md"
    result = rr.run_analysis(path, "rwm-explore", "g", draft_out=out, timeout=7, wandb_module=fake, executor=executor)
    assert seen["RESEARCH_ANALYSIS_LOG"] == "1" and seen["WANDB_RUN_ID"] == result["run_id"] and seen["timeout"] == 7
    assert json.loads(seen["RESEARCH_SOURCE"])["dirty"] in (True, False)
    assert fake.run_path.endswith(f"rwm-explore/{result['run_id']}")
    assert fake.inits[0]["id"] == result["run_id"] and fake.inits[0]["resume"] == "must"
    artifact = fake.artifacts[0]
    assert (artifact.name, artifact.type, artifact.files[0][1]) == (
        "analysis-notebook-g",
        "code",
        "analysis.ipynb",
    )
    assert "status:superseded" in stale.tags and out.read_text() == "### Comparison\nbody\n"

    # dry run: executes with logging off and writes nothing
    again = FakeWandb([])
    other = tmp_path / "dry.md"
    seen.clear()
    dry = rr.run_analysis(
        path, "rwm-explore", "g", draft_out=other, dry_run=True, wandb_module=again, executor=executor
    )
    assert seen["RESEARCH_ANALYSIS_LOG"] == "0" and seen["WANDB_RUN_ID"] == ""
    assert dry["run_id"] is None and not again.inits and not again.artifacts and not other.exists()


def test_run_analysis_refuses_without_side_effects(tmp_path):
    path = tmp_path / "analysis.ipynb"
    nbformat.write(_notebook("open('x')"), path)
    with pytest.raises(SystemExit, match="rejected"):
        rr.run_analysis(path, "rwm-explore", "g", wandb_module=FakeWandb([]), executor=lambda *a, **k: None)
    nbformat.write(_notebook(DRAFT_CELL), path)
    stale = FakeRun("g__analysis", "analysis")
    fake = FakeWandb([stale])

    def failing(nb, *, env, timeout):
        raise RuntimeError("cell 2 failed")

    with pytest.raises(SystemExit, match="nothing was superseded"):
        rr.run_analysis(path, "rwm-explore", "g", wandb_module=fake, executor=failing)
    with pytest.raises(SystemExit, match="no Analysis draft"):
        rr.run_analysis(path, "rwm-explore", "g", wandb_module=fake, executor=lambda nb, **kw: nb)
    fake.logged_run = FakeRun("other", "train", group="g")
    nbformat.write(_notebook(DRAFT_CELL), path)

    def drafts(nb, *, env, timeout):
        nb.cells[1]["outputs"] = [
            nbformat.v4.new_output("stream", name="stdout", text="<<<ANALYSIS_DRAFT\nx\nANALYSIS_DRAFT>>>")
        ]
        return nb

    with pytest.raises(SystemExit, match="did not log"):
        rr.run_analysis(path, "rwm-explore", "g", wandb_module=fake, executor=drafts)
    assert stale.tags == [] and not fake.inits and not fake.artifacts


def test_execute_notebook_uses_an_empty_directory_and_scoped_environment(tmp_path):
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = _notebook(
        'import os\nprint("<<<ANALYSIS_DRAFT\\n" + os.getcwd() + "|" + os.environ["RESEARCH_ANALYSIS_LOG"] +'
        ' "\\nANALYSIS_DRAFT>>>")'
    )
    executed = rr.execute_notebook(nb, env={"RESEARCH_ANALYSIS_LOG": "1"}, timeout=120)
    cwd, flag = rr.analysis_draft(executed).split("|")
    assert flag == "1" and Path(cwd).name.startswith("analysis-") and Path(cwd) != Path.cwd()
    assert "RESEARCH_ANALYSIS_LOG" not in __import__("os").environ


# -- the comparison template, against a fake W&B -------------------------------------------------------------


class _Frame:
    def __init__(self, frame):
        self.frame = frame

    def get(self, name):
        return self

    def get_dataframe(self):
        return self.frame


def _template_world(*, drop=()):
    import numpy as np

    import pandas as pd

    runs, episodes = [], {}
    rng = np.random.default_rng(0)
    for variant, good in (("mlp", 0.4), ("lstm", 0.8)):
        for seed in (1, 2):
            research = {"variant": variant, "seed": seed, "commit": "abc" * 4}
            history = pd.DataFrame({"Perf/learning_time": [1.0 + seed], "Perf/collection_time": [2.0],
                                    "Perf/total_fps": [1e4]})  # fmt: skip
            runs.append(FakeRun(f"2026_{variant}-seed{seed}", "train", ["variant:" + variant], state="finished",
                                config={"train_cfg": {"research": research, "max_iterations": 100}},
                                summary={"_runtime": 3600.0 * (1 + seed)}, lastHistoryStep=99,
                                history=lambda keys, samples, pandas, h=history: h))  # fmt: skip
            if (variant, seed) in drop:
                continue
            runs.append(FakeRun(f"{variant}-seed{seed}__eval", "eval", state="finished",
                                config={"research": {**research, "checkpoint": "x/model_99.pt",
                                                     "protocol": {"slug": "p1", "population": "first"}}},
                                summary={"video/L0-seed41": {}}))  # fmt: skip
            rows = [
                {"cell": cell, "episode_id": 0, "success": rng.random() < good + 0.1 * seed / 10,
                 "fell": rng.random() > good, "lin_vel_tracking_error": 0.2 + 0.1 * (variant == "mlp"),
                 "ang_vel_tracking_error": 0.3}
                for cell in ("L0-seed41", "L5-seed41") for _ in range(40)
            ]  # fmt: skip
            episodes[f"eval-episodes-{variant}-seed{seed}"] = pd.DataFrame(rows)
            frame = episodes[f"eval-episodes-{variant}-seed{seed}"]
            runs[-1].logged_artifacts = lambda v=variant, s=seed, f=frame: [
                _AttachedEpisodes(f"eval-episodes-{v}-seed{s}:v0", f)
            ]
    return runs, episodes


class _AttachedEpisodes:
    type = "eval-episodes"
    digest = "immutable-digest"

    def __init__(self, name, frame):
        self.name, self.frame = name, frame

    def get(self, key):
        assert key == "episodes"
        return _Frame(self.frame).get(key)


class _TemplateWandb:
    def __init__(self, runs, episodes):
        self.runs, self.episodes, self.inits, self.logged, outer = runs, episodes, [], [], self

        class Api:
            default_entity = "team"

            def runs(self, path, filters):
                return outer.runs

            def artifact(self, name):
                return _Frame(outer.episodes[name.split("/")[-1].split(":")[0]])

        class Run:
            summary = {}

            def log(self, data):
                outer.logged.append(data)

            def finish(self):
                pass

        self.Api, self.run = Api, Run()

    def init(self, **kw):
        self.inits.append(kw)
        return self.run

    def Image(self, figure):
        return {"image": figure}

    def Table(self, dataframe):
        return {"table": dataframe}


def _run_template(monkeypatch, runs, episodes, *, log, notes=None, eval_run_ids=None):
    pytest.importorskip("matplotlib").use("Agg")
    pytest.importorskip("pandas")
    nb = nbformat.read(TEMPLATE, as_version=4)
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    code[0] = code[0].replace('"REPLACE-ME"', '"g"', 1).replace('"REPLACE-ME"', '"rwm-explore"', 1)
    if notes:
        code = [c.replace('"evidence": "",', f'"evidence": {notes!r},') for c in code]
    fake = _TemplateWandb(runs, episodes)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("RESEARCH_ANALYSIS_LOG", "1" if log else "0")
    monkeypatch.setenv("RESEARCH_SOURCE", json.dumps({"dirty": True, "commit": "abc", "branch": "dev/agent/g"}))
    if eval_run_ids is not None:
        code = [s.replace("EVAL_RUN_IDS = None", "EVAL_RUN_IDS = " + repr(eval_run_ids)) for s in code]
    scope: dict = {}
    exec(compile("\n".join(code), "analysis.ipynb", "exec"), scope)
    return scope, fake


def test_template_passes_the_guard_and_cleanly_targets_a_group():
    nb = nbformat.read(TEMPLATE, as_version=4)
    nbformat.validate(nb)
    assert rr.notebook_source_problems(nb, group="REPLACE-ME", project="REPLACE-ME") == []
    assert all(not c.outputs and c.execution_count is None for c in nb.cells if c.cell_type == "code")


def test_template_computes_the_comparison_from_wandb_only(monkeypatch, capsys):
    runs, episodes = _template_world()
    scope, fake = _run_template(monkeypatch, runs, episodes, log=True)
    assert scope["BASE"] == "lstm"
    assert all("legacy workflow" in p or "incompatible or unverified comparison" in p for p in scope["PROBLEMS"])
    assert scope["effects"].empty
    assert not scope["per_seed"].empty
    assert set(scope["FIGURES"]) == {"per_seed_by_slice", "training_cost"}
    draft = rr.analysis_draft(type("NB", (), {"cells": [{"outputs": [{"output_type": "stream",
                                                                      "text": capsys.readouterr().out}]}]})())  # fmt: skip
    assert (
        draft == scope["DRAFT"] and "per_seed diagnostics" in draft and "incompatible or unverified comparison" in draft
    )
    assert "Training wall time (h)" in draft and "L5" in draft
    logged = fake.logged[0]
    assert {"analysis/fig/training_cost", "analysis/table/per_seed", "analysis/table/inventory"} <= set(logged)
    init = fake.inits[0]
    assert (init["job_type"], init["group"], init["name"], init["tags"]) == (
        "analysis",
        "g",
        "g__analysis",
        ["dirty"],
    )
    assert fake.run.summary["analysis/draft_md"] == draft


def test_template_flags_missing_evaluations_and_single_seeds(monkeypatch):
    runs, episodes = _template_world(drop={("lstm", 2)})
    scope, fake = _run_template(monkeypatch, runs, episodes, log=False)
    assert any("no scored evaluation for training cells:" in p for p in scope["PROBLEMS"])
    assert scope["effects"].empty
    assert 1 in scope["per_seed"].groupby("variant")["train_seed"].nunique().tolist()
    assert "incompatible or unverified" in scope["DRAFT"] and not fake.inits  # interactive: nothing is written


def test_template_notes_may_not_restate_numbers(monkeypatch):
    runs, episodes = _template_world()
    with pytest.raises(ValueError, match="restate numbers"):
        _run_template(monkeypatch, runs, episodes, log=False, notes="LSTM is 12.5% better")
    scope, _ = _run_template(monkeypatch, runs, episodes, log=False, notes="LSTM is better where it matters")
    assert "LSTM is better where it matters" in scope["DRAFT"]
