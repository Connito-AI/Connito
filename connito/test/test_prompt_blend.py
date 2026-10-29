"""Tests for the prompt blend: the prompt validator's published weights blended into every weight submission."""

from __future__ import annotations

import asyncio
import http.server
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest
import requests
import yaml
from pydantic import ValidationError

from connito.shared import cycle as cycle_mod
from connito.shared.chain import _asubmit_fallback_weights, _submit_fallback_weights, reserve_subnet_owner_share
from connito.shared.config import CycleCfg, EvalCfg
from connito.validator import chain_submitter as chain_submitter_mod
from connito.validator import prompt_blend
from connito.validator.chain_submitter import ChainSubmitter

HOTKEYS = [f"hk{i}" for i in range(8)]  # uid i holds hk{i}; hk1 is this validator, hk2 a peer validator
# Production's 524-block training cycle at a mainnet height, where the local 500-block default would put the
# prompt cycle 49 ahead.
BLOCK = 7_000_000
CYCLE_INDEX = BLOCK // 524
CURRENT_PROMPT_CYCLE = CYCLE_INDEX // prompt_blend.TRAINING_CYCLES_PER_PROMPT_CYCLE
WALLET = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk1"))


class _Metagraph:
    def __init__(self) -> None:
        self.hotkeys = list(HOTKEYS)
        self.block = BLOCK


def _config(mode: str = "on", **eval_overrides):
    evaluation = EvalCfg(**{
        "prompt_blend": mode,
        "prompt_blend_url": "https://prompt-record.test",
        "prompt_blend_rules_sha256": ["a" * 64],
        "prompt_blend_validator_hotkey": "pv",
        **eval_overrides,
    })
    return SimpleNamespace(chain=SimpleNamespace(netuid=1), cycle=CycleCfg(), evaluation=evaluation)


def _payload(**overrides):
    payload = {
        "cycle_id": CURRENT_PROMPT_CYCLE - 1,
        "published_at": "2026-10-27T12:00:00Z",
        "rules_sha256": "a" * 64,
        "validator_hotkey": "pv",
        "weights": {"hk6": 0.4},
        "prompt_weights": {"hk7": 9.0},  # the prompt miners', not read
        "prompt_total": 9.0,
    }
    payload |= overrides
    if "total" not in overrides:
        try:
            payload["total"] = sum(payload["weights"].values())
        except (AttributeError, TypeError):
            payload["total"] = 0.4
    return payload


class _Response:
    def __init__(self, status: int, chunks) -> None:
        self.status_code = status
        self._chunks = iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def raw(self):
        return self

    def read1(self, amt):
        return next(self._chunks, b"")


class _Log:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def info(self, event, **kw):
        self.records.append(("info", event, kw))

    def warning(self, event, **kw):
        self.records.append(("warning", event, kw))

    def debug(self, event, **kw):
        self.records.append(("debug", event, kw))


def _fake_signer(monkeypatch):
    monkeypatch.setattr(prompt_blend, "wallet_signer", lambda wallet: (
        lambda method, target, body: {"X-Signed": f"{wallet.hotkey.ss58_address} {method} {target} {body!r}"}
    ))


@pytest.fixture
def served(monkeypatch):
    """Serve the weights route, signed by a fake signer; returns the list of requests made."""
    fetched: list[dict] = []
    _fake_signer(monkeypatch)

    def _serve(answer, status: int = 200):
        def fake_get(url, **kwargs):
            fetched.append({"url": url, **kwargs})
            assert kwargs["stream"] is True and kwargs["timeout"] == prompt_blend.FETCH_TIMEOUT_S
            if isinstance(answer, BaseException):
                raise answer
            if isinstance(answer, _Response):
                return answer
            body = answer if isinstance(answer, bytes) else json.dumps(answer).encode()
            return _Response(status, [body])

        monkeypatch.setattr(requests, "get", fake_get)
        return fetched

    return _serve


@pytest.fixture(autouse=True)
def cycle_api(monkeypatch):
    """cycle-api's /get_phase and the validator whitelist; set `value` or `whitelist` to None for an outage."""
    state = {"value": {"cycle_index": CYCLE_INDEX, "block": BLOCK}, "whitelist": ["hk1", "hk2"], "calls": []}

    def get_with_retry(url, **kwargs):
        route = url.rsplit("/", 1)[1]
        state["calls"].append((route, kwargs))
        answer = state["value"] if route == "get_phase" else state["whitelist"]
        return None if answer is None else SimpleNamespace(json=lambda: answer)

    monkeypatch.setattr(prompt_blend, "get_with_retry", get_with_retry)
    monkeypatch.setattr(cycle_mod, "get_validator_whitelist_from_api", lambda config: set(state["whitelist"]))
    return state


@pytest.fixture
def log(monkeypatch):
    recorder = _Log()
    monkeypatch.setattr(prompt_blend, "logger", recorder)
    return recorder


def _blend(config, train):
    return prompt_blend.blend_prompt_weights(config, WALLET, train, _Metagraph(), time.monotonic() + 15)


def _ablend(config, train, subtensor):
    return asyncio.run(prompt_blend.ablend_prompt_weights(config, WALLET, train, subtensor))


def _final(config, train):
    return reserve_subnet_owner_share(_blend(config, train))


def test_off_makes_no_fetch_and_keeps_the_vector(chain, served, log):
    fetched = served(_payload())
    train = {3: 0.7, 4: 0.3}
    assert _ablend(_config("off"), train, chain.async_) == train
    assert fetched == [] and log.records == [] and chain.state["metagraph_reads"] == 0


def test_off_by_default():
    assert EvalCfg().prompt_blend == "off"


@pytest.mark.parametrize(("text", "mode"), [("off", "off"), ("on", "on"), ("dry_run", "dry_run")])
def test_the_switch_reads_bare_yaml_words(text, mode):
    assert EvalCfg.model_validate(yaml.safe_load(f"prompt_blend: {text}")).prompt_blend == mode


@pytest.mark.parametrize("age", [1, 2], ids=["one cycle old", "two cycles old"])
def test_on_gives_15_60_25_from_a_signed_get(served, log, age):
    fetched = served(_payload(cycle_id=CURRENT_PROMPT_CYCLE - age))
    assert _final(_config("on"), {3: 1.0}) == pytest.approx({3: 0.60, 6: 0.15, 0: 0.25})
    [request] = fetched
    assert request["url"] == "https://prompt-record.test/v1/validator/weights/latest"
    assert request["headers"] == {
        "X-Signed": "hk1 GET /v1/validator/weights/latest b''", "Accept-Encoding": "identity",
    }
    assert request["allow_redirects"] is False
    assert [r[0] for r in log.records] == ["info"]


def test_the_test_route_is_signed_as_sent_and_skips_freshness(served, log, cycle_api):
    fetched = served(_payload(cycle_id=1_000_000_005))
    assert _final(_config("on", prompt_blend_test_route=True), {3: 1.0}) == pytest.approx({3: 0.60, 6: 0.15, 0: 0.25})
    [request] = fetched
    assert request["url"] == "https://prompt-record.test/v1/test/validator/weights/latest"
    assert request["headers"]["X-Signed"] == "hk1 GET /v1/test/validator/weights/latest b''"
    assert [c for c in cycle_api["calls"] if c[0] == "get_phase"] == []


def test_a_404_on_the_test_route_refuses(served, log):
    served(_Response(404, [b'{"detail": "no_published_cycle"}']))
    assert _blend(_config("dry_run", prompt_blend_test_route=True), {3: 1.0}) == {3: 1.0}
    assert log.records == [("warning", prompt_blend.TRAINING_ONLY,
                            {"mode": "dry_run", "reason": "nothing published yet (404)"})]


def test_dry_run_logs_the_blend_and_submits_training_only(served, log):
    served(_payload())
    train = {3: 1.0}
    assert _blend(_config("dry_run"), train) == train
    [(level, event, fields)] = log.records
    assert level == "info" and "dry run" in event
    assert fields["blended"] == {3: 0.8, 6: 0.2}
    assert fields["with_owner_share"] == {3: 0.6, 0: 0.25, 6: 0.15}
    assert fields["prompt_hotkeys"] == {6: "hk6"}


def test_share_is_configurable(served, log):
    served(_payload())
    assert _final(_config("on", prompt_blend_share=0.5), {3: 1.0}) == pytest.approx({3: 0.375, 6: 0.375, 0: 0.25})


def test_hotkeys_missing_from_the_metagraph_are_dropped(served, log):
    served(_payload(weights={"hk6": 1.0, "ghost-a": 5.0, "ghost-b": 5.0}))
    assert _final(_config("on"), {3: 1.0}) == pytest.approx({3: 0.60, 6: 0.15, 0: 0.25})
    assert log.records[0][2]["dropped_off_metagraph"] == 2


def test_validator_uids_and_our_own_are_dropped(served, log, cycle_api):
    served(_payload(weights={"hk1": 1.0, "hk2": 1.0, "hk6": 1.0}))
    cycle_api["whitelist"] = ["hk2"]  # so only the self-exclusion drops hk1
    assert reserve_subnet_owner_share(_blend(_config("on"), {3: 1.0})) == pytest.approx(
        {3: 0.60, 6: 0.15, 0: 0.25}
    )
    assert log.records[0][2]["dropped_on_validators"] == 2


def test_uid_0_in_the_prompt_weights_adds_to_the_owner(served, log):
    served(_payload(weights={"hk0": 1.0, "hk6": 1.0}))
    assert _final(_config("on"), {3: 1.0}) == pytest.approx({3: 0.60, 6: 0.075, 0: 0.325})


def test_overlap_on_a_uid_normalizes(served, log):
    served(_payload(weights={"hk3": 1.0, "hk6": 1.0}))
    blended = _blend(_config("on"), {3: 2.0, 4: 2.0})
    assert blended == pytest.approx({3: 0.5, 4: 0.4, 6: 0.1})
    assert sum(blended.values()) == pytest.approx(1.0)


def test_nan_or_non_positive_training_weights_are_dropped_before_blending(served, log):
    served(_payload())
    blended = _blend(_config("on"), {3: 1.0, 4: float("nan"), 5: 0.0, 7: -1.0})
    assert blended == pytest.approx({3: 0.8, 6: 0.2})
    assert log.records[0][2]["dropped_training"] == 3


_REFUSALS = {
    "fetch failed": requests.ConnectionError("refused"),
    "timed out": requests.Timeout("slow"),
    "nothing published (404)": _Response(404, [b'{"detail": "no_published_cycle"}']),
    "rate limited (429)": _Response(429, [b""]),
    "an oversized body": _Response(200, [b" " * (prompt_blend.MAX_BODY_BYTES + 1)]),
    "not JSON": b"not json",
    "not an object": ["hk6"],
    "no cycle_id": {k: v for k, v in _payload().items() if k != "cycle_id"},
    "bool cycle_id": _payload(cycle_id=True),
    "no published_at": {k: v for k, v in _payload().items() if k != "published_at"},
    "weights not a map": _payload(weights=[["hk6", 1.0]]),
    "negative weight": _payload(weights={"hk6": -1.0}),
    "string weight": _payload(weights={"hk6": "1"}),
    "an int too large for a float": _payload(weights={"hk6": 10**400}, total=1.0),
    "total disagrees": _payload(total=0.9),
    "no total": {k: v for k, v in _payload().items() if k != "total"},
    "older than 2 prompt cycles": _payload(cycle_id=CURRENT_PROMPT_CYCLE - 3),
    "a future cycle": _payload(cycle_id=CURRENT_PROMPT_CYCLE + 1),
    "rules not allowed": _payload(rules_sha256="b" * 64),
    "another validator": _payload(validator_hotkey="someone-else"),
    "nothing on the metagraph": _payload(weights={"ghost": 1.0}),
    "only validators": _payload(weights={"hk1": 1.0, "hk2": 1.0}),
    "all zero": _payload(weights={"hk6": 0.0}),
    "nothing published, total 0": _payload(weights={}),
}


@pytest.mark.parametrize("answer", list(_REFUSALS.values()), ids=list(_REFUSALS))
def test_each_guard_falls_back_to_training_only(served, log, answer):
    served(answer)
    train = {3: 1.0}
    assert _blend(_config("on"), train) == train
    [(level, event, fields)] = log.records
    assert level == "warning" and event == prompt_blend.TRAINING_ONLY and fields["reason"]


def test_cycle_api_unreachable_falls_back(served, log, cycle_api):
    served(_payload())
    cycle_api["value"] = None
    assert _blend(_config("on"), {3: 1.0}) == {3: 1.0}
    assert log.records[0][2]["reason"] == "cycle-api unreachable"


@pytest.mark.parametrize("whitelist", [None, [], {"hk2": 1}], ids=["outage", "empty", "not a list"])
def test_a_whitelist_outage_refuses(served, log, cycle_api, whitelist):
    served(_payload())
    cycle_api["whitelist"] = whitelist
    assert _blend(_config("on"), {3: 1.0}) == {3: 1.0}
    assert log.records[0][2]["reason"] == "validator whitelist unavailable"


def test_cycle_api_is_asked_once_quickly_and_strictly(served, log, cycle_api):
    served(_payload())
    _blend(_config("on"), {3: 1.0})
    one_short_attempt = {"timeout": prompt_blend.CYCLE_API_TIMEOUT_S, "retries": 0}
    assert cycle_api["calls"] == [("get_phase", one_short_attempt), ("get_validator_whitelist", one_short_attempt)]


class _Trickle(http.server.BaseHTTPRequestHandler):
    """A real server sending an 18 KB body one byte at a time."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "18000")
        self.end_headers()
        try:
            for _ in range(18000):
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.02)
        except OSError:
            pass  # the client hung up

    def log_message(self, *args):
        pass


def test_a_trickling_server_is_cut_off_at_the_deadline(monkeypatch):
    _fake_signer(monkeypatch)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Trickle)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        started = time.monotonic()
        with pytest.raises(prompt_blend.BlendRefused, match="deadline"):
            prompt_blend.fetch_prompt_weights(
                f"http://127.0.0.1:{server.server_port}", prompt_blend.WEIGHTS_ROUTE, WALLET, started + 0.5,
            )
        assert time.monotonic() - started < 2.0
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("url", [
    "http://prompt-record.test",
    "https://prompt-record.test/v1",
    "https://prompt-record.test/?a=1",
    "https://prompt-record.test/#x",
    "https://user@prompt-record.test",
    "https:///v1",
])
def test_the_url_must_be_an_https_origin(url):
    with pytest.raises(ValidationError):
        EvalCfg(prompt_blend_url=url)


@pytest.mark.parametrize("url", ["https://prompt-record.test", "https://prompt-record.test/",
                                 "https://prompt-record.test:8443"])
def test_an_origin_is_accepted(url):
    assert EvalCfg(prompt_blend_url=url).prompt_blend_url == url


# The deploy's evaluation.prompt_blend_rules_sha256: an operator setting (its default refuses all), kept here.
THIRD_RULES_HASH = "TODO THIRD_RULES_HASH: the prompt validator's loss-based rules"
DEPLOY_RULES_SHA256 = [
    "dfa6ca9484210eee72aa99f84b343376616af3cedb8645b7b5ee6d3da07d1111",  # the current rules
    "b435ebcc931c4ac41219a4f4d5a733d1817cb4238d6320703e51589666312a07",  # the previous rules
    THIRD_RULES_HASH,
]


def test_the_deploy_allow_list_holds_three_sha256_hashes():
    # Fails until THIRD_RULES_HASH is filled in.
    allowed = EvalCfg(prompt_blend_rules_sha256=DEPLOY_RULES_SHA256).prompt_blend_rules_sha256
    assert len(set(allowed)) == 3 and all(re.fullmatch("[0-9a-f]{64}", h) for h in allowed)


@pytest.mark.parametrize("unset", ["prompt_blend_rules_sha256", "prompt_blend_validator_hotkey"])
def test_an_empty_allow_list_or_hotkey_refuses_without_a_fetch(served, log, unset):
    fetched = served(_payload())
    config = _config("on", **{unset: [] if unset.endswith("sha256") else ""})
    assert _blend(config, {3: 1.0}) == {3: 1.0}
    assert fetched == [] and log.records[0][0] == "warning"


def test_the_metagraph_read_failing_falls_back(served, log, monkeypatch):
    fetched = served(_payload())

    async def broken(**kwargs):
        raise ConnectionError("ws closed")

    assert _ablend(_config("on"), {3: 1.0}, SimpleNamespace(metagraph=broken)) == {3: 1.0}
    assert fetched == [] and log.records[0][0] == "warning"


@pytest.mark.parametrize("mode", ["on", "dry_run"])
@pytest.mark.parametrize(("hang_s", "grace_s", "order"), [
    (1.0, 0.1, ["warning", "debug"]),  # the caller gives up first; the thread ends later, silent
    (0.3, 1.0, ["debug", "warning"]),  # the thread ends late while the caller still waits
], ids=["abandoned", "late"])
def test_a_late_blend_is_discarded_without_false_logs(chain, served, log, monkeypatch, mode, hang_s, grace_s, order):
    def hang(*args, **kwargs):
        time.sleep(hang_s)
        return _Response(200, [json.dumps(_payload()).encode()])

    served(_payload())
    monkeypatch.setattr(requests, "get", hang)
    monkeypatch.setattr(prompt_blend, "BLEND_DEADLINE_S", 0.1)
    monkeypatch.setattr(prompt_blend, "DEADLINE_GRACE_S", grace_s)
    train = {3: 1.0}

    async def timed():
        started = time.monotonic()
        result = await prompt_blend.ablend_prompt_weights(_config(mode), WALLET, train, chain.async_)
        return result, time.monotonic() - started

    result, waited = asyncio.run(timed())  # asyncio.run itself waits for the abandoned thread
    assert result == train and waited < hang_s + 0.5
    assert [r[0] for r in log.records] == order  # never an info line with a blended vector
    warning = next(r for r in log.records if r[0] == "warning")
    assert warning[2]["reason"] == "past the 0.1 s deadline"


def test_one_window_covers_the_metagraph_read_and_the_blend(served, log, monkeypatch):
    served(_payload())
    monkeypatch.setattr(prompt_blend, "BLEND_DEADLINE_S", 0.3)

    async def slow_metagraph(**kwargs):
        await asyncio.sleep(0.25)
        return _Metagraph()

    def slow_get(*args, **kwargs):
        time.sleep(0.2)  # within 0.3 s alone, past it after the metagraph read
        return _Response(200, [json.dumps(_payload()).encode()])

    monkeypatch.setattr(requests, "get", slow_get)
    assert _ablend(_config("on"), {3: 1.0}, SimpleNamespace(metagraph=slow_metagraph)) == {3: 1.0}
    assert "info" not in [r[0] for r in log.records]


# --- The three submission paths -------------------------------------------------------------------------------------

class _Neuron:
    def __init__(self, weights) -> None:
        self.stake = 100.0
        self.weights = weights


def _neurons(differentiated: bool):
    # The peer (uid 2) votes for miners 3, 4, 5; an even vote sends the fallback to uid 0.
    weights = [(3, 0.5), (4, 0.3), (5, 0.2)] if differentiated else [(3, 0.5), (4, 0.5)]
    return {2: _Neuron(weights)}


@pytest.fixture
def chain():
    """A sync and an async fake subtensor; returns them, the set_weights calls and the async metagraph reads."""
    calls: list[dict[int, float]] = []
    state = {"neurons": _neurons(True), "metagraph_reads": 0}

    def set_weights(*, uids, weights, **kwargs):
        calls.append(dict(zip(uids, weights, strict=True)))
        return True

    async def ametagraph(*, netuid):
        state["metagraph_reads"] += 1
        return _Metagraph()

    async def aneuron_for_uid(*, uid, netuid):
        return state["neurons"].get(uid)

    async def aset_weights(**kwargs):
        return set_weights(**kwargs)

    sync = SimpleNamespace(metagraph=lambda *, netuid: _Metagraph(), set_weights=set_weights,
                           neuron_for_uid=lambda *, uid, netuid: state["neurons"].get(uid))
    async_ = SimpleNamespace(metagraph=ametagraph, neuron_for_uid=aneuron_for_uid, set_weights=aset_weights)
    return SimpleNamespace(calls=calls, state=state, sync=sync, async_=async_)


def _sync_fallback(config, chain):
    return _submit_fallback_weights(config, WALLET, chain.sync)


def _async_fallback(config, chain):
    return asyncio.run(_asubmit_fallback_weights(config, WALLET, chain.async_))


@pytest.mark.parametrize("submit", [_sync_fallback, _async_fallback], ids=["sync", "async"])
@pytest.mark.parametrize(("differentiated", "expected"), [
    (True, {3: 0.25, 4: 0.25, 5: 0.25, 0: 0.25}),
    (False, {0: 1.0}),
], ids=["peer top 3", "uid 0"])
@pytest.mark.parametrize("mode", ["on", "dry_run"])
def test_the_fallbacks_never_blend(chain, served, log, cycle_api, submit, differentiated, expected, mode):
    chain.state["neurons"] = _neurons(differentiated)
    fetched = served(_payload())
    assert submit(_config(mode), chain) is True
    [submitted] = chain.calls
    assert submitted == pytest.approx(expected)
    # The fallback reads the whitelist for its own vote, as before, but the blend makes no call at all.
    assert fetched == [] and cycle_api["calls"] == [] and log.records == []


def _submitter(config, subtensor) -> ChainSubmitter:
    submitter = object.__new__(ChainSubmitter)
    submitter.config = config
    submitter.wallet = WALLET
    submitter.normalize, submitter.top_k = True, None
    submitter.wait_for_inclusion = submitter.wait_for_finalization = True
    submitter.post_submit_delay_s = 0.0
    submitter.observer = False
    submitter._submit_lock = None
    submitter._async_subtensor = subtensor
    return submitter


@pytest.mark.parametrize(("mode", "expected"), [
    ("off", {3: 0.75, 0: 0.25}),
    ("dry_run", {3: 0.75, 0: 0.25}),
    ("on", {3: 0.6, 6: 0.15, 0: 0.25}),
])
def test_round_submission_blends(chain, served, log, monkeypatch, mode, expected):
    fetched = served(_payload())
    submitted: list[dict] = []

    async def fake_submit(**kwargs):
        submitted.append(kwargs["uid_weights"])
        return True

    monkeypatch.setattr(chain_submitter_mod, "submit_weights_async", fake_submit)
    round_obj = SimpleNamespace(round_id=7, weights_submitted=False)
    assert asyncio.run(_submitter(_config(mode), chain.async_)._submit_weight_one(round_obj, {"3": 1.0})) is True
    assert submitted == [expected if mode != "on" else pytest.approx(expected)]  # off and dry_run exactly
    assert round_obj.weights_submitted is True
    blending = mode != "off"
    assert len(fetched) == chain.state["metagraph_reads"] == int(blending)
