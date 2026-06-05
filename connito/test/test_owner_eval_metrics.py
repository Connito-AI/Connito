"""Evaluator behavior tests with stub models/tokenizers (no real model/datasets)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _cfg(**overrides):
    ep = SimpleNamespace(
        enabled_metrics=["gsm8k_ppl", "gsm8k_task", "mmlu"],
        n_samples_per_metric={},
        default_n_samples=2,
        eval_seq_length=128,
        eval_batch_size=1,
        gsm8k_max_new_tokens=16,
        mmlu_length_normalize=True,
        sample_seed=0,
    )
    for k, v in overrides.items():
        setattr(ep, k, v)
    return SimpleNamespace(eval_pipeline=ep)


class _Enc(dict):
    """Dict that mimics HF BatchEncoding's ``.to(device)`` (returns self)."""
    def to(self, device):
        return self


class _FakeTokenizer:
    """Minimal tokenizer: maps each char to its ord; supports the call shapes used."""
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False):
        if isinstance(text, str):
            ids = [min(ord(c), 255) for c in text][: (max_length or 10_000)]
            t = torch.tensor([ids], dtype=torch.long)
            return _Enc(input_ids=t, attention_mask=torch.ones_like(t))
        # batch of strings
        seqs = [[min(ord(c), 255) for c in s][: (max_length or 10_000)] for s in text]
        width = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = width - len(s)
            ids.append(s + [0] * pad)
            mask.append([1] * len(s) + [0] * pad)
        return _Enc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) > 0)


# ---------------------------------------------------------------------------
# gsm8k_ppl
# ---------------------------------------------------------------------------
def test_gsm8k_ppl_exp_of_loss(monkeypatch):
    import connito.owner_eval.metrics.gsm8k_ppl as mod

    rows = [{"question": "q1", "answer": "a1\n#### 1"}, {"question": "q2", "answer": "a2\n#### 2"}]
    monkeypatch.setattr(mod, "load_hf_split", lambda *a, **k: rows)
    monkeypatch.setattr(mod, "evaluate_model", lambda **k: {"val_loss": 0.0})

    out = mod.GSM8KPerplexity().evaluate(model=object(), tokenizer=_FakeTokenizer(),
                                         device=torch.device("cpu"), config=_cfg())
    assert out["gsm8k_ppl"] == pytest.approx(1.0)       # exp(0) == 1
    assert out["gsm8k_ppl_val_loss"] == 0.0


def test_gsm8k_ppl_inf_loss_gives_inf(monkeypatch):
    import connito.owner_eval.metrics.gsm8k_ppl as mod
    monkeypatch.setattr(mod, "load_hf_split", lambda *a, **k: [{"question": "q", "answer": "a"}])
    monkeypatch.setattr(mod, "evaluate_model", lambda **k: {"val_loss": float("inf")})
    out = mod.GSM8KPerplexity().evaluate(object(), _FakeTokenizer(), torch.device("cpu"), _cfg())
    assert out["gsm8k_ppl"] == float("inf")


# ---------------------------------------------------------------------------
# gsm8k_task
# ---------------------------------------------------------------------------
def test_gsm8k_task_accuracy(monkeypatch):
    import connito.owner_eval.metrics.gsm8k_task as mod

    rows = [
        {"question": "qa", "answer": "work\n#### 42"},   # model will say 42 -> correct
        {"question": "qb", "answer": "work\n#### 99"},   # model will say 7  -> wrong
    ]
    monkeypatch.setattr(mod, "load_hf_split", lambda *a, **k: rows)

    completions = iter(["The answer is 42", "The answer is 7"])

    class GenModel:
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None,
                     do_sample=None, pad_token_id=None):
            prompt_len = input_ids.shape[1]
            text = next(completions)
            cont = [min(ord(c), 255) for c in text]
            row = input_ids[0].tolist() + cont
            return torch.tensor([row])

    out = mod.GSM8KTaskScore().evaluate(GenModel(), _FakeTokenizer(), torch.device("cpu"), _cfg())
    assert out["gsm8k_task_n"] == 2.0
    assert out["gsm8k_task_acc"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# mmlu
# ---------------------------------------------------------------------------
def test_mmlu_argmax_over_choice_loglik(monkeypatch):
    import connito.owner_eval.metrics.mmlu as mod

    rows = [
        {"question": "q1", "choices": ["w", "x", "y", "z"], "answer": 2},
        {"question": "q2", "choices": ["w", "x", "y", "z"], "answer": 0},
    ]
    monkeypatch.setattr(mod, "load_hf_split", lambda *a, **k: rows)

    # Make loglikelihood deterministically prefer the gold index for row 1 and a
    # wrong index for row 2, so accuracy is exactly 0.5.
    prefer = {0: 2, 1: 1}
    call = {"row": 0, "choice": 0}

    def fake_ll(model, tokenizer, prompt, continuation, device, normalize=True):
        # continuation is " A"/" B"/" C"/" D"
        idx = " ABCD".strip().index(continuation.strip())
        row = call["row"]
        score = 1.0 if idx == prefer[row] else 0.0
        call["choice"] += 1
        if call["choice"] == 4:
            call["choice"] = 0
            call["row"] += 1
        return score

    monkeypatch.setattr(mod, "loglikelihood", fake_ll)
    out = mod.MMLUAccuracy().evaluate(object(), _FakeTokenizer(), torch.device("cpu"), _cfg())
    assert out["mmlu_n"] == 2.0
    assert out["mmlu_acc"] == pytest.approx(0.5)


def test_mmlu_passes_sample_seed_for_representative_sampling(monkeypatch):
    import connito.owner_eval.metrics.mmlu as mod

    captured = {}

    def capture_load(path, name, split, n, seed=None):
        captured.update(path=path, n=n, seed=seed)
        return []

    monkeypatch.setattr(mod, "load_hf_split", capture_load)
    mod.MMLUAccuracy().evaluate(object(), _FakeTokenizer(), torch.device("cpu"),
                                _cfg(sample_seed=123))
    # MMLU must shuffle with the configured seed (its split is subject-ordered).
    assert captured["path"] == "cais/mmlu"
    assert captured["seed"] == 123


# ---------------------------------------------------------------------------
# loglikelihood primitive
# ---------------------------------------------------------------------------
def test_loglikelihood_prefers_higher_logit_token():
    from connito.owner_eval.metrics.base import loglikelihood

    vocab = 256

    class LogitModel:
        """Returns logits that strongly favor whatever the actual next token is,
        so the continuation gets a high (near-zero) log-likelihood."""
        def __init__(self, favored: bool):
            self.favored = favored

        def __call__(self, input_ids=None):
            T = input_ids.shape[1]
            logits = torch.zeros(1, T, vocab)
            if self.favored:
                for t in range(T - 1):
                    logits[0, t, int(input_ids[0, t + 1])] = 50.0
            return SimpleNamespace(logits=logits)

    tok = _FakeTokenizer()
    favored = loglikelihood(LogitModel(True), tok, "abc", " d", torch.device("cpu"), normalize=True)
    unfavored = loglikelihood(LogitModel(False), tok, "abc", " d", torch.device("cpu"), normalize=True)
    assert favored > unfavored
