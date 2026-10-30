"""btauth/1 signing against vectors made by bittensor 11.3.0's `http_auth.sign`, with a throwaway seed."""

from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from connito.shared import btauth

KEYPAIR = Keypair.create_from_seed("0x" + "01" * 32)
HOTKEY = "5CcyqxXnJucaCnQQvvUg5EPzj1uoNAxACZvzArHw5aVDvgNH"
NONCE = 1_761_700_000_000_000_000
# (method, target, body, nonce): payload and signature from bittensor 11.3.0. sr25519 signatures are randomized,
# so a signature is checked by verifying it, never by comparing it.
VECTORS = [
    (
        ("GET", "/v1/validator/weights/latest", b"", NONCE),
        b"btauth/1\nsr25519\nGET\n/v1/validator/weights/latest\n"
        b"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n1761700000000000000\n" + HOTKEY.encode()
        + b"\n-",
        "0x34ab7da098dd14462c94be90cbebc39ee1b6ea09bc986f020803b437bd44c165"
        "e36a26aa3364f2904ddf5c2a5f59ae2e340d999e6ad896611580896e1a9a3d85",
    ),
    (
        ("post", "/v1/test/x?a=1&b=%2F", b'{"a":1}', NONCE + 1),
        b"btauth/1\nsr25519\nPOST\n/v1/test/x?a=1&b=%2F\n"
        b"015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862\n1761700000000000001\n" + HOTKEY.encode()
        + b"\n-",
        "0x14aa12f419bd52a93d2d1c45d62e1a1cec62a40d04258e101ee6bbe5480c2a0e"
        "05ea09de360b61fda66e2f9a892c04c7690c0d8ecf3b80c6a24c94684f14aa8f",
    ),
]


@pytest.mark.parametrize(("request_", "payload", "signature"), VECTORS, ids=["get", "post with query and body"])
def test_the_payload_is_bittensor_11s(request_, payload, signature):
    assert KEYPAIR.ss58_address == HOTKEY
    assert btauth.btauth_payload(*request_, HOTKEY) == payload
    assert KEYPAIR.verify(payload, bytes.fromhex(signature[2:]))


@pytest.mark.parametrize(("request_", "payload", "signature"), VECTORS, ids=["get", "post with query and body"])
def test_the_headers_are_bittensor_11s(monkeypatch, request_, payload, signature):
    method, target, body, nonce = request_
    monkeypatch.setattr(btauth.time, "time_ns", lambda: nonce)
    headers = btauth.wallet_signer(SimpleNamespace(hotkey=KEYPAIR))(method, target, body)
    ours = headers.pop("X-Bittensor-Signature")
    assert headers == {"X-Bittensor-Version": "1", "X-Bittensor-Hotkey": HOTKEY, "X-Bittensor-Nonce": str(nonce)}
    assert ours.startswith("0x") and len(ours) == len(signature) and ours == ours.lower()
    assert KEYPAIR.verify(payload, bytes.fromhex(ours[2:]))


def test_the_nonce_never_repeats_within_one_nanosecond(monkeypatch):
    monkeypatch.setattr(btauth.time, "time_ns", lambda: NONCE)
    sign = btauth.wallet_signer(SimpleNamespace(hotkey=KEYPAIR))
    nonces = [sign("GET", "/", b"")["X-Bittensor-Nonce"] for _ in range(3)]
    assert nonces == [str(NONCE), str(NONCE + 1), str(NONCE + 2)]
