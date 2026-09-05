"""Stages 6-7 end to end against a real chain: deploy, submit, re-verify, tamper.

These are integration tests, not unit tests: they send genuine transactions to
whatever node ``RPC_URL`` points at (the local Hardhat node by default). No node
listening means the whole module skips with an actionable message rather than
failing -- so ``pytest tests/`` stays green on a machine that has not started one.

Start the node first to exercise them:

    cd blockchain && npx hardhat node          # leave running
    python -m pytest tests/test_chain_integration.py -v

Every record built here carries a unique nonce, so reruns anchor fresh
commitments instead of colliding with earlier ones -- except where a test is
*deliberately* re-submitting to prove the write-once behaviour.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

import record as record_mod
from blockchain import chain, deploy, reverify_record, submit_record
from errors import ChainError

#: Placeholder for the record's ``matched_url``. Nothing here is a real match:
#: these tests never run stage 2, they only need well-formed record bytes to hash.
LOCAL_TEST_URL = "https://example.invalid/chain-integration-test"


@pytest.fixture(scope="module")
def w3() -> Any:
    """A connection to the configured chain, or a skip explaining how to get one."""
    try:
        connection = chain.connect()
    except ChainError as exc:
        pytest.skip(f"no chain at {chain.rpc_url()}: {exc.message}")

    try:
        chain.load_artifact()
    except ChainError as exc:
        pytest.skip(f"contract not compiled: {exc.message}")

    return connection


@pytest.fixture(scope="module")
def contract(w3: Any) -> Any:
    """FaceVerify on this chain, deploying it first if it is not there yet.

    This goes through :func:`blockchain.deploy.deploy`, which reuses a live
    deployment when ``deployment.json`` still points at real code, so a repeat run
    normally sends no transaction at all.
    """
    deploy.deploy(quiet=True)
    return chain.get_contract(w3)


@pytest.fixture
def fresh_record() -> dict[str, Any]:
    """A syntactically valid fingerprint record that has never been anchored."""
    return record_mod.build_record(
        face_hash="sha256:" + uuid.uuid4().hex * 2,
        matched_url=f"{LOCAL_TEST_URL}?run={uuid.uuid4().hex}",
        distance=0.2143,
        threshold=0.6,
        passed=True,
        model="insightface/buffalo_l-arcface-r50-512d",
        source_image_sha256="sha256:" + "11" * 32,
        matched_image_sha256="sha256:" + "22" * 32,
        matched_face_count=1,
    )


# --------------------------------------------------------------------------- #
# The chain itself
# --------------------------------------------------------------------------- #

def test_the_node_looks_like_a_usable_dev_chain(w3: Any):
    account = chain.resolve_account(w3)

    assert w3.is_connected()
    assert chain.require_balance(w3, account) > 0, "the sender must be able to pay gas"
    assert chain.chain_label(w3).endswith(f"(chainId {w3.eth.chain_id})")


def test_deploy_is_idempotent_and_records_where_it_went(w3: Any, contract: Any):
    """A second deploy() reuses the live contract instead of spending gas again."""
    again = deploy.deploy(quiet=True)

    assert again["reused"] is True
    assert w3.to_checksum_address(again["address"]) == contract.address
    assert chain.resolve_address(w3) == contract.address
    assert w3.eth.get_code(contract.address) not in (b"", "0x", None)


# --------------------------------------------------------------------------- #
# Stage 6 -> stage 7: anchor, then verify independently
# --------------------------------------------------------------------------- #

def test_submit_then_reverify_reports_match(w3, contract, fresh_record, tmp_path):
    result = submit_record.submit_record(fresh_record, w3=w3, contract=contract)

    assert result["submitted"] is True
    assert result["already_anchored"] is False
    assert result["commitment"] == record_mod.commitment_hash(fresh_record)
    assert result["tx_hash"].startswith("0x") and len(result["tx_hash"]) == 66
    assert result["block_number"] > 0
    assert result["gas_used"] > 0
    assert result["anchored_at"] > 0

    # Re-verify from disk, the way a third party would: the record is re-read and
    # re-hashed by a separate code path, with nothing carried over from the write.
    path = record_mod.save_record(fresh_record, tmp_path / "fingerprint.json")
    report = reverify_record.reverify(
        record_mod.load_record(path), w3=w3, contract=contract
    )

    assert report["status"] == reverify_record.MATCH
    assert report["anchored"] is True
    assert report["anchored_at"] == result["anchored_at"]
    assert report["commitment"] == result["commitment"]
    assert report["recomputed_from"] == "local record"


def test_the_event_log_confirms_the_anchor_sits_in_a_real_block(
    w3, contract, fresh_record
):
    """State alone could be faked by a bad address; the log proves a real tx."""
    result = submit_record.submit_record(fresh_record, w3=w3, contract=contract)
    report = reverify_record.reverify(fresh_record, w3=w3, contract=contract)

    if not report["event_checked"] or report["block_number"] is None:
        pytest.skip(f"this endpoint would not serve eth_getLogs: {report['notes']}")

    assert report["block_number"] == result["block_number"]
    assert report["tx_hash"] == result["tx_hash"]
    assert report["submitter"] == chain.resolve_account(w3).address
    assert report["block_timestamp_agrees"] is True, report["notes"]
    assert report["notes"] == []


def test_an_unanchored_record_reports_tampered(w3, contract, fresh_record):
    """The negative control: nothing was submitted, so nothing must verify."""
    report = reverify_record.reverify(fresh_record, w3=w3, contract=contract)

    assert report["status"] == reverify_record.TAMPERED
    assert report["anchored"] is False
    assert report["anchored_at"] is None


# --------------------------------------------------------------------------- #
# Stage 8: the tamper demonstration
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "field, value",
    [
        ("matched_url", "https://example.invalid/some-other-post"),
        ("distance", 0.999999),
        ("timestamp", "2000-01-01T00:00:00+00:00"),
        ("passed", False),
    ],
)
def test_changing_any_field_after_anchoring_reports_tampered(
    w3, contract, fresh_record, field, value
):
    submit_record.submit_record(fresh_record, w3=w3, contract=contract)
    assert (
        reverify_record.reverify(fresh_record, w3=w3, contract=contract)["status"]
        == reverify_record.MATCH
    )

    mutated = record_mod.with_field(fresh_record, field, value)
    report = reverify_record.reverify(mutated, w3=w3, contract=contract)

    assert report["status"] == reverify_record.TAMPERED
    assert report["commitment"] != record_mod.commitment_hash(fresh_record)

    # with_field copies, so the untouched original still verifies -- exactly what
    # `pipeline.py --demo-tamper` shows after it restores the record.
    assert (
        reverify_record.reverify(fresh_record, w3=w3, contract=contract)["status"]
        == reverify_record.MATCH
    )


def test_tampering_is_detected_through_the_saved_file_too(
    w3, contract, fresh_record, tmp_path
):
    """The same check via disk, since that is how stage 7 is actually driven."""
    submit_record.submit_record(fresh_record, w3=w3, contract=contract)
    path = record_mod.save_record(fresh_record, tmp_path / "fingerprint.json")

    edited = record_mod.load_record(path)
    edited["distance"] = round(edited["distance"] + 0.000002, 6)
    record_mod.save_record(edited, path)

    report = reverify_record.reverify(
        record_mod.load_record(path), w3=w3, contract=contract
    )
    assert report["status"] == reverify_record.TAMPERED, (
        "a two-millionths change to one float must break the commitment"
    )


# --------------------------------------------------------------------------- #
# Write-once semantics
# --------------------------------------------------------------------------- #

def test_resubmitting_keeps_the_original_timestamp_and_count(w3, contract, fresh_record):
    first = submit_record.submit_record(fresh_record, w3=w3, contract=contract)
    count_after_first = int(contract.functions.totalRecords().call())

    second = submit_record.submit_record(fresh_record, w3=w3, contract=contract)

    assert second["submitted"] is True, "the tx is sent; the contract decides to no-op"
    assert second["already_anchored"] is True
    assert second["anchored_at"] == first["anchored_at"], "an anchor cannot be re-dated"
    assert second["tx_hash"] != first["tx_hash"]
    assert int(contract.functions.totalRecords().call()) == count_after_first


def test_skip_if_anchored_sends_no_transaction(w3, contract, fresh_record):
    first = submit_record.submit_record(fresh_record, w3=w3, contract=contract)
    block_before = w3.eth.block_number

    second = submit_record.submit_record(
        fresh_record, w3=w3, contract=contract, skip_if_anchored=True
    )

    assert second["submitted"] is False
    assert second["already_anchored"] is True
    assert second["tx_hash"] is None
    assert second["anchored_at"] == first["anchored_at"]
    assert w3.eth.block_number == block_before, "no new block means no transaction"


def test_a_fresh_contract_starts_empty_and_counts_up(w3: Any):
    """Deploy a throwaway instance so the counter starts from a known zero."""
    artifact = chain.load_artifact()
    account = chain.resolve_account(w3)
    factory = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
    receipt = chain.send_transaction(w3, account, factory.constructor())
    isolated = w3.eth.contract(
        address=w3.to_checksum_address(receipt["contractAddress"]),
        abi=artifact["abi"],
    )

    assert int(isolated.functions.totalRecords().call()) == 0

    for expected in (1, 2):
        commitment = record_mod.commitment_bytes(
            record_mod.build_record(
                face_hash="sha256:" + uuid.uuid4().hex * 2,
                matched_url=LOCAL_TEST_URL,
                distance=0.1,
                threshold=0.6,
                passed=True,
                model="test",
            )
        )
        result = submit_record.submit_commitment(commitment, w3=w3, contract=isolated)

        assert result["submitted"] is True
        assert isolated.functions.exists(commitment).call() is True
        assert int(isolated.functions.totalRecords().call()) == expected


def test_the_empty_commitment_is_rejected_by_the_contract(w3, contract):
    """A 32-zero-byte commitment is a bug upstream, not a record -- it must revert."""
    with pytest.raises(ChainError):
        submit_record.submit_commitment(b"\x00" * 32, w3=w3, contract=contract)


def test_a_malformed_commitment_never_reaches_the_chain():
    for bad in ("0xnothex", "0x1234", b"too short"):
        with pytest.raises(ChainError):
            chain.as_bytes32(bad)
