"""Independently re-verify a fingerprint record against the chain.

    python blockchain/reverify_record.py --record fingerprint.json
    python blockchain/reverify_record.py --commitment 0x<64 hex chars>

This is deliberately a *separate* entry point from the writer. It re-reads the
record from disk, recomputes ``keccak256(canonical_json(record))`` with its own
code path, and asks the contract whether that exact commitment is anchored:

    MATCH     the recomputed hash resolves on-chain -- the local record is
              byte-for-byte what was anchored, at the reported time.
    TAMPERED  the recomputed hash is absent. Something in the local record
              changed after anchoring (or it was never anchored).

Where the RPC endpoint allows it, the ``RecordSubmitted`` log is fetched as well,
so the anchor is confirmed to sit in a real block whose timestamp matches the
value stored in the mapping -- not just in contract state.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from blockchain import chain  # noqa: E402
from errors import ChainError, PipelineError  # noqa: E402

MATCH = "MATCH"
TAMPERED = "TAMPERED"


def _find_anchor_event(w3: Any, contract: Any, raw: bytes) -> dict[str, Any]:
    """Locate the ``RecordSubmitted`` log for ``raw``, if the endpoint permits it.

    Public RPC providers often cap ``eth_getLogs`` ranges, so a full-history scan
    is attempted first and then narrowed. Failure is reported, never fatal: the
    mapping lookup is the authoritative check.
    """
    event = contract.events.RecordSubmitted
    latest = w3.eth.block_number
    windows = [0]
    if latest > 50_000:
        windows.append(max(0, latest - 50_000))

    error: str | None = None
    for from_block in windows:
        try:
            logs = list(
                event.get_logs(argument_filters={"commitment": raw}, from_block=from_block)
            )
        except Exception as exc:  # noqa: BLE001 - provider-specific log limits
            error = str(exc)
            continue
        error = None
        if logs:
            entry = logs[0]
            return {
                "checked": True,
                "found": True,
                "block_number": int(entry["blockNumber"]),
                "tx_hash": _hex(entry["transactionHash"]),
                "submitter": entry["args"]["submitter"],
                "event_timestamp": int(entry["args"]["timestamp"]),
                "error": None,
            }
        return {"checked": True, "found": False, "error": None}

    return {"checked": False, "found": False, "error": error}


def _hex(value: Any) -> str:
    """Normalise a HexBytes/bytes/str to a 0x-prefixed hex string."""
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    text = str(value)
    return text if text.startswith("0x") else "0x" + text



def reverify_commitment(
    commitment: str | bytes,
    w3: Any = None,
    contract: Any = None,
    check_event: bool = True,
) -> dict[str, Any]:
    """Look ``commitment`` up on-chain and return a verification report."""
    raw = chain.as_bytes32(commitment)

    w3 = w3 or chain.connect()
    contract = contract or chain.get_contract(w3)

    try:
        anchored_at = int(contract.functions.get(raw).call())
    except Exception as exc:  # noqa: BLE001
        raise ChainError(f"Reading the contract failed: {exc}") from exc

    report: dict[str, Any] = {
        "commitment": "0x" + raw.hex(),
        "contract": contract.address,
        "chain_id": w3.eth.chain_id,
        "chain": chain.chain_label(w3),
        "anchored": bool(anchored_at),
        "anchored_at": anchored_at or None,
        "anchored_at_iso": (
            datetime.fromtimestamp(anchored_at, tz=timezone.utc).isoformat(timespec="seconds")
            if anchored_at
            else None
        ),
        "status": MATCH if anchored_at else TAMPERED,
        "block_number": None,
        "tx_hash": None,
        "submitter": None,
        "explorer_url": None,
        "event_checked": False,
        "block_timestamp_agrees": None,
        "notes": [],
    }

    if anchored_at and check_event:
        event = _find_anchor_event(w3, contract, raw)
        report["event_checked"] = event["checked"]
        if event["checked"] and event["found"]:
            report["block_number"] = event["block_number"]
            report["tx_hash"] = event["tx_hash"]
            report["submitter"] = event["submitter"]
            report["explorer_url"] = chain.explorer_url(w3.eth.chain_id, event["tx_hash"])
            try:
                block_timestamp = int(w3.eth.get_block(event["block_number"])["timestamp"])
                report["block_timestamp_agrees"] = block_timestamp == anchored_at
                if not report["block_timestamp_agrees"]:
                    report["notes"].append(
                        f"Stored timestamp {anchored_at} differs from block "
                        f"timestamp {block_timestamp}."
                    )
            except Exception as exc:  # noqa: BLE001
                report["notes"].append(f"Could not read the anchor block: {exc}")
        elif event["checked"]:
            report["notes"].append(
                "State says anchored but no RecordSubmitted log was found."
            )
        else:
            report["notes"].append(
                f"Skipped the event cross-check ({event['error']}); the mapping "
                "lookup above is authoritative."
            )

    return report



def reverify(record: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Recompute a record's commitment locally, then verify it on-chain.

    The hash is derived here from the record's own bytes -- nothing is carried
    over from whatever process wrote it -- so a record altered after anchoring
    reports :data:`TAMPERED`.
    """
    from record import commitment_hash

    report = reverify_commitment(commitment_hash(record), **kwargs)
    report["recomputed_from"] = "local record"
    return report


def print_report(report: Mapping[str, Any], indent: str = "") -> None:
    """Render a verification report for the terminal."""
    print(f"{indent}Commitment: {report['commitment']}")
    print(f"{indent}Contract  : {report['contract']}  ({report['chain']})")

    if report["status"] == MATCH:
        print(f"{indent}On-chain  : anchored at {report['anchored_at']} "
              f"({report['anchored_at_iso']})")
        if report["block_number"] is not None:
            print(f"{indent}Anchor tx : {report['tx_hash']}  "
                  f"in block {report['block_number']}")
            print(f"{indent}Submitter : {report['submitter']}")
        if report["block_timestamp_agrees"] is True:
            print(f"{indent}Block time: agrees with the stored timestamp")
        if report.get("explorer_url"):
            print(f"{indent}Explorer  : {report['explorer_url']}")
    else:
        print(f"{indent}On-chain  : this commitment is not present "
              f"(get() returned 0)")

    for note in report.get("notes", []):
        print(f"{indent}Note      : {note}")

    print(f"\n{indent}RESULT: {report['status']}")



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-verify a fingerprint record against the on-chain anchor.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record", help="Path to a fingerprint record JSON file.")
    source.add_argument("--commitment", help="0x-prefixed 32-byte commitment hash.")
    parser.add_argument(
        "--no-event-check",
        action="store_true",
        help="Skip the eth_getLogs cross-check (faster on rate-limited endpoints).",
    )
    parser.add_argument(
        "--expect",
        choices=[MATCH, TAMPERED],
        help="Exit non-zero unless the result is this. Useful in scripts and CI.",
    )
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    try:
        if args.record:
            from record import load_record

            report = reverify(load_record(args.record), check_event=not args.no_event_check)
        else:
            report = reverify_commitment(
                args.commitment, check_event=not args.no_event_check
            )
        print_report(report)
    except PipelineError as exc:
        sys.stdout.flush()
        print(f"\nERROR: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"\n{exc.hint}", file=sys.stderr)
        return exc.exit_code
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    if args.expect and report["status"] != args.expect:
        # Flush first, or this line lands above the report when stdout is piped.
        sys.stdout.flush()
        print(
            f"\nExpected {args.expect} but got {report['status']}.", file=sys.stderr
        )
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
