"""Write a commitment hash on-chain.

    python blockchain/submit_record.py --record fingerprint.json
    python blockchain/submit_record.py --commitment 0x<64 hex chars>

Prints the transaction hash and block number. Only the 32-byte commitment leaves
this machine -- never the embedding, the photo, or the matched URL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from blockchain import chain  # noqa: E402
from errors import ChainError, PipelineError  # noqa: E402


def submit_commitment(
    commitment: str | bytes,
    w3: Any = None,
    contract: Any = None,
    skip_if_anchored: bool = False,
) -> dict[str, Any]:
    """Anchor ``commitment`` on-chain and return the transaction details.

    Args:
        commitment: 32-byte commitment, as ``0x``-hex or raw bytes.
        w3: Existing connection to reuse (a new one is opened if omitted).
        contract: Existing bound contract to reuse.
        skip_if_anchored: Return early, without sending a transaction, when this
            exact commitment is already on-chain.

    Returns:
        A dict with ``commitment``, ``already_anchored``, ``anchored_at`` and --
        unless the transaction was skipped -- ``tx_hash``, ``block_number``,
        ``gas_used`` and ``explorer_url``.
    """
    raw = chain.as_bytes32(commitment)
    hex_commitment = "0x" + raw.hex()

    w3 = w3 or chain.connect()
    contract = contract or chain.get_contract(w3)
    account = chain.resolve_account(w3)
    chain.require_balance(w3, account)

    try:
        existing = int(contract.functions.get(raw).call())
    except Exception as exc:  # noqa: BLE001
        raise ChainError(f"Reading the contract failed: {exc}") from exc

    if existing and skip_if_anchored:
        return {
            "commitment": hex_commitment,
            "contract": contract.address,
            "chain_id": w3.eth.chain_id,
            "already_anchored": True,
            "anchored_at": existing,
            "submitted": False,
            "tx_hash": None,
            "block_number": None,
            "gas_used": None,
            "explorer_url": None,
        }

    receipt = chain.send_transaction(w3, account, contract.functions.submit(raw))

    tx_hash = receipt["transactionHash"].hex()
    if not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash

    anchored_at = int(contract.functions.get(raw).call())
    if anchored_at == 0:  # pragma: no cover - would mean the write silently failed
        raise ChainError(
            "The transaction succeeded but the commitment still reads back as 0.",
            hint="Confirm the contract address matches the deployed bytecode.",
        )

    return {
        "commitment": hex_commitment,
        "contract": contract.address,
        "chain_id": w3.eth.chain_id,
        "already_anchored": bool(existing),
        "anchored_at": anchored_at,
        "submitted": True,
        "tx_hash": tx_hash,
        "block_number": receipt["blockNumber"],
        "gas_used": receipt["gasUsed"],
        "explorer_url": chain.explorer_url(w3.eth.chain_id, tx_hash),
    }



def submit_record(record: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Hash a fingerprint record and anchor the result."""
    from record import commitment_hash

    return submit_commitment(commitment_hash(record), **kwargs)


def print_result(result: Mapping[str, Any], indent: str = "") -> None:
    """Render a submission result for the terminal."""
    from datetime import datetime, timezone

    print(f"{indent}Commitment: {result['commitment']}")
    print(f"{indent}Contract  : {result['contract']}  (chainId {result['chain_id']})")

    if not result["submitted"]:
        print(f"{indent}Status    : already anchored -- no transaction sent")
    else:
        print(f"{indent}Tx hash   : {result['tx_hash']}")
        print(f"{indent}Block     : {result['block_number']}   "
              f"gas used: {result['gas_used']}")
        if result["already_anchored"]:
            print(f"{indent}Status    : commitment already existed; "
                  f"original timestamp kept")

    stamp = datetime.fromtimestamp(result["anchored_at"], tz=timezone.utc)
    print(f"{indent}Anchored  : {result['anchored_at']}  "
          f"({stamp.isoformat(timespec='seconds')})")
    if result.get("explorer_url"):
        print(f"{indent}Explorer  : {result['explorer_url']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anchor a commitment hash on-chain.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--commitment", help="0x-prefixed 32-byte commitment hash.")
    source.add_argument("--record", help="Path to a fingerprint record JSON file.")
    parser.add_argument(
        "--skip-if-anchored",
        action="store_true",
        help="Do not send a transaction if the commitment is already on-chain.",
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

            result = submit_record(
                load_record(args.record), skip_if_anchored=args.skip_if_anchored
            )
        else:
            result = submit_commitment(
                args.commitment, skip_if_anchored=args.skip_if_anchored
            )
        print_result(result)
    except PipelineError as exc:
        sys.stdout.flush()
        print(f"\nERROR: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"\n{exc.hint}", file=sys.stderr)
        return exc.exit_code
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
