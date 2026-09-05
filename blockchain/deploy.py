"""Deploy FaceVerify.sol and record the address.

    python blockchain/deploy.py            # reuse an existing live deployment
    python blockchain/deploy.py --force    # always deploy a fresh copy

Writes ``blockchain/deployment.json`` so ``submit_record.py``,
``reverify_record.py`` and ``pipeline.py`` can find the contract without being
told the address. Restarting ``npx hardhat node`` wipes chain state, so a saved
address whose code has vanished is detected and re-deployed automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from blockchain import chain  # noqa: E402
from errors import PipelineError  # noqa: E402


def deploy(force: bool = False, quiet: bool = False) -> dict[str, Any]:
    """Deploy (or reuse) FaceVerify and return the deployment record."""
    def log(message: str = "") -> None:
        if not quiet:
            print(message)

    w3 = chain.connect()
    account = chain.resolve_account(w3)
    chain.require_balance(w3, account)

    log(f"Chain     : {chain.chain_label(w3)}  <-  {chain.rpc_url()}")
    log(f"Deployer  : {account.address}  ({account.source})")

    artifact = chain.load_artifact()

    if not force:
        existing = chain.resolve_address(w3)
        if existing:
            log(f"Contract  : {existing}  (already deployed -- use --force to replace)")
            saved = chain.load_deployment() or {}
            saved.update(address=existing, reused=True)
            return saved

    factory = w3.eth.contract(abi=artifact["abi"], bytecode=artifact["bytecode"])
    receipt = chain.send_transaction(w3, account, factory.constructor())

    address = w3.to_checksum_address(receipt["contractAddress"])
    tx_hash = receipt["transactionHash"].hex()
    if not tx_hash.startswith("0x"):
        tx_hash = "0x" + tx_hash

    record = {
        "contract": "FaceVerify",
        "address": address,
        "chain_id": w3.eth.chain_id,
        "chain": chain.chain_label(w3),
        "rpc_url": chain.rpc_url(),
        "deployer": account.address,
        "tx_hash": tx_hash,
        "block_number": receipt["blockNumber"],
        "gas_used": receipt["gasUsed"],
        "bytecode_fingerprint": chain.bytecode_fingerprint(artifact),
        "deployed_at": chain.utc_now(),
        "reused": False,
    }
    chain.save_deployment(record)

    log(f"Contract  : {address}")
    log(f"Tx hash   : {tx_hash}")
    log(f"Block     : {record['block_number']}   gas used: {record['gas_used']}")
    explorer = chain.explorer_url(record["chain_id"], tx_hash)
    if explorer:
        log(f"Explorer  : {explorer}")
    log(f"Saved     : {chain.DEPLOYMENT_PATH}")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deploy the FaceVerify contract.")
    parser.add_argument(
        "--force", action="store_true", help="Deploy a new copy even if one exists."
    )
    parser.add_argument("--quiet", action="store_true", help="Print nothing on success.")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    try:
        deploy(force=args.force, quiet=args.quiet)
    except PipelineError as exc:
        sys.stdout.flush()
        print(f"\nERROR: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"\n{exc.hint}", file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
