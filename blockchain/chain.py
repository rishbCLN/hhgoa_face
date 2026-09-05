"""Shared chain plumbing: connect, resolve an account, load artifacts, send txs.

Two deployment targets, selected purely by environment variable -- no code change
and no separate build:

* **Local Hardhat (default).** ``RPC_URL`` unset or pointing at
  ``http://127.0.0.1:8545``. The node ships 20 pre-funded *unlocked* accounts, so
  transactions are sent with ``eth_sendTransaction`` and no private key is ever
  handled. Free, offline, no signup.
* **Public testnet (optional).** Set ``RPC_URL`` to e.g. a Polygon Amoy endpoint
  and ``PRIVATE_KEY`` to a funded test wallet. Transactions are then signed
  locally and broadcast with ``eth_sendRawTransaction``.

``PRIVATE_KEY`` must only ever hold a throwaway testnet key. It is read from the
environment, never written to disk, and never printed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eth_utils import keccak
from web3 import Web3

from errors import ChainError, ConfigError

#: Hardhat's default JSON-RPC endpoint.
DEFAULT_RPC_URL = "http://127.0.0.1:8545"

#: Chain IDs that mean "a local development node".
LOCAL_CHAIN_IDS = frozenset({31337, 1337})

_HERE = Path(__file__).resolve().parent

#: Written by ``npx hardhat compile``.
ARTIFACT_PATH = _HERE / "artifacts" / "contracts" / "FaceVerify.sol" / "FaceVerify.json"

#: Where deploy.py records the address it deployed to.
DEPLOYMENT_PATH = _HERE / "deployment.json"

_TX_TIMEOUT = float(os.getenv("TX_TIMEOUT", "180"))

#: Human names for the chain IDs this project is likely to touch.
_CHAIN_NAMES = {
    1: "Ethereum mainnet",
    137: "Polygon mainnet",
    1337: "local development chain",
    11155111: "Sepolia testnet",
    31337: "Hardhat local network",
    80002: "Polygon Amoy testnet",
}


@dataclass
class Account:
    """The identity used to send transactions."""

    address: str
    #: Present only in signed (testnet) mode. Never logged.
    private_key: str | None
    #: ``"unlocked node account"`` or ``"PRIVATE_KEY env var"`` -- shown to the user.
    source: str

    @property
    def signs_locally(self) -> bool:
        return self.private_key is not None


def rpc_url() -> str:
    """The configured RPC endpoint."""
    return os.getenv("RPC_URL", "").strip() or DEFAULT_RPC_URL


def connect(url: str | None = None) -> Web3:
    """Open a JSON-RPC connection, or raise a :class:`ChainError` with next steps."""
    endpoint = (url or rpc_url()).strip()
    w3 = Web3(Web3.HTTPProvider(endpoint, request_kwargs={"timeout": 30}))

    try:
        connected = w3.is_connected()
    except Exception:  # noqa: BLE001 - any transport failure means "not connected"
        connected = False

    if not connected:
        is_local = "127.0.0.1" in endpoint or "localhost" in endpoint
        raise ChainError(
            f"No JSON-RPC node answered at {endpoint}.",
            hint=(
                "Start the local chain in a second terminal and leave it running:\n"
                "    cd blockchain && npx hardhat node"
                if is_local
                else "Check RPC_URL, and that the endpoint allows requests from this host."
            ),
        )

    try:
        chain_id = w3.eth.chain_id
    except Exception as exc:  # noqa: BLE001
        raise ChainError(f"Connected to {endpoint} but eth_chainId failed: {exc}") from exc

    # Polygon and other PoA chains put >32 bytes in the block extraData field,
    # which web3's default block formatter rejects.
    if chain_id not in LOCAL_CHAIN_IDS:
        from web3.middleware import ExtraDataToPOAMiddleware

        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    return w3


def chain_label(w3: Web3) -> str:
    """``"Hardhat local network (chainId 31337)"`` -- for the CLI banner."""
    chain_id = w3.eth.chain_id
    return f"{_CHAIN_NAMES.get(chain_id, 'unknown chain')} (chainId {chain_id})"



def resolve_account(w3: Web3) -> Account:
    """Pick the sending account: ``PRIVATE_KEY`` if set, else an unlocked node account."""
    private_key = os.getenv("PRIVATE_KEY", "").strip()

    if private_key:
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        try:
            signer = w3.eth.account.from_key(private_key)
        except Exception as exc:  # noqa: BLE001 - eth-keys raises several types
            raise ConfigError(
                f"PRIVATE_KEY is not a valid 32-byte hex private key: {exc}",
                hint="Expected 64 hex characters, optionally 0x-prefixed.",
            ) from exc
        return Account(
            address=Web3.to_checksum_address(signer.address),
            private_key=private_key,
            source="PRIVATE_KEY env var",
        )

    try:
        accounts = w3.eth.accounts
    except Exception as exc:  # noqa: BLE001
        raise ChainError(f"Could not list node accounts: {exc}") from exc

    if not accounts:
        raise ConfigError(
            "The node exposes no unlocked accounts and PRIVATE_KEY is not set, "
            "so there is no way to sign a transaction.",
            hint=(
                "For local work run 'cd blockchain && npx hardhat node' (it creates "
                "20 funded accounts). For a public testnet set PRIVATE_KEY to a "
                "funded test wallet -- see README -> 'Optional: public testnet'."
            ),
        )

    return Account(
        address=Web3.to_checksum_address(accounts[0]),
        private_key=None,
        source="unlocked node account",
    )


def require_balance(w3: Web3, account: Account) -> int:
    """Return the account balance in wei, refusing to continue if it is zero."""
    balance = w3.eth.get_balance(account.address)
    if balance == 0:
        raise ChainError(
            f"Account {account.address} holds 0 wei, so it cannot pay for gas.",
            hint=(
                "On a public testnet, claim free test funds from a faucet. "
                "On local Hardhat this should never happen -- restart the node."
            ),
        )
    return balance



def load_artifact() -> dict[str, Any]:
    """Read the compiled FaceVerify artifact, or explain how to produce it."""
    if not ARTIFACT_PATH.exists():
        raise ChainError(
            f"Contract artifact not found at {ARTIFACT_PATH}.",
            hint="Compile the contract first:\n    cd blockchain && npx hardhat compile",
        )

    try:
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ChainError(f"Could not read {ARTIFACT_PATH}: {exc}") from exc

    if not artifact.get("abi") or not artifact.get("bytecode"):
        raise ChainError(
            f"{ARTIFACT_PATH} has no abi/bytecode -- the compile may have failed.",
            hint="cd blockchain && npx hardhat clean && npx hardhat compile",
        )
    return artifact


def bytecode_fingerprint(artifact: dict[str, Any] | None = None) -> str:
    """keccak of the deploy bytecode, used to spot a stale saved deployment."""
    artifact = artifact or load_artifact()
    return "0x" + keccak(hexstr=artifact["bytecode"]).hex()


def save_deployment(payload: dict[str, Any]) -> Path:
    """Record a deployment so later runs can find the contract."""
    DEPLOYMENT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return DEPLOYMENT_PATH


def load_deployment() -> dict[str, Any] | None:
    """Previously saved deployment info, or ``None``."""
    if not DEPLOYMENT_PATH.exists():
        return None
    try:
        data = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("address") else None



def send_transaction(w3: Web3, account: Account, call: Any) -> Any:
    """Send a contract call / constructor and wait for a successful receipt.

    ``call`` is a web3 ``ContractFunction`` or ``ContractConstructor``. Unlocked
    node accounts go out via ``eth_sendTransaction`` (the node signs); an account
    from ``PRIVATE_KEY`` is signed here and broadcast raw.
    """
    try:
        if account.signs_locally:
            built = call.build_transaction(
                {
                    "from": account.address,
                    "nonce": w3.eth.get_transaction_count(account.address),
                }
            )
            signed = w3.eth.account.sign_transaction(built, account.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        else:
            tx_hash = call.transact({"from": account.address})
    except Exception as exc:  # noqa: BLE001 - web3 raises many transport/EVM types
        raise ChainError(
            f"Transaction was rejected: {exc}",
            hint=(
                "Confirm the node is still running, the account has funds, and "
                "the contract address matches the current chain."
            ),
        ) from exc

    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=_TX_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        raise ChainError(
            f"Transaction {tx_hash.hex()} was sent but no receipt arrived within "
            f"{_TX_TIMEOUT:.0f}s: {exc}",
            hint="Raise TX_TIMEOUT for a slow public testnet.",
        ) from exc

    if receipt.get("status") != 1:
        raise ChainError(
            f"Transaction {receipt['transactionHash'].hex()} reverted on-chain "
            f"(block {receipt['blockNumber']}).",
        )
    return receipt


def as_bytes32(value: str | bytes) -> bytes:
    """Coerce a commitment (0x-hex or raw bytes) to exactly 32 bytes."""
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        try:
            raw = bytes.fromhex(text[2:] if text.startswith(("0x", "0X")) else text)
        except ValueError as exc:
            raise ChainError(f"Commitment {value!r} is not valid hex.") from exc
    else:
        raise ChainError(f"Commitment must be hex or bytes, got {type(value).__name__}.")

    if len(raw) != 32:
        raise ChainError(f"Commitment must be 32 bytes, got {len(raw)}.")
    return raw



def resolve_address(w3: Web3) -> str | None:
    """Find the FaceVerify address for the connected chain.

    Order: ``FACEVERIFY_ADDRESS`` env var, then ``deployment.json``. A saved
    deployment is ignored when it was made against a different chain, or when the
    address holds no code (typical after restarting ``npx hardhat node``, which
    wipes state).
    """
    override = os.getenv("FACEVERIFY_ADDRESS", "").strip()
    if override:
        try:
            return Web3.to_checksum_address(override)
        except ValueError as exc:
            raise ConfigError(f"FACEVERIFY_ADDRESS is not a valid address: {exc}") from exc

    deployment = load_deployment()
    if not deployment:
        return None

    saved_chain_id = deployment.get("chain_id")
    if saved_chain_id is not None and int(saved_chain_id) != w3.eth.chain_id:
        return None

    address = Web3.to_checksum_address(deployment["address"])
    if w3.eth.get_code(address) in (b"", "0x", None):
        return None
    return address


def get_contract(w3: Web3, address: str | None = None) -> Any:
    """Return a bound FaceVerify contract object."""
    artifact = load_artifact()
    target = address or resolve_address(w3)
    if not target:
        raise ChainError(
            "FaceVerify is not deployed on this chain yet.",
            hint="Deploy it:\n    python blockchain/deploy.py",
        )

    checksummed = Web3.to_checksum_address(target)
    if w3.eth.get_code(checksummed) in (b"", "0x", None):
        raise ChainError(
            f"No contract code at {checksummed} on {chain_label(w3)}.",
            hint=(
                "Restarting 'npx hardhat node' resets the chain and erases "
                "deployments. Re-run: python blockchain/deploy.py"
            ),
        )
    return w3.eth.contract(address=checksummed, abi=artifact["abi"])


def explorer_url(chain_id: int, tx_hash: str) -> str | None:
    """Block-explorer link for a transaction, when one exists for this chain."""
    bases = {
        80002: "https://amoy.polygonscan.com/tx/",
        137: "https://polygonscan.com/tx/",
        11155111: "https://sepolia.etherscan.io/tx/",
        1: "https://etherscan.io/tx/",
    }
    base = bases.get(chain_id)
    return f"{base}{tx_hash}" if base else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "ARTIFACT_PATH",
    "DEFAULT_RPC_URL",
    "DEPLOYMENT_PATH",
    "LOCAL_CHAIN_IDS",
    "Account",
    "as_bytes32",
    "bytecode_fingerprint",
    "chain_label",
    "connect",
    "explorer_url",
    "get_contract",
    "load_artifact",
    "load_deployment",
    "require_balance",
    "resolve_account",
    "resolve_address",
    "rpc_url",
    "save_deployment",
    "send_transaction",
    "utc_now",
]
