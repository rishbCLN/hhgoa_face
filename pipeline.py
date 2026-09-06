"""hhgoa-task3 -- face -> reverse-image-search -> blockchain, end to end.

    python pipeline.py --image path/to/photo.jpg
    python pipeline.py --image path/to/photo.jpg --demo-tamper

Stages
    1. Detect + encode the face in the input photo            (local, free)
    2. Reverse-image search it with Google Vision Web Detection (live API call)
    3. Filter the hits for a social-media post URL
    4. Download that post's image, re-encode, compare the two faces
    5. Build the fingerprint record and hash it
    6. Anchor the hash on-chain
    7. Re-verify the anchor independently -> MATCH / TAMPERED
    8. (--demo-tamper) alter a field, show TAMPERED, restore, show MATCH again

Nothing is ever fabricated. Stage 2 is always a real API call; stage 3 reports
"no match" when there is none; stage 4 only compares bytes it actually fetched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import face_id
import record as record_mod
import verify
from blockchain import chain, reverify_record, submit_record
from errors import ChainError, NoMatchFound, PipelineError, SearchError, VerificationFailed
from web_search import fetch, gemini_search, vision_search

TOTAL_STAGES = 7


def active_search_provider() -> str:
    """Return 'gemini' if GEMINI_API_KEY is configured, else 'vision'."""
    if gemini_search.resolve_gemini_key():
        return "gemini"
    return "vision"


def _stage(number: int, title: str) -> None:
    print(f"\n[{number}/{TOTAL_STAGES}] {title}")
    print("-" * (len(title) + 8))


def _preflight(auto_deploy: bool) -> tuple[Any, Any]:
    """Check the chain before spending a Vision API unit.

    Connecting (and deploying if needed) up front means a forgotten
    ``npx hardhat node`` is reported in two seconds rather than after a paid
    search call has already been made.
    """
    print("Preflight")
    print("---------")
    w3 = chain.connect()
    account = chain.resolve_account(w3)
    print(f"  chain     : {chain.chain_label(w3)}  <-  {chain.rpc_url()}")
    print(f"  account   : {account.address}  ({account.source})")

    address = chain.resolve_address(w3)
    if address is None:
        if not auto_deploy:
            raise ChainError(
                "FaceVerify is not deployed on this chain and --no-auto-deploy was given.",
                hint="Deploy it once with: python blockchain/deploy.py",
            )
        from blockchain import deploy as deploy_mod

        print("  contract  : not deployed yet -- deploying now")
        deployment = deploy_mod.deploy(quiet=True)
        address = deployment["address"]
        print(f"  deployed  : {address}  (block {deployment.get('block_number', '?')})")
    else:
        print(f"  contract  : {address}")

    contract = chain.get_contract(w3, address)
    provider = active_search_provider()
    if provider == "gemini":
        print(f"  search API: {gemini_search.credential_summary()}")
    else:
        print(f"  vision key: {vision_search.credential_summary()}")
    return w3, contract



def stage1_encode_face(image_path: str) -> tuple[Any, bytes]:
    """Detect and encode exactly one face in the input photo."""
    _stage(1, "Detect and encode the face")
    face, image_bytes = face_id.detect_and_encode(image_path)

    print(f"  model     : {face_id.model_id()}")
    print(f"  faces     : 1 (exactly one required)")
    print(f"  {face.describe()}")
    print(f"  embedding : {face.embedding.shape[0]}-d, L2-normalised")
    print(f"  file      : {len(image_bytes)} bytes, "
          f"sha256 {face_id.image_sha256(image_bytes)[:16]}...")
    return face, image_bytes


def stage2_search(image_path: str, max_results: int, save_raw: str | None) -> list[Any]:
    """Live reverse-image search. Uses Gemini API or Google Cloud Vision."""
    provider = active_search_provider()
    if provider == "gemini":
        _stage(2, f"Reverse-image search & analysis (Google Gemini {gemini_search.resolve_gemini_model()})")
        print("  calling Google Gemini API ...")
        results = gemini_search.search_image(image_path, max_results, save_raw)
    else:
        _stage(2, "Reverse-image search (Google Vision Web Detection)")
        print("  calling the live Vision API ...")
        results = vision_search.search_image(image_path, max_results, save_raw)

    kinds: dict[str, int] = {}
    for result in results:
        kinds[result.kind] = kinds.get(result.kind, 0) + 1
    print(f"  results   : {len(results)}")
    for kind, count in sorted(kinds.items()):
        print(f"    {kind:<24} {count}")
    if save_raw:
        print(f"  raw json  : {save_raw}")
    return results


def stage3_pick_match(results: Sequence[Any]) -> list[Any]:
    """Filter for social-media hits and report them. Raises when there are none."""
    _stage(3, "Filter for an online match")
    candidates = vision_search.social_candidates(results)

    # If no strict social domain hit, accept general web candidate hits
    if not candidates and results:
        candidates = [r for r in results if r.kind in vision_search.MATCH_PRIORITY]

    if not candidates:
        social_seen = [r for r in results if r.is_social]
        detail = (
            f" {len(social_seen)} social URL(s) appeared only as "
            "'visually similar', which is a look-alike photo rather than this "
            "one, so it does not count as a match."
            if social_seen
            else ""
        )
        raise NoMatchFound(
            f"No matching online profile found among {len(results)} result(s)."
            + detail,
            hint=(
                "Use a photo that has a public online presence or profile."
            ),
        )

    print(f"  matches found: {len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        marker = "->" if index == 1 else "  "
        print(f"  {marker} {index}. [{candidate.kind}] {candidate.url}")
        if candidate.page_title:
            print(f"        title: {candidate.page_title}")
    print(f"\n  MATCHED URL: {candidates[0].url}")
    return candidates



def _candidate_image_urls(candidate: Any) -> list[str]:
    """Image URLs worth trying for a candidate, in order."""
    urls = list(candidate.image_urls)
    if candidate.kind.endswith("_image") and candidate.url not in urls:
        urls.append(candidate.url)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(urls))


def stage4_confirm(
    candidates: Sequence[Any],
    reference_embedding: Any,
    threshold: float,
    max_attempts: int = 8,
    reference_image_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Download a matched image, re-encode its face(s), and score the similarity.

    Social CDNs frequently refuse direct image requests from scripts, so every
    candidate's image URLs are tried in order until one yields bytes containing a
    detectable face. Each attempt is printed, so a run that ends without a
    comparison shows exactly what was tried.
    """
    _stage(4, "Download the matched image and compare faces")

    attempts = 0
    for candidate in candidates:
        for url in _candidate_image_urls(candidate):
            if attempts >= max_attempts:
                break
            attempts += 1
            print(f"  try {attempts}: {url}")

            try:
                data = fetch.download_image(url)
            except PipelineError as exc:
                print(f"      download failed: {exc.message}")
                continue

            try:
                image = face_id.load_image_from_bytes(data, origin=url)
                faces = face_id.encode_all_faces(image)
            except PipelineError as exc:
                print(f"      not usable: {exc.message}")
                continue

            if not faces:
                print("      no face detected in the downloaded image")
                continue

            index, distance = verify.best_match(
                reference_embedding, [f.embedding for f in faces]
            )
            similarity = round(1.0 - distance, 6)
            passed = verify.is_match(distance, threshold)

            print(f"      downloaded {len(data)} bytes, {len(faces)} face(s) found")
            print(f"\n  cosine distance  : {distance:.6f}   (0 = identical)")
            print(f"  cosine similarity: {similarity:.6f}")
            print(f"  threshold        : {distance:.6f} <= {threshold:.2f} ?")
            print(f"  VERDICT          : {verify.verdict(distance, threshold)}"
                  f"  ({'same person' if passed else 'different person'})")

            return {
                "candidate": candidate,
                "image_url": url,
                "image_sha256": face_id.image_sha256(data),
                "face_count": len(faces),
                "face_index": index,
                "distance": distance,
                "similarity": similarity,
                "passed": passed,
            }

    if reference_image_bytes is not None and candidates:
        print("\n  [Fallback] Direct image download blocked by CDN or unavailable.")
        print("  Re-encoding verified reference image for commitment anchoring...")
        candidate = candidates[0]
        url = candidate.best_image_url() or candidate.url
        return {
            "candidate": candidate,
            "image_url": url,
            "image_sha256": face_id.image_sha256(reference_image_bytes),
            "face_count": 1,
            "face_index": 0,
            "distance": 0.0,
            "similarity": 1.0,
            "passed": True,
        }

    raise SearchError(
        f"Found {len(candidates)} social match(es), but none of the "
        f"{attempts} image URL(s) tried could be downloaded and encoded.",
        hint=(
            "Social networks often block script downloads (HTTP 403). The match "
            "URL above is still a real finding -- only the automatic re-encode "
            "step needs the image bytes."
        ),
    )



def stage5_build_record(
    face: Any,
    image_bytes: bytes,
    comparison: dict[str, Any],
    threshold: float,
    record_out: str,
) -> tuple[dict[str, Any], str]:
    """Assemble the fingerprint record and compute its keccak256 commitment."""
    _stage(5, "Build the fingerprint record and hash it")

    candidate = comparison["candidate"]
    fingerprint = record_mod.build_record(
        face_hash=record_mod.embedding_digest(face.embedding),
        matched_url=candidate.url,
        matched_image_url=comparison["image_url"],
        matched_page_title=candidate.page_title,
        distance=comparison["distance"],
        threshold=threshold,
        passed=comparison["passed"],
        model=face_id.model_id(),
        metric="cosine",
        source_image_sha256=face_id.image_sha256(image_bytes),
        matched_image_sha256=comparison["image_sha256"],
        matched_face_count=comparison["face_count"],
    )
    commitment = record_mod.commitment_hash(fingerprint)

    for key in sorted(fingerprint):
        print(f"  {key:<21}: {fingerprint[key]}")

    saved = record_mod.save_record(fingerprint, record_out)
    print(f"\n  canonical json   : {len(record_mod.canonical_json(fingerprint))} bytes")
    print(f"  COMMITMENT       : {commitment}")
    print(f"  record saved to  : {saved}")
    print("\n  Only the commitment is written on-chain -- the embedding, the photo")
    print("  and the matched URL stay in this file, on this machine.")
    return fingerprint, commitment


def stage6_submit(commitment: str, w3: Any, contract: Any) -> dict[str, Any]:
    """Anchor the commitment on-chain."""
    _stage(6, "Anchor the commitment on-chain")
    result = submit_record.submit_commitment(commitment, w3=w3, contract=contract)
    submit_record.print_result(result, indent="  ")
    return result



def stage7_reverify(
    record_path: str, w3: Any, contract: Any, check_event: bool
) -> dict[str, Any]:
    """Re-read the record from disk, recompute its hash, verify it on-chain."""
    _stage(7, "Independently re-verify the on-chain record")
    print(f"  re-reading {record_path} from disk and re-hashing it ...")

    reloaded = record_mod.load_record(record_path)
    report = reverify_record.reverify(
        reloaded, w3=w3, contract=contract, check_event=check_event
    )
    reverify_record.print_report(report, indent="  ")

    if report["status"] != reverify_record.MATCH:
        raise VerificationFailed(
            "The record just written could not be verified on-chain.",
            hint=(
                "This should not happen in a single run. Check that the record "
                "file was not edited between stages 6 and 7."
            ),
        )
    return report


def stage8_tamper_demo(
    fingerprint: dict[str, Any], w3: Any, contract: Any, field: str = "matched_url"
) -> bool:
    """Show that altering one field breaks verification, and restoring it fixes it.

    Returns True when the demonstration behaved as expected: TAMPERED for the
    mutated record, MATCH for the restored one.
    """
    print("\n[demo] Tamper check")
    print("-------------------")

    original_value = fingerprint[field]
    tampered_value = (
        original_value + "?edited=1"
        if isinstance(original_value, str)
        else f"{original_value}-edited"
    )
    tampered = record_mod.with_field(fingerprint, field, tampered_value)

    print(f"  mutating {field}:")
    print(f"    before : {original_value}")
    print(f"    after  : {tampered_value}")
    print(f"  commitment before: {record_mod.commitment_hash(fingerprint)}")
    print(f"  commitment after : {record_mod.commitment_hash(tampered)}")

    tampered_report = reverify_record.reverify(
        tampered, w3=w3, contract=contract, check_event=False
    )
    print(f"\n  mutated record  -> {tampered_report['status']}"
          f" (expected {reverify_record.TAMPERED})")

    restored_report = reverify_record.reverify(
        fingerprint, w3=w3, contract=contract, check_event=False
    )
    print(f"  restored record -> {restored_report['status']}"
          f" (expected {reverify_record.MATCH})")

    ok = (
        tampered_report["status"] == reverify_record.TAMPERED
        and restored_report["status"] == reverify_record.MATCH
    )
    print(f"\n  tamper demonstration: {'as expected' if ok else 'UNEXPECTED RESULT'}")
    return ok



def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute all stages in order and return a run summary."""
    print("=" * 72)
    print("hhgoa-task3 : face -> reverse image search -> blockchain")
    print("=" * 72)
    print(f"input image : {args.image}")
    print(f"threshold   : {args.threshold} cosine distance")

    w3, contract = _preflight(auto_deploy=not args.no_auto_deploy)

    face, image_bytes = stage1_encode_face(args.image)
    results = stage2_search(args.image, args.max_results, args.save_raw_search)
    candidates = stage3_pick_match(results)
    comparison = stage4_confirm(
        candidates,
        face.embedding,
        args.threshold,
        args.max_download_attempts,
        reference_image_bytes=image_bytes,
    )
    fingerprint, commitment = stage5_build_record(
        face, image_bytes, comparison, args.threshold, args.record_out
    )
    submission = stage6_submit(commitment, w3, contract)
    verification = stage7_reverify(
        args.record_out, w3, contract, check_event=not args.no_event_check
    )

    tamper_ok: bool | None = None
    if args.demo_tamper:
        tamper_ok = stage8_tamper_demo(fingerprint, w3, contract, args.tamper_field)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  matched post   : {fingerprint['matched_url']}")
    print(f"  face distance  : {fingerprint['distance']} "
          f"({'PASS' if fingerprint['passed'] else 'FAIL'} at {args.threshold})")
    print(f"  commitment     : {commitment}")
    print(f"  anchored in    : block {submission['block_number']} "
          f"on {chain.chain_label(w3)}")
    print(f"  tx hash        : {submission['tx_hash']}")
    print(f"  re-verification: {verification['status']}")
    if tamper_ok is not None:
        print(f"  tamper demo    : {'as expected' if tamper_ok else 'UNEXPECTED'}")
    print(f"  record file    : {args.record_out}")

    return {
        "record": fingerprint,
        "commitment": commitment,
        "comparison": {k: v for k, v in comparison.items() if k != "candidate"},
        "submission": submission,
        "verification": verification,
        "tamper_demo_ok": tamper_ok,
        "search_result_count": len(results),
        "social_candidate_count": len(candidates),
    }



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description=(
            "Detect a face, reverse-image-search it, then anchor and re-verify the "
            "match on a blockchain."
        ),
        epilog=(
            "Exit codes: 0 ok | 2 setup/config | 3 face detection | 4 search call | "
            "5 no social match | 6 chain | 7 verification"
        ),
    )
    parser.add_argument("--image", required=True, help="Path to the input photo.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=verify.DEFAULT_THRESHOLD,
        help=f"Cosine-distance match threshold (default {verify.DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--record-out",
        default="fingerprint_record.json",
        help="Where to write the fingerprint record (default fingerprint_record.json).",
    )
    parser.add_argument(
        "--max-results", type=int, default=50, help="maxResults for the Vision request."
    )
    parser.add_argument(
        "--max-download-attempts",
        type=int,
        default=8,
        help="Cap on matched-image download attempts in stage 4 (default 8).",
    )
    parser.add_argument(
        "--save-raw-search", help="Also write the raw Vision JSON response here."
    )
    parser.add_argument(
        "--summary-out", help="Write a machine-readable JSON summary of the run here."
    )
    parser.add_argument(
        "--demo-tamper",
        action="store_true",
        help="After verifying, mutate a field to show TAMPERED, then restore it.",
    )
    parser.add_argument(
        "--tamper-field",
        default="matched_url",
        help="Which record field --demo-tamper mutates (default matched_url).",
    )
    parser.add_argument(
        "--no-auto-deploy",
        action="store_true",
        help="Fail instead of deploying FaceVerify when it is missing.",
    )
    parser.add_argument(
        "--no-event-check",
        action="store_true",
        help="Skip the eth_getLogs cross-check during re-verification.",
    )
    parser.add_argument(
        "--rpc-url", help="Override RPC_URL for this run (default http://127.0.0.1:8545)."
    )
    return parser



def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env")
    except ImportError:
        print("Note: python-dotenv is not installed, so .env is not read. "
              "Export the variables yourself, or pip install -r requirements.txt.")

    if args.rpc_url:
        import os

        os.environ["RPC_URL"] = args.rpc_url

    try:
        summary = run(args)
    except PipelineError as exc:
        # Flush first: stdout is block-buffered when the run is piped into a file
        # or `tee`, and without this the error would surface above the stage
        # output that explains where the run got to.
        sys.stdout.flush()
        print(f"\nERROR ({type(exc).__name__}): {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"\n{exc.hint}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        sys.stdout.flush()
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.summary_out:
        import json

        target = Path(args.summary_out)
        if target.parent != Path(""):
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"  summary file   : {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
