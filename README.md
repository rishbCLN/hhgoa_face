# HH Goa 2026 — Task 3: Face Identification & Blockchain Verification

A command-line pipeline. No website, no hosting, no paid service.

- [What it does](#what-it-does)
- [How to run it](#how-to-run-it)
- [Setup checklist (one-time, human)](#setup-checklist-one-time-human)
- [Which blockchain is used](#which-blockchain-is-used)
- [Which face model, and why](#which-face-model-and-why)
- [Repo layout](#repo-layout)
- [Command reference](#command-reference)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Verification status](#verification-status)

## What it does

`pipeline.py` takes one photo of a consenting subject, detects the face and
encodes it as a 512-dimensional vector locally, then asks Google Cloud Vision's
Web Detection API where that exact image appears online and filters the hits down
to a real social-media post URL. It downloads the matched image, re-encodes the
face in it and scores the two vectors against a threshold; then it hashes the
resulting match record and anchors that hash on a blockchain, and finally
re-verifies the on-chain anchor against the local record from scratch — printing
`MATCH` or `TAMPERED`.

```
photo.jpg
   │
   ├─(1)─ InsightFace SCRFD + ArcFace  ──►  box + 512-d embedding      [local]
   ├─(2)─ Google Vision WEB_DETECTION  ──►  every page hosting it      [live API]
   ├─(3)─ social-domain filter         ──►  the matched post URL
   ├─(4)─ download it, re-encode, compare  ──►  cosine distance, PASS/FAIL
   ├─(5)─ keccak256(canonical JSON)    ──►  32-byte commitment
   ├─(6)─ FaceVerify.submit()          ──►  tx hash + block number     [chain]
   └─(7)─ re-read, recompute, compare  ──►  MATCH / TAMPERED
        (8) --demo-tamper: mutate a field → TAMPERED, restore → MATCH
```

Two rules are enforced throughout, not just documented:

- **Stage 2 is always a real API call.** There is no demo mode, no cached
  response and no fallback that invents a match. With no credential configured
  the pipeline stops there with setup instructions; if the API returns nothing,
  it says so and exits. The only canned Vision JSON in the repo lives in
  `tests/test_search_parser.py`, which is unreachable from the pipeline — and one
  of those tests scans every shipped module to prove no post URL is hardcoded.
- **No biometric data is written on-chain.** Only `keccak256` of the record goes
  on-chain; the embedding, the photo and the matched URL stay on your disk.

## How to run it

From a clean clone. Everything below is free and needs no account.

**Prerequisites:** Python 3.10+ and Node.js 18+ (verified on Python 3.14.2 and
Node 24.13.1). No C++ toolchain and no `cmake` needed — see
[Which face model, and why](#which-face-model-and-why).

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

```bash
cd blockchain && npm install && cd ..
```

The first pipeline run downloads the InsightFace `buffalo_l` model pack
(~275 MB) into `~/.insightface/models` and caches it. To get that out of the way
up front:

```bash
python -c "import face_id; face_id.get_analyzer(); print('model ready')"
```

### 2. Create your .env

```bash
cp .env.example .env
```

Leave it exactly as it is to run everything except the live search. Filling in
the Vision credential is the only human step the pipeline cannot do for itself —
see [Setup checklist](#setup-checklist-one-time-human).

### 3. Compile the contract and start the local chain

```bash
cd blockchain && npx hardhat compile
```

Then, **in a second terminal, and leave it running**:

```bash
cd blockchain && npx hardhat node
```

That is the whole blockchain: a local JSON-RPC chain on `http://127.0.0.1:8545`
with 20 pre-funded accounts. No wallet, no faucet, no signup, no cost.

### 4. Deploy FaceVerify

```bash
python blockchain/deploy.py
```

The address is saved to `blockchain/deployment.json`. Restarting `npx hardhat
node` wipes chain state; the pipeline detects that the saved address holds no code
and re-deploys automatically, so you rarely need to run this by hand.

### 5. Run the pipeline

```bash
python pipeline.py --image sample_data/my_photo.jpg
```

With the tamper demonstration (recommended for the recording):

```bash
python pipeline.py --image sample_data/my_photo.jpg --demo-tamper
```

Supply your own photo — see [`sample_data/README.md`](sample_data/README.md). No
photographs of real people are committed to this repo.

### What you see with no Vision credential

This is the expected outcome until a human adds a key. Stages "preflight" and 1
complete, then the run stops with instructions and exit code 2 — no traceback, no
fabricated match:

```
Preflight
  chain     : Hardhat local network (chainId 31337)  <-  http://127.0.0.1:8545
  account   : 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266  (unlocked node account)
  contract  : 0x5FbDB2315678afecb367f032d93F642f64180aa3
  vision key: none

[1/7] Detect and encode the face
  model     : insightface/buffalo_l/arcface-512d
  faces     : 1 (exactly one required)
  box=(x1=75, y1=97, x2=182, y2=248) size=107x151px score=0.902
  embedding : 512-d, L2-normalised

[2/7] Reverse-image search (Google Vision Web Detection)
  calling the live Vision API ...

ERROR (ConfigError): Google Cloud Vision credentials are not configured.
No Google Cloud Vision credential found. Set ONE of: ...
```

The chain is contacted *before* the search so that a forgotten `npx hardhat node`
is reported in two seconds rather than after a Vision unit has been spent.

## Setup checklist (one-time, human)

These need a person; the code cannot do them and does not try.

- [ ] Create a free Google Cloud account, enable billing on it (Google requires a
      card even for free-tier use — this project stays inside the free 1,000 Web
      Detection calls/month, at 1 call per run), then **enable the Cloud Vision
      API** for the project.
- [ ] Create a credential and put it in `.env` per `.env.example`: either an API
      key restricted to the Vision API (`GOOGLE_VISION_API_KEY=...`, simplest) or
      a service-account JSON key file
      (`GOOGLE_APPLICATION_CREDENTIALS=/abs/path/sa.json`). Both paths are
      implemented; the API key wins if both are set.
- [ ] Choose the test photo: a **consenting** subject (you or a teammate) whose
      photo is **already publicly posted** on a social profile. A never-published
      photo cannot be matched — see [Known limitations](#known-limitations).
- [ ] *(Optional)* For a public testnet instead of local Hardhat: create a
      throwaway wallet, claim free test funds from a faucet, and set `RPC_URL` and
      `PRIVATE_KEY` in `.env`.

No code changes are needed after any of these. Add the key, re-run the same
command, and stage 2 onwards proceeds.

## Which blockchain is used

**A local Hardhat network, by default** — `npx hardhat node` on
`http://127.0.0.1:8545`, chain ID 31337. It is a simulated EVM chain on your own
machine: no signup, no wallet, no faucet, no gas cost, works offline. Transactions
are sent through one of the node's 20 unlocked pre-funded accounts, so **no
private key is handled at all** on the default path.

The contract is [`blockchain/contracts/FaceVerify.sol`](blockchain/contracts/FaceVerify.sol)
(Solidity 0.8.28) — deliberately minimal:

```solidity
mapping(bytes32 => uint256) private records;   // commitment -> block timestamp

function submit(bytes32 commitment) external returns (uint256 anchoredAt);
function get(bytes32 commitment) external view returns (uint256);
function exists(bytes32 commitment) external view returns (bool);
```

Anchors are **write-once**: the first submission of a commitment keeps its
timestamp forever, so an anchor cannot be back-dated or re-dated by a later
caller. Re-submitting emits `RecordAlreadyAnchored` and leaves state untouched.
Verification is `get(keccak256(canonical_json(record))) != 0`, and where the RPC
endpoint allows `eth_getLogs` the `RecordSubmitted` event is fetched too, so the
anchor is confirmed to sit in a real block whose timestamp agrees with the stored
value — not just in contract state.

### Optional: a public testnet

Built and documented, but **not exercised** in this build — deploying to a public
chain needs a wallet and faucet funds a human controls. Switching is two
environment variables and no code change:

```bash
RPC_URL=https://rpc-amoy.polygon.technology
PRIVATE_KEY=<throwaway testnet key, 64 hex chars>
```

`blockchain/chain.py` then signs locally and broadcasts raw transactions instead
of using an unlocked node account, injects PoA middleware for the non-standard
`extraData` those chains use, and prints a Polygonscan link for the transaction.
`PRIVATE_KEY` is read from the environment only — never written to disk, never
logged. Use a throwaway key with nothing but test funds on it.

## Which face model, and why

**InsightFace `buffalo_l`**, run locally through `onnxruntime` on CPU:

| Component | Model | Output |
| --- | --- | --- |
| Detector | `det_10g.onnx` (SCRFD) | bounding box + 5 landmarks + confidence |
| Recogniser | `w600k_r50.onnx` (ArcFace R50) | 512-d embedding, L2-normalised |

`face_recognition`/dlib was the first choice and was **not** usable here: dlib
publishes no binary wheel for the Python interpreter in this environment, so pip
falls back to building from source, which needs `cmake` and a C++ toolchain that
is not installed. InsightFace ships a pure-Python wheel with ONNX weights and
needed no build step, so the pipeline uses it. Both are free, local and
API-key-free; the swap changes only which distance metric is correct.

Because the embeddings are unit-length, similarity is measured as **cosine
distance** (`1 - dot(a, b)`, range 0–2), with a default match threshold of
**0.60**, overridable with `--threshold` or `FACE_MATCH_THRESHOLD`.

> The familiar `0.6` from `face_recognition` is a *Euclidean* distance over 128-d
> vectors — a different metric on a different model. The number coincides, the
> meaning does not. Thresholds are not portable between the two.

Measured locally on the same face at two crops vs. two different people:

```
same person, re-cropped   cosine distance 0.013870   PASS
different person          cosine distance 1.035786   FAIL
```

## Repo layout

```
hhgoa-task3/
├── README.md                     this file
├── .env.example                  every setting, Vision keys left blank
├── .gitignore                    keeps .env, photos and records out of git
├── requirements.txt              pinned, all free/open source
├── pipeline.py                   single CLI entry point, stages 1-8
├── record.py                     canonical JSON + keccak256 commitment
├── errors.py                     typed failures -> clear message + exit code
├── face_id/encode.py             detect_face() / encode_face()          (stage 1)
├── web_search/
│   ├── vision_search.py          search_image() / find_social_match()   (stage 2-3)
│   └── fetch.py                  guarded matched-image download         (stage 4)
├── verify/confirm_match.py       compare_faces() -> cosine distance     (stage 4)
├── blockchain/
│   ├── contracts/FaceVerify.sol  mapping(bytes32 => uint256) + submit/get
│   ├── hardhat.config.js         solc 0.8.28, local chain 31337
│   ├── package.json              Hardhat toolchain (npm install)
│   ├── chain.py                  connect / accounts / artifacts / send tx
│   ├── deploy.py                 deploy, print + save the address       (stage 6)
│   ├── submit_record.py          write the commitment -> tx + block      (stage 6)
│   └── reverify_record.py        read back, recompute -> MATCH/TAMPERED (stage 7)
├── tests/
│   ├── test_hashing.py           commitment determinism + tamper sensitivity
│   ├── test_search_parser.py     Vision JSON parsing, mock-only (see below)
│   ├── test_face_encode.py       stage 1 + distance metric, real inference
│   └── test_chain_integration.py deploy/submit/reverify against local Hardhat
└── sample_data/README.md         placeholder note; bring your own photo
```

### What goes on-chain, and what does not

Stage 5 builds the fingerprint record — `face_hash` (SHA-256 of the normalised
embedding), `matched_url`, `distance`, `threshold`, `passed`, `model`, both image
SHA-256s and a UTC `timestamp` — writes it to `fingerprint_record.json`, and
computes:

```
commitment = keccak256(canonical_json(record))
```

Canonical JSON means sorted keys, no insignificant whitespace, ASCII-escaped,
floats quantised to 6 decimals, NaN/Infinity rejected — so any party hashing the
same record gets the same 32 bytes regardless of platform or key order.
`keccak256` (not SHA3-256) is used precisely because it is what Solidity's
`keccak256` computes, so a third party can verify a commitment with a contract
call and no translation layer.

Only that 32-byte commitment is submitted. The embedding, the photo and the
matched URL never leave your machine — a public ledger is append-only, and
biometric data written there could not be deleted later.

## Command reference

Each stage is also runnable on its own, which is useful when recording a demo.

```bash
python pipeline.py --image photo.jpg --demo-tamper
```

| Flag | Purpose |
| --- | --- |
| `--image PATH` | input photo (required) |
| `--threshold F` | cosine-distance match threshold (default `0.6`) |
| `--demo-tamper` | after verifying, mutate a field → `TAMPERED`, restore → `MATCH` |
| `--tamper-field F` | which field `--demo-tamper` mutates (default `matched_url`) |
| `--record-out PATH` | where to write the record (default `fingerprint_record.json`) |
| `--save-raw-search PATH` | also dump the raw Vision JSON, for auditing |
| `--summary-out PATH` | write a machine-readable JSON summary of the run |
| `--rpc-url URL` | override `RPC_URL` for one run |
| `--max-results N` | `maxResults` for the Vision request (default 50) |
| `--max-download-attempts N` | cap on stage-4 image download attempts (default 8) |
| `--no-auto-deploy` | fail instead of deploying `FaceVerify` when it is missing |
| `--no-event-check` | skip the `eth_getLogs` cross-check |

```bash
# stage 2-3 alone: live search, list the hits, print the best social match
python -m web_search.vision_search --image photo.jpg --all

# stage 6-7 alone, against a record file
python blockchain/submit_record.py   --record fingerprint_record.json
python blockchain/reverify_record.py --record fingerprint_record.json --expect MATCH

# prove the tamper check from the shell: edit any field in the JSON, then
python blockchain/reverify_record.py --record fingerprint_record.json  # -> TAMPERED
```

Exit codes: `0` ok · `2` setup/config · `3` face detection · `4` search call ·
`5` no social match · `6` chain · `7` verification. `--expect MATCH|TAMPERED` on
`reverify_record.py` makes it non-zero unless the result is the one you asked for,
which is what makes it usable in CI.

## Tests

```bash
python -m pytest tests/ -q
# 89 passed  -- with a Hardhat node running and the model pack downloaded
```

Two of the four modules need something from the outside world and **skip with an
explanatory message** rather than failing when it is absent, so a clean checkout
still goes green: `test_face_encode.py` skips without the InsightFace model pack,
`test_chain_integration.py` skips without a node on `RPC_URL`. That leaves 62
tests that always run, offline, with no node and no credentials.

- `test_hashing.py` — the commitment is 32 bytes and deterministic; key order,
  whitespace, pretty-printing and sub-precision float noise cannot change it;
  changing *any* field does change it; the hash really is keccak256, not SHA3-256.
- `test_search_parser.py` — parsing of every Vision result shape from **mock JSON
  defined inside the test file**, plus the constraint-1.2 guards: `search_image`
  raises without a credential, raises on network failure, provably POSTs to the
  real endpoint, and a static scan asserts no shipped module contains a hardcoded
  social post URL.
- `test_face_encode.py` — real InsightFace inference: one face → box + 512-d unit
  vector, same face scores `PASS`, different faces score `FAIL`, zero-face and
  multi-face inputs are rejected. Skips if the model pack is not downloaded.
- `test_chain_integration.py` — deploy → submit → read back → `MATCH`, mutate →
  `TAMPERED`, restore → `MATCH`, and write-once idempotency. Skips with a message
  if no node is listening on `RPC_URL`.

## Known limitations

**1. Reverse image search only finds what Google has already indexed.** Stage 2
asks Google where an image it has *already crawled* appears. A brand-new photo
that has never been posted will return no match, and the pipeline will say
`No matching social-media post found` and exit 5 — a correct, honest outcome, not
a bug. The test subject's photo has to already exist publicly online. Indexing
also lags: a post from this morning may not be findable yet, and heavily
re-compressed or cropped re-uploads may come back only as "visually similar",
which this pipeline deliberately refuses to count as a match (a look-alike photo
is not evidence that *this* photo was posted there).

**2. This is a consent tool, not an identification tool.** It is designed for a
subject verifying their *own* existing online presence. Pointing it at
non-consenting strangers would likely breach the search API's terms of service
and, in many jurisdictions, biometric privacy law (BIPA in Illinois, GDPR Art. 9
in the EU, India's DPDP Act) — where a face template is a special category of
personal data that generally needs explicit consent. Nothing in the code enforces
that; it is on the operator.

**3. Only a hash is anchored — deliberately.** Raw embeddings and matched content
stay off-chain by design. That keeps biometric data off a permanent public ledger,
but it also bounds what the anchor proves: it proves *this exact record existed at
this block time and has not been altered since*. It does not prove the record was
truthful when written. Anyone who can run the pipeline can anchor any record they
like; the guarantee is integrity and timestamping, not honesty.

**4. Local Hardhat is only verifiable by someone running the same local chain.**
The default chain lives on your machine and its state is wiped when the node
restarts, so a third party cannot independently confirm an anchor without
trusting you. That is the right trade-off for a zero-cost build, and it is
genuinely tamper-evident *within* a session. For third-party verification, point
`RPC_URL` at a public testnet (Polygon Amoy) — then the transaction has a real
block-explorer URL anyone can check. That path is implemented and documented but
untested here, since it needs faucet funds a human must claim.

**5. Stage 4 depends on the CDN letting a script download the image.** Social
networks routinely answer `HTTP 403` to non-browser image requests. The pipeline
tries every candidate image URL in order and prints each attempt, but if all of
them are blocked it stops with exit 4 — the matched post URL from stage 3 is still
a real finding; only the automatic re-encode needs the bytes. Workaround: save the
image manually and compare the two files.

**6. One face per input.** Stage 1 refuses zero faces and refuses more than one,
rather than guessing which person you meant. Crop group photos first. Downloaded
match images *may* contain several faces — there the best-scoring one is used, and
the face count is recorded in the fingerprint.

## Verification status

Being explicit about what was actually executed during this build versus what is
complete but could not be run here.

**Verified end to end on this machine:**

- Stage 1 — real InsightFace inference: one face → `box=(75, 97, 182, 248)`,
  score `0.902`, 512-d embedding with L2 norm exactly `1.0`; zero-face and
  multi-face inputs rejected with clear messages.
- Stage 4 metric — same face at a different crop `0.013870` (PASS), different
  people `1.035786` (FAIL).
- Stages 5–8 — `FaceVerify` compiled with solc 0.8.28 and deployed to local
  Hardhat (chain 31337, 182,209 gas); commitment submitted (68,468 gas for the
  first record on a fresh chain, ~51,000 for each one after it); the
  `RecordSubmitted` event located and its block timestamp confirmed to agree with
  the stored value; the record re-read from disk, re-hashed and verified →
  `MATCH`; one field mutated → `TAMPERED`; restored → `MATCH`; re-submitting the
  same commitment left `totalRecords` at 1 and kept the original timestamp.
- Failure paths — no chain running, contract address with no code, unknown
  commitment, missing/multi-face image, and missing credential each produce a
  short message plus the documented exit code, never a traceback.
- `python -m pytest tests/ -q` — **89 passed** (62 offline + 12 real-inference +
  15 live-chain), with the Hardhat node running and the model pack downloaded.

**Complete but not executable here:**

- The live Vision API call. No Google Cloud account exists for this repo, so
  `GOOGLE_VISION_API_KEY` and `GOOGLE_APPLICATION_CREDENTIALS` are intentionally
  blank in `.env.example`. Both credential styles, the request body, the response
  parsing, and every documented HTTP error code are implemented, and the parsing
  is unit-tested against mock JSON — but the round trip itself has never run.
  Adding a key requires no code change.
- Deployment to Polygon Amoy, for the same reason: it needs a funded wallet and
  faucet claim that only a human can do.








