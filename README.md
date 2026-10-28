# Hierarchical QUIC domain classifier

A PyTorch implementation of the supplied five-stage architecture. It predicts a domain label from one observed QUIC connection. This is trainable source code, not a pretrained model. Supply labeled traffic to measure domain classification performance.

## Architecture

```text
Ordered UDP datagrams, grouped into directional bursts
   |
Packet MLP: 76 -> 64 -> 64, shared across datagrams
   |
Within each burst: bidirectional GRU (64 per direction) + masked attention
   | 128-dimensional burst embedding
Concatenate 36 normalized burst features and observation flags
   |
Across bursts: bidirectional GRU (64 per direction) + masked attention
   | 128-dimensional session embedding
   |                                  Connection QUIC features (72)
   |                                             |
   |                                  MLP: 72 -> 64
   +---------------- concatenate ----------------+
                          | 192 dimensions
                    MLP: 192 -> 64 -> K
                          |
                 Softmax over K domains
```

For datagram t in burst b, z_bt = PacketMLP(x_bt). A bidirectional GRU processes z_b1 ... z_bT in order. Additive attention computes a weighted sum of its states. A second bidirectional GRU processes the ordered burst embeddings concatenated with burst statistics; a second attention layer produces the session embedding. The final classifier combines that embedding with the separate QUIC feature encoding. Training uses raw logits and weighted cross-entropy; softmax is applied only for prediction.

The implementation uses the Softmax option from the feature list. `model(rows, return_embedding=True)` exposes the fused 192-dimensional vector for a future XGBoost experiment. An XGBoost training pipeline is not included.

## Input boundary and label definition

Input is JSONL: one JSON object per connection, with `connection_id`, `group_id`, `domain`, and an ordered `datagrams` list. Prediction needs only `datagrams`; `connection_id` is useful for identifying results. Each datagram contains `direction` (`c2s` or `s2c`), `length` (UDP payload bytes), and an ordered `packets` list. Packet objects contain `type` and `length` (the entire QUIC packet, including its header). Packet lengths plus any `udp_padding_length` must sum to the datagram length. See `example.jsonl` for a synthetic format example, not training data.

Supported types: Initial, Handshake, 0-RTT, 1-RTT, Retry, VN. A version-aware upstream parser must decode packet types and split coalesced datagrams correctly. Retry/VN are standalone; a short-header packet must be last in a coalesced datagram. Unsupported versions and malformed/truncated packets must be handled by the upstream parser rather than silently assigned an incorrect type.

Datagrams may also contain `udp_padding_length` (default zero) for appended
UDP-level padding explicitly identified by TShark as `quic.coalesced_padding_data`.
QUIC packet lengths plus this padding must equal the UDP payload length. This is
distinct from encrypted QUIC PADDING frames: it contributes to datagram and burst
byte counts, but not to QUIC packet counts, packet lengths, or Initial frame ratios.

Optional server Initial fields:

- `scid_length`: integer 0..20, or null/missing when unavailable. CID bytes are never used.
- `initial_frames`: the complete decrypted Initial frame list, or null/missing when decryption is unavailable. An empty list means successfully observed with no frames. Each frame has `type`; CRYPTO has a `length` equal to its data byte count, and PADDING has a `length` equal to its padding byte count. ACK/PING only require their type. Consecutive padding can be represented as one PADDING entry.

`pcap_pipeline.py` supplies this input boundary using TShark for capture reading, QUIC connection tracking, packet dissection, and Initial decryption. Server Initial decryption requires connection context from the client's Initial, with version-specific key derivation and Retry handling. No TLS key log is needed for Initial packets. Handshake contents are not used. Consult [RFC 9000 section 12.2](https://www.rfc-editor.org/rfc/rfc9000.html#section-12.2) and [RFC 9001 section 5.2](https://www.rfc-editor.org/rfc/rfc9001.html#section-5.2).

Define `domain` consistently (for example, exact requested hostname or registrable domain) in your collection pipeline. Use independent ground truth such as controlled browsing logs. Hostnames, SNI, DNS names, IP addresses, payload contents, and CID values are excluded from model inputs. A connection serving multiple domains cannot be represented faithfully by a single connection label; exclude those connections or explicitly choose a different labeling/task formulation.

## Feature mapping

Each distribution is summarized by count, mean, population standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum. These summaries are a deliberate fixed-size approximation to a distribution.

| Encoder | Raw feature groups | Raw dimensions |
|---|---|---:|
| Datagram/packet MLP | UDP length; direction; QUIC packet count; six packet-type presence bits (also identify coalesced type sets); server Initial and Handshake length summaries; server Initial CRYPTO length summaries (including frame count); PADDING ratio; CRYPTO/ACK/PADDING/PING presence | 38 |
| Burst metadata | Datagram count, total bytes, datagram length summaries, six type-presence bits, coalescing ratio, direction | 18 |
| Session QUIC | First server Initial SCID length; server Initial and Handshake length summaries (including counts); Handshake total bytes; server coalescing ratio; first server datagram type bits; Retry/VN presence; server 1-RTT-containing datagram length summaries; directional byte share | 36 |

Every raw feature is accompanied by an observed-value flag, doubling the input dimensions. Missing values are imputed to the training mean after signed log1p transformation. Normalization is fitted on training connections only and saved in the checkpoint. No matching packets in the observed window means a zero count; unavailable decryption means missing frame measurements, not zero frame counts.

PADDING ratio = server Initial PADDING bytes / total server Initial packet bytes in that datagram. Directional byte share = server bytes / (server + client bytes), using whole UDP payload lengths for datagrams containing 1-RTT; it is missing if no such bytes occur. This bounded share is the implemented definition of the requested directional byte ratio. Coalescing ratio = fraction of datagrams containing more than one QUIC packet. Type composition means presence, not frame or packet frequencies.

A burst is a maximal consecutive run of datagrams in one direction. There is no time-gap split. The packet BiGRU learns ordered datagram-size/type patterns within a burst. Burst count/direction accompany each ordered burst embedding into the session BiGRU, representing burst-length and directional sequences. Handshake packet-type order is encoded through these ordered packet and burst representations; there is no additional handshake-only encoder. Exact ordering of coalesced packets within one datagram is reduced to a type-presence set. Individual CRYPTO frame lengths are summarized, not fed to an extra sequence encoder.

All branches observe the same first `--max-datagrams` datagrams (default 512), preventing later session summaries from leaking beyond the sequence observation window. The last retained burst may be partial. Counts describe the observed prefix, not an assumed complete connection. The bidirectional architecture operates after collecting this window; it is not a causal per-packet streaming model.

## Install and run

### Extract from PCAP or PCAPNG

Install [Wireshark including TShark](https://www.wireshark.org/download.html) separately (4.2+ recommended), then install `requirements.txt`. TShark is a native executable, not a pip dependency. On Windows the extractor also checks `C:\Program Files\Wireshark\tshark.exe`; elsewhere put `tshark` on PATH or use `--tshark PATH`.

```sh
python pcap_pipeline.py capture.pcapng --output connections.jsonl
python run.py predict --input connections.jsonl --checkpoint model.pt

# Controlled capture: every emitted connection has this known ground-truth label.
python pcap_pipeline.py visit1.pcap visit2.pcapng --output train.jsonl --domain example.org --group-id collection-001

# Explicit client role, and QUIC on a nonstandard UDP port:
python pcap_pipeline.py capture.pcap --output connections.jsonl --client-address 192.0.2.10 --port 8443
```

The pipeline reads TShark PDML incrementally, groups by TShark's QUIC connection number, and retains capture order. It supports IPv4/IPv6 and QUIC v1/v2, including Initial, Handshake, 0-RTT, 1-RTT, Retry, and Version Negotiation. Each QUIC packet keeps its full wire length, independently of other packets coalesced into the same UDP payload. Server Initial frame trees supply CRYPTO data lengths, PADDING byte counts, and ACK/PING presence. Unavailable decryption stays `null`, so `features.extract()` generates missing-value masks instead of false zero measurements. The existing train/predict commands convert the emitted records into the 38/18/36 raw feature dimensions and apply checkpoint normalization.

Direction is inferred only when the first observed datagram contains an Initial with a decoded TLS ClientHello. Otherwise supply `--client-address` (repeat for multiple clients). Port numbers alone never determine direction. Capture the handshake to recover server Initial frame measurements. Connection grouping and Retry context depend on TShark recognizing the traffic; `--port` forces QUIC dissection where automatic recognition fails. Only packets dissected as QUIC are processed.

The default is strict: unsupported QUIC versions, truncation, malformed packets, ambiguous roles, unaccounted payload bytes, and endpoint migration fail extraction. Tunnels with multiple network layers and IPv6 UDP jumbograms are not supported. `--skip-invalid` discards an entire identifiable connection after an error, including its previously buffered datagrams, and reports rejection reasons. Packets without a connection identifier can only be discarded individually. This mode can bias a dataset; inspect its report. Output is replaced only after every capture succeeds and at least one usable connection exists.

`--max-datagrams` defaults to 512 per connection. TShark still processes the full capture to retain protocol context, but only the prefix is saved; later errors can still invalidate a connection. Set this limit at least as high as the model's observation window. Memory scales with the retained prefixes across connections. Reports are printed as JSON to stdout.

Connection IDs combine the capture's SHA-256 hash with its TShark connection number; duplicate input files are rejected. By default the capture hash also becomes `group_id`. Set a shared `--group-id` for related captures from the same collection unit to prevent split leakage. Recaptures/re-encodings of the same traffic require collection-level deduplication. `--domain` labels **all** connections in the supplied captures, so use it only with matching controlled ground truth; for mixed-domain captures, join independent collection labels to the generated connection IDs. SNI/DNS names are never used as labels or features. Endpoint addresses and CID values are not written to the JSONL.

The parser uses documented [TShark PDML output](https://www.wireshark.org/docs/man-pages/tshark.html) and [QUIC dissector fields](https://www.wireshark.org/docs/dfref/q/quic.html). Synthetic PDML tests cover parser behavior and compatibility with the feature extractor; a real capture smoke test also requires the external TShark installation.

### Extract an exported part directory

```sh
python pcap_pipeline.py part-000001 --output connections.jsonl
# Multiple parts can be combined in one invocation:
python pcap_pipeline.py part-000001 part-000002 --output combined.jsonl
```

Directory inputs use `data/verified/<mode>/<domain>/<attempt>/receipt.json`.
The direct layout `data/verified/<defense>/<domain>/<id>/pcaps/traffic.pcapng`
is also supported, with optional receipts. Defense names are unrestricted; all
defense folders are processed. Without a receipt, the domain folder supplies the
label for every connection in the capture, and matching domain/session IDs share
a group across defenses and parts. These captures must contain traffic for that
domain. Client direction is inferred from the ClientHello or supplied with
`--client-address`; receipt-based five-tuple filtering is unavailable without a
receipt. If a receipt is present, its metadata is validated and used, with the
direct canonical capture taking precedence over its archived capture location.
Each verified receipt selects its canonical `pcaps/<artifact_paths.pcap>` file
(`traffic.pcap` or `traffic.pcapng`); diagnostic captures and pending sessions
are excluded. Missing captures, invalid receipts, mismatched folder labels, and
paths outside the session fail extraction. Explicit PCAP inputs remain supported.

The receipt supplies the domain label and client address. Only packets matching
its UDP five-tuple in either direction are retained, preventing unrelated traffic
from receiving the session label. The receipt's server UDP port is explicitly
decoded as QUIC, avoiding collisions with other protocols registered on client
ports; explicit `--port` options override this default.
Labels are collection metadata, not model inputs.
The default group ID hashes the study, domain, and session slot, keeping matching
slots across modes together. Use `--group-id` for broader dependent collection
units. `--domain` must agree with receipt labels when supplied. Duplicate capture
contents are rejected, and output is replaced only after extraction succeeds.

The resulting JSONL contains parsed connection measurements in the existing
classifier schema. Training and prediction compute packet, burst, and session
feature vectors from these records and apply training-fitted normalization.
TShark is required for extraction; folder metadata alone cannot supply the features.
On Windows, capture paths of 260 characters or more are copied temporarily to a
short path for TShark compatibility. Source captures and their digest-based IDs
are unchanged, and the temporary copy is removed after extraction.

### Train and predict

From this directory, in a Python 3.10+ environment:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python run.py train --train train.jsonl --valid valid.jsonl --checkpoint model.pt
python run.py evaluate --input test.jsonl --checkpoint model.pt
python run.py predict --input unlabelled.jsonl --checkpoint model.pt
```

Use `--device cuda` when a compatible GPU/PyTorch build is available. Training supports batch size, learning rate, seed, epochs, patience, and maximum datagrams; see `python run.py train --help`. It uses AdamW, dropout, gradient clipping, inverse-frequency class weights, and early stopping on validation macro-F1. Best weights, preprocessing, labels, model dimensions, and window size are saved together. Evaluation prints accuracy, balanced accuracy, macro and support-weighted precision/recall/F1, sample count, per-domain precision/recall/F1/support, and a confusion matrix (rows = true, columns = predicted, in the reported label order). Each training epoch prints the same full report under `validation_metrics`; early stopping still uses macro-F1. Macro averages include every checkpoint class; weighted averages use ground-truth support. Balanced accuracy averages recall over classes present in the evaluated ground truth. Undefined per-class scores are zero, including for absent classes. Metrics require a nonempty labeled dataset. Prediction prints one JSON object per connection with domain probabilities. Probabilities are not calibrated confidence estimates.

Supply separate train, validation, and test files. `group_id` must identify a whole dependent collection unit, such as a browsing run or capture batch, rather than each connection. The CLI rejects overlapping group/connection IDs and duplicate connection IDs. Validation must include all training classes. Test evaluation rejects training/validation groups and unknown labels; this is a closed-set classifier. ID checks cannot detect duplicated traffic with relabeled IDs, so deduplicate during collection. Use temporal and collection-environment holdouts to test generalization beyond collection artifacts. All records and features are loaded into memory; larger datasets need a streaming dataset implementation.

For deployment, compare against a session-feature-only baseline and run feature ablations (Initial decrypted features, SCID length, and full hierarchy). Domain accuracy cannot be inferred from the architecture: shared hosting, traffic variation, and connection coalescing can make classes difficult to distinguish. Open-set domain rejection requires separately designed and validated thresholds or a rejection model.

## Verification

Verified with Python 3.12 and PyTorch 2.14.1 CPU: seven tests cover feature calculations, missing-value semantics, observation-window consistency, malformed input, prediction invariance to batch padding, gradient propagation/loss reduction, and group-leakage rejection. A separate synthetic CLI smoke test exercised training, checkpoint loading, evaluation, and prediction. Synthetic tests validate execution only; they do not establish accuracy on real domains.
