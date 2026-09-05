// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/**
 * @title FaceVerify
 * @notice Minimal, append-only anchor for face-match "fingerprint records".
 *
 * @dev Privacy by design: NO biometric data ever reaches this contract. The
 *      caller hashes its record off-chain (keccak256 over canonical JSON) and
 *      submits only the resulting 32-byte commitment. Face embeddings, the
 *      matched image and the matched URL all stay on the caller's machine.
 *      Because the chain stores only a one-way hash, an observer cannot learn
 *      anything about the subject, but anyone holding the original record can
 *      recompute the hash and prove the record existed at the anchored time.
 *
 *      Records are write-once: the first submission of a commitment wins and
 *      its timestamp is never overwritten, so an anchor cannot be back-dated or
 *      re-dated by a later caller.
 */
contract FaceVerify {
    /// @dev commitment => block timestamp of the first submission (0 == unknown).
    mapping(bytes32 => uint256) private records;

    /// @notice Number of distinct commitments anchored by this contract.
    uint256 public totalRecords;

    /// @notice Emitted the first time a commitment is anchored.
    event RecordSubmitted(
        bytes32 indexed commitment,
        address indexed submitter,
        uint256 timestamp,
        uint256 blockNumber
    );

    /// @notice Emitted when a commitment that already exists is submitted again.
    event RecordAlreadyAnchored(
        bytes32 indexed commitment,
        address indexed submitter,
        uint256 originalTimestamp
    );

    /**
     * @notice Anchor a commitment hash on-chain.
     * @param commitment keccak256 of the canonical JSON fingerprint record.
     * @return anchoredAt The timestamp now associated with `commitment`. For a
     *         fresh commitment this is `block.timestamp`; for one that already
     *         exists it is the *original* timestamp (state is left untouched).
     */
    function submit(bytes32 commitment) external returns (uint256 anchoredAt) {
        require(commitment != bytes32(0), "FaceVerify: empty commitment");

        uint256 existing = records[commitment];
        if (existing != 0) {
            emit RecordAlreadyAnchored(commitment, msg.sender, existing);
            return existing;
        }

        records[commitment] = block.timestamp;
        totalRecords += 1;
        emit RecordSubmitted(commitment, msg.sender, block.timestamp, block.number);
        return block.timestamp;
    }

    /**
     * @notice Read the anchor timestamp for a commitment.
     * @param commitment The 32-byte commitment to look up.
     * @return The block timestamp of the first submission, or 0 if this exact
     *         commitment was never anchored. A caller that recomputes its record
     *         hash and gets 0 back knows the local record no longer matches what
     *         was anchored (i.e. it was altered).
     */
    function get(bytes32 commitment) external view returns (uint256) {
        return records[commitment];
    }

    /**
     * @notice Convenience predicate over {get}.
     * @param commitment The 32-byte commitment to look up.
     * @return True if `commitment` has been anchored.
     */
    function exists(bytes32 commitment) external view returns (bool) {
        return records[commitment] != 0;
    }
}
