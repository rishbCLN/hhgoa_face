/**
 * Hardhat config for the FaceVerify contract.
 *
 * Only two tasks are needed by this project:
 *   npx hardhat compile   -> writes artifacts/contracts/FaceVerify.sol/FaceVerify.json
 *   npx hardhat node      -> local JSON-RPC chain on http://127.0.0.1:8545 (chainId 31337)
 *
 * The local node ships 20 pre-funded, *unlocked* accounts, so the Python side can
 * send transactions via eth_sendTransaction without holding a private key.
 * Deployment to a public testnet is handled entirely on the Python side
 * (blockchain/chain.py reads RPC_URL + PRIVATE_KEY), so no network entries or
 * secrets are needed in this file.
 */

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: { enabled: true, runs: 200 },
    },
  },
  networks: {
    hardhat: {
      chainId: 31337,
    },
  },
  paths: {
    sources: "./contracts",
    artifacts: "./artifacts",
    cache: "./cache",
  },
};
