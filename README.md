# Subnet 102 — Distributed Mixture-of-Experts

Subnet 102 is a **Bittensor subnet** for collaborative, decentralized training of large language models using a Mixture-of-Experts (MoE) architecture. Miners train small residual experts beside a frozen base model (Qwen3.6-35B-A3B); validators score each miner's residual against the pretrained base, publish each round's best as the next round's starting point, and set weights that drive TAO rewards.

## Documentation

For the most up-to-date and comprehensive documentation, including architecture overviews, setup guides for validators and miners, and reward optimization, please visit our official website:

**[Connito AI](https://connito.ai/)**

- [Running Your Miner](https://connito.ai/docs/miner)
- [Running Your Validator](https://connito.ai/docs/validator)


## Contributing & License

Contributions are welcome. Please open an issue first to discuss significant changes.

See [LICENSE](LICENSE) for details.
