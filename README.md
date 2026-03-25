<p align="center">
  <img src=".github/assets/logo-full-readme.png" alt="LunarGate" width="420" />
</p>

# LunarGate Gateway

Open-source, self-hosted AI gateway for OpenAI-compatible workloads.

## Status

This repository is still a work in progress.

## Documentation

Full documentation lives at `https://docs.lunargate.ai`.

## Quick Start

```bash
cp configs/config.example.yaml ./config.yaml
make build
./bin/lunargate
```

By default, the gateway looks for `config.yaml` in the current working directory.

If your `config.yaml` uses environment placeholders such as `${OPENAI_API_KEY}`, create a `.env` file in the same directory or export the variables before starting the gateway.

Example:

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
EOF
```

If you want to connect the gateway to LunarGate observability, go to `https://lunargate.ai`, create a gateway there, then copy its `gateway_id` and generated gateway API key into your environment:

```bash
cat >> .env <<'EOF'
LUNARGATE_BACKEND_URL=https://api.lunargate.ai/v1
LUNARGATE_GATEWAY_ID=gw_your_gateway_id
LUNARGATE_GATEWAY_API_KEY=lgw_your_gateway_api_key
EOF
```

With that in place, enable `data_sharing` in `config.yaml` and the gateway will start sending metrics/logs to LunarGate.

## Security

Inbound client authentication is not implemented yet.

Do not expose the gateway directly to the public internet without an auth-enforcing edge in front of it.

## License

MIT
