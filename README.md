<p align="center">
  <img src=".github/assets/logo-full-readme.png" alt="LunarGate" width="420" />
</p>

<div align="center">
Self-hosted open-source AI gateway with OpenAI-compatible APIs, Ollama upstream support, routing, fallback, and observability-friendly data sharing.

[Docs](https://docs.lunargate.ai) | [API Reference](https://docs.lunargate.ai/reference/http-api/) | [Examples](https://github.com/lunargate-ai/gateway-examples) | [Website](https://lunargate.ai)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/lunargate-ai/gateway/blob/main/LICENSE)
</div>

---

> [!IMPORTANT]
> LunarGate is under active development. The core gateway is already usable today, but interfaces and configuration may still evolve.

LunarGate is a Go-based AI gateway that lets you expose one stable endpoint to your applications while routing requests to different upstream model providers behind the scenes.

## What LunarGate does

- Exposes an OpenAI-compatible API for client apps
- Routes requests across providers and models with weighted or conditional rules
- Adds retries, fallbacks, and circuit breakers at the gateway layer
- Supports streaming and tool-calling flows
- Keeps a small footprint: the standalone binary is typically around 10-12 MB depending on platform, and the container image is around 12 MB
- Keeps data local by default, with optional prompt/response sharing for observability
- Works with providers such as OpenAI, Anthropic, Ollama, and other OpenAI-compatible local backends

## Supported endpoints

- Chat Completions: create, stored list/retrieve/update/delete, and message listing under `/v1/chat/completions`
- Responses: JSON/SSE `POST /v1/responses`, WebSocket `GET /v1/responses`, retrieve/delete/cancel/input-items, compaction, and input-token counting
- Conversations: create/retrieve/update/delete and item create/list/retrieve/delete under `/v1/conversations`
- `POST /v1/embeddings`
- `GET /v1/models` and `GET /v1/models/{model}`
- Operational endpoints: `GET /health`, `GET /ready`, and `GET /metrics`

Stored Chat Completions lifecycle routes are fail-closed. Enable
`providers.<id>.capabilities.chat_completions_lifecycle` only for an exact
OpenAI-compatible upstream account that implements those routes.

Native Responses lifecycle, cancellation, compaction, input-token counting,
and Conversations require the matching provider capability. Responses sent
to a native upstream also require a route target with
`upstream_request_type: responses`.

## Provider support

LunarGate exposes an OpenAI-compatible client-facing API, but it can route to multiple upstream provider types behind the scenes, including:

- OpenAI
- Anthropic
- Ollama
- other OpenAI-compatible backends

That means your app can keep one stable client integration while the gateway talks to the upstream provider in the format it expects.

## Documentation

Full documentation lives at [docs.lunargate.ai](https://docs.lunargate.ai).

Useful starting points:

- [Quickstart](https://docs.lunargate.ai/getting-started/quickstart/)
- [Routing and fallback](https://docs.lunargate.ai/guides/routing/)
- [Configuration reference](https://docs.lunargate.ai/reference/configuration/)
- [Runnable examples](https://github.com/lunargate-ai/gateway-examples)

## Quick start

Choose one install path:

- [Homebrew on macOS](#homebrew-macos)
- [Install script on Linux](#install-script-linux)
- [Build from source](#build-from-source)

### Homebrew (macOS)

```bash
brew tap lunargate-ai/tap
brew install lunargate
```

### Install script (Linux)

```bash
curl -fsSL https://get.lunargate.ai/install.sh | sh
```

### Build from source

```bash
make build
```

## Minimal configuration

Create `config.yaml`:

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    default_model: "gpt-5.6-terra"

routing:
  routes:
    - name: "default"
      targets:
        - provider: openai
          model: "gpt-5.6-terra"
          weight: 100
```

If you use environment placeholders such as `${OPENAI_API_KEY}`, either export them in your shell or place them in a local `.env` file:

```bash
OPENAI_API_KEY=sk-xxxxxxxxx
```

Then start the gateway:

```bash
lunargate --config ./config.yaml
```

If you omit `--config`, LunarGate will look for `config.yaml` in the current directory.

## Call the gateway

Point your OpenAI-compatible client at:

```text
http://localhost:8080/v1
```

For a runnable client example, see [`gateway-examples/`](https://github.com/lunargate-ai/gateway-examples).

## Observability

By default, LunarGate does not connect to the Dashboard collector or remote-control channel, so ordinary client prompts and responses stay inside your infrastructure.

LunarGate does perform a separate automatic release check. Its JSON payload contains only the running version and CPU architecture and has no installation identifier. Disable requests to `https://get.lunargate.ai/latest` completely with:

```yaml
update_check:
  enabled: false
```

If you want to connect the gateway to LunarGate observability, create a gateway in the `Gateways` section of [app.lunargate.ai](https://app.lunargate.ai) and add the generated gateway API key to your environment:

```bash
LUNARGATE_GATEWAY_API_KEY=lgw_your_gateway_api_key
```

Then configure the shared Dashboard connection and enable `data_sharing` in `config.yaml`:

```yaml
general:
  api_key: "${LUNARGATE_GATEWAY_API_KEY}"

data_sharing:
  enabled: true
  share_prompts: true
  share_responses: true
  remote_control: true
```

Keep `share_prompts` and `share_responses` off for metrics-only collector export. `remote_control` is a separate privacy boundary: each Dashboard sandbox command carries its complete request to the gateway and returns the complete bounded response through the control channel, regardless of those two collector flags. Leave `remote_control: false` unless you want that feature.

## Security

Basic inbound API-key authentication is available via the `security` section in `config.yaml`.

For now, treat that as a solid first layer for trusted internal clients, not as a finished public multi-tenant edge. If the gateway is internet-facing, keep an auth-enforcing proxy, API gateway, or service mesh in front of it.

## License

MIT
