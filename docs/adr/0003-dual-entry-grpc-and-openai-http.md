# ADR-0003: 双入口——gRPC Runtime Interface + HTTP OpenAI 兼容面

- **状态**: 已接受（M4 收官成为正式形态）
- **日期**: 2026-08-07
- **上游依据**: Ultralisk ADR-010（Backend Runtime）

## Context

Zealot 需要对平台 Gateway 提供 Runtime Interface gRPC 契约，同时对终端用户提供 OpenAI 兼容 API。早期设计中 HTTP OpenAI 面「在 Gateway 完成」，这会把协议兼容逻辑散到平台层，且无法在 Zealot 内独立验证 OpenAI 语义（`response_format` / `tools` 等）。

## Decision

Zealot Backend 进程内提供双入口：tonic gRPC Server（Runtime Interface，默认 :9091）+ axum HTTP OpenAI 兼容面（`/v1/chat/completions`，非流式 + SSE）。HTTP 面在 Zealot 内完成（`src/bin/zealot-backend.rs` + `src/openai.rs`），不经 Gateway 转发。Runtime Interface proto 契约（`proto/runtime/`）随本仓库独立维护。

## Consequences

- structured output（`response_format: json_schema`）与 tool calling（`tools`/`tool_choice`）的约束解码接入在 Zealot 内闭环，可独立 e2e 测试（M5-④，issue #18）。
- 错误面统一为 OpenAI 风格 400（schema 缺失/编译失败/未知工具/组合冲突）。
- proto 契约随引擎版本演进，不依赖平台仓库发版节奏。
