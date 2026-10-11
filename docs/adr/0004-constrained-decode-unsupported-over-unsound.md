# ADR-0004: 约束解码 schema→DFA，unsupported > unsound

- **状态**: 已接受（M4-M5 逐步强化）
- **日期**: 2026-08（M5-⑨ 定型，issue #23）
- **相关**: 架构文档 §6.2；issue #15/#17/#23

## Context

vLLM 用 Python `outlines` 库做 JSON schema 约束，有 GIL 开销。Zealot 在 CPU 侧编译 schema → DFA，每次采样时校验 token。核心语义抉择：遇到不支持/无法表达的 schema 特性时，是静默降级（按 Any 放行，产出可能违反 schema）还是显式报错。

## Decision

- schema 编译为字节级自动机；`allowed_tokens` 按真实词表（`Tokenizer::token_bytes`）过滤，`advance` 按 token 字节真实映射状态。
- **unsupported > unsound**：`compile_schema` 统一守卫，对不支持的关键字（`not`/`if`/`then`/`else`/`$ref`/`allOf`/`patternProperties` 等，完整清单见 `src/constrained_decode/schema.rs`）显式编译报错「暂不支持」，不静默降级为 Any。嵌套属性内同样生效，错误整体传播。
- 对抗性防护：`max_states` / `max_compile_ms` / 嵌套深度限制，超限拒绝（`SchemaTooComplex` / `SchemaCompileTimeout`）；「可到达但必死」的前缀不建不接，杜绝 mask 全空。
- 编译缓存为真 LRU（上限 + 最久未用淘汰 + 命中/淘汰统计），key 为 `DefaultHasher(schema_str)`。
- 引擎侧收尾状态 mask 为空时强制放行 EOS，杜绝 AllTokensMasked 退化垃圾。

## Consequences

- 用户得到明确错误而非静默错误的输出；JSON Schema Test Suite（draft-07 子集，151 用例）纳入回归，排除项均为设计性差异并记录于 `tests/json_schema_test_suite/SUBSET.md`。
- DFA 压缩/状态缓存等性能优化留 Phase 3。
- 支持面的扩展（P2 关键字）需逐关键字设计自动机语义，单独开票。
