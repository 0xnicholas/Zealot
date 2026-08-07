# Changelog

All notable changes to this project are documented in this file.

Format: 语义化版本 + 变更类型（Added / Changed / Fixed / Removed）。
发布 tag 命名：`zealot-vX.Y.Z`（与平台 tag 并行，互不干扰）。

## [Unreleased]

- M5-⑨（P2）：不支持关键字显式报错（issue #23）——compile_schema 统一守卫：
  `not`/`if`/`then`/`else`/`dependentSchemas`/`dependentRequired`/`uniqueItems`/
  `contains`/`minContains`/`maxContains`/`propertyNames`/`patternProperties`/
  `unevaluatedProperties`/`unevaluatedItems`/`prefixItems`/`minProperties`/
  `maxProperties`/`const`/`$ref`/`allOf` → 编译报错「暂不支持」（unsupported >
  unsound，不再静默降级 Any）；`additionalProperties` 仅布尔形态、`items` 仅单
  schema 形态（tuple 数组/布尔拒绝）；嵌套属性内同样生效（整体传播）；注解/
  元关键字（title/description/default/examples/readOnly/writeOnly/deprecated/
  $comment/$id/$schema/$defs/definitions/contentEncoding/contentMediaType）
  不约束值 → 放行
- M5-⑧：string pattern 执行（issue #22）——新 regex.rs 引擎：解析（字面量/字符类/
  `.`/转义 `\d\w\s` 等/量词 `*+?{n,m}`/分组/交替）→ Thompson NFA → 子集构造
  DFA → 死状态剪枝；**锚定语义**（业界惯例：llama.cpp 要求 ^…$、outlines #1083）——
  pattern 约束整个字符串，首尾 `^`/`$` 接受并剥离（其余位置报错）；不支持特性
  （环视/反向引用/词边界/命名组/原子组/量词叠加）显式报错不静默；懒量词接受
  （语言等价）；NFA/DFA 状态上限防爆；与 minLength/maxLength 乘积组合
  （复用 format 模板机制 compile_anchored_template）；pattern+format 互斥报错；
  语言含裸 `\`/`"`（JSON 字符串无法表示）→ 显式报错
- M5-⑦：string format 子集（issue #21）——email（local 点不可首尾/连续 +
  域名 label）、date（YYYY-MM-DD，月 01-12 日 01-31 结构校验）、date-time
  （date T time，T/t 分隔，裸日期不可收尾）、time（HH:MM:SS(.f+)?(Z|[+-]HH:MM)?）、
  uuid（8-4-4-4-12 hex）、ipv4（四段 0-255 无前导零）、ipv6（8 组 hex +
  至多一个 `::` 压缩，RFC 4291 结构，不含内嵌 IPv4）、hostname（点分 label
  字母数字首尾）、uri（scheme:非空白/控制字节宽松子集）；与 minLength/maxLength
  乘积组合（状态预算超限显式报错）；不支持 format → 显式编译错误（P2 语义，
  不静默降级）；format 内容不允许转义（`\`/`"` 不在字符集中）
- M5-⑥：string minLength/maxLength 执行（issue #20）——长度计数自动机：
  content_k/esc_k/u1..u4_k（k = 已消费码点数）；码点计数（ASCII/首字节 +1、
  UTF-8 连续字节不计、`\x` 转义 +1、`\uXXXX` +1 且严格 hex 链，代理对按 2 计
  ——文档化偏差）；min 门控收尾 `"`（k ≥ min），max 封顶（k == max 仅可收尾，
  不可再续）；仅 min 时 k == min 冻结为吸收态（任意续写不再计数）；无约束
  （min=0 且无 max）保持紧凑形态零开销；空区间（min > max）/超大上限
  （horizon > 32768）→ 显式编译错误；转义/内容边整块构建避免 O(n²)；
  pattern 组合（长度照常生效，pattern 执行随 #22）；工具参数长度约束端到端
  生效（mock HTTP e2e：minLength 5/maxLength 7 → `"nameage"`）
- M5-⑤：整数数值范围关键字执行（issue #19）——`GrammarType::Integer` 结构化
  （修死代码：integer 与 number 分离，integer 不再接受小数点/指数）；schema 解析
  `minimum`/`maximum`/`exclusiveMinimum`/`exclusiveMaximum`/`multipleOf`（draft-7
  数字形态 + draft-4 布尔形态，exclusive 取更严；小数 multipleOf 十进制展开 ≤6 位）；
  自动机层：有界区间十进制 digit-DP（状态 (pos, ca, cb, rem)，ca/cb 为前缀与
  上下界数字串的 LT/EQ/GT 比较，rem 为 multipleOf 余数；规范形式无前导零，
  负区间 `-`+幅度，`-0` 拒绝），纯 multipleOf 用余数 DFA；活状态剪枝——
  「可到达但必死」前缀（如 [0,100] 的 101）不建不接，杜绝 mask 全空 → 引擎
  fallback 垃圾；空区间/超大 multipleOf（预算超限）→ 显式编译错误（不静默），
  automaton 编译错误透传（不再一律 SchemaTooComplex）；`number` 类型带范围关键字
  → 显式报错（浮点区间暂不支持，请用 integer）；工具参数整数区间端到端生效
  （信封 walk + mock HTTP e2e 验证 [0,1] 取 1）
- M5-④：openai.rs response_format / tools 接入（issue #18）——ChatCompletionRequest
  新增 `response_format`（text/json_object/json_schema）与 `tools`/`tool_choice`；
  json_schema 经 CD 编译链路约束解码（strict 校验：根 object、全属性 required、
  `additionalProperties:false` 递归检查，失败 400）；强制 tool_choice（具名/required）
  编译工具信封 grammar（`{"name":<枚举>,"arguments":<参数 schema>}`，参数同样
  strict 校验，多工具 anyOf 并集）；json_object/auto/none 不约束（通配键 Phase 3）；
  schema.rs 新增 Rust 侧 compile_value（共享 LRU 缓存/计时/上限）；引擎收尾状态
  mask 为空时强制放行 eos（真实词表下 done 状态自然终止，杜绝 AllTokensMasked
  退化垃圾）；自动机 strict 语义——已满足的 required 键从后续键 trie 排除（required
  键唯一）、根出口剥离 token 间空白（嵌套值出口保留，承接 `Ġ,`/`Ġ}`）；dev mock
  runner（`mock:tiny` 固定 byte 词表 + 常量 logits，CI 无模型环境跑 HTTP 端到端）；
  错误面：schema 缺失/编译失败/未知工具/组合冲突 → OpenAI 风格 400
- M5-③：matcher 真实 token 级校验（issue #17）——GrammarType 编译为字节级
  自动机（stay 位集 + 边表 + final 标记，start 恒为 0）；`allowed_tokens` 按真实
  词表（Tokenizer::token_bytes）过滤、`advance` 按 token 字节真实映射状态；
  required 用计数器状态 k 表达（须按 required 数组顺序，`}` 只在 k==r 可达）、
  数组 min/max 计数器、键 trie、union 分支派发、token 间空白（byte-level `Ġ`
  前缀 token）；引擎 mask 钩子接入词表（有 tokenizer 即真实校验，无则 Phase-2
  占位）；自动机超上限（262144 状态）拒绝；PyO3 `load_tokenizer`；性能优化
  （DFA 压缩/状态缓存）留 Phase 3
- M5-②：约束解码参与真实 decode 路径（issue #16）——Sequence 携带 grammar
  状态字段；`Engine::step` 采样前应用 allowed-token mask（`Sampler::sample_with_mask`，
  mask 硬过滤优先于贪心/温度/top-k/p/penalty）；全屏蔽 fallback 为未约束采样
  （tracing warn，不中断生成）；EOS 以 `is_valid_final` 门控（未达收尾状态强制
  续写，可达则收尾 Stop）；grammar 状态随输出 token 推进（prefill 中间 chunk 不推进）
- M5-①：required 收尾语义 + 静默吞错修复 + 递归 state_count（issue #15）——
  `is_valid_final` 改为真状态机（state >= required.len()）；object/union 子
  schema 编译失败整体报错（no-silent-failures）；`count_states` 递归计入嵌套，
  max_states 检查上移到 compile_schema 层级覆盖嵌套/union 聚合
- M4 收官：schema 缓存真 LRU（issue #12，§12-5）——原 HashMap 随机驱逐
  （`keys().next()`）修正为 `lru` 0.18 `LruCache`（O(1) 命中提升/淘汰，
  cache_size 上限）；`cache_size=0` 明确为禁用缓存；新增命中/未命中/淘汰
  统计 getter；Rust 单测 + pytest 双面覆盖淘汰顺序（A,B,A,C → 必逐 B）
- M4 收官：CI 闸门（issue #13）——clippy `-D warnings` 转硬闸（移除
  continue-on-error，21 条既有告警清零 + 全仓 rustfmt 归一）；criterion
  bench job（§8.1 的 8 项每次 CI 跑，HTML 报告 artifact，阈值不达标不挂 CI）；
  pytest job（Python 3.12 + maturin build + wheel 安装 + pytest tests/）
- M4 关闭（issue #4 判定式全部达成）——§12 八项 + proptest 全勾选；
  architecture.md 双入口正式形态（gRPC Runtime Interface + HTTP OpenAI
  兼容面，不再写"在 Gateway 完成"）；README Block Manager / CD 状态同步
- M4 收官：docs/phase2_lifecycle.md（issue #11，§12-8）——构建 → 启动 →
  模型加载 → gRPC/HTTP 请求 → 流式返回 → 关闭全链路文档 + 实测通过
  （本地 tiny LLaMA，16 token 生成，SSE 6 帧）；发现并跟进 #26
  （model_loader 仅 LLaMA 系 / CPU runner 无 GQA）；README/测试
  的 tiny-random-gpt2 引用修正为 LLaMA 系
- M4 收官：proptest BlockManager 随机序列测试（issue #10）——模型对照法
  （free list 栈 + refcount/generation），逐操作断言无重复分配 / stale 拒绝 /
  refcount 归零才回池 / generation 递增 + 引用守恒；mutation 验证有效
- M4 收官：pytest 集成测试面落地（issue #9）——test_block_manager.py
  （接口一致性/错误转换/StaleHandle/池语义）+ test_constrained_decode.py
  （编译/缓存/边界/嵌套深度/对抗性 10001 属性 schema 拒绝 §12-6/超时守卫），
  22 用例全绿；README 修正 pytest 声明与过期 Python API 文档
- M4 收官：§8.1 FFI 微基准 8 项落地（issue #8）——criterion benches/
  BlockManager 6 项全部达标（allocate 36.8ns / free 38.4ns / reference
  10.1ns / stale 4.4ns / GIL 61ns），decode 2 项超阈值标风险
  （详见 architecture.md §8.1）；§12-3 allocate<500ns 已勾选
- M4 收官：Block Manager `loom` 并发模拟测试（issue #7）——free list
  分配/释放、refcount、generation 并发交错全覆盖，mutation 验证有效；
  运行 `ZEALOT_LOOM=1 cargo test --lib loom -- --test-threads=1`
- 项目独立化：Zealot 从 Ultralisk monorepo 拆分为独立闭源仓库
  （历史完整保留，48 commits + Runtime Interface proto 2 commits）
- Runtime Interface proto 契约随本仓库独立维护（`proto/runtime/v1/runtime.proto`）

## [0.1.0] — 初始开发版本

- Block Manager（generation-gated handles，防御跨请求 use-after-free）
- Constrained Decode（JSON schema → DFA 编译 + CPU 侧 token 校验）
- Scheduler（优先级队列、block 预算、OOM 抢占 recompute、chunked prefill）
- Engine 步进循环 + ModelRunner trait（CPU / PyO3 / CUDA 三种 runner）
- CUDA：tiled FlashAttention、PagedAttention（GQA）、RMS Norm、embedding、
  transpose、SiLU gate-mul + element-wise add kernels
- KV Cache（per-layer GPU 内存池 + block table）、BufferPool
- Sampler（temperature、top-k/p、repetition/frequency/presence penalties）
- zealot-backend：Runtime Interface gRPC server（:9091）+ HTTP OpenAI endpoint
- dev-mode CPU 端到端真实推理（PyO3 嵌入 torch）
