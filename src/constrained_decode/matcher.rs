use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::error::ZealotError;
use crate::tokenizer::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GrammarType {
    Any,
    String {
        min_length: usize,
        max_length: Option<u64>,
        pattern: Option<String>,
        enum_values: Option<Vec<String>>,
        /// M5-⑦：format 子集（内容字节级约束；None = 自由字符串）。
        format: Option<StringFormat>,
    },
    Number,
    /// M5-⑤：整数（含范围/整除约束；min/max 为有效闭区间，exclusive* 已在解析层折算）。
    Integer {
        min: Option<i64>,
        max: Option<i64>,
        multiple_of: Option<MultipleOf>,
    },
    Boolean,
    Null,
    Object {
        properties: HashMap<String, GrammarType>,
        required: Vec<String>,
        additional_properties: bool,
    },
    Array {
        min_items: usize,
        max_items: Option<u64>,
        item_type: Option<Box<GrammarType>>,
    },
    Enum {
        values: Vec<JsonValue>,
    },
    Union {
        branches: Vec<GrammarType>,
    },
}

/// M5-⑦：string format 常用子集（字节级约束）。未列出的 format → 编译错误（P2 语义）。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StringFormat {
    Email,
    Date,
    DateTime,
    Time,
    Uuid,
    Ipv4,
    Ipv6,
    Hostname,
    Uri,
}

/// M5-⑤：multipleOf 约束（含小数展开）。`v` 是 divisor 的倍数 ⟺
/// `(scale × v) % divisor == 0`（整数 multipleOf 时 scale = 1）。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultipleOf {
    pub divisor: u64,
    pub scale: u64,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct ConstrainedGrammar {
    pub grammar_type: GrammarType,

    #[pyo3(get)]
    state_count: usize,
    /// M5-③：字节级自动机（词表接入的真实校验核心；None = 编译失败被拒）
    automaton: Option<ByteAutomaton>,
    /// M5-⑤：自动机编译失败的具体原因（空区间/multipleOf 过大/超上限）——
    /// compile_schema 据此透出精确错误，而非一律 SchemaTooComplex。
    automaton_error: Option<String>,
    /// PyO3 存储路径的词表（引擎走参数传递，不占用此处）
    tokenizer: Option<Tokenizer>,
}

impl ConstrainedGrammar {
    pub fn new(grammar_type: GrammarType) -> Self {
        let state_count = count_states(&grammar_type);
        let (automaton, automaton_error) = match ByteAutomaton::compile(&grammar_type) {
            Ok(a) => (Some(a), None),
            Err(e) => (None, Some(e.to_string())),
        };
        Self {
            grammar_type,
            state_count,
            automaton,
            automaton_error,
            tokenizer: None,
        }
    }

    pub fn state_count(&self) -> usize {
        self.state_count
    }

    /// 自动机是否因编译失败被拒（compile_schema 据此报错）。
    pub fn automaton_rejected(&self) -> bool {
        self.automaton.is_none()
    }

    /// 自动机编译失败原因（None = 未失败）。
    pub fn automaton_error(&self) -> Option<&str> {
        self.automaton_error.as_deref()
    }

    /// 设置 PyO3 存储词表。
    pub fn set_tokenizer(&mut self, t: Tokenizer) {
        self.tokenizer = Some(t);
    }

    /// 当前状态允许的 token 集合（M5-③）。
    /// `tok`（或已存储词表）存在 → 真实词表 ID；否则 Phase-2 ASCII 占位语义。
    pub fn allowed_tokens_for(&self, state: usize, tok: Option<&Tokenizer>) -> Vec<i32> {
        let tok = tok.or(self.tokenizer.as_ref());
        match (self.automaton.as_ref(), tok) {
            (Some(a), Some(t)) => self.allowed_tokens_vocab(a, state, t),
            _ => {
                if state < self.state_count {
                    self.compute_allowed_tokens(state)
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// 采样后推进状态（M5-③ 真实映射：按 token 字节走自动机）。
    pub fn advance_state(&self, state: usize, token_id: i64, tok: Option<&Tokenizer>) -> usize {
        let tok = tok.or(self.tokenizer.as_ref());
        if let (Some(a), Some(t)) = (self.automaton.as_ref(), tok) {
            if let Some(bytes) = t.token_bytes(token_id) {
                if let Some(ns) = a.accept(state, &bytes) {
                    return ns;
                }
            }
            // 防御：token 不合法 → 保持原状态（mask 应当已阻止，兜底不崩）
            return state;
        }
        if state >= self.state_count {
            return state;
        }
        (state + 1).min(self.state_count - 1)
    }

    /// 当前状态是否可合法收尾（EOS 门控用）。
    pub fn is_valid_final_state(&self, state: usize, tok: Option<&Tokenizer>) -> bool {
        let tok = tok.or(self.tokenizer.as_ref());
        if let (Some(a), Some(_)) = (self.automaton.as_ref(), tok) {
            return a.is_final(state);
        }
        if state >= self.state_count {
            return false;
        }
        self.compute_is_valid_final(state)
    }

    fn allowed_tokens_vocab(&self, a: &ByteAutomaton, state: usize, tok: &Tokenizer) -> Vec<i32> {
        let mut out = Vec::new();
        for id in 0..tok.vocab_size() as i64 {
            if let Some(bytes) = tok.token_bytes(id) {
                if !bytes.is_empty() && a.accept(state, &bytes).is_some() {
                    out.push(id as i32);
                }
            }
        }
        out
    }
}

fn count_states(gt: &GrammarType) -> usize {
    match gt {
        // Any = 任意 JSON 环骨架（M5-⑩）：3 派发 + 3 套叶子 + 对象/数组骨架 ≈ 64
        GrammarType::Any => 64,
        GrammarType::String { .. }
        | GrammarType::Number
        | GrammarType::Integer { .. }
        | GrammarType::Boolean
        | GrammarType::Null => 1,
        // 递归计入嵌套属性（issue #15）：嵌套结构使 max_states 防护覆盖深层 schema
        GrammarType::Object { properties, .. } => {
            1 + properties.values().map(count_states).sum::<usize>()
        }
        GrammarType::Array { item_type, .. } => {
            1 + item_type.as_ref().map(|t| count_states(t)).unwrap_or(0)
        }
        GrammarType::Enum { values } => values.len().max(1),
        GrammarType::Union { branches } => branches.iter().map(count_states).sum(),
    }
}

// ── 字节级自动机（M5-③ 真实 token 级校验）────────────────────────────────────
//
// 每个 GrammarType 编译为字节级自动机：状态 = 「已消费字节在 grammar 中的位置」，
// 状态转移 = 字节（边表 + stay 位集）。`allowed_tokens(state)` 用真实词表
// （Tokenizer::token_bytes）过滤：token 的字节序列能被自动机完整消费才合法；
// `advance` 即同一次字节游走返回的新状态。
//
// 已知简化（Phase 3 处置，已文档化）：
// - 性能：状态未压缩、mask 未缓存（每次重建）——Phase 3 DFA 压缩/状态缓存
// - 转义序列宽松（`\x` 任意单字节视为合法转义）
// - Union 同一起始字节的分支取先匹配（未做子集构造合并）
// - EOS 门控是 token-id 级：含结构字节的多字节 token（如 `Ġ}`）可绕过 EOS 屏蔽

const MAX_AUTO_STATES: usize = 1 << 18; // 262144：压缩前防爆上限（超限 → SchemaTooComplex）
/// M5-⑥：string 长度计数上限（content/esc/u×4 共 6 状态 × horizon ≤ 预算）。
const MAX_LENGTH_COUNTER: usize = 32_768;
const MAX_ARRAY_COUNTER: usize = 1024; // 数组 min/max 计数器状态上限
const WS_BYTES: [u8; 4] = [0x20, 0x09, 0x0A, 0x0D]; // space / tab / LF / CR

/// 编译后的字节级自动机（M5-③）。
#[derive(Clone, Debug)]
pub struct ByteAutomaton {
    states: Vec<AutoState>,
}

#[derive(Clone, Debug)]
struct AutoState {
    /// 保持当前状态的字节位集（256 bit）
    stay: [u8; 32],
    /// 出边（按字节排序）；边优先于 stay（如 Any 值里 `,` 需跳转而非停留）
    edges: Vec<(u8, usize)>,
    /// Union 值位置：分支起始状态表（step 时逐支尝试，取先匹配）
    union_branches: Option<Vec<usize>>,
    is_final: bool,
}

impl ByteAutomaton {
    /// 从 GrammarType 编译（状态数超上限 → Err）。
    /// 保证 start 状态恒为 0（引擎/调用方以 state 0 为起始）。
    pub fn compile(gt: &GrammarType) -> Result<Self, ZealotError> {
        let mut b = AutoBuilder::new()?;
        let out = b.compile_value(gt)?;
        for &e in &out.exits {
            debug_assert!(b.states[e].is_final, "exit state must be final");
            // M5-④：根出口剥离 token 间空白——收尾后只允许 EOS（引擎侧强制放行），
            // 避免 ws 自循环把序列拖到 Length。嵌套值出口的 ws（Ġ, / Ġ} 承接）
            // 由 compile_value 统一添加，不受影响。
            for &w in &WS_BYTES {
                b.clear_stay(e, w);
            }
        }
        // 重编号：start → 0（其余按序移位），统一起始状态约定
        let n = b.states.len();
        let mut remap = vec![usize::MAX; n];
        remap[out.start] = 0;
        let mut next = 1;
        for slot in remap.iter_mut() {
            if *slot == usize::MAX {
                *slot = next;
                next += 1;
            }
        }
        let mut states: Vec<AutoState> = (0..n)
            .map(|_| AutoState {
                stay: [0u8; 32],
                edges: Vec::new(),
                union_branches: None,
                is_final: false,
            })
            .collect();
        for (old, st) in b.states.iter().enumerate() {
            let ns = &mut states[remap[old]];
            ns.stay = st.stay;
            ns.is_final = st.is_final;
            ns.union_branches = st
                .union_branches
                .as_ref()
                .map(|bs| bs.iter().map(|&x| remap[x]).collect());
            ns.edges = st.edges.iter().map(|&(b, t)| (b, remap[t])).collect();
        }
        Ok(Self { states })
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn is_final(&self, s: usize) -> bool {
        self.states.get(s).is_some_and(|st| st.is_final)
    }

    /// 单字节转移；None = 非法。
    pub fn step(&self, s: usize, b: u8) -> Option<usize> {
        let st = self.states.get(s)?;
        if let Ok(i) = st.edges.binary_search_by_key(&b, |e| e.0) {
            return Some(st.edges[i].1);
        }
        if stay_contains(&st.stay, b) {
            return Some(s);
        }
        if let Some(branches) = &st.union_branches {
            for &bs in branches {
                if let Some(ns) = self.step(bs, b) {
                    return Some(ns);
                }
            }
        }
        None
    }

    /// 消费一段字节序列（一个 token）；成功返回终态，失败 None。
    pub fn accept(&self, s: usize, bytes: &[u8]) -> Option<usize> {
        let mut cur = s;
        for &b in bytes {
            cur = self.step(cur, b)?;
        }
        Some(cur)
    }
}

fn stay_contains(stay: &[u8; 32], b: u8) -> bool {
    stay[(b >> 3) as usize] & (1 << (b & 7)) != 0
}

struct AutoBuilder {
    states: Vec<AutoState>,
}

/// 编译结果：起始状态 + 值完整后的出口状态集（父级在这些状态上接分隔符边）。
struct AutoOut {
    start: usize,
    exits: Vec<usize>,
}

impl AutoBuilder {
    fn new() -> Result<Self, ZealotError> {
        Ok(Self { states: Vec::new() })
    }

    fn new_state(&mut self) -> Result<usize, ZealotError> {
        if self.states.len() >= MAX_AUTO_STATES {
            return Err(ZealotError::SchemaTooComplex {
                states: MAX_AUTO_STATES,
                limit: MAX_AUTO_STATES,
            });
        }
        let s = self.states.len();
        self.states.push(AutoState {
            stay: [0u8; 32],
            edges: Vec::new(),
            union_branches: None,
            is_final: false,
        });
        Ok(s)
    }

    fn add_edge(&mut self, from: usize, b: u8, to: usize) {
        let edges = &mut self.states[from].edges;
        match edges.binary_search_by_key(&b, |e| e.0) {
            Ok(_) => { /* 已存在：保留先写入的目标 */ }
            Err(i) => edges.insert(i, (b, to)),
        }
    }

    fn add_stay(&mut self, s: usize, b: u8) {
        self.states[s].stay[(b >> 3) as usize] |= 1 << (b & 7);
    }

    fn clear_stay(&mut self, s: usize, b: u8) {
        self.states[s].stay[(b >> 3) as usize] &= !(1 << (b & 7));
    }

    fn add_ws(&mut self, s: usize) {
        for &w in &WS_BYTES {
            self.add_stay(s, w);
        }
    }

    /// 边界状态：允许 token 间空白（byte-level BPE 的 `Ġ` 前缀 token 依赖此边）。
    fn boundary(&mut self, s: usize) {
        self.add_ws(s);
    }

    fn set_final(&mut self, s: usize) {
        self.states[s].is_final = true;
    }

    /// 字面量链（前缀共享）；返回（根, 各字面量终点）。
    fn literal_trie(&mut self, lits: &[&[u8]]) -> Result<(usize, Vec<usize>), ZealotError> {
        let root = self.new_state()?;
        let mut ends = Vec::new();
        for lit in lits {
            let mut cur = root;
            for (i, &b) in lit.iter().enumerate() {
                let next = match self.states[cur].edges.iter().find(|(bb, _)| *bb == b) {
                    Some(&(_, to)) => to,
                    None => {
                        let n = self.new_state()?;
                        self.add_edge(cur, b, n);
                        n
                    }
                };
                cur = next;
                if i == lit.len() - 1 {
                    self.set_final(cur);
                    ends.push(cur);
                }
            }
        }
        Ok((root, ends))
    }

    /// 编译一个值 grammar；出口状态统一加 ws（值后空白，如 `Ġ,` token）。
    fn compile_value(&mut self, gt: &GrammarType) -> Result<AutoOut, ZealotError> {
        let out = match gt {
            GrammarType::Any => {
                // M5-⑩（套件修复）：任意 JSON 值用结构骨架 + 状态环，
                // 不用「吸收一切字节」的单状态（会吞掉 `,`/`}`/`]` 绕过 required/minItems 门）。
                self.build_any_json()?
            }
            GrammarType::String {
                min_length,
                max_length,
                pattern,
                format,
                ..
            } => {
                if pattern.is_some() && format.is_some() {
                    // M5-⑦/⑧：两者都指定 → 无法同时满足（不静默丢弃其一）
                    return Err(ZealotError::SchemaCompileError(
                        "string 不能同时指定 pattern 与 format".into(),
                    ));
                }
                if let Some(f) = format {
                    // M5-⑦：format 内容约束（与长度组合见 compile_anchored_template）
                    self.compile_format_string(*min_length, *max_length, f)?
                } else if let Some(pat) = pattern {
                    // M5-⑧：pattern 锚定语义（与长度组合见 compile_anchored_template）
                    self.compile_pattern_string(*min_length, *max_length, pat)?
                } else {
                    // M5-⑥：minLength/maxLength 真正执行。
                    // 无约束（min=0 且无 max）→ 紧凑形态（旧行为，长度不计数）。
                    // 有约束 → 计数自动机：content_k / esc_k / u1..u4_k（k = 已消费码点数）。
                    // 码点计数：ASCII/首字节（<0x80 或 ≥0xC0）= 1，UTF-8 连续字节 = 0；
                    // `\x`（非 u）转义 = 1，`\uXXXX` = 1（代理对按 2 计，文档化偏差）；
                    // min 门控收尾 `"`（k ≥ min），max 封顶（k == max 仅可收尾）；
                    // 仅 min 时 k == min 冻结为吸收态（不再计数）。
                    let min = *min_length;
                    let max = *max_length;
                    if min == 0 && max.is_none() {
                        let s0 = self.new_state()?;
                        let s1 = self.new_state()?;
                        let esc = self.new_state()?;
                        let sx = self.new_state()?;
                        self.add_edge(s0, b'"', s1);
                        for b in 0..=255u8 {
                            if b != b'"' && b != b'\\' {
                                self.add_stay(s1, b);
                            }
                        }
                        self.add_edge(s1, b'"', sx);
                        self.add_edge(s1, b'\\', esc);
                        // esc 是全新状态：整块构建 256 条边（逐条 insert 会 O(n²) 拖慢大 schema）
                        self.states[esc].edges = (0..=255u8).map(|b| (b, s1)).collect();
                        self.set_final(sx);
                        AutoOut {
                            start: s0,
                            exits: vec![sx],
                        }
                    } else {
                        let max = max.unwrap_or(min as u64);
                        if max < min as u64 {
                            return Err(ZealotError::SchemaCompileError(format!(
                                "string 空区间：minLength {min} > maxLength {max}"
                            )));
                        }
                        let horizon = max as usize;
                        if horizon > MAX_LENGTH_COUNTER {
                            return Err(ZealotError::SchemaCompileError(format!(
                                "minLength/maxLength {horizon} 过大（上限 {MAX_LENGTH_COUNTER}）"
                            )));
                        }
                        let free = max_length.is_none(); // 仅 min：k == min 后冻结（吸收态）
                                                         // 预建 content/esc/u 状态（k = 0..=horizon），再建入口 s0
                        let mut content: Vec<usize> = Vec::with_capacity(horizon + 1);
                        let mut esc: Vec<usize> = Vec::with_capacity(horizon + 1);
                        let mut u1: Vec<usize> = Vec::with_capacity(horizon + 1);
                        let mut u2: Vec<usize> = Vec::with_capacity(horizon + 1);
                        let mut u3: Vec<usize> = Vec::with_capacity(horizon + 1);
                        let mut u4: Vec<usize> = Vec::with_capacity(horizon + 1);
                        for _ in 0..=horizon {
                            content.push(self.new_state()?);
                            esc.push(self.new_state()?);
                            u1.push(self.new_state()?);
                            u2.push(self.new_state()?);
                            u3.push(self.new_state()?);
                            u4.push(self.new_state()?);
                        }
                        let s0 = self.new_state()?;
                        let sx = self.new_state()?;
                        self.add_edge(s0, b'"', content[0]);
                        self.set_final(sx);
                        for k in 0..=horizon {
                            let cur = content[k];
                            let absorbing = free && k == horizon;
                            // 内容边整块构建（按字节序自然有序；O(n²) 逐条 insert 拖慢大 horizon）
                            let mut edges: Vec<(u8, usize)> = Vec::with_capacity(256);
                            for b in 0..=255u8 {
                                if b == b'"' {
                                    // min 门控收尾
                                    if k >= min {
                                        edges.push((b, sx));
                                    }
                                } else if b == b'\\' {
                                    // 转义：仅当可回落到活目标时接线（max 封顶处不可再转义）
                                    if absorbing || k < horizon {
                                        edges.push((b, esc[k]));
                                    }
                                } else if !(0x80..0xC0).contains(&b) {
                                    // 码点首字节：+1；吸收态停留
                                    if absorbing {
                                        self.add_stay(cur, b);
                                    } else if k < horizon {
                                        edges.push((b, content[k + 1]));
                                    }
                                } else {
                                    // UTF-8 连续字节：不计数（停留）
                                    self.add_stay(cur, b);
                                }
                            }
                            self.states[cur].edges = edges;
                            // 转义结构：`\x`（非 u）→ +1；`\uXXXX` → +1（hex 链）
                            let e = esc[k];
                            let esc_next = if k < horizon {
                                content[k + 1]
                            } else if free {
                                content[horizon]
                            } else {
                                continue; // max 封顶：esc[k] 不可达（未接线），跳过
                            };
                            let mut esc_edges: Vec<(u8, usize)> =
                                (0..=255u8).map(|b| (b, esc_next)).collect();
                            esc_edges[b'u' as usize].1 = u1[k];
                            self.states[e].edges = esc_edges;
                            // hex 边整块构建（48 个 hex 字节，按字节序）
                            let hex_dst = [u2[k], u3[k], u4[k], esc_next];
                            for (i, src) in [u1[k], u2[k], u3[k], u4[k]].iter().enumerate() {
                                let edges: Vec<(u8, usize)> = (0..=255u8)
                                    .filter(|b| b.is_ascii_hexdigit())
                                    .map(|b| (b, hex_dst[i]))
                                    .collect();
                                self.states[*src].edges = edges;
                            }
                        }
                        AutoOut {
                            start: s0,
                            exits: vec![sx],
                        }
                    }
                }
            }
            GrammarType::Number => {
                // s0 符号/首数字 → s1 整数部分（`.`/`e` 出边）→ s2 小数点 → s3 小数部分；
                // s1/s3 → s4 指数 → s5 指数数字。合法收尾：s1/s3/s5。
                let s0 = self.new_state()?;
                let s1 = self.new_state()?;
                let s2 = self.new_state()?;
                let s3 = self.new_state()?;
                let s4 = self.new_state()?;
                let s5 = self.new_state()?;
                self.add_edge(s0, b'-', s1);
                for b in b'0'..=b'9' {
                    self.add_edge(s0, b, s1);
                }
                for b in b'0'..=b'9' {
                    self.add_stay(s1, b);
                }
                self.add_edge(s1, b'.', s2);
                self.add_edge(s1, b'e', s4);
                self.add_edge(s1, b'E', s4);
                for b in b'0'..=b'9' {
                    self.add_edge(s2, b, s3);
                }
                for b in b'0'..=b'9' {
                    self.add_stay(s3, b);
                }
                self.add_edge(s3, b'e', s4);
                self.add_edge(s3, b'E', s4);
                for b in b'0'..=b'9' {
                    self.add_edge(s4, b, s5);
                }
                self.add_edge(s4, b'-', s5);
                self.add_edge(s4, b'+', s5);
                for b in b'0'..=b'9' {
                    self.add_stay(s5, b);
                }
                self.set_final(s1);
                self.set_final(s3);
                self.set_final(s5);
                AutoOut {
                    start: s0,
                    exits: vec![s1, s3, s5],
                }
            }
            GrammarType::Integer {
                min,
                max,
                multiple_of,
            } => self.compile_integer(*min, *max, multiple_of.as_ref())?,
            GrammarType::Boolean => {
                let (start, mut ends) = self.literal_trie(&[b"true", b"false"])?;
                AutoOut {
                    start,
                    exits: std::mem::take(&mut ends),
                }
            }
            GrammarType::Null => {
                let (start, mut ends) = self.literal_trie(&[b"null"])?;
                AutoOut {
                    start,
                    exits: std::mem::take(&mut ends),
                }
            }
            GrammarType::Enum { values } => {
                // M5-⑩（套件修复）：成员统一走 serde_json 序列化（字符串成员
                // 也转义控制字符——原始字节会与数据的转义形态不匹配，不可达）。
                let lits: Vec<Vec<u8>> = values
                    .iter()
                    .map(|v| serde_json::to_string(v).unwrap_or_default().into_bytes())
                    .collect();
                let refs: Vec<&[u8]> = lits.iter().map(|v| v.as_slice()).collect();
                let (start, mut ends) = self.literal_trie(&refs)?;
                AutoOut {
                    start,
                    exits: std::mem::take(&mut ends),
                }
            }
            GrammarType::Object {
                properties,
                required,
                additional_properties,
            } => self.compile_object(properties, required, *additional_properties)?,
            GrammarType::Array {
                min_items,
                max_items,
                item_type,
            } => self.compile_array(*min_items, *max_items, item_type.as_deref())?,
            GrammarType::Union { branches } => {
                let start = self.new_state()?;
                self.boundary(start);
                let mut branch_starts = Vec::new();
                let mut exits = Vec::new();
                for br in branches {
                    let o = self.compile_value(br)?;
                    branch_starts.push(o.start);
                    exits.extend(o.exits);
                }
                self.states[start].union_branches = Some(branch_starts);
                AutoOut { start, exits }
            }
        };
        // 出口状态统一加 ws（值后的空白由出口承接，如 `Ġ,`/`Ġ}` token）
        for &e in &out.exits {
            self.add_ws(e);
        }
        Ok(out)
    }

    /// 任意 JSON 的叶子集（string/number/boolean/null，无结构骨架）。
    fn any_leaf_set(&mut self) -> Result<(Vec<usize>, Vec<usize>), ZealotError> {
        let plain = GrammarType::String {
            min_length: 0,
            max_length: None,
            pattern: None,
            enum_values: None,
            format: None,
        };
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for gt in [
            plain,
            GrammarType::Number,
            GrammarType::Boolean,
            GrammarType::Null,
        ] {
            let o = self.compile_value(&gt)?;
            starts.push(o.start);
            ends.extend(o.exits);
        }
        Ok((starts, ends))
    }

    /// M5-⑩：任意 JSON 值自动机（JSON Schema Test Suite 子集暴露的修复）。
    ///
    /// 原实现是「吸收一切字节」的单状态——它会把 `,`/`}`/`]` 也吞掉，绕过
    /// required/minItems 门（`{"bar":1}` 在 required foo 时被判合法）→ 换成
    /// 结构骨架 + 状态环（编译无递归）：
    ///
    ///   入口（纯分支派发）：v_top（顶层）/ v_kv（任意对象值）/ v_item（任意数组元素）
    ///     分支：string / number / boolean / null / 对象骨架 / 数组骨架
    ///   对象骨架：`{`→kv0；kv0：`}`→oc（空对象）| `"`→键汤；`,` 后 kvn 只有 `"`→键汤
    ///     （杜绝尾逗号）；键汤 0x20..=0xFF 除 `"`；`"`→ke→`:`→v_kv
    ///   数组骨架：`[`→item0；item0：`]`→ac（空数组）| 值；`,` 后 itemn 只有值
    ///   三套叶子：top（纯出口——父级 required/minItems 门不被绕过）、
    ///     kv（出口接 `,`→kvn / `}`→oc）、item（出口接 `,`→itemn / `]`→ac）
    ///   嵌套任意结构经 2-环（v_kv → 数组骨架 → v_item → 对象骨架 → v_kv）表达。
    fn build_any_json(&mut self) -> Result<AutoOut, ZealotError> {
        let (t_s, t_e) = self.any_leaf_set()?;
        let (k_s, k_e) = self.any_leaf_set()?;
        let (i_s, i_e) = self.any_leaf_set()?;

        let v_top = self.new_state()?;
        self.boundary(v_top);
        let v_kv = self.new_state()?;
        self.boundary(v_kv);
        let v_item = self.new_state()?;
        self.boundary(v_item);

        // 对象骨架
        let oc = self.new_state()?;
        self.set_final(oc);
        let kv0 = self.new_state()?;
        self.boundary(kv0);
        let kvn = self.new_state()?;
        self.boundary(kvn);
        let ks = self.new_state()?;
        self.boundary(ks);
        let ke = self.new_state()?;
        self.boundary(ke);
        let obj_open = self.new_state()?;
        self.boundary(obj_open);
        self.add_edge(obj_open, b'{', kv0);
        self.add_edge(kv0, b'}', oc);
        self.add_edge(kv0, b'"', ks);
        self.add_edge(kvn, b'"', ks);
        for b in 0x20u8..=0xFFu8 {
            if b != b'"' {
                self.add_stay(ks, b);
            }
        }
        self.add_edge(ks, b'"', ke);
        self.add_edge(ke, b':', v_kv);

        // 数组骨架
        let ac = self.new_state()?;
        self.set_final(ac);
        let item0 = self.new_state()?;
        self.boundary(item0);
        let itemn = self.new_state()?;
        self.boundary(itemn);
        let arr_open = self.new_state()?;
        self.boundary(arr_open);
        self.add_edge(arr_open, b'[', item0);
        self.add_edge(item0, b']', ac);
        self.states[item0].union_branches = Some(vec![v_item]);
        self.states[itemn].union_branches = Some(vec![v_item]);

        // 分支派发（边/stay 优先；派发状态无自身边）
        for (v, mut starts) in [(v_top, t_s), (v_kv, k_s), (v_item, i_s)] {
            starts.push(obj_open);
            starts.push(arr_open);
            self.states[v].union_branches = Some(starts);
        }

        // 值出口接线（内部上下文；顶层叶子无内部边）
        for &e in &k_e {
            self.add_ws(e);
            self.add_edge(e, b',', kvn);
            self.add_edge(e, b'}', oc);
        }
        for &e in &i_e {
            self.add_ws(e);
            self.add_edge(e, b',', itemn);
            self.add_edge(e, b']', ac);
        }

        // 顶层出口：top 叶子终点 + oc + ac（ws 由 compile_value 统一加）
        let mut exits = t_e;
        exits.push(oc);
        exits.push(ac);
        Ok(AutoOut {
            start: v_top,
            exits,
        })
    }

    /// Object：required 用计数器状态 k 表达（k = 下一个未满足的 required 键下标）。
    /// 注意：required 键须**按 required 数组顺序**生成——乱序出现不推进计数器
    /// （需按序重发）。生成侧（mask 驱动）天然可控，文档化此约束。
    /// 可选键任意位置出现不改变 k；`}` 只在 k == r 可达。
    fn compile_object(
        &mut self,
        properties: &HashMap<String, GrammarType>,
        required: &[String],
        additional_properties: bool,
    ) -> Result<AutoOut, ZealotError> {
        let mut keys: Vec<&String> = properties.keys().collect();
        // required 去重保序（重复 required 键会让计数器双进）
        let mut req: Vec<&String> = Vec::new();
        for k in required {
            if !req.contains(&k) {
                req.push(k);
            }
        }
        // M5-⑩：required 键未在 properties 中声明时也须有链（值 = 任意 JSON）
        for k in &req {
            if !keys.contains(k) {
                keys.push(k);
            }
        }
        keys.sort();
        let r = req.len();

        let done = self.new_state()?;
        self.set_final(done);

        // 自后向前构建：k 状态引用 k+1 的键起始位置
        let mut trie_root: Vec<usize> = vec![0; r + 1];
        let mut key_root: Vec<usize> = vec![0; r + 1];
        for k in (0..=r).rev() {
            // 键 trie(k)：`"` → 键字节链 → after_key(k,name) → `:` → 值(k,name)。
            // strict 语义（M5-④）：已满足的 required 键 req[0..k] 从后续位置排除
            // （required 键唯一）——否则重复 required 键使计数器 k 永不推进，
            // 贪心采样陷入死循环；optional 键在所有位置保留（可重复）。
            let tr = self.new_state()?;
            self.boundary(tr);
            let kr = self.new_state()?;
            self.add_edge(tr, b'"', kr);
            trie_root[k] = tr;
            key_root[k] = kr;
            let eligible: Vec<&String> = keys
                .iter()
                .copied()
                .filter(|n| !req[..k].contains(n))
                .collect();
            let mut k_chain_states = Vec::new();
            for name in &eligible {
                // M5-⑩：键链用 JSON 转义字节（控制字符/引号键名的序列化形态与数据一致）
                let key_bytes = serde_json::to_string(name.as_str())
                    .map(|s| {
                        let b = s.as_bytes();
                        b[1..b.len() - 1].to_vec()
                    })
                    .unwrap_or_else(|_| name.as_bytes().to_vec());
                let mut chain_states = Vec::new();
                let leaf = self.chain_from_track(key_root[k], &key_bytes, &mut chain_states)?;
                let ak = self.new_state()?;
                self.boundary(ak);
                let vpos = self.new_state()?;
                self.boundary(vpos);
                self.add_edge(leaf, b'"', ak);
                self.add_edge(ak, b':', vpos);
                let k2 = if req.get(k).is_some_and(|rk| rk.as_str() == name.as_str()) {
                    k + 1
                } else {
                    k
                };
                // 值 grammar（按 (k,name) 复制：出口目标与 k 绑定）。
                // vpos 用分支语义派发：首字节由 sub.start 自身的转移消费（避免双消费）
                let sub = self.compile_value(properties.get(*name).unwrap_or(&GrammarType::Any))?;
                self.states[vpos].union_branches = Some(vec![sub.start]);
                // 值出口：`,` → 下一键起始（k2）；`}` 仅在 k2 == r（全部 required 已满足）
                // M5-⑩：`,` 仅当 k2 处仍有键（未排除的 optional / extra）——
                // 否则尾逗号是可达而死路径（reachable-but-dead，issue #19 原则），
                // 生成侧会卡死在 `,` 后的空键位。
                let comma_viable =
                    additional_properties || keys.iter().any(|n| !req[..k2].contains(n));
                for &e in &sub.exits {
                    if comma_viable {
                        self.add_edge(e, b',', trie_root[k2]);
                    }
                    if k2 == r {
                        self.add_edge(e, b'}', done);
                    }
                }
                k_chain_states.extend(chain_states);
            }
            // M5-⑩：additionalProperties=true → 任意键路径：键汤
            // （stay 0x20..=0xFF 除 `"`）→ `"`→eke → `:`→evp → 任意值 →
            // `,`→trie_root[k] / `}`（仅 k==r，required 门不被绕过）。
            // 分支挂 key_root[k] 与全部键链状态（前缀冲突的额外键也可达）。
            if additional_properties {
                let eks = self.new_state()?;
                self.boundary(eks);
                let eke = self.new_state()?;
                self.boundary(eke);
                let evp = self.new_state()?;
                self.boundary(evp);
                for b in 0x20u8..=0xFFu8 {
                    if b != b'"' {
                        self.add_stay(eks, b);
                    }
                }
                self.add_edge(eks, b'"', eke);
                self.add_edge(eke, b':', evp);
                let ev = self.compile_value(&GrammarType::Any)?;
                self.states[evp].union_branches = Some(vec![ev.start]);
                for &e in &ev.exits {
                    self.add_edge(e, b',', trie_root[k]);
                    if k == r {
                        self.add_edge(e, b'}', done);
                    }
                }
                for &cs in std::iter::once(&key_root[k]).chain(k_chain_states.iter()) {
                    let br = &mut self.states[cs].union_branches;
                    if let Some(v) = br {
                        if !v.contains(&eks) {
                            v.push(eks);
                        }
                    } else {
                        *br = Some(vec![eks]);
                    }
                }
            }
        }
        // 只有 k == 0 的对象起始位置被引用（`{` 后）：`"` 直连键内容起始
        let ab = self.new_state()?;
        self.boundary(ab);
        self.add_edge(ab, b'"', key_root[0]);
        if r == 0 {
            // 无 required：空对象 {} 合法
            self.add_edge(ab, b'}', done);
        }
        let start = self.new_state()?;
        self.boundary(start);
        self.add_edge(start, b'{', ab);
        Ok(AutoOut {
            start,
            exits: vec![done],
        })
    }

    /// Array：计数器状态 c（已见 item 数，封顶）。item 出口 `]` 只在 c+1 >= min_items
    /// 可达；max_items 封顶后 `,` 边消失（超量即死）。
    fn compile_array(
        &mut self,
        min_items: usize,
        max_items: Option<u64>,
        item_type: Option<&GrammarType>,
    ) -> Result<AutoOut, ZealotError> {
        let cap = match max_items {
            Some(m) => m as usize,
            None => min_items + 1,
        };
        if cap > MAX_ARRAY_COUNTER {
            return Err(ZealotError::SchemaTooComplex {
                states: MAX_ARRAY_COUNTER,
                limit: MAX_ARRAY_COUNTER,
            });
        }
        let item = item_type.unwrap_or(&GrammarType::Any);
        let exit = self.new_state()?;
        self.set_final(exit);

        let mut vpos: Vec<usize> = vec![0; cap + 1];
        for c in (0..=cap).rev() {
            let v = self.new_state()?;
            self.boundary(v);
            vpos[c] = v;
            // max_items 封顶时 vpos[cap] 是死状态：第 cap+1 个 item 不允许开始
            let at_cap_dead = max_items.is_some() && c == cap;
            if at_cap_dead {
                continue;
            }
            // item grammar 按计数器 c 复制（出口目标与 c 绑定）
            let sub = self.compile_value(item)?;
            self.states[v].union_branches = Some(vec![sub.start]);
            let can_more = max_items.is_none() || (c as u64) < max_items.unwrap();
            for &e in &sub.exits {
                if can_more {
                    let target = if c + 1 > cap { v } else { vpos[c + 1] };
                    self.add_edge(e, b',', target);
                }
                if c + 1 >= min_items {
                    self.add_edge(e, b']', exit);
                }
            }
        }
        let open = self.new_state()?;
        self.add_edge(open, b'[', vpos[0]);
        // `[` 后直接 `]`（空数组）只在 min_items == 0
        if min_items == 0 {
            self.add_edge(vpos[0], b']', exit);
        }
        Ok(AutoOut {
            start: open,
            exits: vec![exit],
        })
    }

    /// 从 cur 起消费字面字节；返回终点。
    /// M5-⑦：format 字符串（内容字节级约束）。与 min/max 长度组合时用
    /// 乘积状态 (format_state, k)——预算超限显式报错；format 内容不允许
    /// 转义（`\\`/`"` 均不在任何格式字符集中）。
    fn compile_format_string(
        &mut self,
        min: usize,
        max: Option<u64>,
        f: &StringFormat,
    ) -> Result<AutoOut, ZealotError> {
        let fstart = self.states.len();
        let (fs, _ffinals) = self.build_format(f)?;
        let template: Vec<(Vec<(u8, usize)>, bool)> = self.states[fstart..]
            .iter()
            .map(|st| (st.edges.clone(), st.is_final))
            .collect();
        self.compile_anchored_template(min, max, template, fs - fstart)
    }

    /// M5-⑧：pattern（锚定语义）——regex DFA 物化 + 与长度计数乘积。
    /// JSON 字符串无法包含裸 `"`/`\\`：这类内容边丢弃（文档化）；
    /// 产物与 format 共用 compile_anchored_template。
    fn compile_pattern_string(
        &mut self,
        min: usize,
        max: Option<u64>,
        pattern: &str,
    ) -> Result<AutoOut, ZealotError> {
        let dfa =
            super::regex::compile_pattern(pattern).map_err(ZealotError::SchemaCompileError)?;
        let mut template: Vec<(Vec<(u8, usize)>, bool)> = Vec::with_capacity(dfa.transitions.len());
        for edges in &dfa.transitions {
            let edges: Vec<(u8, usize)> = edges
                .iter()
                .copied()
                .filter(|&(b, _)| b != b'"' && b != b'\\')
                .collect();
            let is_final = dfa.finals.contains(&(template.len()));
            template.push((edges, is_final));
        }
        // 死状态剪枝（无出边且非终态的 regex 状态）
        let n = template.len();
        let mut alive = vec![false; n];
        for (i, (_, is_final)) in template.iter().enumerate() {
            if *is_final {
                alive[i] = true;
            }
        }
        loop {
            let mut changed = false;
            for i in 0..n {
                if alive[i] {
                    continue;
                }
                if template[i].0.iter().any(|&(_, t)| alive[t]) {
                    alive[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut remap = vec![usize::MAX; n];
        let mut next_id = 0;
        for i in 0..n {
            if alive[i] {
                remap[i] = next_id;
                next_id += 1;
            }
        }
        let mut compact: Vec<(Vec<(u8, usize)>, bool)> = Vec::with_capacity(next_id);
        for i in 0..n {
            if !alive[i] {
                continue;
            }
            let edges = template[i]
                .0
                .iter()
                .filter(|&&(_, t)| alive[t])
                .map(|&(b, t)| (b, remap[t]))
                .collect();
            compact.push((edges, template[i].1));
        }
        let start_fi = remap[dfa.start];
        if start_fi == usize::MAX || compact.is_empty() {
            // 语言为空：pattern 要求内容含裸 `\\`/`"`，JSON 字符串无法表示
            return Err(ZealotError::SchemaCompileError(
                "pattern 要求的内容包含裸 `\\` 或 `\"`（JSON 字符串内无法表示）".into(),
            ));
        }
        self.compile_anchored_template(min, max, compact, start_fi)
    }

    /// 模板自动机（format/regex 共用）× 长度计数乘积：状态 (fi, k)，
    /// k = 已消费码点数；min 门控收尾、max 封顶、仅 min 冻结。
    fn compile_anchored_template(
        &mut self,
        min: usize,
        max: Option<u64>,
        template: Vec<(Vec<(u8, usize)>, bool)>,
        start_fi: usize,
    ) -> Result<AutoOut, ZealotError> {
        let t_count = template.len();
        let min_only = max.is_none();
        let max = max.unwrap_or(min as u64);
        if max < min as u64 {
            return Err(ZealotError::SchemaCompileError(format!(
                "string 空区间：minLength {min} > maxLength {max}"
            )));
        }
        let horizon = max as usize;
        if horizon > MAX_LENGTH_COUNTER {
            return Err(ZealotError::SchemaCompileError(format!(
                "minLength/maxLength {horizon} 过大（上限 {MAX_LENGTH_COUNTER}）"
            )));
        }
        let cols = horizon + 1;
        if t_count * cols > MAX_AUTO_STATES {
            return Err(ZealotError::SchemaCompileError(format!(
                "模板×长度组合过大（{t_count} × {cols} 状态）"
            )));
        }
        let absorbing = min_only; // 仅 min：k == horizon 后冻结
        let s0 = self.new_state()?;
        let sx = self.new_state()?;
        let mut grid: Vec<usize> = Vec::with_capacity(t_count * cols);
        for _ in 0..(t_count * cols) {
            grid.push(self.new_state()?);
        }
        let idx = |fi: usize, k: usize| fi * cols + k;
        self.add_edge(s0, b'"', grid[idx(start_fi, 0)]);
        self.set_final(sx);
        for (fi, (edges, is_final)) in template.iter().enumerate() {
            for k in 0..=horizon {
                let cur = grid[idx(fi, k)];
                for &(b, t) in edges {
                    let k2 = if absorbing {
                        (k + cp_count(b)).min(horizon)
                    } else if k + cp_count(b) <= horizon {
                        k + cp_count(b)
                    } else {
                        continue; // max 封顶：不可再续
                    };
                    self.add_edge(cur, b, grid[idx(t, k2)]);
                }
                if *is_final && k >= min {
                    self.add_edge(cur, b'"', sx);
                }
            }
        }
        Ok(AutoOut {
            start: s0,
            exits: vec![sx],
        })
    }

    fn build_format(&mut self, f: &StringFormat) -> Result<(usize, Vec<usize>), ZealotError> {
        match f {
            StringFormat::Email => self.build_email(),
            StringFormat::Date => self.build_date(),
            StringFormat::DateTime => self.build_datetime(),
            StringFormat::Time => self.build_time(),
            StringFormat::Uuid => self.build_uuid(),
            StringFormat::Ipv4 => self.build_ipv4(),
            StringFormat::Ipv6 => self.build_ipv6(),
            StringFormat::Hostname => self.build_hostname(),
            StringFormat::Uri => self.build_uri(),
        }
    }

    /// email：local（点不可首尾/连续）@ 域名（label：字母数字首尾、中缀连字符）
    fn build_email(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let is_local = |b: u8| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'/'
                        | b'='
                        | b'?'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'{'
                        | b'|'
                        | b'}'
                        | b'~'
                        | b'-'
                )
        };
        let l0 = self.new_state()?;
        let l1 = self.new_state()?;
        let l2 = self.new_state()?;
        for b in 0..=255u8 {
            if is_local(b) && b != b'.' {
                self.add_edge(l0, b, l1);
                self.add_edge(l1, b, l1);
                self.add_edge(l2, b, l1);
            }
        }
        self.add_edge(l1, b'.', l2);
        let d0 = self.new_state()?;
        let d1 = self.new_state()?;
        let d2 = self.new_state()?;
        let d3 = self.new_state()?;
        self.add_edge(l1, b'@', d0);
        for b in 0..=255u8 {
            if b.is_ascii_alphanumeric() {
                self.add_edge(d0, b, d1);
                self.add_edge(d1, b, d1);
                self.add_edge(d2, b, d1);
                self.add_edge(d3, b, d1);
            }
        }
        self.add_edge(d1, b'-', d3);
        self.add_edge(d1, b'.', d2);
        self.set_final(d1);
        Ok((l0, vec![d1]))
    }

    /// date：YYYY-MM-DD（月 01-12、日 01-31，结构校验）
    fn build_date(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let y1 = self.new_state()?;
        let y4 = self.digit_chain(y1, 4)?;
        let dash1 = self.new_state()?;
        self.add_edge(y4, b'-', dash1);
        let ma = self.new_state()?; // 月首 0
        let mb = self.new_state()?; // 月首 1
        let md = self.new_state()?;
        self.add_edge(dash1, b'0', ma);
        self.add_edge(dash1, b'1', mb);
        for b in b'1'..=b'9' {
            self.add_edge(ma, b, md);
        }
        for b in b'0'..=b'2' {
            self.add_edge(mb, b, md);
        }
        let dash2 = self.new_state()?;
        self.add_edge(md, b'-', dash2);
        let da = self.new_state()?; // 日首 0
        let db = self.new_state()?; // 日首 1|2
        let dc = self.new_state()?; // 日首 3
        self.add_edge(dash2, b'0', da);
        self.add_edge(dash2, b'1', db);
        self.add_edge(dash2, b'2', db);
        self.add_edge(dash2, b'3', dc);
        for b in b'1'..=b'9' {
            self.add_edge(da, b, da);
        }
        for b in b'0'..=b'9' {
            self.add_edge(db, b, db);
        }
        for b in b'0'..=b'1' {
            self.add_edge(dc, b, dc);
        }
        self.set_final(da);
        self.set_final(db);
        self.set_final(dc);
        Ok((y1, vec![da, db, dc]))
    }

    /// time：HH:MM:SS(.f+)?(Z|[+-]HH:MM)?（24 小时制）
    fn build_time(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let h1 = self.new_state()?;
        let ha = self.new_state()?; // 时首 0|1
        let hb = self.new_state()?; // 时首 2
        let hx = self.new_state()?; // 时完成
        self.add_edge(h1, b'0', ha);
        self.add_edge(h1, b'1', ha);
        self.add_edge(h1, b'2', hb);
        for b in b'0'..=b'9' {
            self.add_edge(ha, b, hx);
        }
        for b in b'0'..=b'3' {
            self.add_edge(hb, b, hx);
        }
        let c1 = self.new_state()?;
        self.add_edge(hx, b':', c1);
        let m1 = self.new_state()?;
        let m2 = self.new_state()?;
        for b in b'0'..=b'5' {
            self.add_edge(c1, b, m1);
        }
        for b in b'0'..=b'9' {
            self.add_edge(m1, b, m2);
        }
        let c2 = self.new_state()?;
        self.add_edge(m2, b':', c2);
        let s1 = self.new_state()?;
        let s2 = self.new_state()?;
        for b in b'0'..=b'5' {
            self.add_edge(c2, b, s1);
        }
        for b in b'0'..=b'9' {
            self.add_edge(s1, b, s2);
        }
        self.set_final(s2);
        // 小数秒
        let frac = self.new_state()?;
        self.add_edge(s2, b'.', frac);
        for b in b'0'..=b'9' {
            self.add_edge(frac, b, frac);
        }
        self.set_final(frac);
        // 时区：Z / ±HH:MM
        let z_end = self.new_state()?;
        let o1 = self.new_state()?;
        let oa = self.new_state()?; // 偏移时首 0|1
        let ob = self.new_state()?; // 偏移时首 2
        let oc = self.new_state()?;
        let o3 = self.new_state()?;
        let o4 = self.new_state()?;
        for base in [s2, frac] {
            self.add_edge(base, b'Z', z_end);
            self.add_edge(base, b'z', z_end);
            self.add_edge(base, b'+', o1);
            self.add_edge(base, b'-', o1);
        }
        self.add_edge(o1, b'0', oa);
        self.add_edge(o1, b'1', oa);
        self.add_edge(o1, b'2', ob);
        for b in b'0'..=b'9' {
            self.add_edge(oa, b, oc);
        }
        for b in b'0'..=b'3' {
            self.add_edge(ob, b, oc);
        }
        self.add_edge(oc, b':', o3);
        for b in b'0'..=b'5' {
            self.add_edge(o3, b, o4);
        }
        for b in b'0'..=b'9' {
            self.add_edge(o4, b, z_end);
        }
        self.set_final(z_end);
        Ok((h1, vec![s2, frac, z_end]))
    }

    /// date-time：date T time（T/t 分隔，RFC3339）——日期终态降级为中间态（不可裸收尾）
    fn build_datetime(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let (ds, dfs) = self.build_date()?;
        let (ts, tfs) = self.build_time()?;
        for &df in &dfs {
            self.states[df].is_final = false; // 裸日期不可作为 date-time 收尾
            self.add_edge(df, b'T', ts);
            self.add_edge(df, b't', ts);
        }
        Ok((ds, tfs))
    }

    /// uuid：8-4-4-4-12 hex
    fn build_uuid(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let s = self.new_state()?;
        let h8 = self.digit_chain_hex(s, 8)?;
        let d1 = self.new_state()?;
        self.add_edge(h8, b'-', d1);
        let h4a = self.digit_chain_hex(d1, 4)?;
        let d2 = self.new_state()?;
        self.add_edge(h4a, b'-', d2);
        let h4b = self.digit_chain_hex(d2, 4)?;
        let d3 = self.new_state()?;
        self.add_edge(h4b, b'-', d3);
        let h4c = self.digit_chain_hex(d3, 4)?;
        let d4 = self.new_state()?;
        self.add_edge(h4c, b'-', d4);
        let h12 = self.digit_chain_hex(d4, 12)?;
        self.set_final(h12);
        Ok((s, vec![h12]))
    }

    /// ipv4：四段 0-255（无前导零）
    fn build_ipv4(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let s1 = self.new_state()?;
        let mut starts = vec![s1];
        let mut last_ends = Vec::new();
        for i in 0..4 {
            let os = starts[i];
            let oz = self.new_state()?;
            let ot1 = self.new_state()?; // 首 1
            let ot2 = self.new_state()?; // 首 2
            let ot9 = self.new_state()?; // 首 3-9
            let oh = self.new_state()?;
            let oh5 = self.new_state()?;
            let ou = self.new_state()?;
            self.add_edge(os, b'0', oz);
            self.add_edge(os, b'1', ot1);
            self.add_edge(os, b'2', ot2);
            for b in b'3'..=b'9' {
                self.add_edge(os, b, ot9);
            }
            for b in b'0'..=b'9' {
                self.add_edge(ot1, b, oh);
                self.add_edge(ot9, b, oh);
            }
            for b in b'0'..=b'4' {
                self.add_edge(ot2, b, oh);
            }
            self.add_edge(ot2, b'5', oh5);
            for b in b'0'..=b'9' {
                self.add_edge(oh, b, ou);
            }
            for b in b'0'..=b'5' {
                self.add_edge(oh5, b, ou);
            }
            let ends = vec![oz, ot1, ot2, ot9, oh, oh5, ou];
            if i < 3 {
                let next = self.new_state()?;
                for &e in &ends {
                    self.add_edge(e, b'.', next);
                }
                starts.push(next);
            } else {
                for &e in &ends {
                    self.set_final(e);
                }
                last_ends = ends;
            }
        }
        Ok((s1, last_ends))
    }

    /// ipv6：8 组 hex + 至多一个 `::` 压缩（RFC 4291 结构；不含内嵌 IPv4）
    fn build_ipv6(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let mut h: Vec<Vec<usize>> = vec![vec![]; 1]; // H(g, d) g∈1..=8, d∈1..=4（行含 0 位占位）
        h.extend(
            (1..=8)
                .map(|_| -> Result<Vec<usize>, ZealotError> {
                    let mut row = vec![0usize];
                    row.extend(
                        (1..=4)
                            .map(|_| self.new_state())
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    Ok(row)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let b: Vec<usize> = (0..8)
            .map(|_| self.new_state())
            .collect::<Result<Vec<_>, _>>()?; // B(g) g∈0..=7
        let cc: Vec<usize> = (0..8)
            .map(|_| self.new_state())
            .collect::<Result<Vec<_>, _>>()?; // CC(g)
        let mut c: Vec<Vec<usize>> = vec![vec![]; 1]; // C(g, d) g∈1..=7（行含 0 位占位）
        c.extend(
            (1..=7)
                .map(|_| -> Result<Vec<usize>, ZealotError> {
                    let mut row = vec![0usize];
                    row.extend(
                        (1..=4)
                            .map(|_| self.new_state())
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                    Ok(row)
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let cb: Vec<usize> = (1..=7)
            .map(|_| self.new_state())
            .collect::<Result<Vec<_>, _>>()?; // CB(g)
        let s = self.new_state()?;
        self.add_hex_edges(s, h[1][1]);
        self.add_edge(s, b':', b[0]);
        for g in 1..=8 {
            for d in 1..=4 {
                if d < 4 {
                    self.add_hex_edges(h[g][d], h[g][d + 1]);
                }
                if g < 8 {
                    self.add_edge(h[g][d], b':', b[g]);
                }
            }
        }
        for g in 0..8 {
            self.add_hex_edges(b[g], h[g + 1][1]);
            self.add_edge(b[g], b':', cc[g]);
        }
        for g in 0..8 {
            if g < 7 {
                self.add_hex_edges(cc[g], c[g + 1][1]);
            }
            self.set_final(cc[g]);
        }
        for g in 1..=7 {
            for d in 1..=4 {
                if d < 4 {
                    self.add_hex_edges(c[g][d], c[g][d + 1]);
                }
                if g < 7 {
                    self.add_edge(c[g][d], b':', cb[g]);
                }
                self.set_final(c[g][d]);
            }
        }
        for g in 1..=7 {
            if g < 7 {
                self.add_hex_edges(cb[g], c[g + 1][1]);
            }
        }
        for &st in h[8].iter().skip(1) {
            self.set_final(st);
        }
        let mut finals: Vec<usize> = h[8].iter().skip(1).copied().collect();
        finals.extend(cc);
        for row in c.iter().skip(1) {
            finals.extend(row.iter().skip(1).copied());
        }
        Ok((s, finals))
    }

    /// hostname：点分 label（字母数字首尾、中缀连字符，≥1 个 label）
    fn build_hostname(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let l0 = self.new_state()?;
        let l1 = self.new_state()?;
        let l2 = self.new_state()?;
        for b in 0..=255u8 {
            if b.is_ascii_alphanumeric() {
                self.add_edge(l0, b, l1);
                self.add_edge(l1, b, l1);
                self.add_edge(l2, b, l1);
            }
        }
        self.add_edge(l1, b'-', l2);
        self.add_edge(l1, b'.', l0);
        self.set_final(l1);
        Ok((l0, vec![l1]))
    }

    /// uri：scheme（字母开头，字母数字+.-）`:` + 非空白/控制/引号字节（宽松子集）
    fn build_uri(&mut self) -> Result<(usize, Vec<usize>), ZealotError> {
        let u0 = self.new_state()?;
        let u1 = self.new_state()?;
        let u2 = self.new_state()?;
        for b in 0..=255u8 {
            if b.is_ascii_alphabetic() {
                self.add_edge(u0, b, u1);
            }
            if b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.') {
                self.add_edge(u1, b, u1);
            }
            if b > 0x20 && b != 0x7F && b != b'"' && b != b'\\' {
                self.add_edge(u2, b, u2);
            }
        }
        self.add_edge(u1, b':', u2);
        self.set_final(u2);
        Ok((u0, vec![u2]))
    }

    fn add_hex_edges(&mut self, s: usize, t: usize) {
        for b in b'0'..=b'9' {
            self.add_edge(s, b, t);
        }
        for b in b'a'..=b'f' {
            self.add_edge(s, b, t);
        }
        for b in b'A'..=b'F' {
            self.add_edge(s, b, t);
        }
    }

    fn digit_chain(&mut self, from: usize, len: usize) -> Result<usize, ZealotError> {
        let mut cur = from;
        for _ in 0..len {
            let n = self.new_state()?;
            for b in b'0'..=b'9' {
                self.add_edge(cur, b, n);
            }
            cur = n;
        }
        Ok(cur)
    }

    fn digit_chain_hex(&mut self, from: usize, len: usize) -> Result<usize, ZealotError> {
        let mut cur = from;
        for _ in 0..len {
            let n = self.new_state()?;
            self.add_hex_edges(cur, n);
            cur = n;
        }
        Ok(cur)
    }

    /// 字面量链（前缀共享）；沿途状态记录到 out（额外键分支挂载用）。
    fn chain_from_track(
        &mut self,
        cur: usize,
        bytes: &[u8],
        out: &mut Vec<usize>,
    ) -> Result<usize, ZealotError> {
        let mut c = cur;
        for &b in bytes {
            let next = match self.states[c].edges.iter().find(|(bb, _)| *bb == b) {
                Some(&(_, to)) => to,
                None => {
                    let n = self.new_state()?;
                    self.add_edge(c, b, n);
                    n
                }
            };
            c = next;
            out.push(c);
        }
        Ok(c)
    }

    // ── M5-⑤：整数范围/整除约束 ─────────────────────────────────────────────
    //
    // 有界区间：十进制 digit-DP，状态 = (pos, ca, cb, rem)：
    //   ca ∈ {LT,EQ,GT}  前缀与下界幅度 A 的数字串比较（pos ≥ la 时必为 GT）
    //   cb ∈ {LT,EQ,GT}  前缀与上界幅度 B 的比较（超过 lb 位 → 死）
    //   rem              multipleOf 余数：10^k×value mod divisor（scale 为小数展开）
    // 收尾（共 k 位）：k == la → ca 非 LT（且 la == lb 时 cb 非 GT）；
    //   la < k < lb → 任意；k == lb → cb 非 LT（且 lb > la 时自动 ≥ A）；
    //   且 rem == 0。k < la / k > lb → 死。规范形式：无前导零（`0` 单数字）。
    // 纯 multipleOf（无区间）：余数 DFA（任意长度）。
    // 状态数 = (lb+1) × 9 × rem_count ≤ MAX_AUTO_STATES，超限 → 编译错误（不静默）。

    fn compile_integer(
        &mut self,
        min: Option<i64>,
        max: Option<i64>,
        mo: Option<&MultipleOf>,
    ) -> Result<AutoOut, ZealotError> {
        let lo = min.unwrap_or(i64::MIN);
        let hi = max.unwrap_or(i64::MAX);
        if lo > hi {
            return Err(ZealotError::SchemaCompileError(format!(
                "integer 空区间：minimum {lo} > maximum {hi}"
            )));
        }
        let bounded = lo != i64::MIN || hi != i64::MAX;
        let zero_ok = lo <= 0 && hi >= 0;

        let start = self.new_state()?;
        self.boundary(start);
        let mut exits = Vec::new();

        if !bounded {
            if let Some(mo) = mo {
                // 纯 multipleOf：余数 DFA（含正负两侧；`0` 恒为倍数）
                let (div, scale) = (mo.divisor, mo.scale);
                if div > MAX_AUTO_STATES as u64 / 4 {
                    return Err(ZealotError::SchemaCompileError(format!(
                        "multipleOf {div} 过大（无区间余数 DFA 超上限）"
                    )));
                }
                let z0 = self.new_state()?;
                self.add_edge(start, b'0', z0);
                self.set_final(z0);
                let mut rems = Vec::with_capacity(div as usize);
                for _ in 0..div {
                    rems.push(self.new_state()?);
                }
                let rem0 = rems[0];
                for d in 1..=9u8 {
                    let r = ((d as u64 * scale) % div) as usize;
                    self.add_edge(start, b'0' + d, rems[r]);
                }
                for (r, &s) in rems.iter().enumerate() {
                    for d in 0..10u8 {
                        let r2 = ((r as u64 * 10 + d as u64 * scale) % div) as usize;
                        self.add_edge(s, b'0' + d, rems[r2]);
                    }
                    if r == 0 {
                        self.set_final(s);
                    }
                }
                // 负数侧：`-` → 首数字 1-9 → 同一余数 DFA（`-0` 非法）
                let n0 = self.new_state()?;
                self.add_edge(start, b'-', n0);
                for d in 1..=9u8 {
                    let r = ((d as u64 * scale) % div) as usize;
                    self.add_edge(n0, b'0' + d, rems[r]);
                }
                exits.push(z0);
                exits.push(rem0);
                return Ok(AutoOut { start, exits });
            }
            // 无约束 integer：可选 `-` + 数字串（无小数点/指数）——旧宽松语法
            let s1 = self.new_state()?;
            self.add_edge(start, b'-', s1);
            for b in b'0'..=b'9' {
                self.add_edge(start, b, s1);
            }
            for b in b'0'..=b'9' {
                self.add_stay(s1, b);
            }
            self.set_final(s1);
            return Ok(AutoOut {
                start,
                exits: vec![s1],
            });
        }

        // 有界区间：`0`（单数字，恒为倍数）、正数侧 [max(lo,1), hi]、负数侧 `-`+幅度
        if zero_ok {
            let z0 = self.new_state()?;
            self.add_edge(start, b'0', z0);
            self.set_final(z0);
            exits.push(z0);
        }
        if hi >= 1 {
            let a = lo.max(1) as u64;
            let b = hi as u64;
            exits.extend(self.compile_digit_range(start, a, b, mo)?);
        }
        if lo < 0 {
            let a = if hi < 0 {
                (hi as i128).unsigned_abs() as u64
            } else {
                1
            };
            let b = (lo as i128).unsigned_abs() as u64;
            let n0 = self.new_state()?;
            self.add_edge(start, b'-', n0);
            exits.extend(self.compile_digit_range(n0, a, b, mo)?);
        }
        Ok(AutoOut { start, exits })
    }

    /// 正数幅度区间 [a, b]（a ≥ 1）的十进制 digit-DP；entry 直接接首数字 1-9。
    fn compile_digit_range(
        &mut self,
        entry: usize,
        a: u64,
        b: u64,
        mo: Option<&MultipleOf>,
    ) -> Result<Vec<usize>, ZealotError> {
        let a_dig = decimal_digits(a);
        let b_dig = decimal_digits(b);
        let la = a_dig.len();
        let lb = b_dig.len();
        let (div, scale) = mo.map(|m| (m.divisor, m.scale)).unwrap_or((0, 1));
        let rem_count = div.max(1) as usize;
        let cols = 9usize; // ca × cb
        let rows = lb * cols * rem_count; // pos ∈ 1..=lb
        if rows > MAX_AUTO_STATES {
            return Err(ZealotError::SchemaCompileError(format!(
                "multipleOf {div} 过大（区间 digit-DP 需 {rows} 状态 > 上限 {MAX_AUTO_STATES}）"
            )));
        }
        // 预计算 alive（自后向前）：状态可收尾或可续接到活状态；死胡同不建不接——
        // 否则「可到达但必死」前缀（如 [0,10] 的 "11"）导致 mask 全空 → 引擎 fallback。
        let mut alive = vec![false; rows];
        let idxv = |pos: usize, ca: u8, cb: u8, rem: u64| {
            ((pos - 1) * cols + (ca as usize * 3 + cb as usize)) * rem_count + rem as usize
        };
        let cmp3 = |d: u8, r: u8| -> u8 {
            if d < r {
                0
            } else if d == r {
                1
            } else {
                2
            }
        };
        let range_ok = |pos: usize, ca: u8, cb: u8| match pos {
            k if k == la => ca != 0 && (la < lb || cb != 2),
            k if k > la && k < lb => true,
            k if k == lb => cb != 2 && (lb > la || ca != 0),
            _ => false,
        };
        let trans = |pos: usize, ca: u8, cb: u8, d: u8| -> Option<(u8, u8)> {
            let ca2 = if ca == 1 {
                if pos < la {
                    cmp3(d, a_dig[pos])
                } else {
                    2 // 超过 la 位 → 必 > A
                }
            } else {
                ca
            };
            let cb2 = if cb == 1 {
                if pos < lb {
                    cmp3(d, b_dig[pos])
                } else {
                    return None; // 超过 lb 位 → 必 > B，死
                }
            } else {
                cb
            };
            Some((ca2, cb2))
        };
        for ca in 0..3u8 {
            for cb in 0..3u8 {
                for rem in 0..rem_count {
                    alive[idxv(lb, ca, cb, rem as u64)] =
                        range_ok(lb, ca, cb) && (div == 0 || rem == 0);
                }
            }
        }
        for pos in (1..lb).rev() {
            for ca in 0..3u8 {
                for cb in 0..3u8 {
                    for rem in 0..rem_count {
                        let mut a = range_ok(pos, ca, cb) && (div == 0 || rem == 0);
                        if !a {
                            for d in 0..10u8 {
                                if let Some((ca2, cb2)) = trans(pos, ca, cb, d) {
                                    let rem2 = if div == 0 {
                                        0
                                    } else {
                                        (rem as u64 * 10 + d as u64 * scale) % div
                                    };
                                    if alive[idxv(pos + 1, ca2, cb2, rem2)] {
                                        a = true;
                                        break;
                                    }
                                }
                            }
                        }
                        alive[idxv(pos, ca, cb, rem as u64)] = a;
                    }
                }
            }
        }
        // 只创建活状态（顺序建表，索引 O(1)）
        let mut state_of = vec![usize::MAX; rows];
        for (i, a) in alive.iter().enumerate() {
            if *a {
                state_of[i] = self.new_state()?;
            }
        }
        // 首数字（1-9）从 entry 进 DP(1, ·)——仅活目标
        for d in 1..=9u8 {
            let ca = if d < a_dig[0] {
                0
            } else if d == a_dig[0] {
                1
            } else {
                2
            };
            let cb = if d < b_dig[0] {
                0
            } else if d == b_dig[0] {
                1
            } else {
                2
            };
            let rem = if div == 0 {
                0
            } else {
                (d as u64 * scale) % div
            };
            let t = idxv(1, ca, cb, rem);
            if alive[t] {
                self.add_edge(entry, b'0' + d, state_of[t]);
            }
        }
        // 转移（仅活目标）与收尾
        let mut exit_states = Vec::new();
        for pos in 1..=lb {
            for ca in 0..3u8 {
                for cb in 0..3u8 {
                    for rem in 0..rem_count {
                        let i = idxv(pos, ca, cb, rem as u64);
                        if !alive[i] {
                            continue;
                        }
                        let cur = state_of[i];
                        if pos < lb {
                            for d in 0..10u8 {
                                if let Some((ca2, cb2)) = trans(pos, ca, cb, d) {
                                    let rem2 = if div == 0 {
                                        0
                                    } else {
                                        (rem as u64 * 10 + d as u64 * scale) % div
                                    };
                                    let t = idxv(pos + 1, ca2, cb2, rem2);
                                    if alive[t] {
                                        self.add_edge(cur, b'0' + d, state_of[t]);
                                    }
                                }
                            }
                        }
                        if range_ok(pos, ca, cb) && (div == 0 || rem == 0) {
                            self.set_final(cur);
                            exit_states.push(cur);
                        }
                    }
                }
            }
        }
        Ok(exit_states)
    }
}

/// M5-⑥/⑦：字节的码点数贡献（UTF-8 首字节/ASCII = 1，连续字节 = 0）。
fn cp_count(b: u8) -> usize {
    if (0x80..0xC0).contains(&b) {
        0
    } else {
        1
    }
}

fn decimal_digits(mut n: u64) -> Vec<u8> {
    if n == 0 {
        return vec![0];
    }
    let mut v = Vec::new();
    while n > 0 {
        v.push((n % 10) as u8);
        n /= 10;
    }
    v.reverse();
    v
}

#[pymethods]
impl ConstrainedGrammar {
    fn allowed_tokens(&self, state: usize) -> PyResult<Vec<i32>> {
        if state >= self.state_count && self.tokenizer.is_none() {
            return Err(ZealotError::InvalidConstraint(format!(
                "State {} out of range (max {})",
                state, self.state_count
            ))
            .into());
        }
        Ok(self.allowed_tokens_for(state, None))
    }

    fn advance(&self, current_state: usize, token_id: i32) -> PyResult<usize> {
        if current_state >= self.state_count && self.tokenizer.is_none() {
            return Err(ZealotError::InvalidConstraint(format!(
                "State {} out of range (max {})",
                current_state, self.state_count
            ))
            .into());
        }
        Ok(self.advance_state(current_state, token_id as i64, None))
    }

    fn is_valid_final(&self, state: usize) -> PyResult<bool> {
        if state >= self.state_count && self.tokenizer.is_none() {
            return Err(ZealotError::InvalidConstraint(format!(
                "State {} out of range (max {})",
                state, self.state_count
            ))
            .into());
        }
        Ok(self.is_valid_final_state(state, None))
    }

    /// 加载 HuggingFace tokenizer.json 词表（M5-③：allowed_tokens/advance 切换为真实词表）。
    fn load_tokenizer(&mut self, path: &str) -> PyResult<()> {
        let t = Tokenizer::load(path)
            .map_err(|e| ZealotError::SchemaCompileError(format!("tokenizer load failed: {e}")))?;
        self.tokenizer = Some(t);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("ConstrainedGrammar(states={})", self.state_count)
    }
}

impl ConstrainedGrammar {
    fn compute_allowed_tokens(&self, _state: usize) -> Vec<i32> {
        // 无词表时的 ASCII 占位（token ID ≈ 字节值）。真实词表校验走
        // allowed_tokens_vocab（M5-③），本函数仅 PyO3 未 load_tokenizer 时触达。
        match &self.grammar_type {
            GrammarType::Any => {
                // All common tokens: whitespace, punctuation, alphanumerics
                (32..=126).collect()
            }
            GrammarType::String {
                min_length,
                max_length: _,
                pattern: _,
                enum_values,
                format: _,
            } => {
                let mut tokens: Vec<i32> = vec![34]; // opening quote
                if *min_length > 0 {
                    tokens.extend(65..=122); // A-Z, a-z
                    tokens.extend(48..=57); // 0-9
                }
                tokens.push(34); // closing quote
                if let Some(enums) = enum_values {
                    for v in enums {
                        tokens.push(v.len() as i32 + 1000); // placeholder: real impl maps to token IDs
                    }
                }
                tokens
            }
            GrammarType::Number => {
                let mut tokens = vec!['-' as i32];
                tokens.extend(48..=57); // 0-9
                tokens.push('.' as i32);
                tokens.push('e' as i32);
                tokens.push('E' as i32);
                tokens
            }
            GrammarType::Integer { .. } => {
                let mut tokens: Vec<i32> = vec!['-' as i32];
                tokens.extend(48..=57);
                tokens
            }
            GrammarType::Boolean => {
                vec![
                    't' as i32, 'r' as i32, 'u' as i32, 'e' as i32, 'f' as i32, 'a' as i32,
                    'l' as i32, 's' as i32, 'e' as i32,
                ]
            }
            GrammarType::Null => {
                vec!['n' as i32, 'u' as i32, 'l' as i32, 'l' as i32]
            }
            GrammarType::Object {
                properties,
                required: _,
                additional_properties: _,
            } => {
                let mut tokens = vec!['{' as i32, '}' as i32, ',' as i32, ':' as i32, '"' as i32];
                tokens.extend(32..=126);
                for key in properties.keys() {
                    tokens.push(key.len() as i32 + 2000); // placeholder token IDs for property names
                }
                tokens
            }
            GrammarType::Array {
                min_items: _,
                max_items: _,
                item_type: _,
            } => {
                vec!['[' as i32, ']' as i32, ',' as i32]
            }
            GrammarType::Enum { values } => {
                let mut tokens = vec!['"' as i32];
                tokens.extend(65..=122);
                tokens.push('"' as i32);
                for v in values {
                    if let Some(s) = v.as_str() {
                        tokens.push(s.len() as i32 + 1000);
                    }
                }
                tokens
            }
            GrammarType::Union { branches } => {
                let mut tokens = Vec::new();
                for branch in branches {
                    let branch_grammar = ConstrainedGrammar::new(branch.clone());
                    tokens.extend(branch_grammar.compute_allowed_tokens(0));
                }
                tokens.sort_unstable();
                tokens.dedup();
                tokens
            }
        }
    }

    fn compute_is_valid_final(&self, state: usize) -> bool {
        match &self.grammar_type {
            GrammarType::Any => true,
            GrammarType::String { min_length, .. } => {
                // For 1-state grammar: if min_length is 0, string can be empty (valid at state 0)
                *min_length == 0
            }
            GrammarType::Number | GrammarType::Integer { .. } => {
                true // at state 0, any numeric token is valid
            }
            GrammarType::Boolean | GrammarType::Null => {
                true // state 0 = valid final (word complete)
            }
            GrammarType::Object { required, .. } => {
                // 真状态机语义（issue #15）：state = 已遍历的属性槽位数。
                // required 键全部遍历完（state >= required.len()）才可合法收尾；
                // 可选字段（不在 required）允许缺省。
                state >= required.len()
            }
            GrammarType::Array { .. } => {
                true // empty array is valid at state 0
            }
            GrammarType::Enum { .. } => {
                true // at state 0, any enum value is valid
            }
            GrammarType::Union { branches } => branches.iter().any(|b| {
                let g = ConstrainedGrammar::new(b.clone());
                g.compute_is_valid_final(0)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_allowed_tokens_includes_quotes() {
        let g = ConstrainedGrammar::new(GrammarType::String {
            min_length: 1,
            max_length: Some(100),
            pattern: None,
            enum_values: None,
            format: None,
        });
        let tokens = g.allowed_tokens(0).unwrap();
        assert!(tokens.contains(&34)); // opening quote
    }

    #[test]
    fn test_boolean_allowed_tokens() {
        let g = ConstrainedGrammar::new(GrammarType::Boolean);
        let tokens = g.allowed_tokens(0).unwrap();
        assert!(tokens.contains(&('t' as i32)));
        assert!(tokens.contains(&('f' as i32)));
    }

    #[test]
    fn test_number_is_valid_final() {
        let g = ConstrainedGrammar::new(GrammarType::Number);
        // Number with 1 state: state 0 is the only valid state, representing a complete number
        assert!(g.is_valid_final(0).unwrap());
        let err = g.is_valid_final(1).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_object_allows_brace() {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            GrammarType::String {
                min_length: 1,
                max_length: None,
                pattern: None,
                enum_values: None,
                format: None,
            },
        );
        let g = ConstrainedGrammar::new(GrammarType::Object {
            properties: props,
            required: vec!["name".to_string()],
            additional_properties: false,
        });
        let tokens = g.allowed_tokens(0).unwrap();
        assert!(tokens.contains(&('{' as i32)));
        assert!(tokens.contains(&('}' as i32)));
    }

    #[test]
    fn test_state_boundary() {
        let g = ConstrainedGrammar::new(GrammarType::Boolean);
        let err = g.allowed_tokens(999).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    fn object_grammar(props: Vec<(&str, GrammarType)>, required: Vec<&str>) -> ConstrainedGrammar {
        let mut map = HashMap::new();
        for (k, v) in props {
            map.insert(k.to_string(), v);
        }
        ConstrainedGrammar::new(GrammarType::Object {
            properties: map,
            required: required.into_iter().map(String::from).collect(),
            additional_properties: false,
        })
    }

    fn string_leaf() -> GrammarType {
        GrammarType::String {
            min_length: 1,
            max_length: None,
            pattern: None,
            enum_values: None,
            format: None,
        }
    }

    #[test]
    fn test_object_required_keys_must_be_traversed_before_final() {
        // issue #15：required 键全部遍历完（state >= required.len()）才可合法收尾。
        let g = object_grammar(
            vec![
                ("name", string_leaf()),
                (
                    "age",
                    GrammarType::Integer {
                        min: None,
                        max: None,
                        multiple_of: None,
                    },
                ),
            ],
            vec!["name"],
        );
        assert!(
            !g.is_valid_final(0).unwrap(),
            "根状态不能收尾（required 未遍历）"
        );
        assert!(
            g.is_valid_final(1).unwrap(),
            "已遍历 1 个槽位 ≥ required 数"
        );
        assert!(g.is_valid_final(2).unwrap());
    }

    #[test]
    fn test_object_two_required_need_two_slots() {
        let g = object_grammar(
            vec![
                ("a", string_leaf()),
                ("b", string_leaf()),
                ("c", GrammarType::Boolean),
            ],
            vec!["a", "b"],
        );
        assert!(!g.is_valid_final(0).unwrap());
        assert!(!g.is_valid_final(1).unwrap(), "只遍历 1 个槽位，b 未满足");
        assert!(g.is_valid_final(2).unwrap());
        assert!(g.is_valid_final(3).unwrap());
    }

    #[test]
    fn test_object_optional_fields_may_be_omitted() {
        // 无 required：根状态即可合法收尾（空对象 {} 也是合法输出）。
        let g = object_grammar(
            vec![
                ("name", string_leaf()),
                (
                    "age",
                    GrammarType::Integer {
                        min: None,
                        max: None,
                        multiple_of: None,
                    },
                ),
            ],
            vec![],
        );
        assert!(g.is_valid_final(0).unwrap());
        assert!(g.is_valid_final(1).unwrap());
    }

    #[test]
    fn test_object_required_missing_from_properties_never_final() {
        // required 引用不存在的键：任何状态都无法收尾（正确语义——该对象不可构造）。
        let g = object_grammar(vec![], vec!["missing"]);
        assert!(!g.is_valid_final(0).unwrap());
    }

    #[test]
    fn test_count_states_recursive_nested_object() {
        // issue #15：state_count 递归计入嵌套属性。
        let inner = object_grammar(vec![("x", GrammarType::Boolean)], vec![]); // 1 + 1 = 2
        let outer = object_grammar(vec![("inner", inner.grammar_type.clone())], vec![]); // 1 + 2 = 3
        assert_eq!(outer.state_count(), 3);
    }

    #[test]
    fn test_count_states_recursive_array_and_union() {
        let arr = ConstrainedGrammar::new(GrammarType::Array {
            min_items: 0,
            max_items: None,
            item_type: Some(Box::new(
                object_grammar(vec![("x", GrammarType::Boolean)], vec![]).grammar_type,
            )), // 1 + 2 = 3
        });
        assert_eq!(arr.state_count(), 3);

        let uni = ConstrainedGrammar::new(GrammarType::Union {
            branches: vec![string_leaf(), object_grammar(vec![], vec![]).grammar_type], // 1 + 1 = 2
        });
        assert_eq!(uni.state_count(), 2);
    }

    // ── M5-③：字节级自动机 ──────────────────────────────────────────────

    /// 走完字节序列，返回 (成功, 终态是否 final)。
    fn walk(g: &ConstrainedGrammar, bytes: &[u8]) -> (bool, bool) {
        let a = g.automaton.as_ref().expect("automaton");
        match a.accept(0, bytes) {
            Some(s) => (true, a.is_final(s)),
            None => (false, false),
        }
    }

    #[test]
    fn test_automaton_string() {
        let g = ConstrainedGrammar::new(string_leaf());
        assert_eq!(walk(&g, b"\"abc\""), (true, true), "完整字符串");
        assert_eq!(
            walk(&g, b"\"abc"),
            (true, false),
            "中途 token 合法（后续续写）"
        );
        assert_eq!(walk(&g, b"\"a\\\"b\""), (true, true), "转义引号不终止");
        assert_eq!(walk(&g, b"abc"), (false, false), "缺开头引号");
    }

    #[test]
    fn test_automaton_number() {
        let g = ConstrainedGrammar::new(GrammarType::Number);
        for ok in [b"12".as_slice(), b"-1.5e3", b"0", b"1E-2"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "{} 应合法",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"e3".as_slice(), b"--1", b".5"] {
            assert!(
                !walk(&g, bad).1,
                "{} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 中途 token：`1.`/`1e` 可续写（`1.5`/`1e-2`），但作为完整值不合法
        assert_eq!(walk(&g, b"1."), (true, false));
        assert_eq!(walk(&g, b"1e"), (true, false));
    }

    #[test]
    fn test_automaton_integer_unbounded() {
        // M5-⑤：integer 与 number 分离——不接受小数点/指数
        let g = ConstrainedGrammar::new(GrammarType::Integer {
            min: None,
            max: None,
            multiple_of: None,
        });
        for ok in [b"0".as_slice(), b"42", b"-7", b"123456789"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "{} 应合法",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"1.5".as_slice(), b"1e3", b"e3", b"--1", b".5"] {
            assert!(
                !walk(&g, bad).1,
                "{} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn test_automaton_integer_ranges() {
        let mk = |min: Option<i64>, max: Option<i64>| {
            ConstrainedGrammar::new(GrammarType::Integer {
                min,
                max,
                multiple_of: None,
            })
        };
        // [0, 10]
        let g = mk(Some(0), Some(10));
        for ok in [b"0".as_slice(), b"1", b"5", b"10"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "[0,10] {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"-1".as_slice(), b"11", b"00", b"007"] {
            assert!(
                !walk(&g, bad).1,
                "[0,10] {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // [-10, -2]
        let g = mk(Some(-10), Some(-2));
        for ok in [b"-2".as_slice(), b"-10"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "[-10,-2] {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"-1".as_slice(), b"-11", b"0", b"5", b"-0"] {
            assert!(
                !walk(&g, bad).1,
                "[-10,-2] {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // [-3, 5]
        let g = mk(Some(-3), Some(5));
        for ok in [b"-3".as_slice(), b"-2", b"0", b"3", b"5"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "[-3,5] {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"-4".as_slice(), b"6"] {
            assert!(
                !walk(&g, bad).1,
                "[-3,5] {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 单边：min-only / max-only
        let g = mk(Some(5), None);
        for ok in [b"5".as_slice(), b"9223372036854775807"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "min5 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"4".as_slice(), b"-1", b"0"] {
            assert!(
                !walk(&g, bad).1,
                "min5 {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        let g = mk(None, Some(100));
        for ok in [b"-999".as_slice(), b"0", b"100"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "max100 {}",
                String::from_utf8_lossy(ok)
            );
        }
        let bad = b"101";
        assert!(
            !walk(&g, bad).1,
            "max100 {} 应非法收尾",
            String::from_utf8_lossy(bad)
        );
        // 单值
        let g = mk(Some(3), Some(3));
        assert_eq!(walk(&g, b"3"), (true, true));
        for bad in [b"2".as_slice(), b"4", b"03"] {
            assert!(
                !walk(&g, bad).1,
                "[3,3] {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn test_automaton_integer_multiple_of() {
        let mk = |min: Option<i64>, max: Option<i64>, mo: Option<(u64, u64)>| {
            ConstrainedGrammar::new(GrammarType::Integer {
                min,
                max,
                multiple_of: mo.map(|(divisor, scale)| MultipleOf { divisor, scale }),
            })
        };
        // [0, 30] × 3
        let g = mk(Some(0), Some(30), Some((3, 1)));
        for ok in [b"0".as_slice(), b"3", b"9", b"27", b"30"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "[0,30]x3 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"4".as_slice(), b"31", b"-3"] {
            assert!(
                !walk(&g, bad).1,
                "[0,30]x3 {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 无区间 × 7（正负两侧）
        let g = mk(None, None, Some((7, 1)));
        for ok in [b"0".as_slice(), b"7", b"14", b"-7", b"-14", b"77"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "x7 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"8".as_slice(), b"-8", b"6"] {
            assert!(
                !walk(&g, bad).1,
                "x7 {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 小数 multipleOf 2.5 → (divisor 25, scale 10)：整数里即 5 的倍数
        let g = mk(None, None, Some((25, 10)));
        for ok in [b"5".as_slice(), b"-5", b"10", b"15", b"0"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "x2.5 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"4".as_slice(), b"6", b"-4"] {
            assert!(
                !walk(&g, bad).1,
                "x2.5 {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 负区间 × 2
        let g = mk(Some(-10), Some(-2), Some((2, 1)));
        for ok in [b"-10".as_slice(), b"-8", b"-2", b"-4"] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "[-10,-2]x2 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"-9".as_slice(), b"-3", b"-1"] {
            assert!(
                !walk(&g, bad).1,
                "[-10,-2]x2 {} 应非法收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 大数边界：max 10^18 × 1（multipleOf 恒真）
        let g = mk(Some(0), Some(1_000_000_000_000_000_000), Some((1, 1)));
        for ok in [
            b"0".as_slice(),
            b"999999999999999999",
            b"1000000000000000000",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "big {}",
                String::from_utf8_lossy(ok)
            );
        }
    }

    #[test]
    fn test_integer_empty_range_and_oversize_rejected() {
        // 空区间 / multipleOf 过大 → 编译错误（不静默、不产出死自动机）
        let empty = GrammarType::Integer {
            min: Some(10),
            max: Some(5),
            multiple_of: None,
        };
        assert!(ByteAutomaton::compile(&empty).is_err(), "空区间应报错");
        let oversize = GrammarType::Integer {
            min: Some(0),
            max: Some(100),
            multiple_of: Some(MultipleOf {
                divisor: 10000,
                scale: 1,
            }),
        };
        assert!(
            ByteAutomaton::compile(&oversize).is_err(),
            "multipleOf 过大应报错"
        );
        let g = ConstrainedGrammar::new(empty.clone());
        assert!(g.automaton_rejected(), "空区间 grammar 应标记拒绝");
    }

    #[test]
    fn test_automaton_boolean_null() {
        let b = ConstrainedGrammar::new(GrammarType::Boolean);
        assert_eq!(walk(&b, b"true"), (true, true));
        assert_eq!(walk(&b, b"false"), (true, true));
        assert_eq!(walk(&b, b"tru"), (true, false), "中途 token 合法");
        let n = ConstrainedGrammar::new(GrammarType::Null);
        assert_eq!(walk(&n, b"null"), (true, true));
        assert_eq!(walk(&n, b"nil"), (false, false));
    }

    #[test]
    fn test_automaton_string_lengths() {
        // M5-⑥：minLength/maxLength 计数执行（码点计数）
        let mk = |min: usize, max: Option<u64>| {
            ConstrainedGrammar::new(GrammarType::String {
                min_length: min,
                max_length: max,
                pattern: None,
                enum_values: None,
                format: None,
            })
        };
        // 仅 min
        let g = mk(3, None);
        for ok in [b"\"abc\"".as_slice(), b"\"abcdef\"", b"\"\xe4\xb8\xadab\""] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "min3 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [
            b"\"ab\"".as_slice(),
            b"\"\"",
            b"\"\xe4\xb8\xada\"",
            b"\"\\u0041\"",
        ] {
            assert!(
                !walk(&g, bad).1,
                "min3 {} 应不可收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 仅 max
        let g = mk(0, Some(2));
        for ok in [b"\"\"".as_slice(), b"\"ab\""] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "max2 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"\"abc\"".as_slice(), b"\"abcd\""] {
            assert!(
                !walk(&g, bad).0,
                "max2 {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
        // min+max
        let g = mk(2, Some(4));
        for ok in [b"\"ab\"".as_slice(), b"\"abcd\""] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "2..4 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"\"a\"".as_slice(), b"\"abcde\""] {
            assert!(
                !walk(&g, bad).1,
                "2..4 {} 应不可收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // max==0
        let g = mk(0, Some(0));
        assert_eq!(walk(&g, b"\"\""), (true, true));
        assert!(!walk(&g, b"\"a\"").0);
    }

    #[test]
    fn test_automaton_string_lengths_escapes_unicode() {
        let mk = |min: usize, max: Option<u64>| {
            ConstrainedGrammar::new(GrammarType::String {
                min_length: min,
                max_length: max,
                pattern: None,
                enum_values: None,
                format: None,
            })
        };
        // 转义计数：`\\n` = 1、`\\uXXXX` = 1
        let g = mk(2, None);
        assert_eq!(walk(&g, b"\"a\\nb\""), (true, true), "a+\\n+b = 3");
        assert_eq!(
            walk(&g, b"\"\\u0041\""),
            (false, false),
            "\\u0041 单码点 < 2"
        );
        assert_eq!(
            walk(&g, b"\"a\\u0041b\""),
            (true, true),
            "a+\\u0041+b = 3 ≥ 2"
        );
        // 严格 hex：`\\uZZZZ` 拒绝（计数形态）
        assert!(!walk(&g, b"\"a\\uZZZZb\"").0);
        // UTF-8 码点：`中`（3 字节）= 1 码点；孤立连续字节 = 0
        assert_eq!(
            walk(&g, b"\"\xe4\xb8\xad\xe4\xb8\xad\""),
            (true, true),
            "2 码点"
        );
        assert_eq!(
            walk(&g, b"\"\xe4\xb8\xad\""),
            (false, false),
            "1 码点 < 2 不可收尾"
        );
        // max 封顶处不可再转义（续写被拒）
        let g = mk(0, Some(3));
        assert!(!walk(&g, b"\"abc\\n\"").0, "3+1 > 3");
        assert_eq!(walk(&g, b"\"abc\""), (true, true));
        // 前缀语义：min 未满足时中途 token 仍可续写
        assert_eq!(walk(&g, b"\"ab"), (true, false));
        let g = mk(3, None);
        assert_eq!(walk(&g, b"\"ab"), (true, false), "前缀合法");
        // 仅 min 冻结：k == min 后任意续写
        let g = mk(1, None);
        for ok in [b"\"a\"".as_slice(), b"\"abcdefgh\""] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "min1 {}",
                String::from_utf8_lossy(ok)
            );
        }
    }

    #[test]
    fn test_string_length_invalid_ranges_error() {
        // min > max → 空区间错误；过大 horizon → 错误
        let bad1 = GrammarType::String {
            min_length: 5,
            max_length: Some(2),
            pattern: None,
            enum_values: None,
            format: None,
        };
        assert!(ByteAutomaton::compile(&bad1).is_err(), "min > max 应报错");
        let bad2 = GrammarType::String {
            min_length: 0,
            max_length: Some(MAX_LENGTH_COUNTER as u64 + 1),
            pattern: None,
            enum_values: None,
            format: None,
        };
        assert!(ByteAutomaton::compile(&bad2).is_err(), "horizon 过大应报错");
        let bad3 = GrammarType::String {
            min_length: MAX_LENGTH_COUNTER + 1,
            max_length: None,
            pattern: None,
            enum_values: None,
            format: None,
        };
        assert!(ByteAutomaton::compile(&bad3).is_err(), "min 过大应报错");
        let g = ConstrainedGrammar::new(bad1);
        assert!(g.automaton_rejected());
    }

    #[test]
    fn test_string_lengths_compose_with_pattern() {
        // M5-⑥：长度与 pattern 可组合（pattern 执行随 #22；此处长度照常生效）
        let g = ConstrainedGrammar::new(GrammarType::String {
            min_length: 2,
            max_length: Some(3),
            pattern: Some("^[a-z]+$".into()),
            enum_values: None,
            format: None,
        });
        assert_eq!(walk(&g, b"\"ab\""), (true, true));
        assert_eq!(
            walk(&g, b"\"abcd\""),
            (false, false),
            "超 max 直接死（不可续不可收尾）"
        );
        assert_eq!(walk(&g, b"\"a\""), (false, false), "低于 min 不可收尾");
    }

    #[test]
    fn test_automaton_enum() {
        let g = ConstrainedGrammar::new(GrammarType::Enum {
            values: serde_json::from_str(r#"["red", "green", 1]"#).unwrap(),
        });
        assert_eq!(walk(&g, b"\"red\""), (true, true));
        assert_eq!(walk(&g, b"\"green\""), (true, true));
        assert_eq!(walk(&g, b"\"blue\""), (false, false));
        assert_eq!(walk(&g, b"1"), (true, true), "非字符串枚举值");
    }

    #[test]
    fn test_automaton_format_email() {
        let g = fmt_grammar(StringFormat::Email);
        for ok in [
            b"\"a@b.c\"".as_slice(),
            b"\"user.name+tag@example-domain.co.uk\"",
            b"\"a!b@c\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "{}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [
            b"\"a\"".as_slice(),
            b"\"a@\"",
            b"\"@b.c\"",
            b"\".a@b.c\"",
            b"\"a.@b.c\"",
            b"\"a@b.\"",
            b"\"a b@c\"",
        ] {
            assert!(!walk(&g, bad).1, "{} 应非法", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn test_automaton_format_date_time() {
        let g = fmt_grammar(StringFormat::Date);
        for ok in [
            b"\"2024-02-29\"".as_slice(),
            b"\"0000-01-01\"",
            b"\"9999-12-31\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "date {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [
            b"\"2024-13-01\"".as_slice(),
            b"\"2024-00-01\"",
            b"\"2024-01-32\"",
            b"\"2024-01-00\"",
            b"\"24-01-01\"",
            b"\"2024/01/01\"",
        ] {
            assert!(
                !walk(&g, bad).1,
                "date {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
        let g = fmt_grammar(StringFormat::DateTime);
        for ok in [
            b"\"2024-01-01T00:00:00Z\"".as_slice(),
            b"\"2024-01-01t23:59:59+08:00\"",
            b"\"2024-01-01T12:00:00.123Z\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "dt {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [
            b"\"2024-01-01\"".as_slice(),
            b"\"2024-01-01 00:00:00Z\"",
            b"\"2024-01-01T24:00:00Z\"",
            b"\"2024-01-01T12:60:00Z\"",
        ] {
            assert!(
                !walk(&g, bad).1,
                "dt {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
        let g = fmt_grammar(StringFormat::Time);
        for ok in [
            b"\"00:00:00\"".as_slice(),
            b"\"23:59:59\"",
            b"\"12:00:00.5Z\"",
            b"\"01:02:03+05:30\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "time {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"\"24:00:00\"".as_slice(), b"\"12:00\"", b"\"12:00:60\""] {
            assert!(
                !walk(&g, bad).1,
                "time {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn test_automaton_format_uuid_ip() {
        let g = fmt_grammar(StringFormat::Uuid);
        let ok = b"\"123e4567-e89b-12d3-a456-426614174000\"";
        assert_eq!(walk(&g, ok), (true, true));
        for bad in [
            b"\"123e4567e89b12d3a456426614174000\"".as_slice(),
            b"\"123e4567-e89b-12d3-a456-42661417400g\"",
            b"\"123e4567-e89b-12d3-a456-4266141740\"",
        ] {
            assert!(
                !walk(&g, bad).1,
                "uuid {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
        let g = fmt_grammar(StringFormat::Ipv4);
        for ok in [
            b"\"0.0.0.0\"".as_slice(),
            b"\"255.255.255.255\"",
            b"\"192.168.1.1\"",
            b"\"1.2.3.4\"",
            b"\"10.0.0.255\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "ipv4 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [
            b"\"256.0.0.1\"".as_slice(),
            b"\"01.2.3.4\"",
            b"\"1.2.3\"",
            b"\"1.2.3.4.5\"",
            b"\"255.255.255.256\"",
            b"\"1.2.3.-4\"",
        ] {
            assert!(
                !walk(&g, bad).1,
                "ipv4 {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
        let g = fmt_grammar(StringFormat::Ipv6);
        for ok in [
            b"\"2001:0db8:85a3:0000:0000:8a2e:0370:7334\"".as_slice(),
            b"\"::1\"",
            b"\"::\"",
            b"\"fe80::1\"",
            b"\"1:2:3:4:5:6:7:8\"",
            b"\"1::\"",
            b"\"::ffff:1:2\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "ipv6 {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [
            b"\"1:2:3\"".as_slice(),
            b"\"1:2:3:4:5:6:7:8:9\"",
            b"\"1::2::3\"",
            b"\"1:2:3:4:5:6:7:8:\"",
            b"\"12345::\"",
            b"\"1::2:\"",
            b"\"g::1\"",
        ] {
            assert!(
                !walk(&g, bad).1,
                "ipv6 {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn test_automaton_format_hostname_uri() {
        let g = fmt_grammar(StringFormat::Hostname);
        for ok in [
            b"\"localhost\"".as_slice(),
            b"\"example.com\"",
            b"\"sub-domain.example.co.uk\"",
            b"\"a1-b2.c3\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "hn {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [
            b"\"-a.com\"".as_slice(),
            b"\"a-.com\"",
            b"\"a..b\"",
            b"\"a.\"",
            b"\"a b.com\"",
        ] {
            assert!(
                !walk(&g, bad).1,
                "hn {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
        let g = fmt_grammar(StringFormat::Uri);
        for ok in [
            b"\"https://example.com/path?q=1#f\"".as_slice(),
            b"\"mailto:user@example.com\"",
            b"\"urn:isbn:0451450523\"",
            b"\"a:/b\"",
        ] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "uri {}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"\"1http://x\"".as_slice(), b"\"http:\\x\"", b"\"a b:c\""] {
            assert!(
                !walk(&g, bad).1,
                "uri {} 应非法",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn test_automaton_format_with_length() {
        // format × 长度组合
        let g = ConstrainedGrammar::new(GrammarType::String {
            min_length: 8,
            max_length: Some(12),
            pattern: None,
            enum_values: None,
            format: Some(StringFormat::Date),
        });
        assert_eq!(
            walk(&g, b"\"2024-01-01\""),
            (true, true),
            "10 码点 ∈ [8,12]"
        );
        assert_eq!(walk(&g, b"\"2024-1-1\""), (false, false), "非法日期");
        let g = ConstrainedGrammar::new(GrammarType::String {
            min_length: 10,
            max_length: None,
            pattern: None,
            enum_values: None,
            format: Some(StringFormat::Date),
        });
        assert_eq!(walk(&g, b"\"2024-01-01\""), (true, true), "min-only 冻结");
        assert_eq!(walk(&g, b"\"2024-01-0"), (true, false), "前缀可续写");
    }

    /// format 测试辅助
    fn fmt_grammar(f: StringFormat) -> ConstrainedGrammar {
        ConstrainedGrammar::new(GrammarType::String {
            min_length: 0,
            max_length: None,
            pattern: None,
            enum_values: None,
            format: Some(f),
        })
    }

    #[test]
    fn test_automaton_string_pattern() {
        // M5-⑧：锚定语义 + 常用构造
        let mk = |pat: &str, min: usize, max: Option<u64>| {
            ConstrainedGrammar::new(GrammarType::String {
                min_length: min,
                max_length: max,
                pattern: Some(pat.into()),
                enum_values: None,
                format: None,
            })
        };
        let g = mk(r"\d{3}-\d{2}", 0, None);
        assert_eq!(walk(&g, b"\"123-45\""), (true, true));
        assert_eq!(walk(&g, b"\"12-345\""), (false, false));
        assert_eq!(walk(&g, b"\"123-4"), (true, false), "中途 token 可续写");
        let g = mk(r"[a-z]+", 0, None);
        assert_eq!(walk(&g, b"\"abc\""), (true, true));
        assert_eq!(walk(&g, b"\"a1\""), (false, false));
        assert_eq!(walk(&g, b"\"ABC\""), (false, false));
        let g = mk(r"(a|b)c", 0, None);
        for ok in [b"\"ac\"".as_slice(), b"\"bc\""] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "{}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"\"cc\"".as_slice(), b"\"a\""] {
            assert!(
                !walk(&g, bad).1,
                "{} 应不可收尾",
                String::from_utf8_lossy(bad)
            );
        }
        let g = mk(r"a{2,3}", 0, None);
        for ok in [b"\"aa\"".as_slice(), b"\"aaa\""] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "{}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"\"a\"".as_slice(), b"\"aaaa\""] {
            assert!(
                !walk(&g, bad).1,
                "{} 应不可收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 锚定语义：整体匹配（业界惯例）
        let g = mk("abc", 0, None);
        assert_eq!(walk(&g, b"\"abc\""), (true, true));
        for bad in [b"\"xabc\"".as_slice(), b"\"abcx\""] {
            assert!(
                !walk(&g, bad).1,
                "{} 应不可收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 显式首尾锚点等价
        let g = mk("^abc$", 0, None);
        assert_eq!(walk(&g, b"\"abc\""), (true, true));
        // 与长度组合
        let g = mk(r"[0-9]+", 2, Some(4));
        for ok in [b"\"12\"".as_slice(), b"\"1234\""] {
            assert_eq!(
                walk(&g, ok),
                (true, true),
                "{}",
                String::from_utf8_lossy(ok)
            );
        }
        for bad in [b"\"1\"".as_slice(), b"\"12345\""] {
            assert!(
                !walk(&g, bad).1,
                "{} 应不可收尾",
                String::from_utf8_lossy(bad)
            );
        }
        // 字面引号不可表示（JSON 裸引号边丢弃）：a.b 匹配 aXb、拒 a"b
        let g = mk("a.b", 0, None);
        assert_eq!(walk(&g, b"\"aXb\""), (true, true));
        assert!(!walk(&g, b"\"a\"b\"").0, "裸引号不可作为内容");
        // 空 pattern：只匹配空串
        let g = mk("", 0, None);
        assert_eq!(walk(&g, b"\"\""), (true, true));
        assert_eq!(walk(&g, b"\"a\""), (false, false));
        // 转义元字符：`a\.b` = a + 字面点 + b
        let g = mk(r"a\.b", 0, None);
        assert_eq!(walk(&g, b"\"a.b\""), (true, true));
        assert_eq!(walk(&g, b"\"axb\""), (false, false));
    }

    #[test]
    fn test_string_pattern_errors() {
        // 不支持特性 → 编译错误；pattern+format 冲突 → 错误
        let bad = |pat: &str| GrammarType::String {
            min_length: 0,
            max_length: None,
            pattern: Some(pat.into()),
            enum_values: None,
            format: None,
        };
        for pat in [r"(?=a)", r"(?!a)", r"(?<name>a)", r"a\1", r"a\b", "(", "["] {
            let g = bad(pat);
            assert!(
                ByteAutomaton::compile(&g).is_err(),
                "pattern {pat:?} 应报错"
            );
        }
        // 语言为空（要求裸反斜杠字面量）→ 显式报错
        let no_repr = GrammarType::String {
            min_length: 0,
            max_length: None,
            pattern: Some(r"a\\b".into()), // 内容需含字面 `\\`
            enum_values: None,
            format: None,
        };
        assert!(
            ByteAutomaton::compile(&no_repr).is_err(),
            "不可表示的语言应报错"
        );
        let conflict = GrammarType::String {
            min_length: 0,
            max_length: None,
            pattern: Some("a".into()),
            enum_values: None,
            format: Some(StringFormat::Date),
        };
        assert!(
            ByteAutomaton::compile(&conflict).is_err(),
            "pattern+format 应冲突"
        );
    }

    #[test]
    fn test_automaton_object_required_gating() {
        // required=[a, b]，可选 c：`}` 只在 a、b 都出现后可达；任意顺序。
        let g = object_grammar(
            vec![
                ("a", GrammarType::Number),
                ("b", GrammarType::Boolean),
                ("c", string_leaf()),
            ],
            vec!["a", "b"],
        );
        assert_eq!(walk(&g, b"{}"), (false, false), "required 未满足不能空对象");
        assert_eq!(walk(&g, b"{\"a\":1}"), (false, false), "b 缺失");
        assert_eq!(walk(&g, b"{\"a\":1,\"b\":true}"), (true, true));
        // 可选键穿插；required 乱序（b 先于 a）不推进计数器 → `}` 不可达
        assert_eq!(
            walk(&g, b"{\"c\":\"x\",\"b\":true,\"a\":1}"),
            (false, false),
            "required 乱序不计数"
        );
        assert_eq!(
            walk(&g, b"{\"a\":1,\"a\":2,\"b\":true}"),
            (false, false),
            "重复 required 键被拒（strict 唯一性）"
        );
        assert_eq!(
            walk(&g, b"{\"c\":\"x\",\"c\":\"x\",\"a\":1,\"b\":true}"),
            (true, true),
            "optional 键可重复（c 非 required）"
        );
        // 值类型校验
        assert_eq!(
            walk(&g, b"{\"a\":true,\"b\":true}"),
            (false, false),
            "a 必须是数字"
        );
        assert_eq!(
            walk(&g, b"{\"b\":true,\"a\":1,"),
            (true, false),
            "尾逗号是合法前缀（可续写），但不能收尾"
        );
    }

    #[test]
    fn test_automaton_object_optional_only() {
        let g = object_grammar(vec![("x", GrammarType::Boolean)], vec![]);
        assert_eq!(walk(&g, b"{}"), (true, true), "无 required 空对象合法");
        assert_eq!(walk(&g, b"{\"x\":false}"), (true, true));
    }

    #[test]
    fn test_automaton_array_min_max() {
        let arr = |min: usize, max: Option<u64>| {
            ConstrainedGrammar::new(GrammarType::Array {
                min_items: min,
                max_items: max,
                item_type: Some(Box::new(GrammarType::Number)),
            })
        };
        let g1 = arr(1, None);
        assert_eq!(walk(&g1, b"[]"), (false, false), "min=1 空数组非法");
        assert_eq!(walk(&g1, b"[1]"), (true, true));
        assert_eq!(walk(&g1, b"[1,2,3]"), (true, true));
        let g2 = arr(0, Some(2));
        assert_eq!(walk(&g2, b"[]"), (true, true));
        assert_eq!(walk(&g2, b"[1,2]"), (true, true));
        assert_eq!(walk(&g2, b"[1,2,3]"), (false, false), "超 max_items");
    }

    #[test]
    fn test_automaton_whitespace_boundaries() {
        let g = object_grammar(vec![("a", GrammarType::Number)], vec!["a"]);
        // 空白只在 token 边界合法
        assert_eq!(walk(&g, b"{ \"a\" : 1 }"), (true, true));
        assert_eq!(walk(&g, b"{ \"a\": 1}"), (true, true));
        assert_eq!(walk(&g, b"{\"a\": 1 }"), (true, true));
    }

    #[test]
    fn test_automaton_union_dispatch() {
        let g = ConstrainedGrammar::new(GrammarType::Union {
            branches: vec![GrammarType::Boolean, GrammarType::Number],
        });
        assert_eq!(walk(&g, b"true"), (true, true));
        assert_eq!(walk(&g, b"-42"), (true, true));
        assert_eq!(walk(&g, b"x"), (false, false));
    }

    #[test]
    fn test_vocab_mode_allowed_tokens_real_ids() {
        // 词表：{=0, }=1, "=2, :=3, ,=4, name=5, true=6（byte-level）
        let tok = Tokenizer::from_vocab(
            vec![
                Some("{".into()),
                Some("}".into()),
                Some("\"".into()),
                Some(":".into()),
                Some(",".into()),
                Some("name".into()),
                Some("true".into()),
            ],
            true,
        );
        let g = object_grammar(vec![("name", GrammarType::Boolean)], vec!["name"]);
        // state 0：只有 `{` 合法（required 未满足，`}` 被自动机拒绝）
        assert_eq!(g.allowed_tokens_for(0, Some(&tok)), vec![0]);
        // `{` → 真实状态推进
        let s1 = g.advance_state(0, 0, Some(&tok));
        assert_ne!(s1, 0);
        assert!(!g.is_valid_final_state(s1, Some(&tok)));
        // 完整序列：{ " name " : true }
        let mut s = 0;
        for t in [0_i64, 2, 5, 2, 3, 6, 1] {
            s = g.advance_state(s, t, Some(&tok));
        }
        assert!(g.is_valid_final_state(s, Some(&tok)));
        // 非法 token 防御：`{` 后紧跟 `{` 不推进
        let s2 = g.advance_state(0, 0, Some(&tok));
        assert_eq!(
            g.advance_state(s2, 0, Some(&tok)),
            s2,
            "`{{` 后不能再来 `{{`"
        );
    }

    #[test]
    fn test_root_exit_has_no_ws_after_completion() {
        // M5-④：根出口剥离 ws——收尾后 allowed 为空，引擎侧 eos 强制放行。
        // 嵌套值出口保留 ws（`Ġ,`/`Ġ}` 承接 token）。
        use crate::mock_runner::MOCK_TINY_VOCAB;
        let tok = Tokenizer::from_vocab(
            MOCK_TINY_VOCAB
                .iter()
                .map(|s| Some(s.to_string()))
                .collect(),
            true,
        );
        let g = object_grammar(vec![("name", GrammarType::Boolean)], vec!["name"]);
        let mut st = 0usize;
        for t in [9_i64, 11, 1, 11, 12, 3, 10] {
            st = g.advance_state(st, t, Some(&tok));
        }
        assert!(g.is_valid_final_state(st, Some(&tok)));
        // `\n`(14) 不再被允许（根出口 ws 已剥离）
        assert_eq!(g.allowed_tokens_for(st, Some(&tok)), Vec::<i32>::new());
        // 嵌套对象（值位置）的出口仍需 ws：`{"name":{"name":true}}` 走完内层
        // 后 ws 依然被承接（此处仅验证嵌套出口状态仍带 ws 位）
        let outer = object_grammar(
            vec![(
                "name",
                GrammarType::Object {
                    properties: {
                        let mut m = std::collections::HashMap::new();
                        m.insert("name".into(), GrammarType::Boolean);
                        m
                    },
                    required: vec!["name".into()],
                    additional_properties: false,
                },
            )],
            vec!["name"],
        );
        let mut st2 = 0usize;
        for t in [9_i64, 11, 1, 11, 12, 9, 11, 1, 11, 12, 3, 10, 10] {
            st2 = outer.advance_state(st2, t, Some(&tok));
        }
        assert!(outer.is_valid_final_state(st2, Some(&tok)));
        // 内层对象收尾（第 12 个 token，内层 done）——嵌套出口保留 ws（`\n`=14）
        let mut st3 = 0usize;
        for t in [9_i64, 11, 1, 11, 12, 9, 11, 1, 11, 12, 3, 10] {
            st3 = outer.advance_state(st3, t, Some(&tok));
        }
        assert!(
            outer.allowed_tokens_for(st3, Some(&tok)).contains(&14),
            "嵌套出口保留 ws 承接 \u{0122},/\u{0122}"
        );
    }

    #[test]
    fn test_automaton_cap_rejects_oversized_grammar() {
        // 大量长 required 键 → (r+1)×K 展开超 MAX_AUTO_STATES → automaton_rejected
        let mut props = HashMap::new();
        let mut req = Vec::new();
        for i in 0..120 {
            let key = format!("k{i}_{}", "x".repeat(40));
            props.insert(key.clone(), GrammarType::Boolean);
            req.push(key);
        }
        let g = ConstrainedGrammar::new(GrammarType::Object {
            properties: props,
            required: req,
            additional_properties: false,
        });
        assert!(g.automaton_rejected(), "超上限必须拒绝");
    }
}
