use lru::LruCache;
use pyo3::prelude::*;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::matcher::{ConstrainedGrammar, GrammarType, MultipleOf, StringFormat};
use crate::error::ZealotError;

const MAX_NESTING_DEPTH: usize = 32;

/// schema 哈希缓存（§6.2.1 / §12-5）。
///
/// 语义：`cache_size` 上限，容量满时淘汰**最久未使用**条目（真 LRU，`lru::LruCache`，
/// O(1) 命中提升 / 淘汰）；`bypass_cache=True` 不查缓存、不写缓存、不计统计。
/// 统计（`cache_hits` / `cache_misses` / `cache_evictions`）供测试与运行时观测。
#[pyclass]
pub struct JsonSchemaCompiler {
    cache: Mutex<LruCache<u64, Arc<ConstrainedGrammar>>>,
    cache_size: usize,
    max_states: usize,
    max_compile_ms: u64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    cache_evictions: AtomicU64,
}

#[pymethods]
impl JsonSchemaCompiler {
    #[new]
    pub fn new(cache_size: usize, max_states: usize, max_compile_ms: u64) -> Self {
        // `LruCache::new` 要求非零容量；`cache_size == 0` 在 `compile` 中按
        // "禁用缓存"处理（不查不写），这里仅避免构造 panic。
        let capacity = NonZeroUsize::new(cache_size.max(1)).expect("capacity > 0");
        Self {
            cache: Mutex::new(LruCache::new(capacity)),
            cache_size,
            max_states,
            max_compile_ms,
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            cache_evictions: AtomicU64::new(0),
        }
    }

    /// 缓存命中次数（bypass 不计入查找）。
    #[getter]
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// 缓存查找未命中次数（bypass 不计入查找）。
    #[getter]
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// 容量满时淘汰的条目数。
    #[getter]
    pub fn cache_evictions(&self) -> u64 {
        self.cache_evictions.load(Ordering::Relaxed)
    }

    #[pyo3(signature = (schema_str, bypass_cache=None))]
    pub fn compile(
        &self,
        schema_str: &str,
        bypass_cache: Option<bool>,
    ) -> PyResult<ConstrainedGrammar> {
        self.compile_str(schema_str, bypass_cache.unwrap_or(false))
            .map_err(|e| e.into())
    }
}

impl JsonSchemaCompiler {
    /// Rust 侧入口（M5-④ HTTP 链路）：直接从 `serde_json::Value` 编译。
    /// 与 `compile` 共享缓存 / 计时 / 上限检查；缓存键为规范化序列化串
    /// （serde_json 默认 Map 排序，序列化确定性）。
    pub fn compile_value(&self, schema: &JsonValue) -> Result<ConstrainedGrammar, ZealotError> {
        let schema_str = serde_json::to_string(schema)
            .map_err(|e| ZealotError::SchemaCompileError(format!("Invalid JSON schema: {e}")))?;
        self.compile_str(&schema_str, false)
    }

    fn compile_str(
        &self,
        schema_str: &str,
        bypass_cache: bool,
    ) -> Result<ConstrainedGrammar, ZealotError> {
        let hash = schema_hash(schema_str);
        let use_cache = !bypass_cache && self.cache_size > 0;

        if use_cache {
            let mut cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&hash) {
                // 命中：`LruCache::get` 已把该条目提升为最近使用。
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok((**cached).clone());
            }
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        let now = std::time::Instant::now();
        let schema: JsonValue = serde_json::from_str(schema_str)
            .map_err(|e| ZealotError::SchemaCompileError(format!("Invalid JSON schema: {e}")))?;

        let grammar = compile_schema(&schema, self.max_states, 0)?;

        let elapsed = now.elapsed().as_millis() as u64;
        if elapsed > self.max_compile_ms {
            return Err(ZealotError::SchemaCompileTimeout(self.max_compile_ms));
        }

        let grammar = Arc::new(grammar);

        if use_cache {
            let mut cache = self.cache.lock().unwrap();
            // lru 0.18 的 `put` 仅在键已存在（替换）时返回 `Some`，容量满淘汰是静默的；
            // 故在持锁区间内用 len/contains 判定真淘汰：容量满且键不在 → put 必然淘汰 LRU 条目。
            if cache.len() >= self.cache_size && !cache.contains(&hash) {
                self.cache_evictions.fetch_add(1, Ordering::Relaxed);
            }
            cache.put(hash, Arc::clone(&grammar));
        }

        Ok((*grammar).clone())
    }
}

fn schema_hash(schema_str: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    schema_str.hash(&mut h);
    h.finish()
}

fn compile_schema(
    schema: &JsonValue,
    max_states: usize,
    depth: usize,
) -> Result<ConstrainedGrammar, ZealotError> {
    if depth > MAX_NESTING_DEPTH {
        return Err(ZealotError::SchemaCompileError(
            "Maximum nesting depth exceeded".into(),
        ));
    }

    let grammar_type = if let Some(type_str) = schema.get("type").and_then(|t| t.as_str()) {
        match type_str {
            "string" => compile_string(schema, max_states, depth)?,
            "number" => {
                if schema.get("minimum").is_some()
                    || schema.get("maximum").is_some()
                    || schema.get("exclusiveMinimum").is_some()
                    || schema.get("exclusiveMaximum").is_some()
                    || schema.get("multipleOf").is_some()
                {
                    // M5-⑤：范围约束仅 integer 可执行；number（浮点）不静默丢弃——显式报错
                    return Err(ZealotError::SchemaCompileError(
                        "number 类型的 minimum/maximum/exclusive*/multipleOf 约束暂不支持（请改用 integer）"
                            .into(),
                    ));
                }
                GrammarType::Number
            }
            "integer" => compile_integer(schema)?,
            "boolean" => GrammarType::Boolean,
            "null" => GrammarType::Null,
            "object" => compile_object(schema, max_states, depth)?,
            "array" => compile_array(schema, max_states, depth)?,
            _ => GrammarType::Any,
        }
    } else if schema.get("enum").is_some() {
        compile_enum(schema, max_states, depth)?
    } else if schema.get("properties").is_some() {
        compile_object(schema, max_states, depth)?
    } else if schema.get("anyOf").is_some() || schema.get("oneOf").is_some() {
        compile_union(schema, max_states, depth)?
    } else {
        GrammarType::Any
    };

    let grammar = ConstrainedGrammar::new(grammar_type);
    // 层级检查（count_states 递归，覆盖嵌套/union 聚合；issue #15）：
    // 任一子树超限即整体拒绝，max_states 防护不再只挡顶层 object。
    if grammar.state_count() > max_states {
        return Err(ZealotError::SchemaTooComplex {
            states: grammar.state_count(),
            limit: max_states,
        });
    }
    // M5-③：字节级自动机编译失败 → 整体拒绝。M5-⑤：透出具体原因
    // （空区间/multipleOf 过大/状态超限），报错消息保留 SchemaTooComplex 文本。
    if grammar.automaton_rejected() {
        if let Some(msg) = grammar.automaton_error() {
            return Err(ZealotError::SchemaCompileError(msg.to_string()));
        }
        return Err(ZealotError::SchemaTooComplex {
            states: grammar.state_count(),
            limit: max_states,
        });
    }
    Ok(grammar)
}

fn compile_integer(schema: &JsonValue) -> Result<GrammarType, ZealotError> {
    // M5-⑤：minimum/maximum/exclusiveMinimum/exclusiveMaximum/multipleOf。
    // exclusive* 优先（同时出现取更严）；draft-4 布尔形态（exclusiveMinimum: true）
    // 使对应 inclusive 边界排他。
    let num = |k: &str| schema.get(k).and_then(|v| v.as_i64());
    let min = num("minimum");
    let max = num("maximum");
    let e_min = num("exclusiveMinimum");
    let e_max = num("exclusiveMaximum");
    let d4_min = schema
        .get("exclusiveMinimum")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let d4_max = schema
        .get("exclusiveMaximum")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let lo = match (min, e_min) {
        (Some(m), Some(e)) => Some(m.max(e.saturating_add(1))),
        (Some(m), None) if d4_min => Some(m.saturating_add(1)),
        (Some(m), None) => Some(m),
        (None, Some(e)) => Some(e.saturating_add(1)),
        (None, None) => None,
    };
    let hi = match (max, e_max) {
        (Some(m), Some(e)) => Some(m.min(e.saturating_sub(1))),
        (Some(m), None) if d4_max => Some(m.saturating_sub(1)),
        (Some(m), None) => Some(m),
        (None, Some(e)) => Some(e.saturating_sub(1)),
        (None, None) => None,
    };
    let multiple_of = match schema.get("multipleOf") {
        None => None,
        Some(v) => Some(parse_multiple_of(v)?),
    };
    Ok(GrammarType::Integer {
        min: lo,
        max: hi,
        multiple_of,
    })
}

/// multipleOf 解析：正整数直接采用；小数做十进制展开（≤6 位）转整数缩放。
fn parse_multiple_of(v: &JsonValue) -> Result<MultipleOf, ZealotError> {
    if let Some(m) = v.as_u64() {
        if m == 0 {
            return Err(ZealotError::SchemaCompileError(
                "multipleOf 必须为正数".into(),
            ));
        }
        return Ok(MultipleOf {
            divisor: m,
            scale: 1,
        });
    }
    if let Some(f) = v.as_f64() {
        if !f.is_finite() || f <= 0.0 {
            return Err(ZealotError::SchemaCompileError(
                "multipleOf 必须为正的有限数".into(),
            ));
        }
        for k in 1..=6u32 {
            let scaled = f * 10f64.powi(k as i32);
            if (scaled - scaled.round()).abs() < 1e-9 {
                let p = scaled.round() as u64;
                if p == 0 {
                    return Err(ZealotError::SchemaCompileError("multipleOf 过小".into()));
                }
                return Ok(MultipleOf {
                    divisor: p,
                    scale: 10u64.pow(k),
                });
            }
        }
        return Err(ZealotError::SchemaCompileError(
            "multipleOf 小数位数过多（>6 位十进制）或不可精确表示".into(),
        ));
    }
    Err(ZealotError::SchemaCompileError(
        "multipleOf 必须为数字".into(),
    ))
}

fn compile_string(
    schema: &JsonValue,
    _max_states: usize,
    _depth: usize,
) -> Result<GrammarType, ZealotError> {
    let min_len = schema
        .get("minLength")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let max_len = schema.get("maxLength").and_then(|v| v.as_u64());
    let pattern = schema
        .get("pattern")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let format = match schema.get("format").and_then(|v| v.as_str()) {
        None => None,
        // M5-⑦：format 常用子集；未列出 → 显式报错（P2 语义，不静默降级）
        Some("email") => Some(StringFormat::Email),
        Some("date") => Some(StringFormat::Date),
        Some("date-time") => Some(StringFormat::DateTime),
        Some("time") => Some(StringFormat::Time),
        Some("uuid") => Some(StringFormat::Uuid),
        Some("ipv4") => Some(StringFormat::Ipv4),
        Some("ipv6") => Some(StringFormat::Ipv6),
        Some("hostname") => Some(StringFormat::Hostname),
        Some("uri") => Some(StringFormat::Uri),
        Some(other) => {
            return Err(ZealotError::SchemaCompileError(format!(
                "format {other} 暂不支持（支持: email/date/date-time/time/uuid/ipv4/ipv6/hostname/uri）"
            )));
        }
    };
    let enumerated = schema.get("enum").map(|v| {
        v.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|e| e.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    });

    Ok(GrammarType::String {
        min_length: min_len,
        max_length: max_len,
        pattern,
        enum_values: enumerated,
        format,
    })
}

fn compile_enum(
    schema: &JsonValue,
    _max_states: usize,
    _depth: usize,
) -> Result<GrammarType, ZealotError> {
    let values = schema
        .get("enum")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    Ok(GrammarType::Enum { values })
}

fn compile_object(
    schema: &JsonValue,
    max_states: usize,
    depth: usize,
) -> Result<GrammarType, ZealotError> {
    let properties = schema
        .get("properties")
        .and_then(|v| v.as_object())
        .map(
            |props| -> Result<HashMap<String, GrammarType>, ZealotError> {
                let mut prop_grammars = HashMap::new();
                for (name, prop_schema) in props {
                    // no-silent-failures（issue #15）：子 schema 编译失败 → 整体报错，
                    // 不再 filter_map(.ok()) 静默丢弃。
                    let grammar = compile_schema(prop_schema, max_states, depth + 1)?;
                    prop_grammars.insert(name.clone(), grammar.grammar_type);
                }
                Ok(prop_grammars)
            },
        )
        .transpose()?
        .unwrap_or_default();

    let required: Vec<String> = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    let additional_properties = schema
        .get("additionalProperties")
        .map(|v| v.as_bool().unwrap_or(true))
        .unwrap_or(true);

    Ok(GrammarType::Object {
        properties,
        required,
        additional_properties,
    })
}

fn compile_array(
    schema: &JsonValue,
    max_states: usize,
    depth: usize,
) -> Result<GrammarType, ZealotError> {
    let min_items = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let max_items = schema.get("maxItems").and_then(|v| v.as_u64());

    let item_grammar = schema
        .get("items")
        .map(|items| compile_schema(items, max_states, depth + 1))
        .transpose()?;

    Ok(GrammarType::Array {
        min_items,
        max_items,
        item_type: item_grammar.map(|g| Box::new(g.grammar_type)),
    })
}

fn compile_union(
    schema: &JsonValue,
    max_states: usize,
    depth: usize,
) -> Result<GrammarType, ZealotError> {
    let branches = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|s| compile_schema(s, max_states, depth + 1))
                .collect::<Result<Vec<_>, ZealotError>>()
        })
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .map(|g| g.grammar_type)
        .collect();

    Ok(GrammarType::Union { branches })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_string() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let schema = r#"{"type": "string", "minLength": 1, "maxLength": 100}"#;
        let grammar = compiler.compile(schema, None).unwrap();
        assert!(matches!(grammar.grammar_type, GrammarType::String { .. }));
    }

    #[test]
    fn test_compile_object() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name"]}"#;
        let grammar = compiler.compile(schema, None).unwrap();
        if let GrammarType::Object {
            properties,
            required,
            ..
        } = &grammar.grammar_type
        {
            assert_eq!(required.len(), 1);
            assert_eq!(required[0], "name");
            assert!(properties.contains_key("name"));
            assert!(properties.contains_key("age"));
        } else {
            panic!("Expected Object grammar");
        }
    }

    #[test]
    fn test_compile_enum() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let schema = r#"{"enum": ["red", "green", "blue"]}"#;
        let grammar = compiler.compile(schema, None).unwrap();
        if let GrammarType::Enum { values } = &grammar.grammar_type {
            assert_eq!(values.len(), 3);
        } else {
            panic!("Expected Enum grammar");
        }
    }

    #[test]
    fn test_cache_hit_and_miss_stats() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let schema = r#"{"type": "boolean"}"#;
        let g1 = compiler.compile(schema, None).unwrap();
        let g2 = compiler.compile(schema, None).unwrap();
        assert_eq!(g1.state_count(), g2.state_count());
        assert_eq!(compiler.cache_hits(), 1);
        assert_eq!(compiler.cache_misses(), 1);
        assert_eq!(compiler.cache_evictions(), 0);
    }

    #[test]
    fn test_cache_lru_eviction_order() {
        // §12-5 核心语义：cache_size=2 下 A,B,A(命中提升),C → 必须淘汰 B（最久未用），保留 A。
        let compiler = JsonSchemaCompiler::new(2, 10000, 500);
        let a = r#"{"type": "string", "minLength": 1}"#;
        let b = r#"{"type": "integer"}"#;
        let c = r#"{"type": "boolean"}"#;

        compiler.compile(a, None).unwrap(); // miss
        compiler.compile(b, None).unwrap(); // miss
        compiler.compile(a, None).unwrap(); // hit → A 提升为最近使用
        compiler.compile(c, None).unwrap(); // miss → 容量满，淘汰最久未用的 B
        assert_eq!(compiler.cache_evictions(), 1);
        assert_eq!(compiler.cache_hits(), 1);
        assert_eq!(compiler.cache_misses(), 3);

        // C 插入后缓存为 {A, C}：A 存活（hit）——先验证再重插 B，
        // 否则 B 的插入会把当时最久未用的 A 淘汰掉。
        compiler.compile(a, None).unwrap(); // hit → 证明被淘汰的是 B 而非 A
        compiler.compile(b, None).unwrap(); // miss → 证明 B 确实已被淘汰
        assert_eq!(compiler.cache_hits(), 2);
        assert_eq!(compiler.cache_misses(), 4);
    }

    #[test]
    fn test_cache_bypass_does_not_touch_cache() {
        let compiler = JsonSchemaCompiler::new(4, 10000, 500);
        let schema = r#"{"type": "boolean"}"#;
        compiler.compile(schema, Some(true)).unwrap(); // 绕过：不查不写不计
        assert_eq!(compiler.cache_hits(), 0);
        assert_eq!(compiler.cache_misses(), 0);
        assert_eq!(compiler.cache_evictions(), 0);
        // bypass 未写入缓存：之后正常编译同 schema 仍是 miss
        compiler.compile(schema, None).unwrap();
        assert_eq!(compiler.cache_misses(), 1);
    }

    #[test]
    fn test_cache_size_zero_disables_caching() {
        let compiler = JsonSchemaCompiler::new(0, 10000, 500);
        let schema = r#"{"type": "boolean"}"#;
        compiler.compile(schema, None).unwrap();
        compiler.compile(schema, None).unwrap();
        assert_eq!(compiler.cache_hits(), 0);
        assert_eq!(compiler.cache_misses(), 0);
    }

    #[test]
    fn test_cache_size_one_evicts_on_every_new_schema() {
        let compiler = JsonSchemaCompiler::new(1, 10000, 500);
        let a = r#"{"type": "boolean"}"#;
        let b = r#"{"type": "integer"}"#;
        compiler.compile(a, None).unwrap();
        compiler.compile(a, None).unwrap();
        assert_eq!(compiler.cache_hits(), 1);
        compiler.compile(b, None).unwrap(); // 容量 1：新键 → 淘汰 A
        assert_eq!(compiler.cache_evictions(), 1);
        compiler.compile(a, None).unwrap(); // A 已被淘汰 → miss
        assert_eq!(compiler.cache_hits(), 1);
        assert_eq!(compiler.cache_misses(), 3);
    }

    #[test]
    fn test_nested_object_max_states_boundary() {
        // issue #15：count_states 递归后，max_states 覆盖嵌套结构。
        // 每个子对象 1 + 2 叶子 = 3 states；外层 1 + N*3。
        let sub = r#"{"type": "object", "properties": {"x": {"type": "boolean"}, "y": {"type": "integer"}}}"#;
        let compiler = JsonSchemaCompiler::new(16, 10, 500);
        // N=3 → 1 + 9 = 10 = max_states：边界接受
        let schema_ok = format!(
            r#"{{"type": "object", "properties": {{"a": {sub}, "b": {sub}, "c": {sub}}}}}"#
        );
        let g = compiler.compile(&schema_ok, None).unwrap();
        assert_eq!(g.state_count(), 10);
        // N=4 → 1 + 12 = 13 > 10：嵌套子 schema 触发 SchemaTooComplex
        let schema_bad = format!(
            r#"{{"type": "object", "properties": {{"a": {sub}, "b": {sub}, "c": {sub}, "d": {sub}}}}}"#
        );
        let err = compiler.compile(&schema_bad, None).unwrap_err();
        assert!(err.to_string().contains("Schema too complex"));
    }

    #[test]
    fn test_failed_child_schema_propagates_error() {
        // issue #15：子 schema 编译失败 → 整体报错（原实现 filter_map(.ok())
        // 静默丢弃失败属性，返回空对象）。
        let compiler = JsonSchemaCompiler::new(16, 5, 500);
        let schema = r#"{"type": "object", "properties": {"big": {"type": "object", "properties": {"a": {"type": "boolean"}, "b": {"type": "boolean"}, "c": {"type": "boolean"}, "d": {"type": "boolean"}, "e": {"type": "boolean"}}}}}"#;
        // 子对象 1 + 5 = 6 > max_states=5 → 子编译失败必须向整体传播
        let err = compiler.compile(schema, None).unwrap_err();
        assert!(err.to_string().contains("Schema too complex"));
    }

    #[test]
    fn test_union_branch_failure_propagates_error() {
        // issue #15：union 分支超限 → 整体报错（原实现静默丢弃失败分支）。
        let compiler = JsonSchemaCompiler::new(16, 5, 500);
        let schema = r#"{"anyOf": [{"type": "boolean"}, {"type": "object", "properties": {"a": {"type": "boolean"}, "b": {"type": "boolean"}, "c": {"type": "boolean"}, "d": {"type": "boolean"}, "e": {"type": "boolean"}}}]}"#;
        let err = compiler.compile(schema, None).unwrap_err();
        assert!(err.to_string().contains("Schema too complex"));
    }

    // ── M5-⑤：integer 范围/整除关键字 ──────────────────────────────────────

    #[test]
    fn test_integer_keywords_parsed() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        // exclusive 优先：exclusiveMinimum 5 → 有效下界 6；exclusiveMaximum 20 → 19
        let g = compiler
            .compile(
                r#"{"type": "integer", "minimum": 0, "exclusiveMinimum": 5, "maximum": 20, "exclusiveMaximum": 20, "multipleOf": 3}"#,
                None,
            )
            .unwrap();
        match &g.grammar_type {
            GrammarType::Integer {
                min,
                max,
                multiple_of,
            } => {
                assert_eq!(*min, Some(6));
                assert_eq!(*max, Some(19));
                let mo = multiple_of.as_ref().expect("multipleOf");
                assert_eq!((mo.divisor, mo.scale), (3, 1));
            }
            other => panic!("unexpected {other:?}"),
        }
        // draft-4 布尔形态：exclusiveMinimum: true → minimum 0 排他（有效下界 1）
        let g = compiler
            .compile(
                r#"{"type": "integer", "minimum": 0, "exclusiveMinimum": true}"#,
                None,
            )
            .unwrap();
        match &g.grammar_type {
            GrammarType::Integer { min, max, .. } => {
                assert_eq!(*min, Some(1));
                assert_eq!(*max, None);
            }
            other => panic!("unexpected {other:?}"),
        }
        // 小数 multipleOf 2.5 → (divisor 25, scale 10)
        let g = compiler
            .compile(r#"{"type": "integer", "multipleOf": 2.5}"#, None)
            .unwrap();
        match &g.grammar_type {
            GrammarType::Integer {
                multiple_of: Some(mo),
                ..
            } => assert_eq!((mo.divisor, mo.scale), (25, 10)),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn test_integer_invalid_keywords_error() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        // 空区间
        let err = compiler
            .compile(r#"{"type": "integer", "minimum": 10, "maximum": 5}"#, None)
            .unwrap_err();
        assert!(err.to_string().contains("空区间"), "{err}");
        // multipleOf 0
        let err = compiler
            .compile(r#"{"type": "integer", "multipleOf": 0}"#, None)
            .unwrap_err();
        assert!(err.to_string().contains("正数"), "{err}");
        // number + 范围关键字 → 显式报错（不静默丢弃）
        let err = compiler
            .compile(r#"{"type": "number", "minimum": 0}"#, None)
            .unwrap_err();
        assert!(err.to_string().contains("integer"), "{err}");
        // 不可精确小数 multipleOf
        let err = compiler
            .compile(r#"{"type": "integer", "multipleOf": 0.1234567}"#, None)
            .unwrap_err();
        assert!(err.to_string().contains("multipleOf"), "{err}");
    }

    #[test]
    fn test_string_length_ranges_error() {
        // M5-⑥：minLength > maxLength → 空区间；超大 horizon → 显式错误
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let err = compiler
            .compile(
                r#"{"type": "string", "minLength": 5, "maxLength": 2}"#,
                None,
            )
            .unwrap_err();
        assert!(err.to_string().contains("空区间"), "{err}");
        let err = compiler
            .compile(r#"{"type": "string", "maxLength": 99999999}"#, None)
            .unwrap_err();
        assert!(err.to_string().contains("过大"), "{err}");
    }

    #[test]
    fn test_string_lengths_compile_and_state_count() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let g = compiler
            .compile(
                r#"{"type": "string", "minLength": 2, "maxLength": 4}"#,
                None,
            )
            .unwrap();
        assert!(matches!(g.grammar_type, GrammarType::String { .. }));
        assert!(!g.automaton_rejected());
    }

    #[test]
    fn test_string_format_parsed_and_unsupported_errors() {
        // M5-⑦：format 子集解析
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        for f in [
            "email",
            "date",
            "date-time",
            "time",
            "uuid",
            "ipv4",
            "ipv6",
            "hostname",
            "uri",
        ] {
            let g = compiler
                .compile(&format!(r#"{{"type": "string", "format": "{f}"}}"#), None)
                .unwrap_or_else(|e| panic!("format {f} 应支持: {e}"));
            assert!(!g.automaton_rejected(), "format {f} 自动机应可编译");
        }
        // 不支持的 format → 显式报错（P2 语义，不静默降级）
        for f in ["binary", "byte", "password", "date-time-extra"] {
            let err = compiler
                .compile(&format!(r#"{{"type": "string", "format": "{f}"}}"#), None)
                .unwrap_err();
            assert!(err.to_string().contains("暂不支持"), "{f}: {err}");
        }
    }

    #[test]
    fn test_integer_oversize_multiple_of_error() {
        // 区间 + 大 multipleOf → 状态预算超限 → 编译错误
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let err = compiler
            .compile(
                r#"{"type": "integer", "minimum": 0, "maximum": 100, "multipleOf": 10000}"#,
                None,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("过大") || err.to_string().contains("too complex"),
            "{err}"
        );
    }
}
