use lru::LruCache;
use pyo3::prelude::*;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::matcher::{ConstrainedGrammar, GrammarType};
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
        let hash = schema_hash(schema_str);
        let use_cache = !bypass_cache.unwrap_or(false) && self.cache_size > 0;

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
            .map_err(|e| ZealotError::SchemaCompileError(format!("Invalid JSON schema: {}", e)))?;

        let grammar = compile_schema(&schema, self.max_states, 0)?;

        let elapsed = now.elapsed().as_millis() as u64;
        if elapsed > self.max_compile_ms {
            return Err(ZealotError::SchemaCompileTimeout(self.max_compile_ms).into());
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
) -> PyResult<ConstrainedGrammar> {
    if depth > MAX_NESTING_DEPTH {
        return Err(
            ZealotError::SchemaCompileError("Maximum nesting depth exceeded".into()).into(),
        );
    }

    let grammar_type = if let Some(type_str) = schema.get("type").and_then(|t| t.as_str()) {
        match type_str {
            "string" => compile_string(schema, max_states, depth)?,
            "number" | "integer" => GrammarType::Number,
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

    Ok(ConstrainedGrammar::new(grammar_type))
}

fn compile_string(schema: &JsonValue, _max_states: usize, _depth: usize) -> PyResult<GrammarType> {
    let min_len = schema
        .get("minLength")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let max_len = schema.get("maxLength").and_then(|v| v.as_u64());
    let pattern = schema
        .get("pattern")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
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
    })
}

fn compile_enum(schema: &JsonValue, _max_states: usize, _depth: usize) -> PyResult<GrammarType> {
    let values = schema
        .get("enum")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    Ok(GrammarType::Enum { values })
}

fn compile_object(schema: &JsonValue, max_states: usize, depth: usize) -> PyResult<GrammarType> {
    let properties = schema
        .get("properties")
        .and_then(|v| v.as_object())
        .map(|props| {
            let mut prop_grammars = HashMap::new();
            for (name, prop_schema) in props {
                if let Ok(grammar) = compile_schema(prop_schema, max_states, depth + 1) {
                    prop_grammars.insert(name.clone(), grammar.grammar_type);
                }
            }
            prop_grammars
        })
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

    let state_count: usize = properties.len() + 1;
    if state_count > max_states {
        return Err(ZealotError::SchemaTooComplex {
            states: state_count,
            limit: max_states,
        }
        .into());
    }

    Ok(GrammarType::Object {
        properties,
        required,
        additional_properties,
    })
}

fn compile_array(schema: &JsonValue, max_states: usize, depth: usize) -> PyResult<GrammarType> {
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

fn compile_union(schema: &JsonValue, max_states: usize, depth: usize) -> PyResult<GrammarType> {
    let branches = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| compile_schema(s, max_states, depth + 1).ok())
                .map(|g| g.grammar_type)
                .collect()
        })
        .unwrap_or_default();

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
}
