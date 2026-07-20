use pyo3::prelude::*;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::matcher::{ConstrainedGrammar, GrammarType};
use crate::error::ZealotError;

const MAX_NESTING_DEPTH: usize = 32;

#[pyclass]
pub struct JsonSchemaCompiler {
    cache: Mutex<HashMap<u64, Arc<ConstrainedGrammar>>>,
    cache_size: usize,
    max_states: usize,
    max_compile_ms: u64,
}

#[pymethods]
impl JsonSchemaCompiler {
    #[new]
    fn new(cache_size: usize, max_states: usize, max_compile_ms: u64) -> Self {
        Self {
            cache: Mutex::new(HashMap::with_capacity(cache_size.min(256))),
            cache_size,
            max_states,
            max_compile_ms,
        }
    }

    #[pyo3(signature = (schema_str, bypass_cache=None))]
    fn compile(
        &self,
        schema_str: &str,
        bypass_cache: Option<bool>,
    ) -> PyResult<ConstrainedGrammar> {
        let hash = schema_hash(schema_str);

        if !bypass_cache.unwrap_or(false) {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&hash) {
                return Ok((**cached).clone());
            }
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

        if !bypass_cache.unwrap_or(false) {
            let mut cache = self.cache.lock().unwrap();
            if cache.len() >= self.cache_size {
                // Evict oldest entry (simple: just remove a random key)
                if let Some(old_key) = cache.keys().next().copied() {
                    cache.remove(&old_key);
                }
            }
            cache.insert(hash, Arc::clone(&grammar));
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
    fn test_cache_hit() {
        let compiler = JsonSchemaCompiler::new(16, 10000, 500);
        let schema = r#"{"type": "boolean"}"#;
        let g1 = compiler.compile(schema, None).unwrap();
        let g2 = compiler.compile(schema, None).unwrap();
        assert_eq!(g1.state_count(), g2.state_count());
    }
}
