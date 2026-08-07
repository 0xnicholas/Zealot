use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;

use crate::error::ZealotError;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GrammarType {
    Any,
    String {
        min_length: usize,
        max_length: Option<u64>,
        pattern: Option<String>,
        enum_values: Option<Vec<String>>,
    },
    Number,
    Integer,
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

#[pyclass]
#[derive(Clone, Debug)]
pub struct ConstrainedGrammar {
    pub grammar_type: GrammarType,

    #[pyo3(get)]
    state_count: usize,
}

impl ConstrainedGrammar {
    pub fn new(grammar_type: GrammarType) -> Self {
        let state_count = count_states(&grammar_type);
        Self {
            grammar_type,
            state_count,
        }
    }

    pub fn state_count(&self) -> usize {
        self.state_count
    }
}

fn count_states(gt: &GrammarType) -> usize {
    match gt {
        GrammarType::Any
        | GrammarType::String { .. }
        | GrammarType::Number
        | GrammarType::Integer
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

#[pymethods]
impl ConstrainedGrammar {
    fn allowed_tokens(&self, state: usize) -> PyResult<Vec<i32>> {
        if state >= self.state_count {
            return Err(ZealotError::InvalidConstraint(format!(
                "State {} out of range (max {})",
                state, self.state_count
            ))
            .into());
        }
        Ok(self.compute_allowed_tokens(state))
    }

    fn advance(&self, current_state: usize, _token_id: i32) -> PyResult<usize> {
        if current_state >= self.state_count {
            return Err(ZealotError::InvalidConstraint(format!(
                "State {} out of range (max {})",
                current_state, self.state_count
            ))
            .into());
        }
        // For Phase 2: advance by cycling to next state within the grammar.
        // Full token-id → grammar-position mapping requires tokenizer integration (Phase 3).
        Ok((current_state + 1).min(self.state_count - 1))
    }

    fn is_valid_final(&self, state: usize) -> PyResult<bool> {
        if state >= self.state_count {
            return Err(ZealotError::InvalidConstraint(format!(
                "State {} out of range (max {})",
                state, self.state_count
            ))
            .into());
        }
        Ok(self.compute_is_valid_final(state))
    }

    fn __repr__(&self) -> String {
        format!("ConstrainedGrammar(states={})", self.state_count)
    }
}

impl ConstrainedGrammar {
    fn compute_allowed_tokens(&self, _state: usize) -> Vec<i32> {
        // Phase 2: return a broad set of token IDs that match the grammar type.
        // Phase 3+: integrate with tokenizer vocabulary for precise token-level constraints.
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
            GrammarType::Integer => {
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
            GrammarType::Number | GrammarType::Integer => {
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
        }
    }

    #[test]
    fn test_object_required_keys_must_be_traversed_before_final() {
        // issue #15：required 键全部遍历完（state >= required.len()）才可合法收尾。
        let g = object_grammar(
            vec![("name", string_leaf()), ("age", GrammarType::Integer)],
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
            vec![("name", string_leaf()), ("age", GrammarType::Integer)],
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
}
