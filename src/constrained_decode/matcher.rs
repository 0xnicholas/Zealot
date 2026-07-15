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
        GrammarType::Any | GrammarType::String { .. } | GrammarType::Number
        | GrammarType::Integer | GrammarType::Boolean | GrammarType::Null => 1,
        GrammarType::Object { properties, .. } => 1 + properties.len(),
        GrammarType::Array { item_type, .. } => 1 + item_type.as_ref().map(|t| count_states(t)).unwrap_or(0),
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
            GrammarType::String { min_length, max_length: _, pattern: _, enum_values } => {
                let mut tokens: Vec<i32> = vec![34]; // opening quote
                if *min_length > 0 {
                    tokens.extend(65..=122); // A-Z, a-z
                    tokens.extend(48..=57);   // 0-9
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
                    't' as i32, 'r' as i32, 'u' as i32, 'e' as i32,
                    'f' as i32, 'a' as i32, 'l' as i32, 's' as i32, 'e' as i32,
                ]
            }
            GrammarType::Null => {
                vec!['n' as i32, 'u' as i32, 'l' as i32, 'l' as i32]
            }
            GrammarType::Object { properties, required: _, additional_properties: _ } => {
                let mut tokens = vec!['{' as i32, '}' as i32, ',' as i32, ':' as i32, '"' as i32];
                tokens.extend(32..=126);
                for key in properties.keys() {
                    tokens.push(key.len() as i32 + 2000); // placeholder token IDs for property names
                }
                tokens
            }
            GrammarType::Array { min_items: _, max_items: _, item_type: _ } => {
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

    fn compute_is_valid_final(&self, _state: usize) -> bool {
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
            GrammarType::Object { required, properties, .. } => {
                // Valid if all required properties are accounted for
                // State 0 represents the object root — valid if no required props remain
                required.is_empty() || properties.is_empty()
            }
            GrammarType::Array { .. } => {
                true // empty array is valid at state 0
            }
            GrammarType::Enum { .. } => {
                true // at state 0, any enum value is valid
            }
            GrammarType::Union { branches } => {
                branches.iter().any(|b| {
                    let g = ConstrainedGrammar::new(b.clone());
                    g.compute_is_valid_final(0)
                })
            }
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
        props.insert("name".to_string(), GrammarType::String {
            min_length: 1,
            max_length: None,
            pattern: None,
            enum_values: None,
        });
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
}
