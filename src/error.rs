use pyo3::{exceptions::PyRuntimeError, PyErr};

#[derive(Debug, thiserror::Error)]
pub enum ZealotError {
    #[error("Block {0} is already allocated")]
    BlockAlreadyAllocated(usize),

    #[error("Block {0} is not allocated")]
    BlockNotAllocated(usize),

    #[error("Out of blocks: requested {requested}, available {available}")]
    OutOfBlocks { requested: usize, available: usize },

    #[error("Invalid block size: {0}")]
    InvalidBlockSize(usize),

    #[error("JSON schema compile error: {0}")]
    SchemaCompileError(String),

    #[error("Invalid token constraint: {0}")]
    InvalidConstraint(String),

    #[error("Stale handle: block {block_id} was freed (gen {handle_gen}, current {current_gen})")]
    StaleHandle { block_id: usize, handle_gen: u64, current_gen: u64 },

    #[error("Schema too complex: {states} states exceeds limit of {limit}")]
    SchemaTooComplex { states: usize, limit: usize },

    #[error("Schema compilation timed out after {0}ms")]
    SchemaCompileTimeout(u64),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<ZealotError> for PyErr {
    fn from(e: ZealotError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}
