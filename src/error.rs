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

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<ZealotError> for PyErr {
    fn from(e: ZealotError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}
