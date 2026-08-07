pub mod matcher;
pub mod regex;
pub mod schema;

pub use self::matcher::ConstrainedGrammar;
pub use self::schema::JsonSchemaCompiler;
use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<JsonSchemaCompiler>()?;
    m.add_class::<ConstrainedGrammar>()?;
    Ok(())
}
