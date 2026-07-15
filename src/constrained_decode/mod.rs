pub mod matcher;
pub mod schema;

use pyo3::prelude::*;
use self::schema::JsonSchemaCompiler;
use self::matcher::ConstrainedGrammar;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<JsonSchemaCompiler>()?;
    m.add_class::<ConstrainedGrammar>()?;
    Ok(())
}
