pub mod matcher;
pub mod schema;

use self::matcher::ConstrainedGrammar;
use self::schema::JsonSchemaCompiler;
use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<JsonSchemaCompiler>()?;
    m.add_class::<ConstrainedGrammar>()?;
    Ok(())
}
