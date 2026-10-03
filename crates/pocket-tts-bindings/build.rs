fn main() {
    // On macOS an extension module must leave the Python symbols to be
    // resolved at import time; maturin passes these flags itself, this
    // keeps manual `cargo build`s working too.
    pyo3_build_config::add_extension_module_link_args();
}
