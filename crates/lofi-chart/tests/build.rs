//! The build information through the crate's public interface.

use lofi_chart::build::BuildInfo;

#[test]
fn version_has_three_numeric_components() {
    let version = BuildInfo::current().version();
    let components: Vec<&str> = version.split('.').collect();
    assert_eq!(components.len(), 3, "{version} is major.minor.patch");
    assert!(
        components
            .iter()
            .all(|c| !c.is_empty() && c.bytes().all(|b| b.is_ascii_digit())),
        "{version} has numeric components"
    );
}

#[test]
fn current_is_stable_across_calls() {
    assert_eq!(
        BuildInfo::current(),
        BuildInfo::current(),
        "one build, one value"
    );
}
