//! The version and Cargo profile the crate was compiled with.

use std::fmt;

/// The built-in Cargo profile a build inherits from.
///
/// Cargo reports a custom profile as the built-in profile it inherits from, so
/// a profile declared with `inherits = "release"` is [`Profile::Release`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// The `dev` profile, or a profile inheriting from it.
    Debug,
    /// The `release` profile, or a profile inheriting from it.
    Release,
}

impl Profile {
    /// Return the name Cargo gives the profile.
    ///
    /// # Examples
    ///
    /// ```
    /// use lofi_chart::build::Profile;
    ///
    /// assert_eq!(Profile::Release.name(), "release", "the release profile's name");
    /// ```
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Debug => "debug",
            Self::Release => "release",
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(lofi_profile = "debug")]
const PROFILE: Profile = Profile::Debug;

#[cfg(lofi_profile = "release")]
const PROFILE: Profile = Profile::Release;

/// The version and Cargo profile of a compiled build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BuildInfo {
    version: &'static str,
    profile: Profile,
}

impl BuildInfo {
    /// Return the version and profile this crate was compiled with.
    ///
    /// # Examples
    ///
    /// ```
    /// use lofi_chart::build::BuildInfo;
    ///
    /// let info = BuildInfo::current();
    /// assert_eq!(info.version(), env!("CARGO_PKG_VERSION"), "the package version");
    /// ```
    #[must_use]
    pub const fn current() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            profile: PROFILE,
        }
    }

    /// Return the crate version, as Cargo.toml declares it.
    #[must_use]
    pub const fn version(self) -> &'static str {
        self.version
    }

    /// Return the profile the crate was compiled under.
    #[must_use]
    pub const fn profile(self) -> Profile {
        self.profile
    }
}

#[cfg(test)]
mod tests {
    use super::{BuildInfo, Profile};

    #[test]
    fn profile_follows_debug_assertions() {
        let expected = if cfg!(debug_assertions) {
            Profile::Debug
        } else {
            Profile::Release
        };
        assert_eq!(
            BuildInfo::current().profile(),
            expected,
            "the workspace's profiles leave debug assertions at their defaults"
        );
    }

    #[test]
    fn profile_names_are_cargos() {
        assert_eq!(Profile::Debug.name(), "debug", "the dev profile's name");
        assert_eq!(
            Profile::Release.name(),
            "release",
            "the release profile's name"
        );
    }

    #[test]
    fn profile_displays_its_name() {
        for profile in [Profile::Debug, Profile::Release] {
            assert_eq!(
                profile.to_string(),
                profile.name(),
                "{profile:?} displays its name"
            );
        }
    }

    #[test]
    fn version_is_the_package_version() {
        assert_eq!(
            BuildInfo::current().version(),
            env!("CARGO_PKG_VERSION"),
            "the version Cargo.toml declares"
        );
    }
}
