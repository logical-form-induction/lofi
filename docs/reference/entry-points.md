# Entry-point groups

This page lists every entry-point group LoFI's `pyproject.toml` declares, the
declaration family or runtime capability each group extends, and the names LoFI
packages in it.

A distribution extends LoFI by registering an object under one of these groups;
[Writing a plugin](../how-to/extending/plugins.md) shows how. The console
script `lofi` is the only `[project.scripts]` entry and names
`lofi.run._entrypoint:main` (see [Command line](cli.md)).

`lofi.spec.load_plugins` loads the groups in three passes before a document is
validated: the family roots in `lofi.families`, then every declaration family,
then the plugin objects of every component family. A component family's group
is named `lofi.<family>` after its entry in `lofi.families`. The remaining
groups are read by the component or procedure that owns them, when it is
built.

## Family roots

`lofi.families` registers one declaration root per component family. Every
variant of the family derives from that root, and the
[Component roots](configuration/components.md) page renders them.

