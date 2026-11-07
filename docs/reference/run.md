# Run

A [point](../explanation/documents-are-the-dsl.md#point-identity) combines a
resolved scientific configuration with a [seed](../explanation/documents-are-the-dsl.md#point-identity). An experiment collects points
and declares their comparisons. The run layer executes them and records a
[manifest](../explanation/documents-are-the-dsl.md#provenance) describing each run; [sinks](configuration/evaluation.md#sinks)
receive its measurements.

[Launching](#launching-experiments) starts the work,
[ingestion](cli.md#ingest) checks results against the experiment's records,
and [reports](#report-columns-and-renderers) turn accepted rows into tables.
The [comparison tutorial](../tutorials/06-compare-grammars.md) follows that
sequence, including the [commitments](../explanation/experiments-and-commitments.md)
checked along the way.

This page renders [`lofi.run`](#lofi.run), [`lofi.run.launching`](#lofi.run.launching), [`lofi.run.results`](#lofi.run.results), [`lofi.run.fits`](#lofi.run.fits) and [`lofi.report`](#lofi.report): run orchestration, experiment manifests, telemetry sinks, launching, results, and report columns and renderers.

::: lofi.run

## Launching experiments

::: lofi.run.launching

## Experiment results

::: lofi.run.results

## Judgment fit points

::: lofi.run.fits

## Report columns and renderers

::: lofi.report

::: lofi.report.markdown

::: lofi.report.latex
