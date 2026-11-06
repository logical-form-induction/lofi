---
name: Leak report
about: An interface exposes something it should have kept, or an invariant is held only by convention
title: ""
labels: leak
assignees: ""
---

<!--
A leak exposes an implementation detail or leaves an invariant enforced only by
convention. It may prevent one component from being swapped for another.
-->

## Exposed detail

<!--
Name the interface and the detail that reaches through it: a storage layout, an
axis order, a slot index, a substring seam, a dispatch on type, a fact stated in
two places, a device or dtype that travels on a value.
-->

## Where

<!-- Modules, and lines if you have them. -->

## Effect

<!--
Name the swap this prevents or the comparison it distorts. If it blocks neither,
say so.
-->
