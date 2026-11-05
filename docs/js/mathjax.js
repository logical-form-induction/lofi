window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|md-nav__link|md-path__link|md-ellipsis"
  }
};

// instant navigation swaps the article without a reload, so each new article
// is typeset again with its equation numbering reset.
if (typeof document$ !== "undefined") {
  document$.subscribe(function () {
    if (!window.MathJax || !MathJax.typesetPromise) {
      return;
    }
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  });
}
