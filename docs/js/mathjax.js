window.MathJax = {
  loader: {load: ["[tex]/boldsymbol"]},
  chtml: {
    // Keep glyph styles available when instant navigation restores rendered math.
    adaptiveCSS: false
  },
  tex: {
    packages: {"[+]": ["boldsymbol"]},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: "ams"
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex|md-nav__link|md-path__link|md-ellipsis"
  },
  startup: {
    typeset: false,
    ready: function () {
      MathJax.startup.defaultReady();
      MathJax.startup.promise.then(function () {
        let queue = Promise.resolve();
        const typeset = function () {
          // Serialize page changes and skip already rendered navigation labels.
          queue = queue.then(function () {
            const pending = Array.from(document.querySelectorAll(
              ".arithmatex, .md-nav__link, .md-path__link, .md-ellipsis"
            )).filter(function (element) {
              return !element.querySelector("mjx-container");
            });
            MathJax.typesetClear();
            MathJax.texReset();
            return MathJax.typesetPromise(pending);
          }).catch(function (error) {
            console.error("MathJax typesetting failed:", error);
          });
        };
        // Material emits the initial document and each instant-navigation page.
        if (typeof document$ !== "undefined") {
          document$.subscribe(typeset);
        } else {
          typeset();
        }
      });
    }
  }
};
