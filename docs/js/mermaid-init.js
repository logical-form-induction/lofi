(function () {
  function palette() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return dark
      ? {
          background: "#1f2422",
          primaryColor: "#23362f",
          primaryTextColor: "#edf1ee",
          primaryBorderColor: "#6fc2a4",
          lineColor: "#a7b3ad",
          textColor: "#edf1ee",
          edgeLabelBackground: "#1f2422",
          clusterBkg: "#1a1e1c",
          clusterBorder: "#46514c",
        }
      : {
          background: "#ffffff",
          primaryColor: "#e6f3ee",
          primaryTextColor: "#1d2421",
          primaryBorderColor: "#1d6b52",
          lineColor: "#66736d",
          textColor: "#1d2421",
          edgeLabelBackground: "#f7f9f8",
          clusterBkg: "#f7f9f8",
          clusterBorder: "#c9d3ce",
        };
  }

  function sources() {
    document.querySelectorAll("pre.mermaid").forEach(function (pre) {
      const code = pre.querySelector("code");
      const div = document.createElement("div");
      div.className = "mermaid";
      div.dataset.source = ((code ? code.textContent : pre.textContent) || "").trim();
      div.textContent = div.dataset.source;
      pre.parentNode.replaceChild(div, pre);
    });
  }

  async function render(rerender) {
    if (typeof mermaid === "undefined") return;
    sources();
    const diagrams = Array.from(document.querySelectorAll("div.mermaid"));
    if (rerender) {
      diagrams.forEach(function (diagram) {
        diagram.removeAttribute("data-processed");
        diagram.textContent = diagram.dataset.source || diagram.textContent;
      });
    }
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      fontFamily: "IBM Plex Sans, system-ui, sans-serif",
      themeVariables: palette(),
    });
    const pending = diagrams.filter(function (diagram) {
      return !diagram.hasAttribute("data-processed");
    });
    if (pending.length) await mermaid.run({ nodes: pending });
  }

  document.addEventListener("change", function (event) {
    if (event.target.matches('[data-md-component="palette"] input')) {
      window.setTimeout(function () { render(true); }, 0);
    }
  });

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () { render(false); });
  } else {
    document.addEventListener("DOMContentLoaded", function () { render(false); });
  }
})();
