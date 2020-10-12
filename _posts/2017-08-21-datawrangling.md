---
layout: post
title:  "Data Wrangling"
info: "using data wrangling techniques to build and interactive plot"
tech: "d3.js"
img: "/assets/img/wrangling/baseball.png" 
concepts: "Natural Language Processing, SOcial Media mining"
type: "project"
link: "https://observablehq.com/d/c11d2be8dc0ea13b"
tags: ["Data Visualization"]
---

<div class="display"></div>
<div class="viewof-p" style='text-align:center'></div>
<p>Credit: <a href="https://observablehq.com/d/c11d2be8dc0ea13b">Baseball Trades for 2019 by ujb</a></p>

<script type="module">
import {Runtime, Inspector} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@4/dist/runtime.js";
import define from "https://api.observablehq.com/d/c11d2be8dc0ea13b.js?v=3";
(new Runtime).module(define, name => {
  if (name === "display") return Inspector.into(".display")();
  if (name === "viewof p") return Inspector.into(".viewof-p")();
  return ["arcs","ribbons"].includes(name) || null;
});
</script>

