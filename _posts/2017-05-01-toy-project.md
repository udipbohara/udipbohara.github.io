---
layout: post
title:  "Visualizations using d3"
info: "I am just testing to see how the images would align"
tech: "python"
type: Toy 
img: "/assets/img/profile.jpeg" 
type: "blog"
---



<div class="displaysort"></div>
<p>Credit: <a href="https://observablehq.com/@udipbohara/bohara-final-problem-1-attempt-b">Bohara Final Problem 1 Attempt B+  by Udip Bohara</a></p>

<script type="module">
import {Runtime, Inspector} from "https://cdn.jsdelivr.net/npm/@observablehq/runtime@4/dist/runtime.js";
import define from "https://api.observablehq.com/@udipbohara/bohara-final-problem-1-attempt-b.js?v=3";
(new Runtime).module(define, name => {
  if (name === "displaysort") return Inspector.into(".displaysort")();
});
</script>