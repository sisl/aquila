// Move the mike version selector from inside .md-header__topic
// to the end of .md-header__inner so it flows naturally in the
// header flex layout: … | search | repo | version
document.addEventListener("DOMContentLoaded", function () {
  var defined = function (el) { return typeof el !== "undefined" && el !== null; };
  var defined$ = function (sel, ctx) { return defined((ctx || document).querySelector(sel)); };

  var poll = setInterval(function () {
    var version = document.querySelector(".md-version");
    var header = document.querySelector(".md-header__inner");
    if (version && header) {
      clearInterval(poll);
      header.appendChild(version);
    }
  }, 100);

  // Give up after 5 seconds if mike didn't inject the element.
  setTimeout(function () { clearInterval(poll); }, 5000);
});
