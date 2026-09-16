const form = document.getElementById("search");
const box = document.getElementById("q");
const weight = document.getElementById("weight");
const weightValue = document.getElementById("weight-value");
const people = document.getElementById("people");
const albums = document.getElementById("albums");
const since = document.getElementById("since");
const until = document.getElementById("until");
const filters = document.getElementById("filters");
const results = document.getElementById("results");
const hint = document.getElementById("hint");
const themeButton = document.getElementById("theme");
const button = form.querySelector("button");

// Every chip is {kind, name}, where kind is the query parameter it becomes: person or album.
const active = [];
let like = null;
let reference = null;
let inflight = null;
let typing = null;
let slow = null;

// The server renders its configured blend into the slider, so this is the value a URL can leave
// out. Anything else has to travel in the link, or a shared result set is not the one you saw.
const DEFAULT_WEIGHT = weight.value;
const TYPING_PAUSE = 350;
// Below this a spinner state would flash and read as a glitch, so the old results just stay.
const SKELETON_AFTER = 200;
const EXAMPLES = ["blowing out candles", "a train at night", "someone laughing", "a birthday cake"];

weight.addEventListener("input", showWeight);
weight.addEventListener("change", () => {
  if (box.value.trim()) run();
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  clearTimeout(typing);
  if (box.value.trim()) clearLike();
  if (anything()) run();
});

// Typing searches on its own after a pause, so the button is a confirmation rather than a toll.
box.addEventListener("input", () => {
  clearTimeout(typing);
  typing = setTimeout(() => {
    if (box.value.trim()) clearLike();
    rerunOrClear(false);
  }, TYPING_PAUSE);
});

for (const [menu, kind] of [[people, "person"], [albums, "album"]]) {
  menu.addEventListener("change", () => {
    addFilter(kind, menu.value);
    menu.value = "";
  });
}

for (const input of [since, until]) {
  input.addEventListener("change", () => rerunOrClear());
}

themeButton.addEventListener("click", () => {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const now = document.documentElement.dataset.theme || (dark ? "dark" : "light");
  const next = now === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("moments-theme", next);
});

window.addEventListener("keydown", (event) => {
  const inField = event.target.matches("input, select, textarea");
  // The box is autofocused, so "?" has to survive an empty one or the shortcut is unreachable.
  const midQuery = inField && event.target.value !== "";
  if (event.key === "/" && !inField) {
    event.preventDefault();
    box.focus();
    box.select();
  } else if (event.key === "?" && !midQuery) {
    event.preventDefault();
    document.getElementById("shortcuts").togglePopover();
  } else if (event.key === "Escape" && !document.querySelector("[popover]:popover-open")) {
    if (box.value) {
      box.value = "";
      clearTimeout(typing);
      rerunOrClear();
    } else if (anything()) {
      active.length = 0;
      since.value = "";
      until.value = "";
      clearLike();
      rerunOrClear();
    }
  }
});

window.addEventListener("popstate", () => readUrl(false));

loadMenu(people, "/api/people", (data) => data.people.map((p) => ({ name: p.name, count: p.scenes })));
loadMenu(albums, "/api/albums", (data) => data.albums.map((a) => ({ name: a.name, count: a.videos })));
readUrl(false);

function anything() {
  return Boolean(box.value.trim() || active.length || like || since.value || until.value);
}

function readUrl(push) {
  const params = new URLSearchParams(location.search);
  box.value = params.get("q") || "";
  weight.value = params.get("weight") || DEFAULT_WEIGHT;
  showWeight();
  like = params.get("like") ? Number(params.get("like")) : null;
  since.value = params.get("since") || "";
  until.value = params.get("until") || "";
  reference = null;
  active.length = 0;
  for (const kind of ["person", "album"]) {
    for (const name of params.getAll(kind)) active.push({ kind, name });
  }
  drawFilters();
  if (anything()) run(push);
  else showEmpty();
}

function addFilter(kind, name) {
  if (!name || active.some((f) => f.kind === kind && f.name === name)) return;
  active.push({ kind, name });
  drawFilters();
  run();
}

function removeFilter(filter) {
  const at = active.indexOf(filter);
  if (at < 0) return;
  active.splice(at, 1);
  drawFilters();
  rerunOrClear();
}

function showLike(hit) {
  like = hit.scene_id;
  reference = hit;
  box.value = "";
  drawFilters();
  run();
}

function clearLike() {
  like = null;
  reference = null;
  drawFilters();
}

function rerunOrClear(push = true) {
  if (anything()) {
    run(push);
    return;
  }
  showEmpty();
  title("");
  history.pushState({}, "", "/");
}

async function loadMenu(menu, url, entries) {
  try {
    const response = await fetch(url);
    const data = await response.json();
    const options = entries(data);
    for (const entry of options) {
      const option = document.createElement("option");
      option.value = entry.name;
      option.textContent = `${entry.name} (${entry.count})`;
      menu.append(option);
    }
    menu.disabled = options.length === 0;
  } catch {
    menu.disabled = true;
  }
}

function chip(label, title, onClick, className = "chip") {
  const node = document.createElement("button");
  node.type = "button";
  node.className = className;
  node.textContent = label;
  node.title = title;
  node.addEventListener("click", onClick);
  return node;
}

function drawFilters() {
  filters.replaceChildren();
  if (like) {
    const what = reference ? reference.label || reference.file_name : `scene ${like}`;
    filters.append(
      chip(`like ${what} ✕`, "Stop ranking against that scene", () => {
        clearLike();
        rerunOrClear();
      }, "chip active"),
    );
  }
  for (const filter of active) {
    filters.append(
      chip(`${filter.name} ✕`, `Stop filtering on ${filter.name}`, () => removeFilter(filter), "chip active"),
    );
  }
}

function showWeight() {
  weightValue.textContent = Number(weight.value).toFixed(2);
}

function searchParams() {
  const params = new URLSearchParams();
  const query = box.value.trim();
  if (query) params.set("q", query);
  if (weight.value !== DEFAULT_WEIGHT) params.set("weight", weight.value);
  if (like) params.set("like", String(like));
  if (since.value) params.set("since", since.value);
  if (until.value) params.set("until", until.value);
  for (const filter of active) params.append(filter.kind, filter.name);
  return params;
}

function title(what) {
  document.title = what ? `${what} · immich-moments` : "immich-moments";
}

/** Swap the results in one cross-fade where the browser can do it, plainly where it cannot. */
function paint(update) {
  if (!document.startViewTransition || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    update();
    return;
  }
  document.startViewTransition(update);
}

async function run(push = true) {
  const params = searchParams();
  inflight?.abort();
  inflight = new AbortController();
  const mine = inflight;
  button.disabled = true;
  hint.textContent = "Searching…";
  hint.className = "note";
  clearTimeout(slow);
  slow = setTimeout(() => mine === inflight && showSkeletons(), SKELETON_AFTER);
  try {
    const url = `/api/search?${params}&limit=24`;
    const response = await fetch(url, { signal: mine.signal });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || response.statusText);
    clearTimeout(slow);
    paint(() => render(data));
    const here = `/?${params}`;
    if (push) history.pushState({}, "", here);
    else history.replaceState({}, "", here);
  } catch (error) {
    if (error.name === "AbortError") return;
    clearTimeout(slow);
    results.replaceChildren();
    hint.textContent = `Search failed: ${error.message}`;
    hint.className = "note error";
  } finally {
    if (mine === inflight) button.disabled = false;
  }
}

function describe(data) {
  const parts = [];
  if (data.query.trim()) parts.push(`for “${data.query.trim()}”`);
  if (data.like) parts.push(`like “${data.like.label || `scene ${data.like.scene_index}`}” in ${data.like.file_name}`);
  if (data.people.length) parts.push(`with ${data.people.join(" and ")}`);
  if (data.albums.length) parts.push(`in ${data.albums.join(" and ")}`);
  if (data.since) parts.push(`since ${data.since}`);
  if (data.until) parts.push(`until ${data.until}`);
  return parts.join(" ");
}

function render(data) {
  results.replaceChildren();
  if (data.like) {
    reference = data.like;
    drawFilters();
  }
  const what = describe(data);
  title(data.query.trim() || what);
  if (!data.hits.length) {
    hint.textContent = `Nothing matched ${what}.`;
    const panel = state("Nothing matched that", "Try fewer words, a wider range, or index more videos.");
    const loosen = relax();
    if (loosen) panel.append(loosen);
    results.append(panel);
    return;
  }
  hint.textContent = `${data.count} scene${data.count === 1 ? "" : "s"} ${what}`;
  for (const hit of data.hits) results.append(card(hit));
}

function showEmpty() {
  hint.textContent = "Type what you remember, or pick who is in it. Scene vectors and the transcript are searched together.";
  hint.className = "note";
  const panel = state("Search your own footage", "Describe what you saw, quote what was said, or pick a face.");
  const examples = document.createElement("div");
  examples.className = "chips";
  for (const example of EXAMPLES) {
    examples.append(
      chip(example, `Search for ${example}`, () => {
        box.value = example;
        run();
      }),
    );
  }
  panel.append(examples);
  results.replaceChildren(panel);
}

/** One chip per narrowing in play, because dropping one is the usual way out of no results. */
function relax() {
  const ways = [];
  if (like) {
    ways.push(["stop ranking by picture", () => { clearLike(); rerunOrClear(); }]);
  }
  for (const filter of active) {
    ways.push([`without ${filter.name}`, () => removeFilter(filter)]);
  }
  if (since.value || until.value) {
    ways.push(["without the date range", () => {
      since.value = "";
      until.value = "";
      rerunOrClear();
    }]);
  }
  if (!ways.length) return null;
  const row = document.createElement("div");
  row.className = "chips";
  for (const [label, act] of ways) row.append(chip(label, "Search again without it", act));
  return row;
}

function state(heading, detail) {
  const node = document.createElement("div");
  node.className = "empty";
  const title = document.createElement("h2");
  title.textContent = heading;
  node.append(title, text("p", "detail", detail));
  return node;
}

function showSkeletons() {
  const cards = Array.from({ length: 8 }, () => {
    const node = document.createElement("div");
    node.className = "skeleton";
    node.setAttribute("aria-hidden", "true");
    const frame = document.createElement("div");
    frame.className = "frame";
    const body = document.createElement("div");
    body.className = "body";
    body.append(text("div", "line", ""), text("div", "line short", ""));
    node.append(frame, body);
    return node;
  });
  results.replaceChildren(...cards);
}

function card(hit) {
  const node = document.createElement("article");
  node.className = "hit";

  const frame = document.createElement("div");
  frame.className = "frame";
  if (hit.thumb) {
    const image = document.createElement("img");
    image.src = hit.thumb;
    image.alt = hit.label || `scene ${hit.scene_index}`;
    image.loading = "lazy";
    image.decoding = "async";
    image.addEventListener("load", () => image.classList.add("ready"), { once: true });
    if (image.complete) image.classList.add("ready");
    frame.append(image);
  }
  const stamp = text("span", "stamp", hit.timestamp);
  stamp.title = `A ${hit.duration} scene starting at ${hit.timestamp}`;
  frame.append(stamp);
  node.append(frame);

  const body = document.createElement("div");
  body.className = "body";
  if (hit.label) body.append(text("div", "label", hit.label));
  body.append(text("div", "file", hit.file_name));

  if (hit.people.length) {
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const person of hit.people) {
      chips.append(chip(person, `Only scenes with ${person}`, () => addFilter("person", person)));
    }
    body.append(chips);
  }

  if (hit.transcript) {
    const said = document.createElement("blockquote");
    said.className = "said";
    said.textContent = `“${hit.transcript}”`;
    body.append(said);
  }

  body.append(why(hit));

  const links = document.createElement("div");
  links.className = "links";
  const link = document.createElement("a");
  link.href = hit.immich_url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = `Open in Immich at ${hit.timestamp}`;
  const more = document.createElement("button");
  more.type = "button";
  more.className = "more";
  more.textContent = "more like this";
  more.title = "Scenes that look like this one";
  more.addEventListener("click", () => showLike(hit));
  links.append(link, more);
  body.append(links);

  node.append(body);
  return node;
}

/** Which channel found this scene, and the score. Browsing ranks nothing, so it shows the date. */
function why(hit) {
  const row = document.createElement("div");
  row.className = "why";
  const marks = document.createElement("div");
  marks.className = "signals";
  if (hit.visual_score) {
    marks.append(tag("picture", "vision", `CLIP cosine ${hit.visual_score.toFixed(3)}`));
  }
  if (hit.text_score) {
    marks.append(tag("speech", "speech", "matched the transcript"));
  }
  row.append(marks);
  row.append(text("span", "score", hit.score ? hit.score.toFixed(3) : (hit.file_created_at || "").slice(0, 10)));
  return row;
}

function tag(label, className, title) {
  const node = document.createElement("span");
  node.className = `signal ${className}`;
  node.textContent = label;
  node.title = title;
  return node;
}

function text(tag, className, value) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = value;
  return node;
}
