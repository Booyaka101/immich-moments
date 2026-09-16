const form = document.getElementById("search");
const box = document.getElementById("q");
const weight = document.getElementById("weight");
const weightValue = document.getElementById("weight-value");
const people = document.getElementById("people");
const filters = document.getElementById("filters");
const results = document.getElementById("results");
const hint = document.getElementById("hint");
const button = form.querySelector("button");

const active = [];
let like = null;
let reference = null;

weight.addEventListener("input", () => {
  weightValue.textContent = Number(weight.value).toFixed(2);
});
weight.addEventListener("change", () => {
  if (box.value.trim()) run();
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (box.value.trim()) clearLike();
  if (anything()) run();
});

people.addEventListener("change", () => {
  addPerson(people.value);
  people.value = "";
});

window.addEventListener("popstate", () => readUrl(false));

loadPeople();
readUrl(false);

function anything() {
  return Boolean(box.value.trim() || active.length || like);
}

function readUrl(push) {
  const params = new URLSearchParams(location.search);
  box.value = params.get("q") || "";
  like = params.get("like") ? Number(params.get("like")) : null;
  reference = null;
  active.length = 0;
  active.push(...params.getAll("person"));
  drawFilters();
  if (anything()) run(push);
}

function addPerson(name) {
  if (!name || active.includes(name)) return;
  active.push(name);
  drawFilters();
  run();
}

function removePerson(name) {
  const at = active.indexOf(name);
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

function rerunOrClear() {
  if (anything()) {
    run();
    return;
  }
  results.replaceChildren();
  hint.textContent = "Type what you remember, or pick who is in it.";
  history.pushState({}, "", "/");
}

async function loadPeople() {
  try {
    const response = await fetch("/api/people");
    const data = await response.json();
    for (const person of data.people) {
      const option = document.createElement("option");
      option.value = person.name;
      option.textContent = `${person.name} (${person.scenes})`;
      people.append(option);
    }
    people.disabled = data.people.length === 0;
  } catch {
    people.disabled = true;
  }
}

function chip(label, title, onClick) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = "chip active";
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
      }),
    );
  }
  for (const name of active) {
    filters.append(chip(`${name} ✕`, `Stop filtering on ${name}`, () => removePerson(name)));
  }
}

function searchParams() {
  const params = new URLSearchParams();
  const query = box.value.trim();
  if (query) params.set("q", query);
  if (like) params.set("like", String(like));
  for (const name of active) params.append("person", name);
  return params;
}

async function run(push = true) {
  const params = searchParams();
  button.disabled = true;
  hint.textContent = "Searching…";
  hint.className = "note";
  try {
    const url = `/api/search?${params}&limit=24&weight=${weight.value}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || response.statusText);
    render(data);
    if (push) history.pushState({}, "", `/?${params}`);
  } catch (error) {
    results.replaceChildren();
    hint.textContent = `Search failed: ${error.message}`;
    hint.className = "note error";
  } finally {
    button.disabled = false;
  }
}

function describe(data) {
  const parts = [];
  if (data.query.trim()) parts.push(`for “${data.query.trim()}”`);
  if (data.like) parts.push(`like “${data.like.label || `scene ${data.like.scene_index}`}” in ${data.like.file_name}`);
  if (data.people.length) parts.push(`with ${data.people.join(" and ")}`);
  return parts.join(" ");
}

function render(data) {
  results.replaceChildren();
  if (data.like) {
    reference = data.like;
    drawFilters();
  }
  const what = describe(data);
  if (!data.hits.length) {
    hint.textContent = `Nothing matched ${what}. Try fewer words, or index more videos.`;
    return;
  }
  hint.textContent = `${data.count} scene${data.count === 1 ? "" : "s"} ${what}`;
  for (const hit of data.hits) results.append(card(hit));
}

function card(hit) {
  const node = document.createElement("article");
  node.className = "hit";

  if (hit.thumb) {
    const image = document.createElement("img");
    image.src = hit.thumb;
    image.alt = hit.label || `scene ${hit.scene_index}`;
    image.loading = "lazy";
    node.append(image);
  }

  const body = document.createElement("div");
  body.className = "body";

  const top = document.createElement("div");
  top.className = "top";
  // Scores mean nothing when nothing was ranked, so browsing shows the date instead.
  const right = hit.score ? hit.score.toFixed(3) : (hit.file_created_at || "").slice(0, 10);
  top.append(text("span", "at", hit.timestamp), text("span", "score", right));
  body.append(top);

  if (hit.label) body.append(text("div", "label", hit.label));
  body.append(text("div", "file", hit.file_name));

  if (hit.people.length) {
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const person of hit.people) {
      const node = document.createElement("button");
      node.type = "button";
      node.className = "chip";
      node.textContent = person;
      node.title = `Only scenes with ${person}`;
      node.addEventListener("click", () => addPerson(person));
      chips.append(node);
    }
    body.append(chips);
  }

  if (hit.transcript) body.append(text("div", "said", `“${hit.transcript}”`));

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

function text(tag, className, value) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = value;
  return node;
}
