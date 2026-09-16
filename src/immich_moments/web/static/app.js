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

weight.addEventListener("input", () => {
  weightValue.textContent = Number(weight.value).toFixed(2);
});
weight.addEventListener("change", () => {
  if (box.value.trim()) run();
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (box.value.trim() || active.length) run();
});

people.addEventListener("change", () => {
  addPerson(people.value);
  people.value = "";
});

window.addEventListener("popstate", () => readUrl(false));

loadPeople();
readUrl(false);

function readUrl(push) {
  const params = new URLSearchParams(location.search);
  box.value = params.get("q") || "";
  active.length = 0;
  active.push(...params.getAll("person"));
  drawFilters();
  if (box.value.trim() || active.length) run(push);
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
  if (box.value.trim() || active.length) run();
  else {
    results.replaceChildren();
    hint.textContent = "Type what you remember, or pick who is in it.";
    history.pushState({}, "", "/");
  }
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

function drawFilters() {
  filters.replaceChildren();
  for (const name of active) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip active";
    chip.textContent = `${name} ✕`;
    chip.title = `Stop filtering on ${name}`;
    chip.addEventListener("click", () => removePerson(name));
    filters.append(chip);
  }
}

function searchParams() {
  const params = new URLSearchParams();
  const query = box.value.trim();
  if (query) params.set("q", query);
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
  if (data.people.length) parts.push(`with ${data.people.join(" and ")}`);
  return parts.join(" ");
}

function render(data) {
  results.replaceChildren();
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
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = person;
      chip.title = `Only scenes with ${person}`;
      chip.addEventListener("click", () => addPerson(person));
      chips.append(chip);
    }
    body.append(chips);
  }

  if (hit.transcript) body.append(text("div", "said", `“${hit.transcript}”`));

  const link = document.createElement("a");
  link.href = hit.immich_url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = `Open in Immich at ${hit.timestamp}`;
  body.append(link);

  node.append(body);
  return node;
}

function text(tag, className, value) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = value;
  return node;
}
