const form = document.getElementById("search");
const box = document.getElementById("q");
const weight = document.getElementById("weight");
const weightValue = document.getElementById("weight-value");
const results = document.getElementById("results");
const hint = document.getElementById("hint");
const button = form.querySelector("button");

weight.addEventListener("input", () => {
  weightValue.textContent = Number(weight.value).toFixed(2);
});
weight.addEventListener("change", () => {
  if (box.value.trim()) run(box.value.trim());
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = box.value.trim();
  if (query) run(query);
});

window.addEventListener("popstate", () => {
  const query = new URLSearchParams(location.search).get("q") || "";
  box.value = query;
  if (query) run(query, false);
});

const initial = new URLSearchParams(location.search).get("q");
if (initial) {
  box.value = initial;
  run(initial, false);
}

async function run(query, push = true) {
  button.disabled = true;
  hint.textContent = "Searching…";
  hint.className = "note";
  try {
    const url = `/api/search?q=${encodeURIComponent(query)}&limit=24&weight=${weight.value}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || data.detail || response.statusText);
    render(query, data);
    if (push) history.pushState({}, "", `/?q=${encodeURIComponent(query)}`);
  } catch (error) {
    results.replaceChildren();
    hint.textContent = `Search failed: ${error.message}`;
    hint.className = "note error";
  } finally {
    button.disabled = false;
  }
}

function render(query, data) {
  results.replaceChildren();
  if (!data.hits.length) {
    hint.textContent = `Nothing matched “${query}”. Try fewer words, or index more videos.`;
    return;
  }
  hint.textContent = `${data.count} scene${data.count === 1 ? "" : "s"} for “${query}”`;
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
  top.append(text("span", "at", hit.timestamp), text("span", "score", hit.score.toFixed(3)));
  body.append(top);

  if (hit.label) body.append(text("div", "label", hit.label));
  body.append(text("div", "file", hit.file_name));

  if (hit.people.length) {
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const person of hit.people) chips.append(text("span", "chip", person));
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
