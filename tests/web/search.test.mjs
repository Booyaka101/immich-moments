import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { setImmediate } from "node:timers/promises";
import vm from "node:vm";

const source = readFileSync(new URL("../../src/immich_moments/web/static/app.js", import.meta.url), "utf8");

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function browser({ transitions = false } = {}) {
  const nodes = new Map();
  function element() {
    return {
      value: "", textContent: "", disabled: false,
      classList: { toggle() {}, add() {}, remove() {} },
      addEventListener() {}, setAttribute() {}, append() {},
      replaceChildren(...children) { this.children = children; },
      querySelector() { return nodes.get("button"); },
    };
  }
  for (const id of ["search", "q", "weight", "weight-value", "people", "albums", "since", "until", "filters", "results", "hint", "theme", "button"]) {
    nodes.set(id, element());
  }
  nodes.get("weight").value = "0.65";
  const requests = [], timers = new Map(), paints = [], rendered = [], history = [];
  let timerId = 0, skeletons = 0;
  const context = vm.createContext({
    document: {
      getElementById: (id) => nodes.get(id), createElement: element,
      ...(transitions ? { startViewTransition: (callback) => paints.push(callback) } : {}),
    },
    window: { addEventListener() {} }, location: { search: "" },
    history: {
      pushState: (_state, _unused, url) => history.push(url),
      replaceState: (_state, _unused, url) => history.push(url),
    },
    matchMedia: () => ({ matches: false }), URLSearchParams, AbortController,
    setTimeout: (callback) => { timers.set(++timerId, callback); return timerId; },
    clearTimeout: (id) => timers.delete(id),
    fetch: (url, options) => {
      if (!url.startsWith("/api/search?")) return Promise.resolve({ json: async () => ({ people: [], albums: [] }) });
      const pending = deferred();
      requests.push({ ...pending, signal: options.signal });
      return pending.promise;
    },
    recordRender: (data) => rendered.push(data),
    recordSkeletons: () => skeletons++,
  });
  vm.runInContext(source, context);
  vm.runInContext("render = recordRender; showSkeletons = recordSkeletons;", context);
  const evaluate = (code) => vm.runInContext(code, context);
  return {
    nodes, requests, timers, paints, rendered, history, evaluate,
    get skeletons() { return skeletons; },
    search(query) { nodes.get("q").value = query; return evaluate("run()"); },
    clear() { nodes.get("q").value = ""; evaluate("rerunOrClear()"); },
    respond(index, data = {}) { requests[index].resolve({ ok: true, json: async () => data }); },
  };
}

test("clearing search aborts the request and removes its skeleton timer", async () => {
  const page = browser();
  const pending = page.search("candles");
  page.clear();
  assert.equal(page.requests[0].signal.aborted, true);
  assert.equal(page.timers.size, 0);
  assert.equal(page.nodes.get("button").disabled, false);
  page.respond(0);
  await pending;
  assert.equal(page.rendered.length, 0);
  assert.deepEqual(page.history, ["/"]);
});

test("a late response body cannot restore cleared results", async () => {
  const page = browser();
  const body = deferred();
  const pending = page.search("candles");
  page.requests[0].resolve({ ok: true, json: () => body.promise });
  await setImmediate();
  page.clear();
  body.resolve({ query: "candles" });
  await pending;
  assert.equal(page.rendered.length, 0);
  assert.deepEqual(page.history, ["/"]);
});

for (const fails of [false, true]) {
  test(`a superseded ${fails ? "failure" : "response"} leaves the current search alone`, async () => {
    const page = browser();
    const old = page.search("old");
    const current = page.search("new");
    if (fails) page.requests[0].reject(new Error("old failure"));
    else page.respond(0, { query: "old" });
    await old;
    assert.equal(page.nodes.get("hint").textContent, "Searching…");
    assert.equal(page.nodes.get("button").disabled, true);
    assert.equal(page.timers.size, 1);
    for (const callback of page.timers.values()) callback();
    assert.equal(page.skeletons, 1);
    page.respond(1, { query: "new" });
    await current;
    assert.deepEqual(page.rendered, [{ query: "new" }]);
    assert.deepEqual(page.history, ["/?q=new"]);
    assert.equal(page.nodes.get("button").disabled, false);
  });
}

for (const next of ["clear", "search"]) {
  test(`a queued view transition cannot paint after a newer ${next}`, async () => {
    const page = browser({ transitions: true });
    const pending = page.search("old");
    page.respond(0, { query: "old" });
    await pending;
    assert.equal(page.paints.length, 1);
    let current;
    if (next === "clear") page.clear();
    else current = page.search("new");
    page.paints.shift()();
    assert.equal(page.rendered.length, 0);
    if (current) {
      page.respond(1, { query: "new" });
      await current;
      page.paints.shift()();
      assert.deepEqual(page.rendered, [{ query: "new" }]);
    }
  });
}

test("back navigation to an empty URL cancels pending search", async () => {
  const page = browser();
  const pending = page.search("candles");
  page.evaluate('location.search = ""; readUrl(false)');
  assert.equal(page.requests[0].signal.aborted, true);
  assert.equal(page.timers.size, 0);
  page.respond(0);
  await pending;
  assert.equal(page.rendered.length, 0);
  assert.deepEqual(page.history, []);
});

test("returning to an empty search also clears a pending typing timer", () => {
  const page = browser();
  page.evaluate("typing = setTimeout(() => run(), TYPING_PAUSE)");
  page.clear();
  assert.equal(page.timers.size, 0);
});

test("the current search still reports errors and restores the button", async () => {
  const page = browser();
  const pending = page.search("candles");
  page.requests[0].reject(new Error("service unavailable"));
  await pending;
  assert.equal(page.nodes.get("hint").textContent, "Search failed: service unavailable");
  assert.equal(page.nodes.get("hint").className, "note error");
  assert.equal(page.nodes.get("button").disabled, false);
  assert.equal(page.timers.size, 0);
});
