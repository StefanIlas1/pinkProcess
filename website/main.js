const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, value));
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

let scrollSpeed = 0;

function setupFabric() {
  const canvas = $(".fabric");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const cell = 26;
  let columns = 0;
  let rows = 0;
  let ratio = 1;
  let glow = new Float32Array(0);
  let pulses = [];

  const spawnPulse = (anywhere) => ({
    row: Math.floor(Math.random() * rows),
    x: anywhere ? Math.random() * columns : -2,
    speed: 0.06 + Math.random() * 0.12,
    hot: Math.random() < 0.25
  });

  const resize = () => {
    ratio = Math.min(2, devicePixelRatio || 1);
    canvas.width = innerWidth * ratio;
    canvas.height = innerHeight * ratio;
    columns = Math.ceil(innerWidth / cell) + 1;
    rows = Math.ceil(innerHeight / cell) + 1;
    glow = new Float32Array(columns * rows);
    pulses = Array.from({ length: Math.round((columns * rows) / 260) }, () => spawnPulse(true));
  };

  const draw = () => {
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, innerWidth, innerHeight);
    const boost = 1 + Math.min(6, scrollSpeed * 0.08);
    pulses.forEach((pulse, index) => {
      pulse.x += pulse.speed * boost;
      if (pulse.x > columns + 2) pulses[index] = spawnPulse(false);
      const column = Math.floor(pulse.x);
      if (column >= 0 && column < columns) {
        const slot = pulse.row * columns + column;
        glow[slot] = Math.max(glow[slot], pulse.hot ? 1 : 0.6);
      }
    });
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const slot = row * columns + column;
        const level = glow[slot];
        const x = column * cell + 9;
        const y = row * cell + 9;
        if (level > 0.02) {
          context.fillStyle = `rgba(251,100,141,${(level * 0.4).toFixed(3)})`;
          context.beginPath();
          context.roundRect(x - 1, y - 1, 8, 8, 2.5);
          context.fill();
          glow[slot] = level * 0.94;
        } else {
          context.fillStyle = "rgba(253,206,223,0.055)";
          context.fillRect(x + 2, y + 2, 2, 2);
        }
      }
    }
    scrollSpeed *= 0.9;
  };

  resize();
  addEventListener("resize", resize);
  if (reducedMotion) { draw(); return; }
  let lastTime = 0;
  const loop = (time) => {
    if (time - lastTime > 33 && !document.hidden) { lastTime = time; draw(); }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function setupNav() {
  const nav = $(".nav");
  if (!nav) return;
  let lastY = scrollY;
  addEventListener("scroll", () => {
    scrollSpeed = Math.max(scrollSpeed, Math.abs(scrollY - lastY));
    nav.classList.toggle("tuck", scrollY > lastY && scrollY > 240);
    lastY = scrollY;
  }, { passive: true });
}

function setupReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in");
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -12% 0px" });
  $$("[data-reveal]").forEach((element) => observer.observe(element));
}

const scrubbers = [];
function addScrubber(element, measure, onProgress) {
  scrubbers.push({ element, measure, onProgress, last: -1 });
}
function pinnedProgress(element) {
  const box = element.getBoundingClientRect();
  return clamp(-box.top / Math.max(1, box.height - innerHeight));
}
function passingProgress(element) {
  const box = element.getBoundingClientRect();
  return clamp((innerHeight - box.top) / (innerHeight + box.height));
}
function runScrubbers() {
  scrubbers.forEach((scrubber) => {
    const progress = scrubber.measure(scrubber.element);
    if (Math.abs(progress - scrubber.last) < 0.0005) return;
    scrubber.last = progress;
    scrubber.element.style.setProperty("--p", progress.toFixed(4));
    if (scrubber.onProgress) scrubber.onProgress(progress);
  });
}
function startScrubLoop() {
  let ticking = false;
  const request = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; runScrubbers(); });
  };
  addEventListener("scroll", request, { passive: true });
  addEventListener("resize", () => { scrubbers.forEach((scrubber) => (scrubber.last = -1)); request(); });
  runScrubbers();
}

function setupMarquee() {
  const track = $(".marquee-track");
  if (!track) return;
  track.innerHTML += track.innerHTML;
  const chips = $$("span", track);
  const half = chips.length / 2;
  let index = 0;
  setInterval(() => {
    chips.forEach((chip) => chip.classList.remove("on"));
    index = (index + 3) % half;
    chips[index].classList.add("on");
    chips[index + half].classList.add("on");
  }, 1400);
}

const hotPathStages = [
  { name: "Data", note: "Your CSV or JSON lines file is read on your Mac. Columns, types and the time column get spotted." },
  { name: "Profile", note: "Signals like queue, latency, errors or temperature are found, so the app knows what it is looking at." },
  { name: "Model", note: "A small quantized model that fits the data is picked for you. Swap it any time from the library." },
  { name: "PYNQ-Z2", note: "On the board, the FPGA runs the INT8 matrix math and the ARM core drives generation." },
  { name: "Answer", note: "Short lands first, then Summary and Final. Every answer shows the readings it rests on." }
];

function setupHotPath() {
  const path = $(".path");
  if (!path) return;
  const road = $(".road", path);
  const svg = $("svg.track", road);
  const lane = $(".lane", svg);
  const laneDash = $(".lane-dash", svg);
  const laneFill = $(".lane-fill", svg);
  const stations = $$(".station", road);
  const nowName = $(".path-now strong", path);
  const nowNote = $(".path-now span", path);
  const stops = [0.01, 0.255, 0.5, 0.745, 0.99];
  let length = 0;
  let activeIndex = -1;

  const layout = () => {
    const width = road.clientWidth;
    const height = road.clientHeight;
    const inset = Math.min(70, width * 0.08);
    const span = width - inset * 2;
    const at = (fraction) => inset + span * fraction;
    const middle = height / 2;
    const swing = height * 0.22;
    const shape = `M ${at(0)} ${middle} C ${at(0.15)} ${middle - swing}, ${at(0.3)} ${middle + swing}, ${at(0.5)} ${middle} S ${at(0.85)} ${middle - swing}, ${at(1)} ${middle}`;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    [lane, laneDash, laneFill].forEach((element) => element.setAttribute("d", shape));
    length = lane.getTotalLength();
    laneFill.style.strokeDasharray = `${length} ${length}`;
    stations.forEach((station, index) => {
      const point = lane.getPointAtLength(length * stops[index]);
      station.style.left = point.x + "px";
      station.style.top = point.y + "px";
    });
  };

  const show = (progress) => {
    if (!length) return;
    const travel = clamp((progress - 0.06) / 0.84);
    const along = stops[0] + travel * (stops[stops.length - 1] - stops[0]);
    laneFill.style.strokeDashoffset = length * (1 - along);
    let reached = -1;
    stops.forEach((stop, index) => {
      const lit = along >= stop - 0.001;
      if (lit) reached = index;
      stations[index].classList.toggle("lit", lit);
    });
    if (reached === activeIndex) return;
    if (reached > activeIndex && reached >= 0) {
      stations[reached].classList.remove("pop");
      void stations[reached].offsetWidth;
      stations[reached].classList.add("pop");
    }
    activeIndex = reached;
    const stage = hotPathStages[Math.max(0, reached)];
    nowName.textContent = reached < 0 ? "Ready" : stage.name;
    nowNote.textContent = reached < 0 ? "Scroll to send a question down the path." : stage.note;
  };

  layout();
  addEventListener("resize", () => { layout(); show(pinnedProgress(path)); });
  addScrubber(path, pinnedProgress, show);
}

function setupSteps() {
  const steps = $$(".step");
  if (!steps.length) return;
  const mock = createMock();
  let current = -1;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const index = steps.indexOf(entry.target);
      if (index === current) return;
      current = index;
      steps.forEach((step, stepIndex) => step.classList.toggle("on", stepIndex === index));
      if (mock) mock.play(index);
    });
  }, { rootMargin: "-45% 0px -45% 0px" });
  steps.forEach((step) => observer.observe(step));
}

const shortAnswer = "waiting is steady and now 0; it peaked at 58 at 36.9 s. It spiked 1 time.";
const summaryAnswer = shortAnswer + "\nThe file has 600 rows covering 0.0 s to 59.9 s.\nTypical waiting is 7.90. Over the most recent readings it is steady.";

function createMock() {
  const mock = $(".mock");
  if (!mock) return null;
  const fit = mock.parentElement;
  const cursor = $(".m-cursor", mock);
  const pick = (selector) => $(selector, mock);
  let scale = 1;
  let runId = 0;
  const resize = () => {
    scale = fit.clientWidth / 760;
    mock.style.setProperty("--s", scale.toFixed(4));
  };
  new ResizeObserver(resize).observe(fit);
  resize();

  const stopped = () => { const error = new Error("stopped"); error.stopped = true; return error; };
  const pointOf = (element, offsetX = 0.5, offsetY = 0.5) => {
    const box = element.getBoundingClientRect();
    const frame = mock.getBoundingClientRect();
    return { x: (box.left - frame.left + box.width * offsetX) / scale, y: (box.top - frame.top + box.height * offsetY) / scale };
  };
  const setCursor = (point, instant) => {
    cursor.style.transition = instant ? "none" : "";
    cursor.style.setProperty("--x", point.x - 4 + "px");
    cursor.style.setProperty("--y", point.y - 3 + "px");
  };

  function scene(id) {
    const alive = () => id === runId;
    const wait = (ms) => new Promise((resolve, reject) => setTimeout(() => (alive() ? resolve() : reject(stopped())), reducedMotion ? 0 : ms));
    const moveTo = async (selector, offsetX, offsetY) => {
      const target = typeof selector === "string" ? pick(selector) : selector;
      setCursor(pointOf(target, offsetX, offsetY));
      await wait(800);
      return target;
    };
    const click = async (selector, offsetX, offsetY) => {
      const target = await moveTo(selector, offsetX, offsetY);
      const point = pointOf(target, offsetX, offsetY);
      const ripple = document.createElement("span");
      ripple.className = "m-ripple";
      ripple.style.left = point.x + "px";
      ripple.style.top = point.y + "px";
      mock.appendChild(ripple);
      setTimeout(() => ripple.remove(), 600);
      target.classList.add("m-press");
      await wait(140);
      target.classList.remove("m-press");
      return target;
    };
    const type = async (element, text, speed = 55) => {
      for (let index = 1; index <= text.length; index++) {
        element.textContent = text.slice(0, index);
        await wait(speed + (text[index - 1] === " " ? 40 : Math.random() * 40));
      }
    };
    return { alive, wait, moveTo, click, type };
  }

  const states = [
    { classes: [], model: "FLAN-T5-small INT8", by: "by PYNQ team · Runs on the board" },
    { classes: ["connected", "imported", "recommended"], model: "Queue-onset logistic", by: "by Stefan Ilas · Runs on the board" },
    { classes: ["connected", "imported", "recommended"], model: "Queue-onset logistic", by: "by Stefan Ilas · Runs on the board" },
    { classes: ["connected", "imported", "recommended", "asked", "answered"], model: "Queue-onset logistic", by: "by Stefan Ilas · Runs on the board" },
    { classes: ["connected", "imported", "recommended", "asked", "answered"], model: "Queue-onset logistic", by: "by Stefan Ilas · Runs on the board" }
  ];
  const tabs = $$(".m-tabs span", mock);
  const pathRows = $$(".m-path-row", mock);
  const setTab = (name) => tabs.forEach((tab) => tab.classList.toggle("on", tab.dataset.tab === name));

  const reset = (index) => {
    const state = states[index];
    mock.className = ["mock", ...state.classes].join(" ");
    pick(".m-status span").textContent = state.classes.includes("connected") ? "PYNQ-Z2 connected" : "Not connected";
    pick(".m-model-name").textContent = state.model;
    pick(".m-by").textContent = state.by;
    pick(".m-typed").textContent = "";
    pick(".m-search-typed").textContent = "";
    pick(".m-answer-skill em").textContent = "original";
    $$(".m-row", mock).forEach((row) => row.classList.remove("gone", "hover"));
    $$(".m-rail-tabs span", mock).forEach((tab) => tab.classList.toggle("on", tab.dataset.rtab === "notes"));
    $$(".m-hover", mock).forEach((element) => element.classList.remove("m-hover"));
    $$(".pressed, .m-press, .hover", mock).forEach((element) => element.classList.remove("pressed", "m-press", "hover"));
    $$(".m-ripple", mock).forEach((element) => element.remove());
    const answered = state.classes.includes("answered");
    tabs.forEach((tab) => tab.classList.toggle("wait", !answered));
    setTab(answered ? "Short" : "");
    pick(".m-text").textContent = answered ? shortAnswer : "";
    pathRows.forEach((row) => {
      row.classList.toggle("lit", answered);
      $("b", row).textContent = answered ? Number(row.querySelector("b").dataset.ms).toLocaleString("en-US") + " ms" : "0 ms";
    });
  };

  const scenes = [
    async ({ wait, click, moveTo }) => {
      await wait(300);
      await click(".m-find");
      mock.classList.add("searching");
      pick(".m-status span").textContent = "Searching";
      await wait(1200);
      mock.classList.remove("searching");
      mock.classList.add("connected");
      pick(".m-status span").textContent = "PYNQ-Z2 connected";
      await wait(700);
      const sample = await moveTo(".m-target-sample");
      sample.classList.add("m-hover");
      await click(".m-target-sample");
      mock.classList.add("imported");
      await wait(500);
      pick(".m-model-name").textContent = "Queue-onset logistic";
      pick(".m-by").textContent = "by Stefan Ilas · Runs on the board";
      mock.classList.add("recommended");
      await wait(1400);
      await moveTo(".m-rec");
      await wait(1600);
    },
    async ({ wait, click, moveTo, type }) => {
      await wait(300);
      await click(".m-search");
      mock.classList.add("library");
      await wait(500);
      await click(".m-search-field", 0.2);
      mock.classList.add("searchfocus");
      await type(pick(".m-search-typed"), "queue", 90);
      pick(".m-row-2").classList.add("gone");
      await wait(400);
      const row = await moveTo(".m-row-1", 0.4);
      row.classList.add("hover");
      await wait(1100);
      await click(".m-close");
      mock.classList.remove("library", "searchfocus");
      await wait(1400);
    },
    async ({ wait, click, moveTo, type }) => {
      await wait(300);
      await click(".m-input", 0.3);
      mock.classList.add("focus", "typing");
      await type(pick(".m-typed"), "Why did the queue spike?");
      await wait(300);
      const enter = await moveTo(".m-return");
      enter.classList.add("pressed");
      await click(".m-return");
      await wait(120);
      enter.classList.remove("pressed");
      pick(".m-typed").textContent = "";
      mock.classList.remove("typing");
      mock.classList.add("asked", "thinking");
      tabs.forEach((tab) => tab.classList.add("wait"));
      await moveTo(".m-chat", 0.75, 0.75);
      for (const row of pathRows) {
        row.classList.add("lit");
        await wait(320);
      }
      mock.classList.remove("thinking");
      tabs[0].classList.remove("wait");
      setTab("Short");
      await type(pick(".m-text"), shortAnswer, 12);
      tabs[1].classList.remove("wait");
      await wait(600);
      await click(".m-tabs [data-tab=Summary]");
      setTab("Summary");
      pick(".m-text").textContent = summaryAnswer;
      tabs[2].classList.remove("wait");
      tabs[3].classList.remove("wait");
      await wait(2000);
    },
    async ({ wait, moveTo }) => {
      pathRows.forEach((row) => { row.classList.remove("lit"); $("b", row).textContent = "0 ms"; });
      await wait(300);
      await moveTo(".m-strip", 0.5, 0.55);
      mock.classList.add("peek");
      await wait(450);
      for (const row of pathRows) {
        await moveTo(row, 0.55);
        row.classList.add("lit");
        const label = $("b", row);
        const target = Number(label.dataset.ms);
        const frames = target > 10 ? 18 : 2;
        for (let frame = 1; frame <= frames; frame++) {
          label.textContent = Math.round((target * frame) / frames).toLocaleString("en-US") + " ms";
          await wait(28);
        }
      }
      await wait(1400);
      await moveTo(".m-chat", 0.6, 0.5);
      mock.classList.remove("peek");
      await wait(1200);
    },
    async ({ wait, click, moveTo }) => {
      await wait(300);
      await click(".m-rail-tabs [data-rtab=skills]");
      $$(".m-rail-tabs span", mock).forEach((tab) => tab.classList.toggle("on", tab.dataset.rtab === "skills"));
      mock.classList.add("skills");
      await wait(500);
      await click(".m-answer-skill");
      mock.classList.add("options");
      await wait(300);
      const option = await moveTo(".m-v2");
      option.classList.add("hover");
      await click(".m-v2");
      option.classList.remove("hover");
      mock.classList.remove("options");
      pick(".m-answer-skill em").textContent = "Queue history v2";
      mock.classList.add("publishable");
      await wait(500);
      await click(".m-publish");
      mock.classList.add("toasted");
      await wait(1800);
      mock.classList.remove("toasted");
      await wait(1000);
    }
  ];

  return {
    async play(index) {
      const id = ++runId;
      const tools = scene(id);
      while (tools.alive()) {
        reset(index);
        if (!tools.alive()) return;
        try {
          await scenes[index](tools);
          await tools.wait(900);
        } catch (error) {
          if (!error.stopped) throw error;
          return;
        }
      }
    }
  };
}

function setupTrace() {
  const trace = $(".trace");
  if (!trace) return;
  const values = trace.dataset.series.split(",").map(Number);
  const svg = $(".trace-chart", trace);
  const line = $(".trace-line", svg);
  const area = $(".trace-area", svg);
  const reveal = $(".trace-reveal", svg);
  const scan = $(".trace-scan", svg);
  const peak = $(".trace-peak", trace);
  const question = $(".trace-q", trace);
  const answer = $(".trace-a", trace);
  const answerText = $(".trace-a-text", trace);
  const width = values.length - 1;
  const top = Math.max(...values);
  const peakIndex = values.indexOf(top);
  const yOf = (value) => 96 - (value / top) * 78;
  const points = values.map((value, index) => `${index} ${yOf(value).toFixed(2)}`);
  svg.setAttribute("viewBox", `0 0 ${width} 100`);
  line.setAttribute("d", "M" + points.join(" L"));
  area.setAttribute("d", `M0 100 L${points.join(" L")} L${width} 100 Z`);

  const placePeak = () => {
    const box = svg.getBoundingClientRect();
    const frame = trace.getBoundingClientRect();
    peak.style.left = box.left - frame.left + (peakIndex / width) * box.width + "px";
    peak.style.top = box.top - frame.top + (yOf(top) / 100) * box.height + "px";
  };
  placePeak();
  addEventListener("resize", placePeak);

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, reducedMotion ? 0 : ms));
  const draw = (duration) => new Promise((resolve) => {
    const start = performance.now();
    const step = (now) => {
      const progress = reducedMotion ? 1 : Math.min(1, (now - start) / duration);
      const x = progress * width;
      reveal.setAttribute("width", x);
      scan.setAttribute("x1", x);
      scan.setAttribute("x2", x);
      if (x >= peakIndex) peak.classList.add("on");
      if (progress < 1) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });

  const loop = async () => {
    while (true) {
      peak.classList.remove("on");
      question.classList.remove("on");
      answer.classList.remove("on");
      answerText.textContent = "";
      reveal.setAttribute("width", 0);
      scan.style.opacity = "";
      await sleep(500);
      question.classList.add("on");
      await sleep(600);
      await draw(3000);
      scan.style.opacity = "0";
      answer.classList.add("on");
      for (let index = 1; index <= shortAnswer.length; index++) {
        answerText.textContent = shortAnswer.slice(0, index);
        await sleep(24);
      }
      await sleep(5200);
      if (reducedMotion) return;
    }
  };
  loop();
}

function setupStack() {
  const row = $(".stack-row");
  if (!row) return;
  const svg = $(":scope > svg", row);
  const line = $("path", svg);
  const layers = $$(".layer", row);
  const draw = () => {
    svg.setAttribute("viewBox", `0 0 ${row.clientWidth} 20`);
    const first = layers[0].offsetLeft + layers[0].offsetWidth / 2;
    const last = layers[layers.length - 1].offsetLeft + layers[layers.length - 1].offsetWidth / 2;
    line.setAttribute("d", `M ${first} 2 L ${last} 2`);
    row.style.setProperty("--len", Math.ceil(last - first));
  };
  draw();
  addEventListener("resize", draw);
  addScrubber(row, (element) => clamp((passingProgress(element) - 0.25) / 0.35), (progress) => {
    layers.forEach((layer, index) => layer.classList.toggle("lit", progress >= index / (layers.length - 1) - 0.02));
  });
}

function setupCta() {
  $$(".cta").forEach((cta) => addScrubber(cta, passingProgress));
}

function burst(originX, originY) {
  if (reducedMotion) return;
  const layer = document.createElement("div");
  layer.className = "confetti";
  document.body.appendChild(layer);
  const colors = ["#fdcedf", "#fb648d", "#fdcedf", "#ffffff"];
  const pieces = Array.from({ length: 34 }, (_, index) => {
    const piece = document.createElement("i");
    const size = 8 + Math.random() * 14;
    piece.style.width = piece.style.height = size + "px";
    piece.style.background = colors[index % colors.length];
    piece.style.left = originX + "px";
    piece.style.top = originY + "px";
    layer.appendChild(piece);
    const angle = Math.random() * Math.PI * 2;
    const speed = 5 + Math.random() * 9;
    return { piece, x: 0, y: 0, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed - 7, life: 0, spin: Math.random() * 360 };
  });
  const step = () => {
    let alive = false;
    pieces.forEach((item) => {
      item.life++;
      item.vy += 0.42;
      item.vx *= 0.985;
      item.x += item.vx;
      item.y += item.vy;
      const fade = Math.max(0, 1 - item.life / 80);
      if (fade > 0) alive = true;
      item.piece.style.transform = `translate(${item.x}px, ${item.y}px) rotate(${item.spin + item.life * 6}deg) scale(${0.4 + fade * 0.6})`;
      item.piece.style.opacity = fade;
    });
    if (alive) requestAnimationFrame(step);
    else layer.remove();
  };
  requestAnimationFrame(step);
}

function setupDownload() {
  const buttons = $$("[data-download]");
  if (!buttons.length) return;
  fetch(buttons[0].getAttribute("href"), { method: "HEAD" }).then((response) => {
    const bytes = +response.headers.get("content-length");
    if (response.ok && bytes > 100000) $$("[data-size]").forEach((label) => (label.textContent = (bytes / 1048576).toFixed(1) + " MB"));
  }).catch(() => {});
  const toast = $(".toast");
  buttons.forEach((button) => button.addEventListener("click", () => {
    const box = button.getBoundingClientRect();
    burst(box.left + box.width / 2, box.top + box.height / 2);
    toast.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => toast.classList.remove("show"), 4200);
  }));
}

setupFabric();
setupNav();
setupReveal();
setupTrace();
setupMarquee();
setupHotPath();
setupSteps();
setupStack();
setupCta();
setupDownload();
startScrubLoop();
