const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (value, low = 0, high = 1) => Math.min(high, Math.max(low, value));
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function setupNav() {
  const nav = $(".nav");
  if (!nav) return;
  const links = $(".nav-links", nav);
  const blob = $(".nav-blob", nav);
  const current = $("a[aria-current]", links);
  const moveBlob = (target) => {
    if (!target) { blob.style.opacity = 0; return; }
    blob.style.opacity = 1;
    blob.style.left = target.offsetLeft + "px";
    blob.style.width = target.offsetWidth + "px";
  };
  moveBlob(current);
  $$("a", links).forEach((link) => link.addEventListener("pointerenter", () => moveBlob(link)));
  links.addEventListener("pointerleave", () => moveBlob(current));
  addEventListener("resize", () => moveBlob(current));
  $(".nav-toggle", nav).addEventListener("click", () => nav.classList.toggle("open"));
  let lastY = scrollY;
  addEventListener("scroll", () => {
    const goingDown = scrollY > lastY && scrollY > 240;
    nav.classList.toggle("tuck", goingDown && !nav.classList.contains("open"));
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

function scatterBubbles(container, count, seed) {
  if (!container) return;
  let state = seed;
  const random = () => ((state = (state * 9301 + 49297) % 233280) / 233280);
  const kinds = ["", "", "bow", "hollow", "hollow"];
  for (let index = 0; index < count; index++) {
    const bubble = document.createElement("i");
    const size = 10 + random() * 46;
    bubble.className = kinds[Math.floor(random() * kinds.length)];
    bubble.style.width = bubble.style.height = size + "px";
    const edge = random() * 20;
    bubble.style.left = (random() < 0.5 ? edge : 80 + edge) + "%";
    bubble.style.top = random() * 100 + "%";
    bubble.dataset.depth = (0.2 + random()).toFixed(2);
    container.appendChild(bubble);
  }
  if (reducedMotion) return;
  const bubbles = $$("i", container);
  addEventListener("pointermove", (event) => {
    const offsetX = event.clientX / innerWidth - 0.5;
    const offsetY = event.clientY / innerHeight - 0.5;
    bubbles.forEach((bubble) => {
      const depth = +bubble.dataset.depth;
      bubble.style.setProperty("--mx", (offsetX * 40 * depth).toFixed(1) + "px");
      bubble.style.setProperty("--my", (offsetY * 40 * depth).toFixed(1) + "px");
    });
  }, { passive: true });
  addScrubber(container, passingProgress, (progress) => {
    bubbles.forEach((bubble) => bubble.style.setProperty("--sy", (-(progress - 0.5) * 260 * +bubble.dataset.depth).toFixed(1) + "px"));
  });
}

function setupHero() {
  const hero = $(".hero");
  if (!hero) return;
  addScrubber(hero, () => clamp(scrollY / (innerHeight * 0.55)));
  scatterBubbles($(".bubbles", hero), 18, 7);
  if (reducedMotion) return;
  const floats = $$(".float", hero);
  addEventListener("pointermove", (event) => {
    const offsetX = event.clientX / innerWidth - 0.5;
    const offsetY = event.clientY / innerHeight - 0.5;
    floats.forEach((float, index) => {
      const depth = 14 + index * 8;
      float.style.transform = `translate(${(offsetX * depth).toFixed(1)}px, ${(offsetY * depth).toFixed(1)}px)`;
    });
  }, { passive: true });
}

function setupMarquee() {
  const track = $(".marquee-track");
  if (!track) return;
  track.innerHTML += track.innerHTML;
  let index = 0;
  const chips = $$("span", track);
  setInterval(() => {
    chips.forEach((chip) => chip.classList.remove("on"));
    index = (index + 3) % (chips.length / 2);
    chips[index].classList.add("on");
    chips[index + chips.length / 2].classList.add("on");
  }, 1400);
}

const hotPathStages = [
  { name: "Data", time: "1 ms", note: "Your CSV or JSON lines file is read on your Mac. Columns, types and the time column get spotted." },
  { name: "Profile", time: "0 ms", note: "Signals like queue, latency, errors or temperature are found, so the app knows what it is looking at." },
  { name: "Model", time: "1 ms", note: "A small quantized model that fits the data is picked for you. Swap it any time from the library." },
  { name: "PYNQ-Z2", time: "9,340 ms", note: "On the board, the FPGA runs the INT8 matrix math and the ARM core drives generation." },
  { name: "Answer", time: "17,570 ms", note: "Short lands first, then Summary and Final. Every answer shows the readings it rests on." }
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
  const rider = $(".rider", road);
  const nowName = $(".path-now strong", path);
  const nowNote = $(".path-now span", path);
  const counter = $(".path-count", path);
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
    const point = lane.getPointAtLength(length * along);
    rider.style.left = point.x + "px";
    rider.style.top = point.y + "px";
    let reached = -1;
    stops.forEach((stop, index) => {
      const lit = along >= stop - 0.001;
      if (lit) reached = index;
      stations[index].classList.toggle("lit", lit);
    });
    if (reached !== activeIndex) {
      if (reached > activeIndex && reached >= 0) {
        stations[reached].classList.remove("pop");
        void stations[reached].offsetWidth;
        stations[reached].classList.add("pop");
      }
      activeIndex = reached;
      const stage = hotPathStages[Math.max(0, reached)];
      nowName.textContent = reached < 0 ? "Ready" : stage.name;
      nowNote.textContent = reached < 0 ? "Scroll to send a question down the path." : stage.note;
      counter.textContent = `${Math.max(0, reached + 1)} / 5`;
    }
  };

  layout();
  addEventListener("resize", () => { layout(); show(pinnedProgress(path)); });
  addScrubber(path, pinnedProgress, show);
}

function setupGallery() {
  const gallery = $(".gallery");
  if (!gallery) return;
  const row = $(".gallery-row", gallery);
  const measure = () => gallery.style.setProperty("--travel", Math.max(0, row.scrollWidth - innerWidth) + "px");
  measure();
  addEventListener("resize", measure);
  addEventListener("load", measure);
  addScrubber(gallery, pinnedProgress);
}

const answerStyles = {
  Short: "waiting is steady and now 0; it peaked at 58 at 36.9 s. It spiked 1 time.",
  Summary: "waiting is steady and now 0; it peaked at 58 at 36.9 s. It spiked 1 time.\nThe file has 600 rows covering 0.0 s to 59.9 s.\nTypical waiting is 7.90. Over the most recent readings it is steady.\n1 unusual stretch stood out.",
  Transcript: "0.0 s   START  waiting 0\n34.6 s  SPIKE  waiting jumps to 41\n36.9 s  PEAK   highest waiting, 58\n40.3 s  CALM   waiting eases to 39\n59.9 s  NOW    waiting 0, steady",
  Table: "Time     Event   waiting\n0.0 s    start   waiting 0\n34.6 s   spike   jumps to 41\n36.9 s   peak    highest, 58\n59.9 s   now     0, steady"
};

function setupAnswerDemo() {
  const answer = $(".answer");
  if (!answer) return;
  const tabs = $$(".tabs button", answer);
  const body = $(".body", answer);
  const order = Object.keys(answerStyles);
  let ready = 0;
  let typingTimer = 0;
  let cycleTimer = 0;
  let userPicked = false;

  const type = (text, onDone) => {
    clearInterval(typingTimer);
    const monospaced = text.includes("  ");
    body.style.fontFamily = monospaced ? "var(--mono)" : "";
    body.style.fontSize = monospaced ? "14px" : "";
    if (reducedMotion) { body.textContent = text; onDone && onDone(); return; }
    let shown = 0;
    typingTimer = setInterval(() => {
      shown = Math.min(text.length, shown + 3);
      body.innerHTML = "";
      body.append(text.slice(0, shown));
      if (shown < text.length) {
        const caret = document.createElement("span");
        caret.className = "caret";
        body.append(caret);
      } else {
        clearInterval(typingTimer);
        onDone && onDone();
      }
    }, 18);
  };

  const select = (name) => {
    tabs.forEach((tab) => tab.classList.toggle("on", tab.dataset.style === name));
    type(answerStyles[name]);
  };

  const skeleton = () => {
    body.innerHTML = '<div class="skel"><i style="width:94%"></i><i style="width:81%"></i><i style="width:58%"></i></div>';
  };

  const play = () => {
    ready = 0;
    tabs.forEach((tab) => tab.classList.add("wait"));
    tabs.forEach((tab) => tab.classList.remove("on"));
    skeleton();
    setTimeout(() => {
      const unlock = () => {
        if (ready >= order.length) return;
        tabs[ready].classList.remove("wait");
        if (!userPicked) select(order[ready]);
        ready++;
        cycleTimer = setTimeout(unlock, 2600);
      };
      unlock();
    }, 1100);
  };

  tabs.forEach((tab) => tab.addEventListener("click", () => {
    if (tab.classList.contains("wait")) return;
    userPicked = true;
    select(tab.dataset.style);
  }));

  const observer = new IntersectionObserver((entries) => {
    if (!entries[0].isIntersecting) return;
    observer.disconnect();
    play();
  }, { threshold: 0.4 });
  observer.observe(answer);
}

function setupSteps() {
  const steps = $$(".step");
  if (!steps.length) return;
  const windows = $$(".steps-media .window");
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const index = steps.indexOf(entry.target);
      steps.forEach((step, stepIndex) => step.classList.toggle("on", stepIndex === index));
      windows.forEach((window, windowIndex) => window.classList.toggle("on", windowIndex === index));
    });
  }, { rootMargin: "-45% 0px -45% 0px" });
  steps.forEach((step) => observer.observe(step));
}

function setupStack() {
  const row = $(".stack-row");
  if (!row) return;
  const line = $("path", row);
  const svg = $("svg", row);
  const layers = $$(".layer", row);
  const draw = () => {
    const width = row.clientWidth;
    svg.setAttribute("viewBox", `0 0 ${width} 20`);
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

function setupCtaBear() {
  $$(".cta").forEach((cta) => {
    addScrubber(cta, passingProgress);
    scatterBubbles($(".bubbles", cta), 14, 31);
  });
}

function formatSize(bytes) {
  return (bytes / 1048576).toFixed(1) + " MB";
}

function burst(originX, originY) {
  if (reducedMotion) return;
  const layer = document.createElement("div");
  layer.className = "confetti";
  document.body.appendChild(layer);
  const colors = ["#fdcedf", "#fb648d", "#fdcedf", "#ffffff"];
  const pieces = Array.from({ length: 34 }, (_, index) => {
    const piece = document.createElement("i");
    const size = 8 + Math.random() * 18;
    piece.style.width = piece.style.height = size + "px";
    piece.style.background = colors[index % colors.length];
    piece.style.left = originX + "px";
    piece.style.top = originY + "px";
    layer.appendChild(piece);
    const angle = Math.random() * Math.PI * 2;
    const speed = 5 + Math.random() * 9;
    return { piece, x: 0, y: 0, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed - 7, life: 0 };
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
      item.piece.style.transform = `translate(${item.x}px, ${item.y}px) scale(${0.4 + fade * 0.6})`;
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
  const sizeLabels = $$("[data-size]");
  fetch(buttons[0].getAttribute("href"), { method: "HEAD" }).then((response) => {
    const bytes = +response.headers.get("content-length");
    if (response.ok && bytes > 100000) sizeLabels.forEach((label) => (label.textContent = formatSize(bytes)));
  }).catch(() => {});
  const toast = $(".toast");
  buttons.forEach((button) => button.addEventListener("click", () => {
    const box = button.getBoundingClientRect();
    burst(box.left + box.width / 2, box.top + box.height / 2);
    if (!toast) return;
    toast.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => toast.classList.remove("show"), 4200);
  }));
}

function setupMagnets() {
  if (reducedMotion || matchMedia("(hover: none)").matches) return;
  $$(".btn.huge, [data-magnet]").forEach((element) => {
    element.addEventListener("pointermove", (event) => {
      const box = element.getBoundingClientRect();
      const offsetX = (event.clientX - box.left - box.width / 2) * 0.18;
      const offsetY = (event.clientY - box.top - box.height / 2) * 0.25;
      element.style.translate = `${offsetX.toFixed(1)}px ${offsetY.toFixed(1)}px`;
    });
    element.addEventListener("pointerleave", () => (element.style.translate = ""));
  });
}

setupNav();
setupReveal();
setupHero();
setupMarquee();
setupHotPath();
setupGallery();
setupAnswerDemo();
setupSteps();
setupStack();
setupCtaBear();
setupDownload();
setupMagnets();
startScrubLoop();
