/* 우주쓰레기 분류 서비스 — 브라우저 안에서 도는 3단계 파이프라인
   ① 위치 정하기: ✍️ 내가 박스를 그리거나 / 🤖 AI(위치 찾기 모델)가 찾기
   ② 종류 판별(ResNet18) → ③ 판정 규칙(내가 설계)
   서버가 없어요. 사진은 이 브라우저 밖으로 나가지 않아요. */
(() => {
  "use strict";

  const CLASSES = ["cheops", "debris", "double_start", "earth_observation_sat_1", "lisa_pathfinder",
    "proba_2", "proba_3_csc", "proba_3_ocs", "smart_1", "soho", "xmm_newton"];
  const LOC_SIZE = 224;   // 위치 찾기 모델 입력 크기 (학습 때와 같아야 해요)
  const CLS_SIZE = 160;   // 종류 판별 모델 입력 크기
  const PAD = 0.2;        // 박스를 오릴 때 주는 여유 (Day 2와 같은 20%)
  const STORAGE_KEY = "debris-framework-settings-v1";
  const LEVEL_NAMES = { info: "정보", warn: "주의", alert: "경고" };
  const SOURCE_NAMES = { human: "✍️ 내가 그린 박스", ai: "🤖 AI가 찾은 박스", whole: "🖼️ 사진 전체" };

  const $ = (id) => document.getElementById(id);
  const clone = (o) => JSON.parse(JSON.stringify(o));

  // ------------------------------------------------------------------ 설정 (기본값 + 내가 바꾼 값)
  function mergeDeep(base, extra) {
    for (const k of Object.keys(extra)) {
      if (extra[k] && typeof extra[k] === "object" && !Array.isArray(extra[k]) && base[k] && typeof base[k] === "object") {
        base[k] = mergeDeep(base[k], extra[k]);
      } else {
        base[k] = extra[k];
      }
    }
    return base;
  }
  function loadSettings() {
    let s = clone(window.APP_CONFIG);
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (saved) s = mergeDeep(s, saved);
    } catch (e) { /* 저장 공간을 못 쓰는 환경이어도 괜찮아요 */ }
    return s;
  }
  function saveSettings() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(settings)); } catch (e) { /* 무시 */ }
  }
  let settings = loadSettings();

  // 지금 화면에 올라온 사진과 결과
  const state = {
    img: null, sq: null,           // 원본 사진, 정사각형으로 맞춘 캔버스 정보
    humanBox: null, aiBox: null,   // 박스는 정사각형 기준 0~1 비율 [cx, cy, w, h]
    source: null,                  // 지금 판별에 쓴 박스: "human" | "ai" | "whole"
    probs: null, ms: [null, null],
    drag: null, busy: false,
  };

  // ------------------------------------------------------------------ 상태 표시
  function setStatus(text, kind) {
    $("statusText").textContent = text;
    $("status").className = "status" + (kind ? " " + kind : "");
  }
  function setChips(active) {
    ["c0", "c1", "c2", "c3"].forEach((id, i) => {
      $(id).className = "chip" + (i < active ? " done" : i === active ? " run" : "");
    });
  }
  function setHint(text) { $("hint").textContent = text; }
  function mode() { return document.querySelector("input[name=mode]:checked").value; }

  // ------------------------------------------------------------------ 모델 불러오기 (위치 찾기는 없어도 돼요)
  let locSession = null, clsSession = null;
  async function loadModels() {
    if (location.protocol === "file:") {
      setStatus("이 페이지는 웹 서버로 열어야 해요 (파일을 더블클릭하면 AI 모델을 불러올 수 없어요)", "error");
      return false;
    }
    setStatus("AI 모델 불러오는 중… (처음 한 번은 조금 걸려요)");
    // 엔진 파일(.wasm/.mjs) 위치는 '페이지 기준' 절대 주소로 줘야 해요 (상대 주소는 엔진 스크립트 기준으로 해석돼서 /ort/ort/ 를 찾음)
    ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
    ort.env.wasm.numThreads = 1;
    const t0 = performance.now();
    const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const [cls, loc] = await Promise.allSettled([
      ort.InferenceSession.create("models/classifier.onnx", opts),
      ort.InferenceSession.create("models/localizer.onnx", opts),
    ]);
    if (cls.status !== "fulfilled") {
      console.error(cls.reason);
      const missing = await fetch("models/classifier.onnx", { method: "HEAD" }).then((r) => !r.ok).catch(() => true);
      setStatus(missing
        ? "AI 모델 파일이 아직 없어요 (models/classifier.onnx). 사진 판별은 모델을 넣은 뒤에 돼요 — 🧑‍💻 코드 실험실은 지금도 쓸 수 있어요!"
        : "종류 판별 모델을 불러오지 못했어요: " + (cls.reason && cls.reason.message), "error");
      return false;
    }
    clsSession = cls.value;
    locSession = loc.status === "fulfilled" ? loc.value : null;
    const aiRadio = $("modeAi");
    aiRadio.disabled = !locSession;
    $("modeAiLabel").className = "radio" + (locSession ? "" : " disabled");
    $("modeAiLabel").title = locSession ? "" : "위치 찾기 모델이 아직 없어요 (다음 버전에서 추가돼요)";
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`준비 완료 (${secs}초) · 종류 판별 ✅ · AI 위치 찾기 ${locSession ? "✅" : "⏳ 준비 중"}`, "ready");
    return true;
  }

  // ------------------------------------------------------------------ 사진 → 숫자(텐서)
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error("사진을 열 수 없어요"));
      im.src = src;
    });
  }
  // 정사각형이 아닌 사진은 어두운 배경으로 정사각형을 만들어 찌그러지지 않게 해요
  function toSquare(img) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const side = Math.max(w, h);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = side;
    const g = canvas.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "#202020";
    g.fillRect(0, 0, side, side);
    const ox = Math.floor((side - w) / 2), oy = Math.floor((side - h) / 2);
    g.drawImage(img, ox, oy);
    const data = g.getImageData(0, 0, side, side).data;   // 원본 크기 그대로(1:1)의 픽셀 값
    return { canvas, data, side, ox, oy, w, h };
  }

  // 사진 줄이기: 모델은 파이썬 PIL의 BILINEAR 축소로 학습했기 때문에, 브라우저도 '똑같은 계산'으로 줄여야
  // 같은 답을 내요. (브라우저 기본 축소는 계산 방식이 달라서 박스 위치가 최대 8%까지 어긋났었음)
  // Pillow의 Resample.c(precompute_coeffs + 8비트 고정소수점)를 그대로 옮긴 것.
  const PREC = 22;
  function pilCoeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const filterscale = Math.max(scale, 1.0);
    const support = 1.0 * filterscale;              // BILINEAR 필터의 반경
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2), kk = new Int32Array(outSize * ksize);
    const ss = 1.0 / filterscale;
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      const w = new Float64Array(xmax);
      let ww = 0;
      for (let x = 0; x < xmax; x++) {
        const t = Math.abs((x + xmin - center + 0.5) * ss);
        w[x] = t < 1 ? 1 - t : 0;
        ww += w[x];
      }
      for (let x = 0; x < xmax; x++) {
        const v = ww !== 0 ? w[x] / ww : w[x];
        kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PREC)) : Math.trunc(0.5 + v * (1 << PREC));
      }
      bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
    }
    return { bounds, kk, ksize };
  }
  const clip8 = (v) => { v = v >> PREC; return v < 0 ? 0 : v > 255 ? 255 : v; };
  // src(정사각형 캔버스의 RGBA 픽셀) 중 (x0, y0, w, h) 영역을 size×size로 줄여 0~1 텐서(1×3×size×size)로 만든다
  function toTensor(sq, x0, y0, w, h, size) {
    const src = sq.data, srcW = sq.side;
    const H = pilCoeffs(w, size), V = pilCoeffs(h, size);
    const tmp = new Uint8Array(h * size * 3);              // 1) 가로 방향 먼저
    for (let yy = 0; yy < h; yy++) {
      const rowBase = ((y0 + yy) * srcW + x0) * 4;
      for (let xx = 0; xx < size; xx++) {
        const xmin = H.bounds[2 * xx], xmax = H.bounds[2 * xx + 1], kb = xx * H.ksize;
        let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
        for (let x = 0; x < xmax; x++) {
          const k = H.kk[kb + x], p = rowBase + (xmin + x) * 4;
          s0 += src[p] * k; s1 += src[p + 1] * k; s2 += src[p + 2] * k;
        }
        const o = (yy * size + xx) * 3;
        tmp[o] = clip8(s0); tmp[o + 1] = clip8(s1); tmp[o + 2] = clip8(s2);
      }
    }
    const plane = size * size, out = new Float32Array(3 * plane);   // 2) 세로 방향
    for (let yy = 0; yy < size; yy++) {
      const ymin = V.bounds[2 * yy], ymax = V.bounds[2 * yy + 1], kb = yy * V.ksize;
      for (let xx = 0; xx < size; xx++) {
        let s0 = 1 << (PREC - 1), s1 = s0, s2 = s0;
        for (let y = 0; y < ymax; y++) {
          const k = V.kk[kb + y], p = ((ymin + y) * size + xx) * 3;
          s0 += tmp[p] * k; s1 += tmp[p + 1] * k; s2 += tmp[p + 2] * k;
        }
        const i = yy * size + xx;
        out[i] = clip8(s0) / 255; out[plane + i] = clip8(s1) / 255; out[2 * plane + i] = clip8(s2) / 255;
      }
    }
    return new ort.Tensor("float32", out, [1, 3, size, size]);
  }
  // 박스(0~1 비율)를 오릴 영역으로 바꿔요. pad만큼 넓혀요 (학습 때 쓴 crop_with_pad 와 같은 방식)
  function cropRect(box, side, pad) {
    const [cx, cy, w, h] = box;
    const bw = w * side, bh = h * side;
    let x0 = cx * side - bw / 2, y0 = cy * side - bh / 2, x1 = x0 + bw, y1 = y0 + bh;
    x0 = Math.floor(Math.max(0, x0 - bw * pad)); y0 = Math.floor(Math.max(0, y0 - bh * pad));
    x1 = Math.floor(Math.min(side, x1 + bw * pad)); y1 = Math.floor(Math.min(side, y1 + bh * pad));
    if (x1 - x0 < 2 || y1 - y0 < 2) return [0, 0, side, side];
    return [x0, y0, x1 - x0, y1 - y0];
  }
  function softmax(logits) {
    const m = Math.max(...logits);
    const e = logits.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  function iou(a, b) {
    const ax0 = a[0] - a[2] / 2, ay0 = a[1] - a[3] / 2, ax1 = a[0] + a[2] / 2, ay1 = a[1] + a[3] / 2;
    const bx0 = b[0] - b[2] / 2, by0 = b[1] - b[3] / 2, bx1 = b[0] + b[2] / 2, by1 = b[1] + b[3] / 2;
    const iw = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)), ih = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
    const inter = iw * ih;
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9);
  }

  // ------------------------------------------------------------------ ① 위치 / ② 종류
  async function findBox(sq) {
    const t0 = performance.now();
    const out = await locSession.run({ input: toTensor(sq, 0, 0, sq.side, sq.side, LOC_SIZE) });
    return { box: Array.from(out.box.data), ms: performance.now() - t0 };
  }
  async function classify(sq, box, pad) {
    const t0 = performance.now();
    const [x, y, w, h] = cropRect(box, sq.side, pad);
    const out = await clsSession.run({ input: toTensor(sq, x, y, w, h, CLS_SIZE) });
    return { probs: softmax(Array.from(out.logits.data)), ms: performance.now() - t0 };
  }
  // 사진 전체 박스 (정사각형 기준으로 원본 사진 영역만)
  function wholeBox(sq) {
    return [(sq.ox + sq.w / 2) / sq.side, (sq.oy + sq.h / 2) / sq.side, sq.w / sq.side, sq.h / sq.side];
  }

  async function run(source) {
    if (!clsSession || !state.img || state.busy) return;
    state.busy = true;
    try {
      setChips(1);
      await new Promise((r) => setTimeout(r, 20)); // 화면이 먼저 갱신되도록 잠깐 양보
      let box, msLoc = null, pad = PAD;
      if (source === "ai") {
        const r = await findBox(state.sq);
        state.aiBox = r.box; box = r.box; msLoc = r.ms;
      } else if (source === "human") {
        box = state.humanBox;
      } else {
        box = wholeBox(state.sq); pad = 0;
      }
      setChips(2);
      const c = await classify(state.sq, box, pad);
      state.source = source; state.probs = c.probs; state.ms = [msLoc, c.ms];
      setChips(4);
      render();
      setHint(source === "ai"
        ? "AI가 박스를 찾았어요. 내가 직접 드래그해서 다시 그리면, 내 박스와 AI 박스를 비교할 수 있어요."
        : "다른 위치에 박스를 다시 그리면 바로 다시 판별해요. 박스를 크게/작게 그려보며 결과가 어떻게 바뀌는지 실험해보세요!");
    } catch (e) {
      console.error(e);
      setStatus("분석 중 문제가 생겼어요: " + e.message, "error");
    } finally {
      state.busy = false;
    }
  }

  async function setImage(img) {
    state.img = img; state.sq = toSquare(img);
    state.humanBox = null; state.aiBox = null; state.source = null; state.probs = null; state.ms = [null, null];
    $("btnWhole").disabled = !clsSession;
    setChips(0);
    resetPanels();
    render();
    if (mode() === "ai" && locSession) {
      await run("ai");
    } else {
      setHint("✍️ 물체 둘레를 마우스로 드래그해서 박스를 그려주세요. (물체만 꽉 찬 사진이면 '사진 전체로 판별')");
    }
  }

  // ------------------------------------------------------------------ ③ 판정 규칙 (내 설정으로 바뀌어요)
  function label(cls) { return (settings.classLabels && settings.classLabels[cls]) || cls; }
  // 지금 판별에 쓴 박스가 '원본 사진'에서 차지하는 비율(%)
  function boxPercentNow() {
    const sq = state.sq;
    const b = state.source === "ai" ? state.aiBox : state.source === "human" ? state.humanBox : null;
    if (!b) return 100;
    return Math.round((b[2] * sq.side / sq.w) * (b[3] * sq.side / sq.h) * 1000) / 10;
  }
  function judge() {
    const probs = state.probs;
    const top = probs.indexOf(Math.max(...probs));
    const conf = probs[top], cls = CLASSES[top];
    // 🧑‍💻 '내 코드로 판정하기'가 켜져 있으면 멘티가 짠 judge 함수가 판정해요
    if (lab.enabled && lab.fn) {
      const confPct = Math.round(conf * 100), boxPct = boxPercentNow();
      const r = lab.call(cls, confPct, boxPct);
      const inputs = `kind = "${cls}", confidence = ${confPct}, box_percent = ${boxPct}`;
      if (r.ok) {
        const icons = { info: "✅", warn: "⚠️", alert: "🚨", review: "👀" };
        const title = r.level === "review" ? "👀 사람 확인 필요" : `${icons[r.level]} ${label(cls)}`;
        return { level: r.level, title, msg: r.message || settings.messages[r.level] || "", cls, conf, byCode: true,
          reason: `🧑‍💻 내 코드(judge 함수)가 "${r.raw}"(이)라고 대답했어요. ← 들어간 값: ${inputs}` };
      }
      labConsole("err", "서비스 판정 중 내 코드에서 문제가 생겼어요", `${r.error}\n(들어간 값: ${inputs})\n그래서 이번 판정은 '코드 없이 정하는 규칙'으로 대신 했어요.`);
    }
    const box = state.source === "ai" ? state.aiBox : null;
    if (box && settings.checkBox) {
      const area = box[2] * box[3];
      if (area < settings.minBoxArea || area > settings.maxBoxArea) {
        return { level: "review", title: "👀 사람 확인 필요", msg: settings.messages.review,
          reason: `AI가 찾은 박스가 사진의 ${(area * 100).toFixed(1)}%예요. 내가 정한 범위(${(settings.minBoxArea * 100).toFixed(1)}~${(settings.maxBoxArea * 100).toFixed(0)}%)를 벗어나서 위치가 불확실해요.`, cls, conf };
      }
    }
    if (conf < settings.confThreshold) {
      return { level: "review", title: "👀 사람 확인 필요", msg: settings.messages.review,
        reason: `AI의 1순위 답은 '${label(cls)}'이지만 확신도 ${(conf * 100).toFixed(0)}%로, 내가 정한 기준(${(settings.confThreshold * 100).toFixed(0)}%)보다 낮아요.`, cls, conf };
    }
    const level = (settings.levels && (settings.levels[cls] || settings.levels.default)) || "info";
    const icons = { info: "✅", warn: "⚠️", alert: "🚨" };
    return { level, title: `${icons[level] || ""} ${label(cls)}`, msg: settings.messages[level] || "",
      reason: `확신도 ${(conf * 100).toFixed(0)}%. '${label(cls)}'의 경고 수준은 ${LEVEL_NAMES[level] || level}(으)로 정해져 있어요.`, cls, conf };
  }

  // ------------------------------------------------------------------ 화면 그리기
  const LEVEL_COLORS = { info: "#3ecf8e", warn: "#f5b73b", alert: "#ff5d6c", review: "#9aa6c8" };
  const view = { scale: 1 };
  function boxToCanvas(box) {
    const [cx, cy, w, h] = box, sq = state.sq, s = view.scale;
    return [((cx - w / 2) * sq.side - sq.ox) * s, ((cy - h / 2) * sq.side - sq.oy) * s, w * sq.side * s, h * sq.side * s];
  }
  function drawTag(g, x, y, text, color) {
    const canvas = g.canvas;
    g.font = `bold ${Math.max(13, Math.round(canvas.width / 38))}px sans-serif`;
    const tw = g.measureText(text).width + 10, th = Math.max(20, Math.round(canvas.width / 30));
    const ty = y - th < 0 ? y + th : y;
    g.fillStyle = color;
    g.fillRect(x, ty - th, tw, th);
    g.fillStyle = "#0b1020";
    g.fillText(text, x + 5, ty - th * 0.28);
  }
  function render() {
    const canvas = $("view");
    if (!state.img) return;
    const sq = state.sq;
    view.scale = Math.min(1, 640 / Math.max(sq.w, sq.h));
    canvas.width = Math.round(sq.w * view.scale);
    canvas.height = Math.round(sq.h * view.scale);
    const g = canvas.getContext("2d");
    g.drawImage(state.img, 0, 0, canvas.width, canvas.height);
    canvas.hidden = false;
    $("placeholder").hidden = true;
    const lw = Math.max(2, canvas.width / 220);

    const verdict = state.probs ? judge() : null;
    const color = verdict ? LEVEL_COLORS[verdict.level] : "#5aa9ff";
    // AI 박스 (점선) — 내 박스와 비교할 때도 보이게
    if (state.aiBox) {
      const [x, y, w, h] = boxToCanvas(state.aiBox);
      g.setLineDash([8, 6]); g.lineWidth = lw; g.strokeStyle = state.source === "ai" ? color : "#9aa6c8";
      g.strokeRect(x, y, w, h); g.setLineDash([]);
    }
    // 내 박스 (실선)
    if (state.humanBox) {
      const [x, y, w, h] = boxToCanvas(state.humanBox);
      g.lineWidth = lw; g.strokeStyle = state.source === "human" ? color : "#5aa9ff";
      g.strokeRect(x, y, w, h);
    }
    // 드래그 중인 박스
    if (state.drag && state.drag.moved) {
      const d = state.drag;
      g.setLineDash([6, 4]); g.lineWidth = lw; g.strokeStyle = "#5aa9ff";
      g.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
      g.setLineDash([]);
    }
    if (!verdict) return;
    const active = state.source === "ai" ? state.aiBox : state.source === "human" ? state.humanBox : null;
    if (active) {
      const [x, y] = boxToCanvas(active);
      drawTag(g, x, y, `${label(verdict.cls)} ${(verdict.conf * 100).toFixed(0)}%`, color);
    } else {
      drawTag(g, 0, 0, `${label(verdict.cls)} ${(verdict.conf * 100).toFixed(0)}%`, color);
    }

    // 단계 카드
    $("ms1").textContent = state.ms[0] != null ? `${state.ms[0].toFixed(0)} ms` : "";
    $("ms2").textContent = `${state.ms[1].toFixed(0)} ms`;
    const used = active || wholeBox(sq);
    let desc = `${SOURCE_NAMES[state.source]}: 사진의 ${(used[2] * sq.side / sq.w * 100).toFixed(0)}% × ${(used[3] * sq.side / sq.h * 100).toFixed(0)}% 크기.`;
    if (state.humanBox && state.aiBox) {
      desc += ` 내 박스와 AI 박스가 겹치는 정도(IoU): ${iou(state.humanBox, state.aiBox).toFixed(2)} (1이면 완전히 같음)`;
    }
    $("b1").textContent = desc;
    const order = state.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
    const box2 = $("b2");
    box2.textContent = "";
    order.forEach(([p, i]) => {
      const row = document.createElement("div"); row.className = "bar";
      const name = document.createElement("span"); name.textContent = label(CLASSES[i]);
      const track = document.createElement("div"); track.className = "track";
      const fill = document.createElement("div"); fill.className = "fill"; fill.style.width = `${(p * 100).toFixed(1)}%`;
      track.appendChild(fill);
      const val = document.createElement("span"); val.className = "val"; val.textContent = `${(p * 100).toFixed(0)}%`;
      row.append(name, track, val);
      box2.appendChild(row);
    });
    $("verdict").className = "verdict " + verdict.level;
    $("vt").textContent = verdict.title;
    $("vm").textContent = verdict.msg;
    $("vr").textContent = (verdict.byCode ? `🧑‍💻 판정: ${settings.author}의 코드 · ` : "") + "왜 이렇게 판정했나요? " + verdict.reason;
  }
  function resetPanels() {
    $("ms1").textContent = ""; $("ms2").textContent = "";
    $("b1").textContent = "내가 그리거나(✍️) AI가 찾은(🤖) 박스로 물체의 위치를 정해요.";
    $("b2").textContent = "박스 안을 오려서 11개 클래스 중 무엇인지 맞혀요.";
    $("verdict").className = "verdict";
    $("vt").textContent = "③ 판정 결과";
    $("vm").textContent = "박스가 정해지면 내가 정한 규칙으로 판정해요.";
    $("vr").textContent = "";
  }

  // ------------------------------------------------------------------ ✍️ 박스 그리기 (마우스·터치)
  const canvasEl = $("view");
  function canvasPoint(e) {
    const r = canvasEl.getBoundingClientRect();
    return [(e.clientX - r.left) * canvasEl.width / r.width, (e.clientY - r.top) * canvasEl.height / r.height];
  }
  canvasEl.addEventListener("pointerdown", (e) => {
    if (!state.img || state.busy) return;
    const [x, y] = canvasPoint(e);
    state.drag = { x0: x, y0: y, x1: x, y1: y, moved: false };
    canvasEl.setPointerCapture(e.pointerId);
  });
  canvasEl.addEventListener("pointermove", (e) => {
    if (!state.drag) return;
    const [x, y] = canvasPoint(e);
    state.drag.x1 = Math.max(0, Math.min(canvasEl.width, x));
    state.drag.y1 = Math.max(0, Math.min(canvasEl.height, y));
    state.drag.moved = Math.abs(state.drag.x1 - state.drag.x0) > 4 || Math.abs(state.drag.y1 - state.drag.y0) > 4;
    render();
  });
  canvasEl.addEventListener("pointerup", () => {
    const d = state.drag;
    state.drag = null;
    if (!d || Math.abs(d.x1 - d.x0) < 8 || Math.abs(d.y1 - d.y0) < 8) { render(); return; }
    // 캔버스 좌표 → 정사각형 기준 0~1 비율
    const sq = state.sq, s = view.scale;
    const x0 = Math.min(d.x0, d.x1) / s + sq.ox, x1 = Math.max(d.x0, d.x1) / s + sq.ox;
    const y0 = Math.min(d.y0, d.y1) / s + sq.oy, y1 = Math.max(d.y0, d.y1) / s + sq.oy;
    state.humanBox = [(x0 + x1) / 2 / sq.side, (y0 + y1) / 2 / sq.side, (x1 - x0) / sq.side, (y1 - y0) / sq.side];
    run("human");
  });

  // ------------------------------------------------------------------ 🧑‍💻 코드 실험실 (🐍 파이썬)
  // 멘티의 파이썬 코드는 브라우저 안의 진짜 파이썬(Pyodide)에서 실행돼요. 서버로 나가지 않아요.
  const LAB_KEY = "debris-codelab-py-v1";
  const STARTER = `# 🧑‍💻 판정 규칙 함수 — 이 함수가 서비스의 최종 판정을 정해요.
#
# AI가 사진을 보고 이 함수에 넣어주는 값 3가지:
#   kind        : 물체의 종류. "debris"(우주쓰레기), "soho", "cheops" ... 중 하나
#   confidence  : AI의 확신도. 0 ~ 100 사이 숫자 (예: 87)
#   box_percent : 박스가 사진에서 차지하는 크기(%) (예: 12.5)
#
# 이 함수는 "정보", "주의", "경고", "사람 확인" 중 하나를 return(돌려주기) 해야 해요.

def judge(kind, confidence, box_percent):
    return "정보"
`;
  const LEVEL_OF = { "정보": "info", "주의": "warn", "경고": "alert", "사람 확인": "review",
    info: "info", warn: "warn", alert: "alert", review: "review" };
  // 미션: 정해진 상황(cases)마다 기대하는 답이 있어요. 도전 미션은 정답이 없어요.
  const MISSIONS = [
    { title: "미션 1", name: "함수가 대답하게", learn: "함수 · return",
      goal: "어떤 사진이 와도 <b>\"정보\"</b>라고 대답하게 하기. (처음 코드를 그대로 ▶ 실행해보세요!)",
      hint: "이미 정답이에요! <code>return</code>은 함수가 '대답을 돌려주는' 명령이에요. 따옴표 안의 글자를 바꾸면 대답이 바뀌어요.",
      cases: [["soho", 95, 20, "정보"], ["debris", 30, 2, "정보"], ["cheops", 60, 35, "정보"]] },
    { title: "미션 2", name: "자신 없으면 사람에게", learn: "if · 비교 <",
      goal: "확신도(confidence)가 <b>80보다 낮으면 \"사람 확인\"</b>, 아니면 \"정보\"라고 대답하게 하기.",
      hint: "<code>if</code>는 '만약 ~라면'이에요. 줄 끝에 <b>콜론(:)</b>, 다음 줄은 <b>스페이스 4칸 들여쓰기</b>(Tab 키)!<pre>if confidence &lt; 80:\n    return \"사람 확인\"\nreturn \"정보\"</pre>79는? 80은? 경계값을 테스트해보세요.",
      cases: [["soho", 95, 20, "정보"], ["soho", 40, 20, "사람 확인"], ["cheops", 79, 30, "사람 확인"], ["cheops", 80, 30, "정보"]] },
    { title: "미션 3", name: "우주쓰레기 경보", learn: "문자열 비교 == · 순서",
      goal: "미션 2 규칙에 더해서: 확신도가 80 이상이고 <b>우주쓰레기(\"debris\")면 \"경고\"</b>.",
      hint: "두 값이 같은지 볼 땐 <code>==</code>를 써요. (<code>=</code> 하나는 '넣기'라서 비교가 아니에요!)<pre>if kind == \"debris\":\n    return \"경고\"</pre>어떤 <code>if</code>를 먼저 쓰느냐에 따라 답이 달라져요. 확신도가 낮은 우주쓰레기는 어떻게 돼야 할까요?",
      cases: [["debris", 95, 2, "경고"], ["debris", 40, 2, "사람 확인"], ["soho", 95, 20, "정보"], ["soho", 40, 20, "사람 확인"]] },
    { title: "미션 4", name: "확신도 구간 나누기", learn: "elif",
      goal: "확신도 <b>50 미만은 \"사람 확인\"</b>, <b>50~79는 \"주의\"</b>, 80 이상은 미션 3처럼 (debris면 \"경고\", 아니면 \"정보\").",
      hint: "<code>elif</code>는 '그게 아니라 만약 ~라면'이에요. 위에서부터 차례로 검사해요.<pre>if confidence &lt; 50:\n    return \"사람 확인\"\nelif confidence &lt; 80:\n    return \"주의\"</pre>",
      cases: [["soho", 30, 20, "사람 확인"], ["soho", 50, 20, "주의"], ["soho", 65, 20, "주의"], ["debris", 65, 2, "주의"],
        ["debris", 90, 2, "경고"], ["xmm_newton", 90, 25, "정보"], ["soho", 80, 20, "정보"]] },
    { title: "🏆 도전", name: "나만의 규칙", learn: "and · or · 내 설계",
      goal: "<b>box_percent(박스 크기)</b>까지 써서 나만의 규칙 만들기. 예: 박스가 너무 작으면 위치가 불확실하니 \"사람 확인\". 정답은 없어요 — 내가 정한 이유를 말할 수 있으면 성공!",
      hint: "<code>and</code>는 '그리고', <code>or</code>는 '또는'이에요.<pre>if kind == \"debris\" and box_percent &lt; 1:\n    return \"사람 확인\"</pre>대답과 함께 내 문구도 정할 수 있어요 (쉼표로 두 개를 돌려주기):<pre>return \"경고\", f\"우주쓰레기 발견! (확신도 {confidence}%)\"</pre><code>print(kind, confidence)</code>로 값을 찍어보면 콘솔에 보여요.",
      cases: null },
  ];
  const FREE_CASES = [["debris", 95, 0.6], ["debris", 92, 3], ["debris", 55, 2], ["soho", 97, 18], ["soho", 45, 18], ["cheops", 88, 60], ["proba_2", 70, 30]];
  // 멘티 함수의 대답을 정리해주는 도우미 (문자열 / (대답, 문구) / {"level":..., "message":...} 모두 받아줘요)
  const HELPER = `
def __call_judge(k, c, b):
    r = judge(k, c, b)
    if isinstance(r, (tuple, list)) and len(r) == 2:
        return ["pair", str(r[0]), str(r[1])]
    if isinstance(r, dict):
        m = r.get("message")
        return ["pair", str(r.get("level")), None if m is None else str(m)]
    if isinstance(r, str):
        return ["str", r, None]
    return ["bad", repr(r), None]
`;

  const lab = { fn: null, enabled: false, current: 0, passed: {}, code: STARTER, call: null, ns: null };
  try {
    const saved = JSON.parse(localStorage.getItem(LAB_KEY) || "null");
    if (saved) Object.assign(lab, { code: saved.code || STARTER, current: saved.current || 0, passed: saved.passed || {}, enabled: !!saved.enabled });
  } catch (e) { /* 무시 */ }
  function saveLab() {
    try { localStorage.setItem(LAB_KEY, JSON.stringify({ code: lab.code, current: lab.current, passed: lab.passed, enabled: lab.enabled })); } catch (e) { /* 무시 */ }
  }

  // ---- 파이썬 엔진 준비 (처음 한 번만, 몇 초)
  let py = null, pyReady = null, prints = [];
  function loadPython() {
    if (!pyReady) {
      pyReady = (async () => {
        if (location.protocol === "file:") throw new Error("파이썬은 웹 서버로 연 페이지에서만 돌아가요 (파일 더블클릭으로는 안 돼요).");
        const t0 = performance.now();
        py = await loadPyodide({ indexURL: new URL("py/", location.href).href,
          stdout: (s) => prints.push(s), stderr: (s) => prints.push(s) });
        return ((performance.now() - t0) / 1000).toFixed(1);
      })();
    }
    return pyReady;
  }

  // ---- 파이썬 에러를 쉬운 말로
  function parsePyError(msg) {
    const text = String(msg);
    const lines = text.trim().split("\n");
    const last = lines[lines.length - 1].trim();
    const all = [...text.matchAll(/File "my_judge\.py", line (\d+)/g)];
    const line = all.length ? Number(all[all.length - 1][1]) : null;
    return { line, raw: last };
  }
  function friendly(raw, code) {
    const m = String(raw);
    // 자바스크립트식 else if 는 파이썬이 "콜론이 없다"고만 해서 헷갈려요 → 먼저 잡아줘요
    if (/SyntaxError/.test(m) && /^\s*else\s+if\b/m.test(code || "")) return "파이썬에서는 else if 대신 elif 를 써요. 예: elif confidence < 80:";
    const rules = [
      [/expected ':'/, "줄 끝에 콜론(:)이 빠졌어요. def, if, elif, else 줄은 : 로 끝나야 해요."],
      [/expected an indented block/, "if 나 def 다음 줄은 스페이스 4칸 들여써야 해요. (Tab 키를 누르면 4칸이 들어가요)"],
      [/unindent does not match|inconsistent use of tabs/, "들여쓰기 칸 수가 위 줄들과 안 맞아요. 같은 묶음의 줄은 같은 칸만큼 들여써야 해요."],
      [/unexpected indent/, "필요 없는 들여쓰기가 있어요. 그 줄 앞의 빈칸을 위 줄과 맞춰보세요."],
      [/unterminated string literal|EOL while scanning/, "따옴표가 닫히지 않았어요. \"정보\" 처럼 앞뒤에 따옴표가 있어야 해요."],
      [/invalid character '[“”‘’]'/, "한글 따옴표(“ ” ‘ ’)가 들어갔어요. 키보드의 \" 로 바꿔주세요."],
      [/Maybe you meant '==' or ':=' instead of '='/, "if 안에서 = 하나는 '넣기'예요. 같은지 비교하려면 == 를 써요."],
      [/was never closed/, "여는 괄호나 따옴표가 닫히지 않았어요. 짝을 확인해보세요."],
      [/unmatched '[)\]}]'/, "필요 없는 닫는 괄호가 있어요."],
      [/NameError: name '(\w+)' is not defined(?:\. Did you mean: '(\w+)'\?)?/, (x) =>
        `'${x[1]}'라는 이름을 찾을 수 없어요.` + (x[2] ? ` 혹시 '${x[2]}'를 쓰려던 거 아닐까요?` : "") + " (쓸 수 있는 이름: kind, confidence, box_percent)"],
      [/not supported between instances of 'str' and '(int|float)'|not supported between instances of '(int|float)' and 'str'/,
        "글자(문자열)와 숫자를 크기 비교했어요. confidence, box_percent 같은 숫자끼리 비교해야 해요."],
      [/invalid syntax/, () => (/===/.test(code) ? "파이썬에서 같은지 비교는 == 예요. (= 를 세 개 쓰지 않아요)"
        : /else\s+if/.test(code) ? "파이썬에서는 else if 대신 elif 를 써요."
        : "문법이 맞지 않는 부분이 있어요. 콜론(:), 따옴표, 괄호, 철자를 확인해보세요.")],
    ];
    for (const [re, text] of rules) {
      const x = m.match(re);
      if (x) return typeof text === "function" ? text(x) : text;
    }
    return "";
  }
  function lintTips(code) {
    const tips = [];
    if (/kind\s*==\s*["'](우주쓰레기|쓰레기|Debris|DEBRIS)["']/.test(code)) tips.push("💡 kind 에는 영어 소문자 \"debris\" 가 들어와요.");
    if (/\t/.test(code)) tips.push("💡 탭 문자가 섞여 있어요. 들여쓰기는 스페이스 4칸으로 맞추는 게 안전해요.");
    return tips;
  }

  function labConsole(kind, title, body) {
    const c = $("console");
    c.className = "console " + (kind || "");
    c.textContent = "";
    const t = document.createElement("div"); t.className = "t"; t.textContent = title;
    c.append(t, document.createTextNode(body || ""));
  }
  function appendPrints() {
    if (!prints.length) return;
    const shown = prints.slice(0, 20);
    $("console").appendChild(document.createTextNode(`\n\n🖨️ print 로 출력한 것:\n${shown.join("\n")}${prints.length > 20 ? `\n… (${prints.length - 20}줄 더)` : ""}`));
  }
  function markGutter(badLine) {
    const n = lab.code.split("\n").length;
    const g = $("gutter");
    g.textContent = "";
    for (let i = 1; i <= n; i++) {
      const s = document.createElement("div");
      s.textContent = String(i);
      if (i === badLine) s.className = "bad";
      g.appendChild(s);
    }
  }

  // ---- 멘티 코드를 파이썬으로 실행해서 judge 함수를 꺼내요
  async function compileUserCode(code) {
    await loadPython();
    if (lab.ns) { try { lab.ns.destroy(); } catch (e) { /* 무시 */ } lab.ns = null; }
    const ns = py.globals.get("dict")();
    prints = [];
    try {
      py.runPython(code, { globals: ns, filename: "my_judge.py" });
    } catch (e) {
      ns.destroy();
      return { ok: false, ...parsePyError(e.message) };
    }
    if (!py.runPython('callable(globals().get("judge"))', { globals: ns })) {
      ns.destroy();
      return { ok: false, line: null, raw: "judge 함수를 찾을 수 없어요. def judge(kind, confidence, box_percent): 모양이 있어야 해요." };
    }
    py.runPython(HELPER, { globals: ns });
    lab.ns = ns;
    return { ok: true, fn: ns.get("__call_judge") };
  }
  // 멘티 함수를 안전하게 한 번 불러요 (동기 호출 — 파이썬이 준비된 뒤에만 불려요)
  lab.call = (kind, confidence, box) => {
    let arr;
    try {
      const r = lab.fn(kind, confidence, box);
      arr = r.toJs();
      r.destroy();
    } catch (e) {
      const p = parsePyError(e.message);
      const f = friendly(p.raw, lab.code);
      return { ok: false, error: `실행 중 에러${p.line ? ` (${p.line}번째 줄)` : ""}: ${p.raw}${f ? "\n👉 " + f : ""}` };
    }
    const [shape, raw, message] = arr;
    const level = shape === "bad" ? null : LEVEL_OF[raw];
    if (!level) {
      return { ok: false, error: `judge가 돌려준 값 ${shape === "bad" ? raw : JSON.stringify(raw)} 은(는) 쓸 수 없어요. "정보", "주의", "경고", "사람 확인" 중 하나여야 해요.` +
        (raw === "None" ? "\n👉 return 을 빼먹었거나, 어떤 경우에도 return 에 도달하지 못했어요." : "") };
    }
    return { ok: true, level, message, raw };
  };

  function renderMissions() {
    const box = $("missions");
    box.textContent = "";
    MISSIONS.forEach((m, i) => {
      const b = document.createElement("button");
      b.className = "mission" + (i === lab.current ? " current" : "") + (lab.passed[i] ? " passed" : "");
      const t = document.createElement("div"); t.className = "mt"; t.textContent = `${m.title} · ${m.name}`;
      const l = document.createElement("div"); l.className = "ml"; l.textContent = m.learn;
      b.append(t, l);
      b.onclick = () => { lab.current = i; saveLab(); renderMissions(); runTests(false); };
      box.appendChild(b);
    });
    const m = MISSIONS[lab.current];
    $("goal").innerHTML = `🎯 <b>${m.title}</b> — ${m.goal}`;   // 미션 문구는 우리가 쓴 고정 텍스트
    $("hintBody").innerHTML = m.hint;
    $("hintBox").open = false;
  }
  // 테스트: 현재 미션의 상황들을 내 코드에 넣어보고 기대한 답과 비교해요
  function runTests(announce) {
    const holder = $("tests");
    holder.textContent = "";
    if (!lab.fn) return null;
    prints = [];
    const m = MISSIONS[lab.current];
    const cases = m.cases || FREE_CASES.map((c) => [...c, null]);
    const table = document.createElement("table");
    const head = document.createElement("tr");
    ["상황", "kind", "confidence", "box_percent", "내 코드의 답", m.cases ? "기대한 답" : "", m.cases ? "결과" : ""].forEach((h) => {
      const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
    });
    table.appendChild(head);
    let pass = 0, firstError = null;
    cases.forEach(([k, c, b, expect], idx) => {
      const r = lab.call(k, c, b);
      const got = r.ok ? String(r.raw) : "⚠️ 에러";
      if (!r.ok && !firstError) firstError = r.error;
      const ok = expect != null && r.ok && LEVEL_OF[expect] === r.level;
      if (ok) pass++;
      const tr = document.createElement("tr");
      [String(idx + 1), `"${k}"`, String(c), String(b), got, expect == null ? "" : expect, expect == null ? "" : (ok ? "✅" : "❌")].forEach((v, j) => {
        const td = document.createElement("td"); td.textContent = v;
        if (j === 6 && expect != null) td.className = ok ? "pass" : "fail";
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    holder.appendChild(table);
    if (!m.cases) {
      if (announce) {
        labConsole(firstError ? "err" : "ok", firstError ? "에러가 난 상황이 있어요" : "🏆 도전 미션: 내 규칙의 결과예요",
          firstError || "정답은 없어요. 위 표를 보고 '왜 이렇게 정했는지' 설명할 수 있으면 성공!\n아래 '직접 시험해보기'로 다른 상황도 넣어보세요.");
        appendPrints();
      }
      return null;
    }
    const all = pass === cases.length;
    if (announce) {
      if (all) {
        lab.passed[lab.current] = true;
        const next = lab.current + 1 < MISSIONS.length ? MISSIONS[lab.current + 1] : null;
        labConsole("ok", `🎉 ${m.title} 통과! (${pass}/${cases.length})`, next ? `다음은 ${next.title} · ${next.name}! 위의 미션 버튼을 눌러 넘어가세요.` : "모든 미션 완료!");
        saveLab(); renderMissions();
      } else {
        labConsole("err", `아직이에요: ${pass}/${cases.length} 통과`, firstError ? firstError : "❌ 표시된 상황에서 내 코드의 답과 기대한 답을 비교해보세요. 어느 if에서 갈렸을까요?");
      }
      appendPrints();
    }
    return all;
  }
  let running = false;
  async function runCode() {
    if (running) return;
    running = true;
    lab.code = $("code").value;
    saveLab();
    try {
      if (!py) labConsole("", "🐍 파이썬 준비 중…", "처음 한 번은 몇 초 걸려요.");
      const res = await compileUserCode(lab.code);
      if (!res.ok) {
        lab.fn = null;
        markGutter(res.line);
        const f = friendly(res.raw, lab.code);
        labConsole("err", res.line ? `❌ ${res.line}번째 줄에 문제가 있어요` : "❌ 실행할 수 없어요", `${res.raw}${f ? "\n👉 " + f : ""}`);
        appendPrints();
        $("tests").textContent = "";
        return;
      }
      lab.fn = res.fn;
      markGutter(null);
      if (!lab.enabled) { lab.enabled = true; $("useCode").checked = true; saveLab(); }
      runTests(true);
      const tips = lintTips(lab.code);
      if (tips.length) $("console").appendChild(document.createTextNode("\n\n" + tips.join("\n")));
      updateRulesNote();
      render();
    } catch (e) {
      labConsole("err", "파이썬을 준비하지 못했어요", e.message);
    } finally {
      running = false;
    }
  }
  function updateRulesNote() {
    $("rulesNote").textContent = lab.enabled && lab.fn ? "— 지금은 🧑‍💻 내 코드가 판정 중이라 아래 설정은 쓰이지 않아요" : "";
  }
  function initLab() {
    const ta = $("code");
    ta.value = lab.code;
    markGutter(null);
    ta.addEventListener("input", () => { lab.code = ta.value; saveLab(); markGutter(null); });
    ta.addEventListener("scroll", () => { $("gutter").scrollTop = ta.scrollTop; });
    const insert = (text, back = 0) => {
      const s = ta.selectionStart, en = ta.selectionEnd;
      ta.value = ta.value.slice(0, s - back) + text + ta.value.slice(en);
      ta.selectionStart = ta.selectionEnd = s - back + text.length;
      lab.code = ta.value; saveLab(); markGutter(null);
    };
    ta.addEventListener("keydown", (e) => {
      const s = ta.selectionStart;
      const lineStart = ta.value.lastIndexOf("\n", s - 1) + 1;
      const before = ta.value.slice(lineStart, s);
      if (e.key === "Tab") {                       // Tab = 스페이스 4칸
        e.preventDefault(); insert("    ");
      } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault(); runCode();
      } else if (e.key === "Enter") {              // 엔터: 들여쓰기 이어가기, 콜론(:) 뒤에는 4칸 더
        e.preventDefault();
        const indent = before.match(/^ */)[0];
        insert("\n" + indent + (before.trimEnd().endsWith(":") ? "    " : ""));
      } else if (e.key === "Backspace" && s === ta.selectionEnd && /^( {4})+$/.test(before)) {
        e.preventDefault(); insert("", 4);         // 들여쓰기 한 칸(4칸)씩 지우기
      }
    });
    $("btnRun").onclick = runCode;
    $("btnTest").onclick = () => { if (!lab.fn) runCode(); else runTests(true); };
    $("btnCodeReset").onclick = () => {
      if (!confirm("처음 코드로 되돌릴까요? 지금 쓴 코드는 사라져요. (먼저 '코드 내려받기'로 저장해두면 안전해요)")) return;
      ta.value = STARTER; lab.code = STARTER; lab.fn = null; saveLab(); markGutter(null);
      labConsole("", "콘솔", "처음 코드로 되돌렸어요. ▶ 실행을 눌러보세요.");
      $("tests").textContent = ""; updateRulesNote(); render();
    };
    $("btnCodeSave").onclick = () => {
      const blob = new Blob([lab.code], { type: "text/x-python" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = "my_judge.py"; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    };
    $("useCode").checked = lab.enabled;
    $("useCode").onchange = (e) => { lab.enabled = e.target.checked; saveLab(); updateRulesNote(); render(); };
    // kind 목록 / 직접 시험해보기
    $("kindList").textContent = CLASSES.map((c) => `"${c}" = ${label(c)}`).join("\n");
    $("kindList").style.whiteSpace = "pre";
    const sel = $("tKind");
    CLASSES.forEach((c) => { const o = document.createElement("option"); o.value = c; o.textContent = c; sel.appendChild(o); });
    sel.value = "debris";
    $("btnMyTest").onclick = async () => {
      if (!lab.fn) await runCode();
      if (!lab.fn) { $("tOut").textContent = "(먼저 코드의 에러를 고쳐주세요)"; return; }
      prints = [];
      const r = lab.call(sel.value, Number($("tConf").value), Number($("tBox").value));
      $("tOut").textContent = (r.ok ? `"${r.raw}"` + (r.message ? ` (문구: ${r.message})` : "") : "⚠️ " + r.error) +
        (prints.length ? `   🖨️ ${prints.slice(0, 3).join(" / ")}` : "");
    };
    renderMissions();
    // 파이썬을 미리 준비해두고, 저장해둔 코드가 있으면 조용히 불러와요
    labConsole("", "🐍 파이썬 준비 중…", "처음 한 번은 몇 초 걸려요. 그동안 코드를 써도 돼요.");
    loadPython().then(async (secs) => {
      const res = await compileUserCode(lab.code);
      if (res.ok) { lab.fn = res.fn; runTests(false); } else { lab.enabled = false; $("useCode").checked = false; }
      updateRulesNote();
      labConsole("ok", `🐍 파이썬 준비 완료 (${secs}초)`, "코드를 쓰고 ▶ 실행을 눌러보세요. (단축키: Ctrl + Enter)");
      render();
    }).catch((e) => labConsole("err", "파이썬을 준비하지 못했어요", e.message));
    updateRulesNote();
  }

  // ------------------------------------------------------------------ 설정 칸 연결
  function applyTexts() {
    $("title").textContent = settings.title;
    document.title = settings.title;
    $("subtitle").textContent = settings.subtitle;
    $("fAuthor").textContent = settings.author;
    $("cardView").textContent = settings.modelCard;
  }
  function buildSettingsUI() {
    const conf = $("conf");
    conf.value = settings.confThreshold;
    $("confVal").textContent = `${Math.round(settings.confThreshold * 100)}%`;
    conf.oninput = () => { settings.confThreshold = parseFloat(conf.value); $("confVal").textContent = `${Math.round(settings.confThreshold * 100)}%`; changed(); };

    $("checkBox").checked = !!settings.checkBox;
    $("checkBox").onchange = (e) => { settings.checkBox = e.target.checked; changed(); };
    $("minArea").value = (settings.minBoxArea * 100).toFixed(1);
    $("maxArea").value = (settings.maxBoxArea * 100).toFixed(0);
    $("minArea").oninput = (e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) { settings.minBoxArea = v / 100; changed(); } };
    $("maxArea").oninput = (e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) { settings.maxBoxArea = v / 100; changed(); } };

    const levels = $("levels");
    levels.textContent = "";
    CLASSES.forEach((cls) => {
      const wrap = document.createElement("div"); wrap.className = "lv";
      const name = document.createElement("span"); name.textContent = label(cls);
      const sel = document.createElement("select");
      Object.entries(LEVEL_NAMES).forEach(([k, v]) => { const o = document.createElement("option"); o.value = k; o.textContent = v; sel.appendChild(o); });
      sel.value = (settings.levels[cls] || settings.levels.default || "info");
      sel.onchange = () => { settings.levels[cls] = sel.value; changed(); };
      wrap.append(name, sel);
      levels.appendChild(wrap);
    });

    const bindText = (id, get, set) => { const el = $(id); el.value = get(); el.oninput = () => { set(el.value); changed(); }; };
    bindText("tTitle", () => settings.title, (v) => (settings.title = v));
    bindText("tSub", () => settings.subtitle, (v) => (settings.subtitle = v));
    bindText("tAuthor", () => settings.author, (v) => (settings.author = v));
    bindText("mInfo", () => settings.messages.info, (v) => (settings.messages.info = v));
    bindText("mWarn", () => settings.messages.warn, (v) => (settings.messages.warn = v));
    bindText("mAlert", () => settings.messages.alert, (v) => (settings.messages.alert = v));
    bindText("mReview", () => settings.messages.review, (v) => (settings.messages.review = v));
    bindText("tCard", () => settings.modelCard, (v) => (settings.modelCard = v));
  }
  function changed() { saveSettings(); applyTexts(); render(); }

  $("btnSave").onclick = () => {
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "my-framework-settings.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $("fileSettings").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      settings = mergeDeep(clone(window.APP_CONFIG), JSON.parse(await f.text()));
      saveSettings(); buildSettingsUI(); applyTexts(); render();
    } catch (err) { setStatus("설정 파일을 읽을 수 없어요: " + err.message, "error"); }
    e.target.value = "";
  };
  $("btnReset").onclick = () => {
    settings = clone(window.APP_CONFIG);
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) { /* 무시 */ }
    buildSettingsUI(); applyTexts(); render();
  };
  $("btnWhole").onclick = () => run("whole");
  document.querySelectorAll("input[name=mode]").forEach((r) => r.addEventListener("change", () => {
    if (mode() === "ai" && state.img && locSession) run("ai");
    else if (mode() === "human") setHint("✍️ 물체 둘레를 마우스로 드래그해서 박스를 그려주세요.");
  }));

  // ------------------------------------------------------------------ 사진 넣기 (선택 / 끌어놓기 / 예시)
  async function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) { setStatus("이미지 파일을 올려주세요", "error"); return; }
    const url = URL.createObjectURL(file);
    try { await setImage(await loadImage(url)); } finally { URL.revokeObjectURL(url); }
  }
  $("file").onchange = (e) => { handleFile(e.target.files[0]); e.target.value = ""; };
  const stage = $("stage");
  ["dragenter", "dragover"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove("drag"); }));
  stage.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });

  async function loadExamples() {
    try {
      const manifest = await (await fetch("examples/manifest.json")).json();
      const holder = $("examples");
      manifest.forEach((ex) => {
        const im = document.createElement("img");
        im.src = "examples/" + ex.file;
        im.alt = ex.cls;
        im.title = `예시: ${label(ex.cls)}`;
        im.onclick = async () => setImage(await loadImage(im.src));
        holder.appendChild(im);
      });
      return manifest;
    } catch (e) { return []; }
  }

  // ------------------------------------------------------------------ 시작 + (개발용) 자가 점검: ?selftest=1
  // 예시 사진마다 ① 정답 박스로 판별(= 사람이 정확히 그린 경우), ② AI 위치 찾기(모델이 있을 때)를 돌려 JSON으로 남겨요
  async function selfTest(manifest) {
    const out = [];
    for (const ex of manifest) {
      const sq = toSquare(await loadImage("examples/" + ex.file));
      const row = { file: ex.file, cls: ex.cls };
      // 모델 입력값 자체가 파이썬(PIL)과 같은지 비교하기 위한 요약 (0~255 정수로 되돌린 합과 앞부분)
      const summarize = (t) => { const d = t.data; let s = 0; for (let i = 0; i < d.length; i++) s += Math.round(d[i] * 255); return { sum: s, head: Array.from(d.slice(0, 8), (v) => Math.round(v * 255)) }; };
      row.locInput = summarize(toTensor(sq, 0, 0, sq.side, sq.side, LOC_SIZE));
      const [cx0, cy0, cw, ch] = cropRect(ex.gt_box, sq.side, PAD);
      row.clsInput = summarize(toTensor(sq, cx0, cy0, cw, ch, CLS_SIZE));
      row.gt = await classify(sq, ex.gt_box, PAD);
      if (locSession) {
        const f = await findBox(sq);
        row.ai = { box: f.box, ms: f.ms, ...(await classify(sq, f.box, PAD)) };
      }
      out.push(row);
    }
    window.__selftest = out;
    const pre = document.createElement("pre");
    pre.id = "selftest";
    pre.hidden = true;
    pre.textContent = JSON.stringify(out);
    document.body.appendChild(pre);
  }

  (async function init() {
    applyTexts();
    buildSettingsUI();
    initLab();
    setChips(0);
    const manifest = await loadExamples();
    const ok = await loadModels();
    if (ok && new URLSearchParams(location.search).has("selftest")) await selfTest(manifest);
  })();
})();
