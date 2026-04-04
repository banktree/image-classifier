import { useState, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen } from "@tauri-apps/api/event";
import { load, Store } from "@tauri-apps/plugin-store";
import "./App.css";

type Category = "2d" | "3d" | "unknown";

interface ImageResult {
  path: string;
  filename: string;
  category: Category;
  thumbnail: string;
}

// Each item carries its stable index so we never need indexOf
interface IndexedResult {
  item: ImageResult;
  idx: number;
}

interface ProgressEvent {
  current: number;
  total: number;
  filename: string;
  category: Category;
  thumbnail: string;
  elapsed_secs: number;
}

interface DragState {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

type DedupState = "idle" | "scanning" | "reviewing" | "applying" | "done";

interface DupImage {
  path: string;
  filename: string;
  size: number;
  width: number;
  height: number;
  thumbnail: string;
}

interface DupGroup {
  id: number;
  images: DupImage[];
  keep_idx: number;
}

type AppState = "idle" | "scanning" | "paused" | "reviewing" | "applying" | "done";

export default function App() {
  const [folderPath, setFolderPath]     = useState("");
  const [folder2d, setFolder2d]         = useState("");
  const [folder3d, setFolder3d]         = useState("");
  const [folderUnknown, setFolderUnknown] = useState("");

  const storeRef = useRef<Store | null>(null);
  const [ollamaOk, setOllamaOk]       = useState<boolean | null>(null);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("llava");
  const [appState, setAppState]       = useState<AppState>("idle");
  const [progress, setProgress]       = useState({ current: 0, total: 0, filename: "", elapsed: 0 });
  const [counts, setCounts]           = useState({ "2d": 0, "3d": 0, unknown: 0 });
  const [results, setResults]         = useState<ImageResult[]>([]);
  const [filterCategory, setFilterCategory] = useState<Category | "all">("all");
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [lastClickedIdx, setLastClickedIdx]   = useState<number | null>(null);
  const [dragState, setDragState]     = useState<DragState | null>(null);
  const [realtimeMode, setRealtimeMode] = useState(false);

  // ── Dedup state ──────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<"classify" | "dedup">("classify");
  const [dedupFolder, setDedupFolder] = useState("");
  const [dedupDestFolder, setDedupDestFolder] = useState("");
  const [dedupRecursive, setDedupRecursive] = useState(false);
  const [dedupHashMethod, setDedupHashMethod] = useState<"exact" | "perceptual">("exact");
  const [dedupState, setDedupState] = useState<DedupState>("idle");
  const [dedupGroups, setDedupGroups] = useState<DupGroup[]>([]);
  const [dedupProgress, setDedupProgress] = useState({ current: 0, total: 0, filename: "" });
  const dedupGroupsRef = useRef<DupGroup[]>([]);
  const dedupUnlistenRef = useRef<(() => void) | null>(null);

  const [showWelcome, setShowWelcome]   = useState(false);
  const [dontShowAgain, setDontShowAgain] = useState(false);

  const resultsRef  = useRef<ImageResult[]>([]);
  const countsRef   = useRef({ "2d": 0, "3d": 0, unknown: 0 });
  const gridRef     = useRef<HTMLDivElement>(null);
  const cancelledRef = useRef(false);
  const unlistenRef  = useRef<(() => void) | null>(null);

  useEffect(() => { init(); }, []);

  // 폴더 경로 변경 시 자동 저장 (store 초기화 후에만)
  useEffect(() => { savePaths(); }, [folderPath, folder2d, folder3d, folderUnknown, dedupFolder, dedupDestFolder]);

  async function init() {
    // 저장된 경로 불러오기
    try {
      const store = await load("settings.json");
      storeRef.current = store;
      const p  = await store.get<string>("folderPath")    ?? "";
      const d2 = await store.get<string>("folder2d")      ?? "";
      const d3 = await store.get<string>("folder3d")      ?? "";
      const du = await store.get<string>("folderUnknown") ?? "";
      if (p)  setFolderPath(p);
      if (d2) setFolder2d(d2);
      if (d3) setFolder3d(d3);
      if (du) setFolderUnknown(du);
      const dd  = await store.get<string>("dedupFolder")    ?? "";
      const ddd = await store.get<string>("dedupDestFolder") ?? "";
      if (dd)  setDedupFolder(dd);
      if (ddd) setDedupDestFolder(ddd);
    } catch { /* store 초기화 실패 시 무시 */ }

    // 시작 안내 팝업 표시 여부 확인
    try {
      const store = storeRef.current;
      if (store) {
        const hideWelcome = await store.get<boolean>("hideWelcome") ?? false;
        if (!hideWelcome) setShowWelcome(true);
      }
    } catch { setShowWelcome(true); }

    // Ollama 연결 확인
    try {
      const ok = await invoke<boolean>("check_ollama");
      setOllamaOk(ok);
      if (ok) await refreshModels();
    } catch { setOllamaOk(false); }
  }

  async function closeWelcome() {
    if (dontShowAgain && storeRef.current) {
      await storeRef.current.set("hideWelcome", true);
      await storeRef.current.save();
    }
    setShowWelcome(false);
  }

  async function savePaths() {
    const store = storeRef.current;
    if (!store) return;
    await store.set("folderPath",    folderPath);
    await store.set("folder2d",      folder2d);
    await store.set("folder3d",      folder3d);
    await store.set("folderUnknown", folderUnknown);
    await store.set("dedupFolder",    dedupFolder);
    await store.set("dedupDestFolder", dedupDestFolder);
    await store.save();
  }

  async function refreshModels() {
    try {
      const models = await invoke<string[]>("list_ollama_models");
      setOllamaModels(models);
      if (models.length > 0) setSelectedModel(prev => models.includes(prev) ? prev : models[0]);
    } catch { /* ignore */ }
  }

  // ── Drag selection ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!dragState) return;

    const handleMouseMove = (e: MouseEvent) =>
      setDragState(prev => prev ? { ...prev, currentX: e.clientX, currentY: e.clientY } : null);

    const handleMouseUp = (e: MouseEvent) => {
      const dx = Math.abs(e.clientX - dragState.startX);
      const dy = Math.abs(e.clientY - dragState.startY);

      if ((dx > 5 || dy > 5) && gridRef.current) {
        const sel = {
          left:   Math.min(dragState.startX, e.clientX),
          top:    Math.min(dragState.startY, e.clientY),
          right:  Math.max(dragState.startX, e.clientX),
          bottom: Math.max(dragState.startY, e.clientY),
        };
        const newSel = new Set<number>();
        gridRef.current.querySelectorAll<HTMLElement>("[data-realidx]").forEach(card => {
          const r = card.getBoundingClientRect();
          if (r.left < sel.right && r.right > sel.left && r.top < sel.bottom && r.bottom > sel.top) {
            const idx = parseInt(card.getAttribute("data-realidx") ?? "-1");
            if (idx >= 0) newSel.add(idx);
          }
        });
        setSelectedIndices(newSel);
      }
      setDragState(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragState]);

  // ── Dedup ────────────────────────────────────────────────────────────────
  async function startDedupScan() {
    if (!dedupFolder) return;
    setDedupState("scanning");
    setDedupGroups([]);
    dedupGroupsRef.current = [];
    setDedupProgress({ current: 0, total: 0, filename: "" });

    const unlistenProgress = await listen<{ current: number; total: number; filename: string }>(
      "dup-scan-progress",
      (event) => {
        const e = event.payload;
        setDedupProgress({ current: e.current, total: e.total, filename: e.filename });
      }
    );

    const unlistenGroup = await listen<DupGroup>("dup-group-found", (event) => {
      const group = event.payload;
      dedupGroupsRef.current = [...dedupGroupsRef.current, group];
      setDedupGroups([...dedupGroupsRef.current]);
    });

    dedupUnlistenRef.current = () => { unlistenProgress(); unlistenGroup(); };

    try {
      await invoke("scan_duplicates", {
        folderPath: dedupFolder,
        recursive: dedupRecursive,
        hashMethod: dedupHashMethod,
      });
      setDedupState("reviewing");
    } catch (err) {
      console.error(err);
      setDedupState("idle");
    } finally {
      unlistenProgress();
      unlistenGroup();
      dedupUnlistenRef.current = null;
    }
  }

  async function cancelDedupScan() {
    dedupUnlistenRef.current?.();
    dedupUnlistenRef.current = null;
    await invoke("cancel_dup_scan");
    setDedupState("idle");
  }

  async function applyDedupMoves() {
    if (!dedupDestFolder) return;
    setDedupState("applying");
    try {
      const toMove: string[] = [];
      for (const group of dedupGroupsRef.current) {
        group.images.forEach((img, i) => {
          if (i !== group.keep_idx) toMove.push(img.path);
        });
      }
      await invoke("apply_duplicate_moves", { toMove, destFolder: dedupDestFolder });
      setDedupState("done");
    } catch (err) {
      console.error(err);
      setDedupState("reviewing");
    }
  }

  function resetDedup() {
    setDedupState("idle");
    setDedupGroups([]);
    dedupGroupsRef.current = [];
    setDedupProgress({ current: 0, total: 0, filename: "" });
  }

  function changeKeepIdx(groupId: number, newKeepIdx: number) {
    const updated = dedupGroupsRef.current.map(g =>
      g.id === groupId ? { ...g, keep_idx: newKeepIdx } : g
    );
    dedupGroupsRef.current = updated;
    setDedupGroups([...updated]);
  }

  function formatBytes(bytes: number) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // ── Classification ────────────────────────────────────────────────────────
  async function pickFolder(setter: (p: string) => void) {
    const sel = await open({ directory: true, multiple: false });
    if (sel && typeof sel === "string") setter(sel);
  }

  const allFoldersSet = folderPath && folder2d && folder3d && folderUnknown;

  async function startClassification() {
    if (!folderPath) return;
    cancelledRef.current = false;
    setAppState("scanning");
    setResults([]);
    setSelectedIndices(new Set());
    setLastClickedIdx(null);
    resultsRef.current = [];
    countsRef.current = { "2d": 0, "3d": 0, unknown: 0 };
    setCounts({ "2d": 0, "3d": 0, unknown: 0 });
    setProgress({ current: 0, total: 0, filename: "", elapsed: 0 });

    const unlisten = await listen<ProgressEvent>("classification-progress", (event) => {
      const e = event.payload;
      setProgress({ current: e.current, total: e.total, filename: e.filename, elapsed: e.elapsed_secs });
      const newItem: ImageResult = {
        path: e.filename,
        filename: e.filename.split(/[\\/]/).pop() ?? e.filename,
        category: e.category,
        thumbnail: e.thumbnail,
      };
      resultsRef.current = [...resultsRef.current, newItem];
      setResults([...resultsRef.current]);
      countsRef.current = { ...countsRef.current, [e.category]: countsRef.current[e.category] + 1 };
      setCounts({ ...countsRef.current });
    });
    unlistenRef.current = unlisten;

    try {
      await invoke("classify_images", {
        folderPath,
        dest2d: folder2d,
        dest3d: folder3d,
        destUnknown: folderUnknown,
        realtime: realtimeMode,
        modelName: selectedModel,
      });
      if (!cancelledRef.current) setAppState(realtimeMode ? "done" : "reviewing");
    } catch (err) {
      if (!cancelledRef.current) { console.error(err); setAppState("idle"); }
    } finally {
      unlisten();
      unlistenRef.current = null;
    }
  }

  async function pauseClassification() {
    await invoke("pause_classification");
    setAppState("paused");
  }

  async function resumeClassification() {
    await invoke("resume_classification");
    setAppState("scanning");
  }

  async function cancelClassification() {
    cancelledRef.current = true;
    unlistenRef.current?.();
    unlistenRef.current = null;
    await invoke("cancel_classification");
    setAppState("reviewing");
  }

  async function applyMoves() {
    setAppState("applying");
    try {
      await invoke("apply_moves", {
        dest2d: folder2d,
        dest3d: folder3d,
        destUnknown: folderUnknown,
        results: resultsRef.current.map(r => ({ path: r.path, category: r.category })),
      });
      setAppState("done");
    } catch (err) { console.error(err); setAppState("reviewing"); }
  }

  function resetApp() {
    setAppState("idle");
    setResults([]);
    setSelectedIndices(new Set());
    setLastClickedIdx(null);
    setProgress({ current: 0, total: 0, filename: "", elapsed: 0 });
    setCounts({ "2d": 0, "3d": 0, unknown: 0 });
  }

  // ── Category change ───────────────────────────────────────────────────────
  function changeCategory(idx: number, category: Category) {
    const updated = [...resultsRef.current];
    updated[idx] = { ...updated[idx], category };
    resultsRef.current = updated;
    setResults([...updated]);
    const nc = { "2d": 0, "3d": 0, unknown: 0 };
    updated.forEach(r => nc[r.category]++);
    setCounts(nc);
    countsRef.current = nc;
  }

  function bulkChangeCategory(category: Category) {
    if (selectedIndices.size === 0) return;
    const updated = [...resultsRef.current];
    selectedIndices.forEach(idx => {
      if (idx >= 0 && idx < updated.length) {
        updated[idx] = { ...updated[idx], category };
      }
    });
    resultsRef.current = updated;
    setResults([...updated]);
    const nc = { "2d": 0, "3d": 0, unknown: 0 };
    updated.forEach(r => nc[r.category]++);
    setCounts(nc);
    countsRef.current = nc;
    setSelectedIndices(new Set());
  }

  // ── Selection ─────────────────────────────────────────────────────────────
  const isInteractive = appState === "reviewing";

  function handleCardClick(e: React.MouseEvent, realIdx: number) {
    if (!isInteractive) return;
    e.stopPropagation();

    if (e.shiftKey && lastClickedIdx !== null) {
      const min = Math.min(lastClickedIdx, realIdx);
      const max = Math.max(lastClickedIdx, realIdx);
      const ns = new Set(selectedIndices);
      for (let i = min; i <= max; i++) ns.add(i);
      setSelectedIndices(ns);
    } else if (e.ctrlKey || e.metaKey) {
      const ns = new Set(selectedIndices);
      ns.has(realIdx) ? ns.delete(realIdx) : ns.add(realIdx);
      setSelectedIndices(ns);
      setLastClickedIdx(realIdx);
    } else {
      if (selectedIndices.size === 1 && selectedIndices.has(realIdx)) {
        setSelectedIndices(new Set());
        setLastClickedIdx(null);
      } else {
        setSelectedIndices(new Set([realIdx]));
        setLastClickedIdx(realIdx);
      }
    }
  }

  function handleGridMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (!isInteractive) return;
    if ((e.target as HTMLElement).closest(".image-card")) return;
    e.preventDefault();
    setDragState({ startX: e.clientX, startY: e.clientY, currentX: e.clientX, currentY: e.clientY });
    setSelectedIndices(new Set());
    setLastClickedIdx(null);
  }

  function selectAll() {
    setSelectedIndices(new Set(filteredIndexed.map(fi => fi.idx)));
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  // Carry the stable index with each item — no indexOf needed anywhere
  const filteredIndexed: IndexedResult[] = results
    .map((item, idx) => ({ item, idx }))
    .filter(({ item }) => filterCategory === "all" || item.category === filterCategory);

  const progressPercent = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;
  const avgSecs = progress.current > 0 ? progress.elapsed / progress.current : 0;
  const remaining = Math.max(0, (progress.total - progress.current) * avgSecs);

  const rubberBand = dragState &&
    (Math.abs(dragState.currentX - dragState.startX) > 3 || Math.abs(dragState.currentY - dragState.startY) > 3)
    ? {
        left:   Math.min(dragState.startX, dragState.currentX),
        top:    Math.min(dragState.startY, dragState.currentY),
        width:  Math.abs(dragState.currentX - dragState.startX),
        height: Math.abs(dragState.currentY - dragState.startY),
      }
    : null;

  const isInProgress = appState === "scanning" || appState === "paused";
  const isReviewVisible = isInProgress || appState === "reviewing" || appState === "applying";

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-title">
          <span className="header-icon">🗂</span>
          <h1>Image Auto Classifier</h1>
        </div>
        <div className="header-right">
          <div className={`ollama-status ${ollamaOk === true ? "ok" : ollamaOk === false ? "error" : "checking"}`}>
            <span className="status-dot" />
            <span>{ollamaOk === true ? "Ollama 연결됨" : ollamaOk === false ? "Ollama 오프라인" : "확인 중..."}</span>
          </div>
        </div>
      </header>

      {/* ── Tab bar ── */}
      <div className="tab-bar">
        <button
          className={`tab-btn ${activeTab === "classify" ? "active" : ""}`}
          onClick={() => setActiveTab("classify")}
        >
          🗂 이미지 분류
        </button>
        <button
          className={`tab-btn ${activeTab === "dedup" ? "active" : ""}`}
          onClick={() => setActiveTab("dedup")}
        >
          🔍 중복 이미지
        </button>
      </div>

      {/* ── Classify tab ── */}
      {activeTab === "classify" && appState === "done" && (
        <div className="done-screen">
          <div className="done-card">
            <div className="done-icon">✅</div>
            <h2>분류 완료!</h2>
            <div className="done-stats">
              <div className="done-stat"><span className="badge badge-2d">2D</span><strong>{counts["2d"]}</strong>장</div>
              <div className="done-stat"><span className="badge badge-3d">3D</span><strong>{counts["3d"]}</strong>장</div>
              <div className="done-stat"><span className="badge badge-unknown">?</span><strong>{counts.unknown}</strong>장</div>
            </div>
            <p className="done-desc">파일이 각 하위 폴더로 이동되었습니다.</p>
            <button className="btn-primary" onClick={resetApp}>새 폴더 분류하기</button>
          </div>
        </div>
      )}

      {activeTab === "classify" && appState !== "done" && (
        <main className="main">
          {/* ── Folder + Mode ── */}
          <section className="folder-section">
            <div className="folder-grid">
              {/* Source */}
              <span className="folder-label">스캔 폴더</span>
              <button className="btn-secondary folder-btn"
                onClick={() => pickFolder(setFolderPath)}
                disabled={isInProgress || appState === "applying"}>
                📁 선택
              </button>
              <span className="folder-path">{folderPath || "이미지가 있는 폴더"}</span>

              {/* 2D destination */}
              <span className="folder-label dest-label badge-2d-text">2D 저장 폴더</span>
              <button className="btn-secondary folder-btn"
                onClick={() => pickFolder(setFolder2d)}
                disabled={isInProgress || appState === "applying"}>
                📁 선택
              </button>
              <span className="folder-path">{folder2d || "2D 이미지를 보낼 폴더"}</span>

              {/* 3D destination */}
              <span className="folder-label dest-label badge-3d-text">3D 저장 폴더</span>
              <button className="btn-secondary folder-btn"
                onClick={() => pickFolder(setFolder3d)}
                disabled={isInProgress || appState === "applying"}>
                📁 선택
              </button>
              <span className="folder-path">{folder3d || "3D 이미지를 보낼 폴더"}</span>

              {/* Unknown destination */}
              <span className="folder-label dest-label badge-unknown-text">미분류 저장 폴더</span>
              <button className="btn-secondary folder-btn"
                onClick={() => pickFolder(setFolderUnknown)}
                disabled={isInProgress || appState === "applying"}>
                📁 선택
              </button>
              <span className="folder-path">{folderUnknown || "미분류 이미지를 보낼 폴더"}</span>
            </div>

            {/* ── Model selection ── */}
            <div className="model-row">
              <div className="model-row-left">
                <span className="model-row-label">🤖 AI 모델</span>
                {ollamaOk === true ? (
                  ollamaModels.length > 0 ? (
                    <select
                      className="model-select-main"
                      value={selectedModel}
                      onChange={e => setSelectedModel(e.target.value)}
                      disabled={isInProgress || appState === "applying"}
                    >
                      {ollamaModels.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                  ) : (
                    <span className="model-none-hint">비전 모델 없음 — <code>ollama pull llava</code> 로 설치하세요</span>
                  )
                ) : (
                  <span className="model-none-hint">Ollama 연결 필요</span>
                )}
                <button
                  className="btn-refresh-models"
                  onClick={refreshModels}
                  disabled={isInProgress || appState === "applying" || ollamaOk !== true}
                  title="모델 목록 새로고침"
                >🔄</button>
              </div>
              <div className="model-row-right">
                <span className="model-tip">💡 더 정확한 분류: <code>ollama pull llama3.2-vision</code> 또는 <code>llava:13b</code></span>
              </div>
            </div>

            <div className="mode-row">
              <label className={`mode-toggle ${realtimeMode ? "on" : ""}`}>
                <input type="checkbox" checked={realtimeMode}
                  onChange={e => setRealtimeMode(e.target.checked)}
                  disabled={isInProgress || appState === "applying"} />
                <span className="toggle-slider" />
                <span className="toggle-label">
                  {realtimeMode ? "⚡ 실시간 정리 (분류 즉시 이동)" : "📋 검토 후 정리 (분류 후 확인)"}
                </span>
              </label>
            </div>
          </section>

          {/* ── Progress ── */}
          {(isInProgress || appState === "reviewing" || appState === "applying") && (
            <section className="progress-section">
              <div className="progress-header">
                <span className="progress-label">
                  {appState === "applying" ? "파일 이동 중..."
                    : appState === "reviewing" ? "분류 완료"
                    : appState === "paused"    ? "⏸ 일시정지됨"
                    : "분류 중..."}
                </span>
                <span className="progress-count">{progress.current} / {progress.total}</span>
              </div>
              <div className="progress-bar-bg">
                <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }} />
              </div>
              <div className="progress-meta">
                <span className="progress-file" title={progress.filename}>
                  {progress.filename.split(/[\\/]/).pop()}
                </span>
                {appState === "scanning" && remaining > 0 && (
                  <span>남은 시간: 약 {remaining < 60 ? `${Math.ceil(remaining)}초` : `${Math.ceil(remaining / 60)}분`}</span>
                )}
              </div>
              <div className="counts-row">
                <span className="badge badge-2d">2D {counts["2d"]}</span>
                <span className="badge badge-3d">3D {counts["3d"]}</span>
                <span className="badge badge-unknown">? {counts.unknown}</span>
              </div>
            </section>
          )}

          {/* ── Start button ── */}
          {appState === "idle" && (
            <section className="action-section">
              <button className="btn-primary btn-start" onClick={startClassification}
                disabled={!allFoldersSet || !ollamaOk}>
                ▶ 분류 시작
              </button>
              {!allFoldersSet && (
                <p className="start-hint">스캔 폴더와 저장 폴더 4개를 모두 선택해주세요</p>
              )}
            </section>
          )}

          {/* ── Pause / Resume / Cancel ── */}
          {isInProgress && (
            <section className="action-section scanning-controls">
              {appState === "scanning"
                ? <button className="btn-pause" onClick={pauseClassification}>⏸ 일시정지</button>
                : <button className="btn-resume" onClick={resumeClassification}>▶ 재개</button>
              }
              <button className="btn-cancel-scan" onClick={cancelClassification}>✕ 취소 (지금까지 분류 결과 확인)</button>
            </section>
          )}

          {/* ── Results ── */}
          {isReviewVisible && results.length > 0 && (
            <section className="review-section">

              {/* ── Sticky controls bar ── */}
              <div className="sticky-controls">
                <div className="review-header">
                  <h2>결과 확인 <span className="result-count">({results.length}장)</span></h2>
                  <div className="filter-row">
                    {(["all", "2d", "3d", "unknown"] as const).map(cat => (
                      <button key={cat}
                        className={`filter-btn ${filterCategory === cat ? "active" : ""}`}
                        onClick={() => setFilterCategory(cat)}>
                        {cat === "all" ? `전체 ${results.length}` : cat === "unknown" ? `미분류 ${counts.unknown}` : `${cat.toUpperCase()} ${counts[cat]}`}
                      </button>
                    ))}
                  </div>
                </div>

                {/* ── Bulk toolbar ── */}
                {isInteractive && selectedIndices.size > 0 && (
                  <div className="bulk-toolbar">
                    <span className="bulk-count">{selectedIndices.size}개 선택됨</span>
                    <div className="bulk-actions">
                      <button className="bulk-btn bulk-2d" onClick={() => bulkChangeCategory("2d")}>→ 2D</button>
                      <button className="bulk-btn bulk-3d" onClick={() => bulkChangeCategory("3d")}>→ 3D</button>
                      <button className="bulk-btn bulk-unknown" onClick={() => bulkChangeCategory("unknown")}>→ 미분류</button>
                    </div>
                    <button className="bulk-deselect" onClick={() => setSelectedIndices(new Set())}>✕ 선택 해제</button>
                  </div>
                )}

                {/* ── Selection hint ── */}
                {isInteractive && selectedIndices.size === 0 && (
                  <div className="selection-hint">
                    <span>클릭 · Shift+클릭(범위) · Ctrl+클릭(추가) · 드래그(다중)</span>
                    <button className="select-all-btn" onClick={selectAll}>전체 선택 ({filteredIndexed.length}장)</button>
                  </div>
                )}

                {/* ── Scanning hint ── */}
                {isInProgress && (
                  <div className="scanning-hint">분류 완료 또는 취소 후 카테고리 변경 가능</div>
                )}
              </div>

              {/* ── Grid ── */}
              <div ref={gridRef}
                className={`image-grid ${isInteractive ? "selectable" : ""}`}
                onMouseDown={handleGridMouseDown}>
                {filteredIndexed.map(({ item, idx: realIdx }) => {
                  const isSelected = selectedIndices.has(realIdx);
                  return (
                    <div key={item.path} data-realidx={realIdx}
                      className={`image-card ${isSelected ? "selected" : ""}`}
                      onClick={e => handleCardClick(e, realIdx)}>
                      <div className="image-thumb-wrap">
                        {item.thumbnail
                          ? <img src={`data:image/jpeg;base64,${item.thumbnail}`} alt={item.filename} className="image-thumb" />
                          : <div className="image-thumb-placeholder">🖼</div>}
                        <span className={`badge-overlay badge-${item.category}`}>
                          {item.category === "unknown" ? "?" : item.category.toUpperCase()}
                        </span>
                        {isInteractive && (
                          <div className={`card-checkbox ${isSelected ? "checked" : ""}`}>
                            {isSelected && <span>✓</span>}
                          </div>
                        )}
                      </div>
                      <div className="image-info">
                        <span className="image-name" title={item.filename}>{item.filename}</span>
                        <select className="category-select" value={item.category}
                          onChange={e => { e.stopPropagation(); changeCategory(realIdx, e.target.value as Category); }}
                          onClick={e => e.stopPropagation()}
                          disabled={!isInteractive}>
                          <option value="2d">2D</option>
                          <option value="3d">3D</option>
                          <option value="unknown">미분류</option>
                        </select>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* ── Review actions ── */}
              {(appState === "reviewing" || appState === "applying") && (
                <div className="review-actions">
                  <button className="btn-cancel" onClick={resetApp} disabled={appState === "applying"}>취소</button>
                  <button className="btn-apply" onClick={applyMoves} disabled={appState === "applying"}>
                    {appState === "applying"
                      ? <><span className="spinner" /> 적용 중...</>
                      : `✅ 적용 (${results.length}장 이동)`}
                  </button>
                </div>
              )}
            </section>
          )}
        </main>
      )}

      {/* ── Dedup tab ── */}
      {activeTab === "dedup" && (
        <main className="main">
          {/* Done */}
          {dedupState === "done" && (
            <div className="done-screen">
              <div className="done-card">
                <div className="done-icon">✅</div>
                <h2>중복 정리 완료!</h2>
                <p className="done-desc">
                  {dedupGroups.reduce((acc, g) => acc + g.images.length - 1, 0)}개의 중복 파일이 이동되었습니다.
                </p>
                <button className="btn-primary" onClick={resetDedup}>다시 검사하기</button>
              </div>
            </div>
          )}

          {dedupState !== "done" && (
            <>
              {/* 폴더 설정 */}
              <section className="folder-section">
                <div className="folder-grid">
                  <span className="folder-label">검사 폴더</span>
                  <button className="btn-secondary folder-btn"
                    onClick={() => pickFolder(setDedupFolder)}
                    disabled={dedupState === "scanning" || dedupState === "applying"}>
                    📁 선택
                  </button>
                  <span className="folder-path">{dedupFolder || "중복을 검사할 폴더"}</span>

                  <span className="folder-label dest-label">이동 폴더</span>
                  <button className="btn-secondary folder-btn"
                    onClick={() => pickFolder(setDedupDestFolder)}
                    disabled={dedupState === "scanning" || dedupState === "applying"}>
                    📁 선택
                  </button>
                  <span className="folder-path">{dedupDestFolder || "중복 파일을 보낼 폴더"}</span>
                </div>

                {/* 옵션 */}
                <div className="dedup-options">
                  <label className={`mode-toggle ${dedupRecursive ? "on" : ""}`}>
                    <input type="checkbox" checked={dedupRecursive}
                      onChange={e => setDedupRecursive(e.target.checked)}
                      disabled={dedupState === "scanning"} />
                    <span className="toggle-slider" />
                    <span className="toggle-label">
                      {dedupRecursive ? "📂 하위 폴더 포함" : "📁 선택 폴더만"}
                    </span>
                  </label>

                  <div className="hash-method-row">
                    <span className="hash-method-label">중복 기준</span>
                    <label className={`hash-radio ${dedupHashMethod === "exact" ? "selected" : ""}`}>
                      <input type="radio" name="hashMethod" value="exact"
                        checked={dedupHashMethod === "exact"}
                        onChange={() => setDedupHashMethod("exact")}
                        disabled={dedupState === "scanning"} />
                      파일 해시 (완전 동일)
                    </label>
                    <label className={`hash-radio ${dedupHashMethod === "perceptual" ? "selected" : ""}`}>
                      <input type="radio" name="hashMethod" value="perceptual"
                        checked={dedupHashMethod === "perceptual"}
                        onChange={() => setDedupHashMethod("perceptual")}
                        disabled={dedupState === "scanning"} />
                      유사 이미지 (dHash)
                    </label>
                  </div>
                </div>
              </section>

              {/* 진행 상태 */}
              {dedupState === "scanning" && (
                <section className="progress-section">
                  <div className="progress-header">
                    <span className="progress-label">중복 검사 중...</span>
                    <span className="progress-count">{dedupProgress.current} / {dedupProgress.total}</span>
                  </div>
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill"
                      style={{ width: `${dedupProgress.total > 0 ? (dedupProgress.current / dedupProgress.total) * 100 : 0}%` }} />
                  </div>
                  <div className="progress-meta">
                    <span className="progress-file">{dedupProgress.filename}</span>
                  </div>
                  <div className="action-section scanning-controls" style={{ marginTop: 8 }}>
                    <button className="btn-cancel-scan" onClick={cancelDedupScan}>✕ 취소</button>
                  </div>
                </section>
              )}

              {/* 시작 버튼 */}
              {dedupState === "idle" && (
                <section className="action-section">
                  <button className="btn-primary btn-start" onClick={startDedupScan}
                    disabled={!dedupFolder}>
                    🔍 중복 검사 시작
                  </button>
                  {!dedupFolder && <p className="start-hint">검사할 폴더를 선택해주세요</p>}
                </section>
              )}

              {/* 결과 */}
              {(dedupState === "scanning" || dedupState === "reviewing" || dedupState === "applying") && dedupGroups.length > 0 && (
                <section className="review-section">
                  <div className="sticky-controls">
                    <div className="review-header">
                      <h2>중복 검사 결과
                        <span className="result-count">
                          ({dedupGroups.length}개 그룹 ·&nbsp;
                          {dedupGroups.reduce((acc, g) => acc + g.images.length - 1, 0)}개 삭제 대상)
                        </span>
                      </h2>
                    </div>
                  </div>

                  {dedupGroups.length === 0 ? (
                    <div className="dedup-empty">중복 이미지가 없습니다 🎉</div>
                  ) : (
                    dedupGroups.map(group => (
                      <div key={group.id} className="dup-group">
                        <div className="dup-group-header">
                          그룹 {group.id + 1} — {group.images.length}개 중 1개 보존
                        </div>
                        <div className="dup-group-images">
                          {group.images.map((img, i) => {
                            const isKeeper = i === group.keep_idx;
                            return (
                              <div key={img.path}
                                className={`dup-image-card ${isKeeper ? "keeper" : "to-delete"}`}
                                onClick={() => dedupState === "reviewing" && changeKeepIdx(group.id, i)}>
                                <div className="image-thumb-wrap">
                                  {img.thumbnail
                                    ? <img src={`data:image/jpeg;base64,${img.thumbnail}`} alt={img.filename} className="image-thumb" />
                                    : <div className="image-thumb-placeholder">🖼</div>}
                                  <span className={`dup-badge ${isKeeper ? "dup-keep" : "dup-del"}`}>
                                    {isKeeper ? "✓ 유지" : "삭제"}
                                  </span>
                                </div>
                                <div className="image-info">
                                  <span className="image-name" title={img.filename}>{img.filename}</span>
                                  <span className="dup-meta">{img.width}×{img.height} · {formatBytes(img.size)}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))
                  )}

                  <div className="review-actions">
                    <button className="btn-cancel" onClick={resetDedup} disabled={dedupState === "scanning" || dedupState === "applying"}>취소</button>
                    <button className="btn-apply" onClick={applyDedupMoves}
                      disabled={dedupState === "scanning" || dedupState === "applying" || !dedupDestFolder || dedupGroups.length === 0}>
                      {dedupState === "applying"
                        ? <><span className="spinner" /> 이동 중...</>
                        : `✅ 적용 (${dedupGroups.reduce((acc, g) => acc + g.images.length - 1, 0)}개 이동)`}
                    </button>
                  </div>
                </section>
              )}
            </>
          )}
        </main>
      )}

      {/* ── Welcome modal ── */}
      {showWelcome && (
        <div className="modal-overlay" onClick={closeWelcome}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span className="modal-icon">🗂</span>
              <h2>Image Auto Classifier 시작 안내</h2>
            </div>
            <div className="modal-body">
              <p className="modal-intro">
                이 프로그램은 <strong>Ollama</strong> 로컬 AI 서버를 통해 이미지를 분류합니다.<br />
                사용 전 아래 두 가지를 준비해주세요.
              </p>

              <div className="modal-step">
                <div className="modal-step-num">1</div>
                <div className="modal-step-content">
                  <strong>Ollama 설치</strong>
                  <p>아래 링크에서 운영체제에 맞는 버전을 다운로드하여 설치하세요.</p>
                  <a className="modal-link" href="https://ollama.com/download" target="_blank" rel="noreferrer">
                    🔗 https://ollama.com/download
                  </a>
                </div>
              </div>

              <div className="modal-step">
                <div className="modal-step-num">2</div>
                <div className="modal-step-content">
                  <strong>비전 모델 설치</strong>
                  <p>터미널(명령 프롬프트)에서 아래 중 하나를 실행하세요.</p>
                  <div className="modal-models">
                    <div className="modal-model recommended">
                      <span className="model-badge">추천</span>
                      <code>ollama pull qwen2.5vl</code>
                      <span className="model-desc">~6 GB · 분류 정확도 최고</span>
                    </div>
                    <div className="modal-model recommended">
                      <span className="model-badge">추천</span>
                      <code>ollama pull llama3.2-vision</code>
                      <span className="model-desc">~7.8 GB · 안정적이고 범용적</span>
                    </div>
                    <div className="modal-model">
                      <span className="model-badge light">경량</span>
                      <code>ollama pull gemma3:4b</code>
                      <span className="model-desc">~3 GB · 느린 PC에서 사용</span>
                    </div>
                    <div className="modal-model">
                      <span className="model-badge light">경량</span>
                      <code>ollama pull llava</code>
                      <span className="model-desc">~4.7 GB · 기본 모델</span>
                    </div>
                  </div>
                </div>
              </div>

              <p className="modal-note">
                💡 설치 후 Ollama가 실행 중인 상태에서 앱을 사용하세요. 상단에 <strong>Ollama 연결됨</strong> 표시가 뜨면 준비 완료입니다.
              </p>
            </div>

            <div className="modal-footer">
              <label className="modal-dont-show">
                <input type="checkbox" checked={dontShowAgain}
                  onChange={e => setDontShowAgain(e.target.checked)} />
                다음부터 이 창을 표시하지 않음
              </label>
              <button className="btn-primary modal-confirm" onClick={closeWelcome}>
                확인
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Rubber band ── */}
      {rubberBand && (
        <div className="rubber-band" style={{
          left: rubberBand.left, top: rubberBand.top,
          width: rubberBand.width, height: rubberBand.height,
        }} />
      )}
    </div>
  );
}
