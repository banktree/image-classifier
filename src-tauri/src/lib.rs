use anyhow::Result;
use base64::{engine::general_purpose, Engine};
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::{Mutex, Notify, Semaphore};

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    #[serde(rename = "2d")]
    TwoD,
    #[serde(rename = "3d")]
    ThreeD,
    Unknown,
}

impl Category {
    fn folder_name(&self) -> &str {
        match self {
            Category::TwoD => "2d",
            Category::ThreeD => "3d",
            Category::Unknown => "unknown",
        }
    }
}

#[derive(Serialize, Clone)]
struct ProgressEvent {
    current: usize,
    total: usize,
    filename: String,
    category: String,
    thumbnail: String,
    elapsed_secs: f64,
}

#[derive(Deserialize)]
pub struct MoveItem {
    pub path: String,
    pub category: String,
}

// ── Classification control (pause / cancel) ───────────────────────────────────

pub struct ClassificationControl {
    pub cancelled: AtomicBool,
    pub paused: AtomicBool,
    pub resume_notify: Notify,
}

impl ClassificationControl {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            cancelled: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            resume_notify: Notify::new(),
        })
    }

    /// Wait here if paused, return true if cancelled.
    async fn wait_if_paused(&self) -> bool {
        loop {
            if self.cancelled.load(Ordering::Relaxed) {
                return true;
            }
            if !self.paused.load(Ordering::Relaxed) {
                return false;
            }
            self.resume_notify.notified().await;
        }
    }
}

// Tauri managed state: holds the current session's control handle.
pub struct ControlState(pub Mutex<Option<Arc<ClassificationControl>>>);

// ── Duplicate scan state ──────────────────────────────────────────────────────
pub struct DupScanState(pub Mutex<Option<Arc<AtomicBool>>>);

#[derive(Serialize, Clone)]
struct DupScanProgress {
    current: usize,
    total: usize,
    filename: String,
}

#[derive(Serialize, Clone)]
struct DupImageInfo {
    path: String,
    filename: String,
    size: u64,
    width: u32,
    height: u32,
    thumbnail: String,
}

#[derive(Serialize, Clone)]
struct DupGroup {
    id: usize,
    images: Vec<DupImageInfo>,
    keep_idx: usize,
}

// ── Ollama ────────────────────────────────────────────────────────────────────

const OLLAMA_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MODEL: &str = "llava";
const CLASSIFY_PROMPT: &str = "\
You are an image style classifier. Your job is to determine whether an image is '2d' artwork or '3d' photographic content.

RULES — read carefully:
- Answer '2d' if the image is: anime, manga, cartoon, webtoon, comic, illustration, digital drawing, pixel art, hand-drawn art, watercolor, sketch, or ANY content that was drawn/painted by a human or AI art generator (like Stable Diffusion, NovelAI, Midjourney in art style).
- Answer '3d' if the image is: a real photograph, a selfie, a photo of a person or place, photorealistic CGI/3D render, or any image that looks like it was captured by a camera or rendered to look like a real photo.
- Answer 'unknown' ONLY for: pure text screenshots, blank images, logos, charts, diagrams, or abstract patterns with no clear subject.

IMPORTANT: Be decisive. Do NOT default to 'unknown'. Most images are clearly 2d or 3d.
- When in doubt about art style → answer '2d'
- When in doubt about photos → answer '3d'

Reply with exactly ONE word: 2d, 3d, or unknown";

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: &'static str,
    images: Vec<String>,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

#[derive(Serialize, Deserialize)]
struct OllamaModelEntry {
    name: String,
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelEntry>,
}

async fn classify_with_ollama(client: &reqwest::Client, image_b64: &str, model: &str) -> Result<Category> {
    let req = OllamaRequest {
        model: model.to_string(),
        prompt: CLASSIFY_PROMPT,
        images: vec![image_b64.to_string()],
        stream: false,
    };

    let resp = client
        .post(format!("{}/api/generate", OLLAMA_URL))
        .json(&req)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await?
        .json::<OllamaResponse>()
        .await?;

    let answer = resp.response.trim().to_lowercase();
    let category = if answer.contains("2d") {
        Category::TwoD
    } else if answer.contains("3d") {
        Category::ThreeD
    } else {
        Category::Unknown
    };

    Ok(category)
}

// ── Image helpers ─────────────────────────────────────────────────────────────

fn make_thumbnail_b64(path: &Path, size: u32) -> Result<String> {
    let img = image::open(path)?;
    let thumb = img.thumbnail(size, size);
    let mut buf = Vec::new();
    thumb.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)?;
    Ok(general_purpose::STANDARD.encode(&buf))
}

fn make_ollama_b64(path: &Path) -> Result<String> {
    let img = image::open(path)?;
    let resized = img.resize(768, 768, FilterType::Lanczos3);
    let mut buf = Vec::new();
    resized.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)?;
    Ok(general_purpose::STANDARD.encode(&buf))
}

/// Move a file into dest_dir, handling name collisions automatically.
fn move_file_to_dir(src: &Path, dest_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dest_dir)?;
    let filename = src.file_name().ok_or_else(|| anyhow::anyhow!("invalid filename"))?;
    let mut dst = dest_dir.join(filename);
    if dst.exists() {
        let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("file");
        let ext  = src.extension().and_then(|s| s.to_str()).unwrap_or("");
        let mut n = 1u32;
        loop {
            dst = dest_dir.join(format!("{}_{}.{}", stem, n, ext));
            if !dst.exists() { break; }
            n += 1;
        }
    }
    std::fs::rename(src, &dst)?;
    Ok(())
}

fn collect_images(folder: &Path) -> Vec<PathBuf> {
    collect_images_depth(folder, false)
}

fn collect_images_depth(folder: &Path, recursive: bool) -> Vec<PathBuf> {
    let extensions = ["jpg", "jpeg", "png", "webp", "bmp", "gif"];
    let max_depth = if recursive { usize::MAX } else { 1 };
    walkdir::WalkDir::new(folder)
        .max_depth(max_depth)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| extensions.contains(&s.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn compute_file_hash(path: &Path) -> Result<String> {
    let data = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let result = hasher.finalize();
    Ok(result.iter().map(|b| format!("{:02x}", b)).collect())
}

/// dHash (difference hash): 9×8 grayscale → 64-bit hash
fn compute_dhash(img: &image::DynamicImage) -> u64 {
    let small = img
        .resize_exact(9, 8, FilterType::Lanczos3)
        .grayscale();
    let pixels = small.to_luma8().into_raw();
    let mut hash: u64 = 0;
    for row in 0..8usize {
        for col in 0..8usize {
            let left = pixels[row * 9 + col] as i32;
            let right = pixels[row * 9 + col + 1] as i32;
            if left > right {
                hash |= 1u64 << (row * 8 + col);
            }
        }
    }
    hash
}

fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

fn find_root(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        let px = parent[x];
        parent[x] = parent[px];
        x = parent[x];
    }
    x
}

fn group_by_edges(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    for &(a, b) in edges {
        let ra = find_root(&mut parent, a);
        let rb = find_root(&mut parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find_root(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    groups.into_values().filter(|g| g.len() >= 2).collect()
}

/// 해상도×크기 기준으로 보존할 이미지 인덱스 결정
fn determine_keeper(images: &[DupImageInfo]) -> usize {
    images
        .iter()
        .enumerate()
        .max_by_key(|(_, img)| img.size + (img.width as u64 * img.height as u64) * 1000)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Tauri commands ────────────────────────────────────────────────────────────

#[tauri::command]
async fn check_ollama() -> bool {
    let client = reqwest::Client::new();
    client
        .get(format!("{}/api/tags", OLLAMA_URL))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

#[tauri::command]
async fn list_ollama_models() -> Vec<String> {
    let client = reqwest::Client::new();
    let Ok(resp) = client
        .get(format!("{}/api/tags", OLLAMA_URL))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    else {
        return vec![];
    };
    let Ok(tags) = resp.json::<OllamaTagsResponse>().await else {
        return vec![];
    };
    // Vision-capable model keywords
    let vision_keys = ["llava", "moondream", "minicpm", "bakllava", "cogvlm",
                       "phi3.5-vision", "qwen2-vl", "qwen2.5vl", "qwen3-vl",
                       "llama3.2-vision", "llama4", "internvl", "gemma3",
                       "mistral-small3.2", "granite3.2-vision", "kimi"];
    let mut vision: Vec<String> = tags.models.iter()
        .map(|m| m.name.clone())
        .filter(|n| vision_keys.iter().any(|k| n.to_lowercase().contains(k)))
        .collect();
    if vision.is_empty() {
        // Fallback: return all models
        vision = tags.models.iter().map(|m| m.name.clone()).collect();
    }
    vision
}

#[tauri::command]
async fn classify_images(
    app: AppHandle,
    folder_path: String,
    dest_2d: String,
    dest_3d: String,
    dest_unknown: String,
    realtime: bool,
    model_name: Option<String>,
    control_state: State<'_, ControlState>,
) -> Result<(), String> {
    let model_name = model_name.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let folder = PathBuf::from(&folder_path);
    let images = collect_images(&folder);
    let total = images.len();

    if total == 0 {
        return Err("선택한 폴더에 이미지가 없습니다.".to_string());
    }

    // Pre-create destination dirs when realtime mode is on
    if realtime {
        for dest in [&dest_2d, &dest_3d, &dest_unknown] {
            std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;
        }
    }

    let ctrl = ClassificationControl::new();
    *control_state.0.lock().await = Some(ctrl.clone());

    let client     = Arc::new(reqwest::Client::new());
    let semaphore  = Arc::new(Semaphore::new(3));
    let start      = Instant::now();
    let current    = Arc::new(AtomicUsize::new(0));
    let model_name = Arc::new(model_name);
    let dest_2d    = Arc::new(dest_2d);
    let dest_3d    = Arc::new(dest_3d);
    let dest_unknown = Arc::new(dest_unknown);

    let mut handles = vec![];

    for image_path in images {
        let client       = client.clone();
        let sem          = semaphore.clone();
        let app          = app.clone();
        let current      = current.clone();
        let ctrl         = ctrl.clone();
        let model_name   = model_name.clone();
        let dest_2d      = dest_2d.clone();
        let dest_3d      = dest_3d.clone();
        let dest_unknown = dest_unknown.clone();

        let handle = tokio::spawn(async move {
            if ctrl.wait_if_paused().await { return; }
            let _permit = sem.acquire().await.unwrap();
            if ctrl.wait_if_paused().await { return; }

            let thumb_b64 = make_thumbnail_b64(&image_path, 120).unwrap_or_default();
            let ollama_b64 = match make_ollama_b64(&image_path) {
                Ok(b) => b,
                Err(_) => return,
            };

            if ctrl.cancelled.load(Ordering::Relaxed) { return; }

            let category = classify_with_ollama(&client, &ollama_b64, &model_name)
                .await
                .unwrap_or(Category::Unknown);

            if ctrl.cancelled.load(Ordering::Relaxed) { return; }

            // Realtime mode: move to the specified destination folder immediately
            if realtime {
                let dest_dir = match category {
                    Category::TwoD    => dest_2d.as_str(),
                    Category::ThreeD  => dest_3d.as_str(),
                    Category::Unknown => dest_unknown.as_str(),
                };
                let _ = move_file_to_dir(&image_path, Path::new(dest_dir));
            }

            let idx     = current.fetch_add(1, Ordering::Relaxed) + 1;
            let elapsed = start.elapsed().as_secs_f64();

            let _ = app.emit("classification-progress", ProgressEvent {
                current: idx,
                total,
                filename: image_path.to_string_lossy().to_string(),
                category: category.folder_name().to_string(),
                thumbnail: thumb_b64,
                elapsed_secs: elapsed,
            });
        });

        handles.push(handle);
    }

    for h in handles { let _ = h.await; }
    *control_state.0.lock().await = None;
    Ok(())
}

#[tauri::command]
async fn pause_classification(control_state: State<'_, ControlState>) -> Result<(), String> {
    if let Some(ctrl) = control_state.0.lock().await.as_ref() {
        ctrl.paused.store(true, Ordering::Relaxed);
    }
    Ok(())
}

#[tauri::command]
async fn resume_classification(control_state: State<'_, ControlState>) -> Result<(), String> {
    if let Some(ctrl) = control_state.0.lock().await.as_ref() {
        ctrl.paused.store(false, Ordering::Relaxed);
        ctrl.resume_notify.notify_waiters();
    }
    Ok(())
}

#[tauri::command]
async fn cancel_classification(control_state: State<'_, ControlState>) -> Result<(), String> {
    if let Some(ctrl) = control_state.0.lock().await.as_ref() {
        ctrl.cancelled.store(true, Ordering::Relaxed);
        // Wake up any paused tasks so they can exit
        ctrl.paused.store(false, Ordering::Relaxed);
        ctrl.resume_notify.notify_waiters();
    }
    Ok(())
}

#[tauri::command]
async fn apply_moves(
    dest_2d: String,
    dest_3d: String,
    dest_unknown: String,
    results: Vec<MoveItem>,
) -> Result<(), String> {
    // Ensure destination directories exist
    for dest in [&dest_2d, &dest_3d, &dest_unknown] {
        std::fs::create_dir_all(dest).map_err(|e| e.to_string())?;
    }

    for item in results {
        let src = PathBuf::from(&item.path);
        if !src.exists() { continue; }

        let dest_dir = match item.category.as_str() {
            "2d" => dest_2d.as_str(),
            "3d" => dest_3d.as_str(),
            _    => dest_unknown.as_str(),
        };

        move_file_to_dir(&src, Path::new(dest_dir)).map_err(|e| e.to_string())?;
    }

    Ok(())
}

// ── Duplicate detection commands ──────────────────────────────────────────────

#[tauri::command]
async fn scan_duplicates(
    app: AppHandle,
    folder_path: String,
    recursive: bool,
    hash_method: String, // "exact" | "perceptual"
    dup_scan_state: State<'_, DupScanState>,
) -> Result<(), String> {
    let folder = PathBuf::from(&folder_path);
    let images = collect_images_depth(&folder, recursive);
    let total = images.len();

    if total == 0 {
        return Err("선택한 폴더에 이미지가 없습니다.".to_string());
    }

    let cancelled = Arc::new(AtomicBool::new(false));
    *dup_scan_state.0.lock().await = Some(cancelled.clone());

    // Phase 1: 이미지별 메타데이터 + 썸네일 + 해시 수집
    struct ScanItem {
        info: DupImageInfo,
        exact_hash: Option<String>,
        dhash: Option<u64>,
    }

    let mut scan_results: Vec<ScanItem> = Vec::with_capacity(total);

    for (idx, path) in images.iter().enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            *dup_scan_state.0.lock().await = None;
            return Ok(());
        }

        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let _ = app.emit(
            "dup-scan-progress",
            DupScanProgress { current: idx + 1, total, filename: filename.clone() },
        );

        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let img_result = image::open(path);

        let (width, height, thumbnail) = match &img_result {
            Ok(img) => {
                let w = img.width();
                let h = img.height();
                let thumb = img.thumbnail(120, 120);
                let mut buf = Vec::new();
                let _ = thumb.write_to(
                    &mut std::io::Cursor::new(&mut buf),
                    image::ImageFormat::Jpeg,
                );
                (w, h, general_purpose::STANDARD.encode(&buf))
            }
            Err(_) => (0, 0, String::new()),
        };

        let (exact_hash, dhash) = if hash_method == "exact" {
            (compute_file_hash(path).ok(), None)
        } else {
            let dh = img_result.as_ref().ok().map(compute_dhash);
            (None, dh)
        };

        scan_results.push(ScanItem {
            info: DupImageInfo { path: path.to_string_lossy().to_string(), filename, size, width, height, thumbnail },
            exact_hash,
            dhash,
        });
    }

    *dup_scan_state.0.lock().await = None;

    // Phase 2: 중복 그룹 분류
    let index_groups: Vec<Vec<usize>> = if hash_method == "exact" {
        let mut hash_map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, item) in scan_results.iter().enumerate() {
            if let Some(h) = &item.exact_hash {
                hash_map.entry(h.clone()).or_default().push(i);
            }
        }
        hash_map.into_values().filter(|g| g.len() >= 2).collect()
    } else {
        let threshold = 5u32;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..scan_results.len() {
            for j in (i + 1)..scan_results.len() {
                if let (Some(h1), Some(h2)) = (scan_results[i].dhash, scan_results[j].dhash) {
                    if hamming_distance(h1, h2) <= threshold {
                        edges.push((i, j));
                    }
                }
            }
        }
        group_by_edges(scan_results.len(), &edges)
    };

    let mut result: Vec<DupGroup> = index_groups
        .into_iter()
        .enumerate()
        .map(|(id, group_indices)| {
            let group_images: Vec<DupImageInfo> = group_indices.iter().map(|&i| scan_results[i].info.clone()).collect();
            let keep_idx = determine_keeper(&group_images);
            DupGroup { id, images: group_images, keep_idx }
        })
        .collect();

    // 중복 수 많은 그룹 먼저
    result.sort_by(|a, b| b.images.len().cmp(&a.images.len()));

    // 그룹을 하나씩 이벤트로 emit (실시간 표시)
    for group in result {
        let _ = app.emit("dup-group-found", group);
    }
    Ok(())
}

#[tauri::command]
async fn cancel_dup_scan(dup_scan_state: State<'_, DupScanState>) -> Result<(), String> {
    if let Some(c) = dup_scan_state.0.lock().await.as_ref() {
        c.store(true, Ordering::Relaxed);
    }
    Ok(())
}

#[tauri::command]
async fn apply_duplicate_moves(
    to_move: Vec<String>,
    dest_folder: String,
) -> Result<(), String> {
    std::fs::create_dir_all(&dest_folder).map_err(|e| e.to_string())?;
    for path_str in to_move {
        let src = PathBuf::from(&path_str);
        if src.exists() {
            move_file_to_dir(&src, Path::new(&dest_folder)).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

// ── App entry ─────────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ControlState(Mutex::new(None)))
        .manage(DupScanState(Mutex::new(None)))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .invoke_handler(tauri::generate_handler![
            check_ollama,
            list_ollama_models,
            classify_images,
            pause_classification,
            resume_classification,
            cancel_classification,
            apply_moves,
            scan_duplicates,
            cancel_dup_scan,
            apply_duplicate_moves,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
