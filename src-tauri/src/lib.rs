use anyhow::Result;
use base64::{engine::general_purpose, Engine};
use image::imageops::FilterType;
use serde::{Deserialize, Serialize};
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
    let extensions = ["jpg", "jpeg", "png", "webp", "bmp", "gif"];
    walkdir::WalkDir::new(folder)
        .max_depth(1)
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

// ── App entry ─────────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ControlState(Mutex::new(None)))
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
