# Image Auto Classifier — 프로젝트 컨텍스트

## 프로젝트 개요
Tauri v2 + React + TypeScript 기반 크로스플랫폼 네이티브 데스크탑 앱.
Ollama 로컬 AI(LLaVA 등 비전 모델)로 이미지를 2D(일러스트/애니/만화) / 3D(실사/사진) / 미분류로 자동 분류하고, 지정한 폴더로 이동시킨다.

## 기술 스택
- **프론트엔드**: React 18 + TypeScript + Vite (src/App.tsx, src/App.css)
- **백엔드**: Rust + Tauri v2 (src-tauri/src/lib.rs)
- **AI**: Ollama HTTP API (127.0.0.1:11434) — 비전 모델 사용
- **상태 저장**: tauri-plugin-store → settings.json (폴더 경로 영구 저장)
- **동시성**: tokio Semaphore (최대 3개 동시 분류)

## 구현된 기능 (완성)
1. 이미지 폴더 스캔 + Ollama 비전 모델로 2D/3D/미분류 자동 분류
2. 실시간 썸네일 미리보기 + 진행 표시 (progress bar, 남은 시간)
3. 일시정지 / 재개 / 취소 기능 (AtomicBool + Notify 기반)
4. 분류 완료 후 결과 검토 화면 (카테고리 개별 변경, 다중 선택, 드래그 선택, 일괄 변경)
5. 필터 버튼 (전체/2D/3D/미분류) sticky 고정 — 스크롤해도 상단에 유지
6. 4개 폴더 직접 지정 (스캔 소스 / 2D 저장 / 3D 저장 / 미분류 저장)
7. 폴더 경로 앱 재시작 후에도 기억 (tauri-plugin-store)
8. AI 모델 선택 드롭다운 + 🔄 새로고침 버튼 (Ollama에서 비전 모델 자동 감지)
9. 앱 시작 시 Ollama 설치 안내 팝업 (다시 보지 않기 옵션 포함)
10. 검토 후 파일 이동 적용 (apply_moves) 또는 실시간 즉시 이동 모드

## 파일 구조
```
image-classifier/
├── src/
│   ├── App.tsx          # React 프론트엔드 전체
│   └── App.css          # 다크 테마 스타일
├── src-tauri/
│   ├── src/
│   │   └── lib.rs       # Rust 백엔드 (Tauri 커맨드, Ollama 연동, 파일 이동)
│   ├── capabilities/
│   │   └── default.json # Tauri 권한 설정
│   ├── tauri.conf.json  # 앱 설정 (identifier: com.imageclassifier.desktop)
│   └── Cargo.toml       # Rust 의존성
├── .github/
│   └── workflows/
│       └── build-mac.yml # GitHub Actions macOS 자동 빌드
├── CLAUDE.md            # 이 파일
└── .gitignore
```

## 주요 Rust 구조체 / 커맨드
- `ClassificationControl` — pause/cancel용 AtomicBool + Notify
- `ControlState(Mutex<Option<Arc<ClassificationControl>>>)` — Tauri managed state
- `classify_images(app, folder_path, dest_2d, dest_3d, dest_unknown, realtime, model_name, control_state)`
- `apply_moves(dest_2d, dest_3d, dest_unknown, results: Vec<MoveItem>)`
- `list_ollama_models()` — Ollama API에서 비전 모델 필터링
- `move_file_to_dir()` — 파일 이동 (충돌 시 자동 rename)

## 비전 모델 감지 키워드 (lib.rs)
```rust
let vision_keys = ["llava", "moondream", "minicpm", "bakllava", "cogvlm",
                   "phi3.5-vision", "qwen2-vl", "qwen2.5vl", "qwen3-vl",
                   "llama3.2-vision", "llama4", "internvl", "gemma3",
                   "mistral-small3.2", "granite3.2-vision", "kimi"];
```

## 추천 AI 모델 (정확도 순)
1. `ollama pull qwen2.5vl` — 6GB, 분류 정확도 최고
2. `ollama pull llama3.2-vision` — 7.8GB, 안정적
3. `ollama pull gemma3:4b` — 3GB, 경량 (느린 PC용)
4. `ollama pull llava` — 4.7GB, 기본

## 개발 실행
```bash
npm install
cargo tauri dev       # 개발 모드
cargo tauri build     # 배포 빌드
```

## 배포 현황
- Windows 11 x64: 빌드 완료 (`src-tauri/target/release/bundle/nsis/*.exe`)
- macOS (arm64/x86_64): GitHub Actions 자동 빌드 설정 완료 (`.github/workflows/build-mac.yml`)

## GitHub
https://github.com/banktree/image-classifier

## 알려진 이슈 / 향후 개선 가능 항목
- Windows 코드 서명 없음 (배포 시 SmartScreen 경고 뜰 수 있음)
- macOS 코드 서명 없음 (배포 시 Gatekeeper 경고 → 우클릭 → 열기로 우회)
- 분류 AI 프롬프트 추가 튜닝 가능
