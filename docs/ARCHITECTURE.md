# Synth — Architecture Deep-Dive

## System Overview

```
            ┌─────────────────────────┐
            │      CLI Layer          │
            │  Typer + Rich TUI       │
            │  cli/main.py            │
            │  cli/display.py         │
            └──────────┬──────────────┘
                       │
            ┌──────────▼──────────────┐
            │      Core Layer         │
            │  ┌───────┐ ┌─────────┐  │
            │  │  OCR  │→│  Auth   │  │
            │  │ocr.py │ │ auth.py │  │
            │  └───┬───┘ └────┬────┘  │
            │      └──────────┘       │
            │      device.py          │
            └─────────────────────────┘
```

## Data Flow

```
image.png → OpenCV Preprocess → EasyOCR Extract → AI Detect → Rich Display
              (grayscale,         (readtext)       (Strategy)   (table,
               threshold,                                        panel)
               denoise)
```

## Hardware Auto-Detection

**File:** `src/synth/core/device.py`

```python
def detect_device() -> str:
    if torch.cuda.is_available():          return "cuda"   # NVIDIA
    elif torch.backends.mps.is_available(): return "mps"   # Apple Silicon
    else:                                   return "cpu"   # Fallback
```

| Component | CUDA | MPS | CPU |
|---|:---:|:---:|:---:|
| EasyOCR | ✅ gpu=True | ❌ CPU fallback | ✅ |
| HuggingFace Transformers | ✅ | ✅ Native | ✅ |
| OpenCV | N/A | N/A | ✅ |

**Key nuance:** On Apple Silicon, OCR runs on CPU (EasyOCR limitation) while the transformer model runs on GPU via MPS.

## OCR Pipeline — OpenCV Filters

**File:** `src/synth/core/ocr.py`

### Stage 1: Grayscale

`cv2.cvtColor(img, COLOR_BGR2GRAY)` — Reduces 3 channels to 1. Eliminates colour noise and is required by thresholding.

### Stage 2: Adaptive Thresholding

```python
cv2.adaptiveThreshold(gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize=11, C=2)
```

**Why adaptive over global?** Global thresholding fails on uneven lighting, shadows, and coloured paper. Adaptive calculates a local threshold per 11×11 pixel neighbourhood, handling all of these.

### Stage 3: Denoising

`cv2.fastNlMeansDenoising(thresh, h=10)` — Non-local means denoising. Compares patches across the image to suppress scanner noise and JPEG artifacts without destroying text edges.

### Config

All parameters are tunable via the frozen `PreprocessConfig` dataclass.

## Strategy Pattern — Authentication

**File:** `src/synth/core/auth.py`

### Class Hierarchy

```
    BaseAuthenticator (ABC)
    ├── detect(text) → AuthResult
    └── name → str
          │
   ┌──────┴──────┐
   │              │
LocalHF       UniversalAPI
(HuggingFace)   (Any HTTP)
                  ├── OpenAI
                  ├── Anthropic
                  ├── Ollama
                  └── Custom
```

### Why Strategy Pattern?

The detection backend is the most variable component. Strategy Pattern decouples "what to detect" from "how to detect" — users swap backends via `--engine local|api` without touching code.

### AuthResult — Universal Output

```python
@dataclass(frozen=True)
class AuthResult:
    score: float      # 0.0 (human) → 1.0 (AI)
    verdict: str      # "human" | "ai" | "mixed"
    reasoning: str
    model: str
```

### LocalHFAuthenticator

- Default model: `roberta-base-openai-detector`
- Supports heterogeneous label formats (LABEL_0/1, Real/Fake)
- Defers `transformers` import for fast cold-starts
- Places model on best device via `get_torch_device()`

### UniversalAPIAuthenticator

- Generic `httpx.Client` wrapper
- Config from `.env`, JSON, or programmatic `APIEndpointConfig`
- `{text}` placeholder substitution in payload templates
- Dot-notation response paths (e.g. `choices.0.message.content`)
- Context-manager for clean HTTP shutdown

### DetectorFactory

```python
DetectorFactory.create("local")   # HuggingFace
DetectorFactory.create("api")     # Remote HTTP
DetectorFactory.register("custom", MyClass)  # Plugin
```

## Extension Points

1. **New strategy** — Subclass `BaseAuthenticator`, register with `DetectorFactory.register()`
2. **New languages** — `synth verify --lang en,ja,ko` (80+ EasyOCR languages)
3. **Tune preprocessing** — Pass a custom `PreprocessConfig` to `DocumentScanner`
