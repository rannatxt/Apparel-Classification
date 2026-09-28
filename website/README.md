# Apparel Classification Project — Website

This folder contains the project showcase website.

## Files

| File | Purpose |
|------|---------|
| `index.html` | The single-page website |
| `style.css` | All styling (dark/light mode, responsive) |
| `script.js` | Reads `results.json` and renders all data dynamically |
| `results.json` | **All numbers come from here** — edit this file to update the site |
| `assets/` | Place confusion matrices, Grad-CAM images, and edge-case sample images here |

---

## How to Open Locally

Just double-click `index.html`. It opens in any modern browser.

> **Note:** The browser needs to be able to load `results.json` via `fetch()`.  
> If you see no data, open with a local server instead:
>
> ```bash
> # Option A — Python
> py -3 -m http.server 8080
> # Then visit: http://localhost:8080
>
> # Option B — VS Code Live Server extension (right-click index.html → Open with Live Server)
> ```

---

## How to Update Results After Training

Open `results.json` and fill in the model metrics:

```json
"models": {
  "MobileNetV3": {
    "accuracy":         "92.4%",
    "precision":        "91.8%",
    "recall":           "92.1%",
    "f1":               "91.9%",
    "balanced_accuracy":"91.5%"
  },
  ...
}
```

Save the file — refresh the browser and the table updates automatically.

---

## How to Add Images (Confusion Matrices, Grad-CAM, Edge Cases)

1. Copy your images into the `assets/` folder.
2. In `results.json`, add entries like:

```json
"assets": {
  "confusion_matrices": [
    { "src": "assets/cm_mobilenetv3.png", "caption": "MobileNetV3 Confusion Matrix" }
  ],
  "gradcam_examples": [
    { "src": "assets/gradcam_indian.png", "caption": "Indian — Saree (ResNet50)" }
  ]
}
```

3. The website will display them automatically.

---

## How to Publish (GitHub Pages)

1. Push the `website/` folder contents to a GitHub repository.
2. Go to **Settings → Pages → Source → main branch → /root**.
3. Your site will be live at `https://<your-username>.github.io/<repo-name>/`.

---

## Dataset Stats (from validate_dataset.py — 16/16 PASS)

| Class | Total | Train | Val | Test |
|-------|-------|-------|-----|------|
| Indian | 1,494 | 1,046 | 224 | 224 |
| Western | 1,500 | 1,050 | 225 | 225 |
| Indo-Western | 200 | 140 | 30 | 30 |
| **Total** | **3,194** | **2,236** | **479** | **479** |

Seed: 42 · Image size: 224×224 JPEG · Duplicates removed: 6 (1 exact + 5 near)
