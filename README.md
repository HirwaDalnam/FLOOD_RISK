# SW27 Flood Forecast — Deployment Guide

## 1. Export your trained model from Colab

Run `export_cell_for_colab.py` as a new cell at the end of your training
notebook (after the model is trained and `rmse` is computed). It produces
`flood_model_export.zip` containing:

- `lstm_model.keras`
- `feature_scaler.pkl`
- `target_scaler.pkl`
- `config.json`
- `seed_data.csv`

Download and unzip it locally — you'll upload these files in step 2.

## 2. Deploy the API to Hugging Face Spaces (free tier)

1. Create a free account at huggingface.co if you don't have one.
2. Create a new Space → choose **Docker** as the Space SDK.
3. Upload these files to the Space's repo (via the web UI or git):
   - `app.py`
   - `requirements.txt`
   - `Dockerfile`
   - `lstm_model.keras`, `feature_scaler.pkl`, `target_scaler.pkl`, `config.json`, `seed_data.csv` (from step 1)
4. The Space will build automatically. Once it says "Running", your API is live at:
   `https://<your-username>-<space-name>.hf.space`
5. Test it directly in the browser:
   `https://<your-username>-<space-name>.hf.space/health`
   You should see `{"status": "ok", "is_live_feed": false, ...}`

## 3. Embed the widget in Blogger

1. Open `blogger_widget.html`, find the line:
   ```js
   var API_BASE_URL = "https://YOUR-SPACE-URL-HERE.hf.space";
   ```
   Replace it with your actual Space URL from step 2.
2. In Blogger: create or edit a post/page → switch the editor to **HTML view**
   (not the visual "Compose" view) → paste the entire contents of
   `blogger_widget.html`.
3. Publish. The widget will show a date/time picker and horizon selector;
   clicking "Run Forecast" calls your live API and renders the chart in-page.

## Honest labeling note

The widget explicitly tells visitors whether they're seeing historical
scenario data or a live feed (`is_live_feed` from the API), and flags when
the seed data is stale relative to the requested start time. Keep this
visible — don't remove it — until real telemetry is wired in via
`load_seed_data()` in `app.py`.

## When you get a live sensor feed

Only `load_seed_data()` in `app.py` needs to change — replace the CSV read
with a call to whatever endpoint/database serves live SW27 readings, keeping
the same return shape (DataFrame indexed by naive datetime, columns matching
`FEATURE_COLS`). Also flip `IS_LIVE_FEED = True`. Nothing else in the API or
the widget needs to change.
