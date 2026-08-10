# AI Outfit Recommendation

Streamlit dissertation prototype combining SigLIP garment recognition, colour
analysis, live weather and outfit recommendation.

Recommendation has two safe execution paths:

- `trained_siglip_compatibility_ranker_v1` when a trained compatibility
  checkpoint and embedded product catalogue are available;
- `lightweight_content_ranker_v1` as an automatic, interpretable fallback.

Training and data preparation commands are documented in
[`recommendation_training/README.md`](recommendation_training/README.md).

Run the application with:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
