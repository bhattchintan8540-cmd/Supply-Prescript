# Convenience targets for beginners — run from the project root.
# Example:  make setup && make train && make api

.PHONY: setup data explore train demo demo-ui api test retrain evaluate smoke phase2 phase2-export clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	python week1/generate_mock_data.py

explore:
	python week1/explore_data.py

train:
	python week1/train_model.py

# Presentation demos (model comparison + prescribe story)
demo: train
	python week1/demo_model.py
	python week2/demo_prescribe.py

demo-ui: train
	@echo "Open http://127.0.0.1:8000/ui/  — click Demo A / B / C buttons"
	uvicorn week3.main:app --reload

api:
	uvicorn week3.main:app --reload

test:
	python -m pytest -q

retrain:
	python week4/retrain.py

evaluate:
	python week1/evaluate_xgboost.py

smoke:
	python week5/smoke_loop.py

# Weeks 6 and 7 — Phase 2 stopped at the midpoint
phase2:
	python week6/phase2_midpoint.py
	python week7/recommend.py

# Export only: grid CSV + recommendation JSON under data/
phase2-export: phase2
	@echo "Exported data/phase2_grid.csv and data/phase2_recommendation.json"

clean:
	rm -f data/*.csv data/*.joblib data/*.db data/metrics.json
	rm -rf data/ml_evaluation exports
	# Keep committed presentation figures under docs/figures/.
