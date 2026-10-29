# Thirty thousand rows, five hundred and seventy patients

Exploratory data analysis of the liver patient records (BCSE331L Exploratory Data Analysis course project, VIT Vellore).
Shlok Kumar Goenka (23BDS0316), faculty: Balaji G N.

The 30,691-row course file turns out to be built from the 583-record Indian Liver Patient Dataset (UCI) by
copying each column from a random patient of the same class. Lab values therefore leak the label: random forest
scores 99.7% on a random split, while on the 570 unique real patients the best models (RBF SVM, logistic
regression) reach a cross-validated ROC-AUC of about 0.73.

## Contents
- `report/Liver_EDA_Report.pdf` (and `.docx`): the full report, chapters follow the eight syllabus modules
- `report/Liver_EDA_Presentation.pptx`: 7-slide deck, narration in the speaker notes (`report/video_script.txt`)
- `src/analysis.py`: every analysis step; writes `figures/` and `results/results.json`
- `src/build_report.py`, `src/docx_theme.py`, `src/build_deck.py`: generate the report and the deck from the results
- `data/liver.csv` (course file), `data/ilpd_original_583.csv` (UCI release, used for the provenance check)

## Reproduce
```
pip install -r requirements.txt
python src/analysis.py
python src/build_report.py
python src/build_deck.py
```
