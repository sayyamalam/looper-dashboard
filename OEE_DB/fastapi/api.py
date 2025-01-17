from fastapi import FastAPI, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, List
from model import MaschinenData
from database import SessionLocal
from joblib import load
from sqlalchemy import func
from datetime import datetime, timedelta
import asyncio
import numpy as np

# ------------------------------------------------------------------------------
# 1) Modelle laden
# ------------------------------------------------------------------------------
random_forest   = load("random-forest-model-smote.pkl")
neural_network  = load("neural-network-model-smote.pkl")  # MLPClassifier

# ------------------------------------------------------------------------------
# 2) CLASS_MAPPING (aktualisiert)
# ------------------------------------------------------------------------------
CLASS_MAPPING = {
    0: "Block Loading",
    1: "Remove Side Skin",
    2: "Gluing",
    3: "Remove Top Skin",
    4: "Production",
    5: "Remove Bottom Skin",
    6: "Idle",
}

# ------------------------------------------------------------------------------
# 3) Spaltenreihenfolge
# ------------------------------------------------------------------------------
COLUMN_ORDER = [
    'AggHoeheIst', 'AutomTurmverstellungEin', 'AutomatikLaeuft',
    'BSR_Satznummer', 'BSR_Schnittstaerke', 'BSR_StueckzahlIst',
    'BSR_StueckzahlSoll', 'BSVE_LaengeIst', 'BSVE_LaengeSoll',
    'BSVE_Satznummer', 'BSVE_Schnittstaerke', 'BSVE_StueckzahlIst',
    'BSVE_StueckzahlSoll', 'BetriebsartBSR', 'BetriebsartBSVE',
    'BetriebsartHalbautomat', 'BetriebsartManuell', 'BetriebsartService',
    'Blocklaenge120m', 'DrwHoeheIst', 'DrwHoeheOffsetAutomatik', 'HTBVIst',
    'HTB_OffsetSoll', 'HTB_StromIst', 'HTB_TemperaturIst', 'HTB_VIst',
    'Halbautomat_Satznummer', 'Halbautomat_Schnittstaerke',
    'Halbautomat_StueckzahlIst', 'Halbautomat_StueckzahlSoll', 'Reserve01',
    'Reserve02', 'Reserve03', 'Reserve04', 'SAOAbstandIst', 'SAUAbstandIst',
    'VDrwIst', 'VMesserIst', 'VMesserSoll', 'VWicklerIst', 'WinkelIst',
    'WinkelSoll'
]

# ------------------------------------------------------------------------------
# 4) FastAPI Initialisierung
# ------------------------------------------------------------------------------
app = FastAPI()

# ------------------------------------------------------------------------------
# 5) Generator-Variable
# ------------------------------------------------------------------------------
row_generator = None

# ------------------------------------------------------------------------------
# 6) DB-Session Handling
# ------------------------------------------------------------------------------
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# ------------------------------------------------------------------------------
# 7) Generator-Funktion
# ------------------------------------------------------------------------------
def row_generator_function(db: Session, start_id: Optional[int] = None):
    query = db.query(MaschinenData)
    if start_id is not None:
        query = query.filter(MaschinenData.id > start_id)
    query = query.order_by(MaschinenData.id.asc())

    for row in query:
        yield row

# ------------------------------------------------------------------------------
# 8) Hilfsfunktion NN (statt decision_function => predict_proba)
# ------------------------------------------------------------------------------
def neural_network_predict_and_proba(X: np.ndarray):
    """
    Für MLPClassifier:
      - preds: predicted labels
      - probas: predict_proba => Wahrscheinlichkeiten je Klasse
    """
    nn_preds  = neural_network.predict(X)
    nn_probas = neural_network.predict_proba(X)
    return nn_preds, nn_probas

# ------------------------------------------------------------------------------
# 9) Endpoint: Einzelner Datensatz + Predictions
# ------------------------------------------------------------------------------
@app.get("/data/get_next_row_with_predictions/")
async def get_next_row_with_predictions(
    db: Session = Depends(get_db),
    start_id: Optional[int] = Query(None, description="Start ID für die Datenabfrage"),
):
    global row_generator

    # Generator (re-)initialisieren
    if row_generator is None or start_id is not None:
        row_generator = row_generator_function(db, start_id=start_id)

    try:
        next_row = next(row_generator)
    except StopIteration:
        return {"message": "No more rows to process"}

    # Daten aus DB (Original für Ausgabe)
    data_dict = next_row.to_dict()

    # 1) Nur die benötigten Spalten in korrekter Reihenfolge entnehmen
    input_data = [[data_dict[col] for col in COLUMN_ORDER]]
    input_arr = np.array(input_data, dtype=np.float32)

    # 2) Vorhersagen (ohne Skaler)
    # a) Random Forest
    rf_pred  = random_forest.predict(input_arr)[0]
    rf_proba = random_forest.predict_proba(input_arr)[0]
    rf_conf  = float(rf_proba.max())

    # b) Neural Network (MLP)
    nn_preds, nn_probas = neural_network_predict_and_proba(input_arr)
    nn_pred  = nn_preds[0]
    nn_proba = nn_probas[0]
    nn_conf  = float(nn_proba.max())

    return {
        "data": data_dict,  # Original-Row bleibt so
        "predictions": {
            "random_forest": {
                "prediction": CLASS_MAPPING[rf_pred],
                "confidence_score": rf_conf,
                "predict_proba": {
                    CLASS_MAPPING[idx]: float(prob) for idx, prob in enumerate(rf_proba)
                },
            },
            "neural_network": {
                "prediction": CLASS_MAPPING[nn_pred],
                "confidence_score": nn_conf,
                "predict_proba": {
                    CLASS_MAPPING[idx]: float(prob) for idx, prob in enumerate(nn_proba)
                },
            },
        },
    }

# ------------------------------------------------------------------------------
# 10) Endpoint: Batch / Zeitfenster
# ------------------------------------------------------------------------------
@app.get("/data/get_predictions_for_timeframe/")
async def get_predictions_for_timeframe(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    start_time: Optional[datetime] = Query(None, description="Start time (YYYY-MM-DDTHH:MM:SS)"),
    end_time:   Optional[datetime] = Query(None, description="End time (YYYY-MM-DDTHH:MM:SS)"),
    chunk_size: int = Query(5000, description="Number of records to process per chunk")
):
    # a) Zeitbereich ermitteln
    if not start_time or not end_time:
        minmax = await asyncio.to_thread(
            lambda: db.query(
                func.min(MaschinenData._time),
                func.max(MaschinenData._time)
            )
            .filter(MaschinenData._time.isnot(None))
            .first()
        )
        start_time = start_time or minmax[0]
        end_time   = end_time   or minmax[1]

    # b) Erforderliche Spalten
    required_cols = ["_time", "AggHoeheIst", "HTBVIst"] + [
        c for c in COLUMN_ORDER 
        if c not in ["_time", "AggHoeheIst", "HTBVIst"]
    ]

    # c) Daten abfragen
    rows = await asyncio.to_thread(
        lambda: db.query(
            *[getattr(MaschinenData, col) for col in required_cols]
        )
        .filter(MaschinenData._time.between(start_time, end_time))
        .order_by(MaschinenData._time.asc())
        .yield_per(chunk_size)
        .all()
    )

    if not rows:
        return {"message": "No data found for the given timeframe."}

    # d) Hilfsfunktion zur Chunk-Verarbeitung
    async def process_chunk(chunk):
        # 1) Nur die Spalten in COLUMN_ORDER als Input-Array
        input_arr = np.array([
            [getattr(row, col) for col in COLUMN_ORDER]
            for row in chunk
        ], dtype=np.float32)

        # 2) Vorhersagen (ohne Skaler)
        rf_preds  = await asyncio.to_thread(lambda: random_forest.predict(input_arr))
        rf_probas = await asyncio.to_thread(lambda: random_forest.predict_proba(input_arr))
        nn_preds, nn_probas = neural_network_predict_and_proba(input_arr)

        # 3) Ergebnisse pro Zeile bauen
        results_chunk = []
        for i, row in enumerate(chunk):
            rf_label = rf_preds[i]
            nn_label = nn_preds[i]
            results_chunk.append({
                "_time": row._time,
                "AggHoeheIst": row.AggHoeheIst,
                "HTBVIst": row.HTBVIst,
                "predictions": {
                    "random_forest": CLASS_MAPPING[rf_label],
                    "neural_network": CLASS_MAPPING[nn_label],
                },
            })
        return results_chunk

    # e) Chunking + parallel
    all_results = []
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    processed_lists = await asyncio.gather(*[process_chunk(ch) for ch in chunks])
    for sublist in processed_lists:
        all_results.extend(sublist)

    # f) Zustandszeiten (Anteile pro vorhergesagtem Zustand)
    total_seconds = (end_time - start_time).total_seconds()

    def calc_shares(pred_list):
        unique_states, counts = np.unique(pred_list, return_counts=True)
        shares_sec = (counts / len(pred_list)) * total_seconds
        out = {}
        for st, sec in zip(unique_states, shares_sec):
            if sec > 0:
                out[st] = timedelta(seconds=int(sec))
        return out

    rf_list = [r["predictions"]["random_forest"] for r in all_results]
    nn_list = [r["predictions"]["neural_network"] for r in all_results]

    rf_shares = calc_shares(rf_list)
    nn_shares = calc_shares(nn_list)

    # g) Antwort
    return {
        "timeframe": {
            "start_time": start_time,
            "end_time":   end_time,
            "duration":   timedelta(seconds=int(total_seconds)),
        },
        "state_time_shares": {
            "random_forest": rf_shares,
            "neural_network": nn_shares,
        },
        "results": all_results
    }
