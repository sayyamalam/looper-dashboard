-- init.sql

-- 1) Tabelle anlegen (Schema identisch zu model.py in FastAPI)
CREATE TABLE IF NOT EXISTS maschinendata (
    id                SERIAL PRIMARY KEY,
    _time             TIMESTAMP NOT NULL,
    "AggHoeheIst"     DOUBLE PRECISION,
    "AutomTurmverstellungEin" DOUBLE PRECISION,
    "AutomatikLaeuft" DOUBLE PRECISION,
    "BSR_Satznummer"  DOUBLE PRECISION,
    "BSR_Schnittstaerke" DOUBLE PRECISION,
    "BSR_StueckzahlIst" DOUBLE PRECISION,
    "BSR_StueckzahlSoll" DOUBLE PRECISION,
    "BSVE_LaengeIst"  DOUBLE PRECISION,
    "BSVE_LaengeSoll" DOUBLE PRECISION,
    "BSVE_Satznummer" DOUBLE PRECISION,
    "BSVE_Schnittstaerke" DOUBLE PRECISION,
    "BSVE_StueckzahlIst" DOUBLE PRECISION,
    "BSVE_StueckzahlSoll" DOUBLE PRECISION,
    "BetriebsartBSR"  DOUBLE PRECISION,
    "BetriebsartBSVE" DOUBLE PRECISION,
    "BetriebsartHalbautomat" DOUBLE PRECISION,
    "BetriebsartManuell" DOUBLE PRECISION,
    "BetriebsartService" DOUBLE PRECISION,
    "Blocklaenge120m" DOUBLE PRECISION,
    "DrwHoeheIst"     DOUBLE PRECISION,
    "DrwHoeheOffsetAutomatik" DOUBLE PRECISION,
    "HTBVIst"         DOUBLE PRECISION,
    "HTB_OffsetSoll"  DOUBLE PRECISION,
    "HTB_StromIst"    DOUBLE PRECISION,
    "HTB_TemperaturIst" DOUBLE PRECISION,
    "HTB_VIst"        DOUBLE PRECISION,
    "Halbautomat_Satznummer" DOUBLE PRECISION,
    "Halbautomat_Schnittstaerke" DOUBLE PRECISION,
    "Halbautomat_StueckzahlIst" DOUBLE PRECISION,
    "Halbautomat_StueckzahlSoll" DOUBLE PRECISION,
    "Reserve01"       DOUBLE PRECISION,
    "Reserve02"       DOUBLE PRECISION,
    "Reserve03"       DOUBLE PRECISION,
    "Reserve04"       DOUBLE PRECISION,
    "SAOAbstandIst"   DOUBLE PRECISION,
    "SAUAbstandIst"   DOUBLE PRECISION,
    "VDrwIst"         DOUBLE PRECISION,
    "VMesserIst"      DOUBLE PRECISION,
    "VMesserSoll"     DOUBLE PRECISION,
    "VWicklerIst"     DOUBLE PRECISION,
    "WinkelIst"       DOUBLE PRECISION,
    "WinkelSoll"      DOUBLE PRECISION
    );

-- 2) Optional: Index
CREATE INDEX IF NOT EXISTS ix_maschinendata__time ON maschinendata ("_time");

-- 3) COPY-Daten aus CSV
--    (Nur ein einziges COPY-Statement, 
--     hier mit Komma als Spaltentrenner und Header=true)
COPY maschinendata(
    "_time",
    "AggHoeheIst",
    "AutomTurmverstellungEin",
    "AutomatikLaeuft",
    "BSR_Satznummer",
    "BSR_Schnittstaerke",
    "BSR_StueckzahlIst",
    "BSR_StueckzahlSoll",
    "BSVE_LaengeIst",
    "BSVE_LaengeSoll",
    "BSVE_Satznummer",
    "BSVE_Schnittstaerke",
    "BSVE_StueckzahlIst",
    "BSVE_StueckzahlSoll",
    "BetriebsartBSR",
    "BetriebsartBSVE",
    "BetriebsartHalbautomat",
    "BetriebsartManuell",
    "BetriebsartService",
    "Blocklaenge120m",
    "DrwHoeheIst",
    "DrwHoeheOffsetAutomatik",
    "HTBVIst",
    "HTB_OffsetSoll",
    "HTB_StromIst",
    "HTB_TemperaturIst",
    "HTB_VIst",
    "Halbautomat_Satznummer",
    "Halbautomat_Schnittstaerke",
    "Halbautomat_StueckzahlIst",
    "Halbautomat_StueckzahlSoll",
    "Reserve01",
    "Reserve02",
    "Reserve03",
    "Reserve04",
    "SAOAbstandIst",
    "SAUAbstandIst",
    "VDrwIst",
    "VMesserIst",
    "VMesserSoll",
    "VWicklerIst",
    "WinkelIst",
    "WinkelSoll"
)
FROM '/docker-entrypoint-initdb.d/2024_11_12_auszug.csv'
WITH (
    FORMAT csv,
    HEADER true,
    DELIMITER ','
);
