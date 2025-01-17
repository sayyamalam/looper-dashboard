from database import Base 
from sqlalchemy import Column, Float, Integer, DateTime

class MaschinenData(Base):
    __tablename__ = "maschinendata"
    
    id = Column(Integer, primary_key=True)  # Dummy-Primärschlüssel
    _time = Column(DateTime, index= True)  # Neue Spalte für Zeitstempel
    AggHoeheIst = Column(Float)
    AutomTurmverstellungEin = Column(Float)
    AutomatikLaeuft = Column(Float)
    BSR_Satznummer = Column(Float)
    BSR_Schnittstaerke = Column(Float)
    BSR_StueckzahlIst = Column(Float)
    BSR_StueckzahlSoll = Column(Float)
    BSVE_LaengeIst = Column(Float)
    BSVE_LaengeSoll = Column(Float)
    BSVE_Satznummer = Column(Float)
    BSVE_Schnittstaerke = Column(Float)
    BSVE_StueckzahlIst = Column(Float)
    BSVE_StueckzahlSoll = Column(Float)
    BetriebsartBSR = Column(Float)
    BetriebsartBSVE = Column(Float)
    BetriebsartHalbautomat = Column(Float)
    BetriebsartManuell = Column(Float)
    BetriebsartService = Column(Float)
    Blocklaenge120m = Column(Float)
    DrwHoeheIst = Column(Float)
    DrwHoeheOffsetAutomatik = Column(Float)
    HTBVIst = Column(Float)
    HTB_OffsetSoll = Column(Float)
    HTB_StromIst = Column(Float)
    HTB_TemperaturIst = Column(Float)
    HTB_VIst = Column(Float)
    Halbautomat_Satznummer = Column(Float)
    Halbautomat_Schnittstaerke = Column(Float)
    Halbautomat_StueckzahlIst = Column(Float)
    Halbautomat_StueckzahlSoll = Column(Float)
    Reserve01 = Column(Float)
    Reserve02 = Column(Float)
    Reserve03 = Column(Float)
    Reserve04 = Column(Float)
    SAOAbstandIst = Column(Float)
    SAUAbstandIst = Column(Float)
    VDrwIst = Column(Float)
    VMesserIst = Column(Float)
    VMesserSoll = Column(Float)
    VWicklerIst = Column(Float)
    WinkelIst = Column(Float)
    WinkelSoll = Column(Float)

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}