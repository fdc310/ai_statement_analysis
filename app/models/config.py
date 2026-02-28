from sqlalchemy import Column, Integer, String, JSON
from app.core.database import Base

class SystemConfig(Base):
    __tablename__ = "system_configs"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(String, nullable=True)
