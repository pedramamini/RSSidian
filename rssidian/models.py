from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

Base = declarative_base()


class Feed(Base):
    """Model for RSS feed subscriptions."""
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    website_url = Column(String(512), nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    last_checked = Column(DateTime, default=datetime.utcnow)
    muted = Column(Boolean, default=False)
    muted_reason = Column(Text, nullable=True)  # Reason for muting the feed
    peer_through = Column(Boolean, default=False)  # Whether to peer through aggregator to origin article
    error_count = Column(Integer, default=0)  # Count of consecutive errors
    no_entries_count = Column(Integer, default=0)  # Count of consecutive checks with no entries
    last_error = Column(Text, nullable=True)
    
    # Statistics
    article_count = Column(Integer, default=0)  # Total count of ingested articles
    avg_quality_score = Column(Float, nullable=True)  # Average quality score of articles
    quality_tier_counts = Column(Text, nullable=True)  # JSON string of quality tier counts {"S": 5, "A": 10, ...}
    
    # Relationships
    articles = relationship("Article", back_populates="feed", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Feed(id={self.id}, title='{self.title}', muted={self.muted}, peer_through={self.peer_through})>"


class Article(Base):
    """Model for articles from RSS feeds."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    title = Column(String(512), nullable=False)
    url = Column(String(512), nullable=False)
    guid = Column(String(512), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    author = Column(String(255), nullable=True)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)
    quality_tier = Column(String(1), nullable=True)  # S, A, B, C, D
    quality_score = Column(Integer, nullable=True)  # 1-100
    labels = Column(String(255), nullable=True)
    embedding_generated = Column(Boolean, default=False)
    embedding_vector = Column(Text, nullable=True)  # JSON-serialized embedding vector
    word_count = Column(Integer, nullable=True)
    jina_enhanced = Column(Boolean, default=False)  # Flag to indicate if content was fetched from Jina.ai
    
    # Relationships
    feed = relationship("Feed", back_populates="articles")

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:30]}...', quality_tier='{self.quality_tier or 'None'}')>"


def init_db(db_path: str) -> Session:
    """Initialize database and return session."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    # Migrate: add embedding_vector column if missing (existing databases)
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("articles")]
    if "embedding_vector" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE articles ADD COLUMN embedding_vector TEXT"))
            conn.commit()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def get_db_session(db_path: str) -> Session:
    """Get a database session."""
    engine = create_engine(f"sqlite:///{db_path}")

    # Migrate: add embedding_vector column if missing (existing databases)
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("articles")]
    if "embedding_vector" not in columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE articles ADD COLUMN embedding_vector TEXT"))
            conn.commit()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()