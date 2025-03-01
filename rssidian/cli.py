import os
import sys
import click
import uvicorn
from datetime import datetime, timedelta
from typing import Optional
import logging

from .config import Config, init_config
from .models import init_db, get_db_session, Feed, Article
from .core import RSSProcessor
from .opml import import_feeds_from_opml
from .backup import get_backup_list, create_backup, restore_backup
from .markdown import write_digest_to_obsidian
from .api import create_api

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def cli(ctx):
    """RSSidian - Bridge between RSS feeds and Obsidian with AI-powered features."""
    # Create a config object and add it to context
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config()


@cli.command()
def init():
    """Initialize configuration files and database."""
    init_config()
    
    # Initialize database
    config = Config()
    db_session = init_db(config.db_path)
    db_session.close()
    
    click.echo("RSSidian initialized successfully.")


@cli.command()
@click.argument("opml_file", type=click.Path(exists=True))
@click.pass_context
def import_opml(ctx, opml_file):
    """Import feeds from OPML file."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    try:
        new_count, updated_count = import_feeds_from_opml(opml_file, db_session)
        click.echo(f"Imported {new_count} new feeds and updated {updated_count} existing feeds.")
    except Exception as e:
        click.echo(f"Error importing feeds: {str(e)}", err=True)
    finally:
        db_session.close()


@cli.group()
def subscriptions():
    """Manage feed subscriptions."""
    pass


@subscriptions.command("list")
@click.option("--sort", type=click.Choice(["title", "updated", "articles"]), default="title",
              help="Sort subscriptions by field")
@click.pass_context
def list_subscriptions(ctx, sort):
    """List all feed subscriptions."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    # Query feeds with different sorting
    if sort == "title":
        feeds = db_session.query(Feed).order_by(Feed.title).all()
    elif sort == "updated":
        feeds = db_session.query(Feed).order_by(Feed.last_updated.desc()).all()
    elif sort == "articles":
        # This is more complex as it requires counting articles per feed
        feeds = db_session.query(Feed).all()
        
        # Add article count for each feed
        feeds_with_counts = []
        for feed in feeds:
            article_count = db_session.query(Article).filter_by(feed_id=feed.id).count()
            feeds_with_counts.append((feed, article_count))
        
        # Sort by article count
        feeds_with_counts.sort(key=lambda x: x[1], reverse=True)
        feeds = [f[0] for f in feeds_with_counts]
    
    # Display feeds
    if not feeds:
        click.echo("No subscriptions found.")
        db_session.close()
        return
    
    click.echo(f"Total subscriptions: {len(feeds)}")
    click.echo()
    
    for feed in feeds:
        article_count = db_session.query(Article).filter_by(feed_id=feed.id).count()
        muted_status = "[MUTED] " if feed.muted else ""
        last_updated = feed.last_updated.strftime("%Y-%m-%d") if feed.last_updated else "Never"
        
        click.echo(f"{muted_status}{feed.title}")
        click.echo(f"  URL: {feed.url}")
        click.echo(f"  Articles: {article_count}")
        click.echo(f"  Last Updated: {last_updated}")
        click.echo()
    
    db_session.close()


@subscriptions.command("mute")
@click.argument("feed_title")
@click.pass_context
def mute_subscription(ctx, feed_title):
    """Mute a feed subscription."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    # Find feed by title
    feed = db_session.query(Feed).filter(Feed.title.like(f"%{feed_title}%")).first()
    
    if not feed:
        click.echo(f"No feed found matching title: {feed_title}", err=True)
        db_session.close()
        return
    
    feed.muted = True
    db_session.commit()
    click.echo(f"Muted feed: {feed.title}")
    db_session.close()


@subscriptions.command("unmute")
@click.argument("feed_title")
@click.pass_context
def unmute_subscription(ctx, feed_title):
    """Unmute a feed subscription."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    # Find feed by title
    feed = db_session.query(Feed).filter(Feed.title.like(f"%{feed_title}%")).first()
    
    if not feed:
        click.echo(f"No feed found matching title: {feed_title}", err=True)
        db_session.close()
        return
    
    feed.muted = False
    db_session.commit()
    click.echo(f"Unmuted feed: {feed.title}")
    db_session.close()


@cli.command()
@click.option("--lookback", type=int, help="Number of days to look back for articles")
@click.option("--debug/--no-debug", default=False, help="Enable debug output")
@click.pass_context
def ingest(ctx, lookback, debug):
    """Ingest articles from RSS feeds."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    processor = RSSProcessor(config, db_session)
    
    try:
        # Step 1: Fetch articles from feeds
        result = processor.ingest_feeds(lookback_days=lookback, debug=debug)
        click.echo(f"Processed {result['feeds_processed']} feeds, found {result['new_articles']} new articles.")
        
        # Step 2: Process fetched articles
        if result['new_articles'] > 0:
            click.echo("Processing new articles...")
            process_result = processor.process_articles()
            click.echo(f"Processed {process_result['articles_processed']} articles:")
            click.echo(f"  With summaries: {process_result['with_summary']}")
            click.echo(f"  With value analysis: {process_result['with_value']}")
            click.echo(f"  With embeddings: {process_result['with_embedding']}")
        
        # Step 3: Generate digest
        click.echo("Generating digest...")
        lookback_period = lookback or config.default_lookback
        digest = processor.generate_digest(lookback_days=lookback_period)
        
        # Step 4: Write digest to Obsidian
        file_path = write_digest_to_obsidian(digest, config)
        if file_path:
            click.echo(f"Wrote digest to {file_path}")
        else:
            click.echo("Failed to write digest to Obsidian.")
        
    except Exception as e:
        click.echo(f"Error during ingestion: {str(e)}", err=True)
        if debug:
            import traceback
            click.echo(traceback.format_exc())
    finally:
        db_session.close()


@cli.command()
@click.argument("query")
@click.option("--relevance", type=float, default=0.6, help="Minimum relevance threshold (0-1)")
@click.option("--refresh/--no-refresh", default=False, help="Refresh search index before searching")
@click.pass_context
def search(ctx, query, relevance, refresh):
    """Search through article content using natural language."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    processor = RSSProcessor(config, db_session)
    
    try:
        results = processor.search(query, relevance_threshold=relevance, refresh=refresh)
        
        if not results:
            click.echo("No matching articles found.")
            db_session.close()
            return
        
        click.echo(f"Found {len(results)} matching articles:")
        click.echo()
        
        # Group results by feed
        feeds = {}
        for result in results:
            feed = result["feed"]
            if feed not in feeds:
                feeds[feed] = []
            feeds[feed].append(result)
        
        # Display results grouped by feed
        for feed, feed_results in feeds.items():
            click.echo(f"## {feed}")
            click.echo()
            
            for result in feed_results:
                title = result["title"]
                published = result["published_at"].strftime("%Y-%m-%d") if result["published_at"] else "Unknown date"
                relevance_pct = int(result["relevance"] * 100)
                quality = f"{result['quality_tier']}-Tier" if result["quality_tier"] else ""
                
                click.echo(f"### {title}")
                click.echo(f"*{published} | Relevance: {relevance_pct}% | {quality}*")
                click.echo(f"URL: {result['url']}")
                click.echo()
                
                if result["excerpt"]:
                    click.echo(f"Excerpt: {result['excerpt']}")
                    click.echo()
                
                if result["summary"]:
                    click.echo(f"Summary: {result['summary']}")
                    click.echo()
    
    except Exception as e:
        click.echo(f"Error during search: {str(e)}", err=True)
    finally:
        db_session.close()


@cli.command()
@click.option("--port", type=int, default=8080, help="Port to run the API server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to bind the API server to")
@click.pass_context
def mcp(ctx, port, host):
    """Start the MCP (Message Control Program) API service."""
    config = ctx.obj["config"]
    
    # Create the FastAPI app
    app = create_api(config)
    
    click.echo(f"Starting MCP service on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")
    
    # Run uvicorn server
    uvicorn.run(app, host=host, port=port)


@cli.group()
def backup():
    """Manage database backups."""
    pass


@backup.command("create")
@click.pass_context
def backup_create(ctx):
    """Create a new database backup."""
    config = ctx.obj["config"]
    
    backup_path = create_backup(config)
    if backup_path:
        click.echo(f"Created backup at {backup_path}")
    else:
        click.echo("Failed to create backup.", err=True)


@backup.command("list")
@click.pass_context
def backup_list(ctx):
    """List all available backups."""
    config = ctx.obj["config"]
    
    backups = get_backup_list(config)
    
    if not backups:
        click.echo("No backups found.")
        return
    
    click.echo(f"Found {len(backups)} backups:")
    click.echo()
    
    for backup in backups:
        click.echo(f"{backup['date']} - {backup['size_mb']:.2f} MB")
    
    click.echo()
    click.echo(f"Backup location: {config.backup_dir}")


@backup.command("restore")
@click.argument("backup_date")
@click.option("--force/--no-force", default=False, help="Force restore without confirmation")
@click.pass_context
def backup_restore(ctx, backup_date, force):
    """Restore database from a backup."""
    config = ctx.obj["config"]
    
    backups = get_backup_list(config)
    
    # Find the backup by date
    target_backup = None
    for backup in backups:
        if backup_date in backup["date"]:
            target_backup = backup
            break
    
    if not target_backup:
        click.echo(f"No backup found for date: {backup_date}", err=True)
        return
    
    # Get current database info
    db_path = config.db_path
    current_size = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
    
    # Show comparison between current and backup
    click.echo(f"Current database size: {current_size:.2f} MB")
    click.echo(f"Backup database size: {target_backup['size_mb']:.2f} MB")
    click.echo()
    
    # Confirm restore
    if not force:
        if not click.confirm("Are you sure you want to restore from this backup? This will overwrite your current database."):
            click.echo("Restore cancelled.")
            return
    
    # Perform restore
    success = restore_backup(backup_date, config, force=force)
    
    if success:
        click.echo(f"Successfully restored database from {target_backup['date']} backup.")
    else:
        click.echo("Failed to restore backup.", err=True)


@cli.command()
@click.pass_context
def show_config(ctx):
    """Show current configuration and system status."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    # Database stats
    article_count = db_session.query(Article).count()
    feed_count = db_session.query(Feed).count()
    muted_feed_count = db_session.query(Feed).filter_by(muted=True).count()
    processed_article_count = db_session.query(Article).filter_by(processed=True).count()
    with_embedding_count = db_session.query(Article).filter_by(embedding_generated=True).count()
    
    # Quality tier stats
    quality_counts = {
        "S": db_session.query(Article).filter_by(quality_tier="S").count(),
        "A": db_session.query(Article).filter_by(quality_tier="A").count(),
        "B": db_session.query(Article).filter_by(quality_tier="B").count(),
        "C": db_session.query(Article).filter_by(quality_tier="C").count(),
        "D": db_session.query(Article).filter_by(quality_tier="D").count(),
    }
    
    # Configuration info
    click.echo("=== RSSidian Configuration ===")
    click.echo(f"Config path: {config.config_path}")
    click.echo(f"Database path: {config.db_path}")
    click.echo(f"Obsidian vault: {config.obsidian_vault_path}")
    click.echo(f"Vector index: {config.annoy_index_path}")
    click.echo(f"OpenRouter API configured: {'Yes' if config.openrouter_api_key else 'No'}")
    click.echo(f"Default lookback period: {config.default_lookback} days")
    click.echo(f"Minimum quality tier: {config.minimum_quality_tier}")
    click.echo()
    
    # System status
    click.echo("=== System Status ===")
    click.echo(f"Total feeds: {feed_count} ({muted_feed_count} muted)")
    click.echo(f"Total articles: {article_count}")
    click.echo(f"Processed articles: {processed_article_count}")
    click.echo(f"Articles with embeddings: {with_embedding_count}")
    click.echo()
    
    # Quality distribution
    click.echo("=== Quality Distribution ===")
    for tier in ["S", "A", "B", "C", "D"]:
        click.echo(f"{tier}-Tier: {quality_counts[tier]}")
    
    db_session.close()


if __name__ == "__main__":
    cli()