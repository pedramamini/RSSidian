import os
import sys
import click
import uvicorn
from datetime import datetime, timedelta
from typing import Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

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

# Set up rich console
console = Console()


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
    
    console.print(Panel.fit(
        "[bold green]RSSidian initialized successfully[/bold green]", 
        border_style="green"
    ))


@cli.command()
@click.argument("opml_file", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Force import even if feeds already exist")
@click.pass_context
def import_opml(ctx, opml_file, force):
    """Import feeds from OPML file."""
    config = ctx.obj["config"]
    db_session = get_db_session(config.db_path)
    
    console.print(f"Importing feeds from [bold]{opml_file}[/bold]...")
    
    try:
        # Parse the OPML file first to get a count of feeds
        from .opml import parse_opml
        feeds_data = parse_opml(opml_file)
        console.print(f"Found [bold]{len(feeds_data)}[/bold] feeds in OPML file.")
        
        # Import the feeds
        new_count, updated_count = import_feeds_from_opml(opml_file, db_session)
        
        # Show summary
        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column("Statistic", style="cyan")
        summary_table.add_column("Value")
        
        if new_count > 0 or updated_count > 0:
            summary_table.add_row("New feeds", f"[green]{new_count}[/green]")
            summary_table.add_row("Updated feeds", f"[yellow]{updated_count}[/yellow]")
            console.print(Panel.fit(
                "[bold green]Import Successful[/bold green]", 
                border_style="green"
            ))
        else:
            console.print("[yellow]No new feeds were imported.[/yellow]")
        
        # Show total feeds in database
        total_feeds = db_session.query(Feed).count()
        summary_table.add_row("Total feeds in database", f"[bold]{total_feeds}[/bold]")
        
        console.print(summary_table)
        
    except Exception as e:
        console.print(f"[bold red]Error importing feeds:[/bold red] {str(e)}")
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
        console.print("[bold red]No subscriptions found.[/bold red]")
        db_session.close()
        return
    
    console.print(f"[bold green]Total subscriptions:[/bold green] {len(feeds)}")
    console.print()
    
    # Create a table for better visualization
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Feed")
    table.add_column("Articles", justify="right")
    table.add_column("Last Updated")
    table.add_column("URL", style="dim")
    
    for feed in feeds:
        article_count = db_session.query(Article).filter_by(feed_id=feed.id).count()
        muted_status = "[dim italic]MUTED[/dim italic] " if feed.muted else ""
        last_updated = feed.last_updated.strftime("%Y-%m-%d") if feed.last_updated else "Never"
        
        feed_title = f"{muted_status}{feed.title}"
        table.add_row(
            feed_title,
            str(article_count),
            last_updated,
            feed.url
        )
    
    console.print(table)
    
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
        console.print(f"[green]Created backup at {backup_path}[/green]")
    else:
        console.print("[bold red]Failed to create backup.[/bold red]")


@backup.command("list")
@click.pass_context
def backup_list(ctx):
    """List all available backups."""
    config = ctx.obj["config"]
    
    backups = get_backup_list(config)
    
    if not backups:
        console.print("[yellow]No backups found.[/yellow]")
        return
    
    console.print(Panel.fit(
        f"[bold blue]Found {len(backups)} backups[/bold blue]", 
        border_style="blue"
    ))
    
    backup_table = Table(box=box.SIMPLE)
    backup_table.add_column("Date")
    backup_table.add_column("Size", justify="right")
    
    for backup in backups:
        backup_table.add_row(
            backup['date'],
            f"{backup['size_mb']:.2f} MB"
        )
    
    console.print(backup_table)
    console.print()
    console.print(f"[dim]Backup location: {config.backup_dir}[/dim]")


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
    info_table = Table(show_header=False, box=box.SIMPLE)
    info_table.add_column("Info", style="cyan")
    info_table.add_column("Value")
    
    info_table.add_row("Current database size", f"{current_size:.2f} MB")
    info_table.add_row("Backup database size", f"{target_backup['size_mb']:.2f} MB")
    
    console.print(info_table)
    console.print()
    
    # Confirm restore
    if not force:
        if not click.confirm("Are you sure you want to restore from this backup? This will overwrite your current database."):
            console.print("[yellow]Restore cancelled.[/yellow]")
            return
    
    # Perform restore
    success = restore_backup(backup_date, config, force=force)
    
    if success:
        console.print(f"[green]Successfully restored database from {target_backup['date']} backup.[/green]")
    else:
        console.print("[bold red]Failed to restore backup.[/bold red]")


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
    console.print(Panel.fit(
        "[bold blue]RSSidian Configuration[/bold blue]", 
        border_style="blue"
    ))
    
    config_table = Table(show_header=False, box=box.SIMPLE)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value")
    
    config_table.add_row("Config path", config.config_path)
    config_table.add_row("Database path", config.db_path)
    config_table.add_row("Obsidian vault", config.obsidian_vault_path)
    config_table.add_row("Vector index", config.annoy_index_path)
    config_table.add_row(
        "OpenRouter API", 
        "[green]Configured[/green]" if config.openrouter_api_key else "[red]Not configured[/red]"
    )
    config_table.add_row("Default lookback period", f"{config.default_lookback} days")
    config_table.add_row("Minimum quality tier", f"[bold]{config.minimum_quality_tier}[/bold]")
    
    console.print(config_table)
    
    # System status
    console.print("\n", Panel.fit(
        "[bold blue]System Status[/bold blue]", 
        border_style="blue"
    ))
    
    stats_table = Table(show_header=False, box=box.SIMPLE)
    stats_table.add_column("Statistic", style="cyan")
    stats_table.add_column("Value")
    
    stats_table.add_row("Total feeds", f"{feed_count} ([dim]{muted_feed_count} muted[/dim])")
    stats_table.add_row("Total articles", str(article_count))
    stats_table.add_row("Processed articles", str(processed_article_count))
    stats_table.add_row("Articles with embeddings", str(with_embedding_count))
    
    console.print(stats_table)
    
    # Quality distribution
    console.print("\n", Panel.fit(
        "[bold blue]Quality Distribution[/bold blue]", 
        border_style="blue"
    ))
    
    tier_table = Table(box=box.SIMPLE)
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("Count", justify="right")
    
    tier_colors = {
        "S": "bright_green",
        "A": "green",
        "B": "yellow",
        "C": "red",
        "D": "bright_red"
    }
    
    for tier in ["S", "A", "B", "C", "D"]:
        count = quality_counts[tier]
        tier_table.add_row(
            f"[{tier_colors[tier]}]{tier}[/{tier_colors[tier]}]",
            str(count)
        )
    
    console.print(tier_table)
    
    db_session.close()


if __name__ == "__main__":
    cli()