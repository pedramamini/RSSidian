import os
import sys
import json
import click
from datetime import datetime, timedelta
from typing import Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
from rich import box

# Only import the lightweight config module at startup
from .config import Config, init_config
from . import __version__

# Import backup functions at the top level since they're lightweight
from .backup import create_backup, get_backup_list, restore_backup

# Defer heavy imports until they're needed in specific commands
# These will be imported within each command function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up rich console
console = Console()


def ensure_db_exists(config, initialize=True):
    """Ensure the database exists and is initialized.
    
    Args:
        config: Configuration object
        initialize: Whether to initialize the database (default: True)
            Set to False for commands that just need to check if the DB exists
    """
    # Ensure the directory exists
    db_dir = os.path.dirname(config.db_path)
    if not os.path.exists(db_dir):
        console.print(f"[yellow]Creating database directory: {db_dir}[/yellow]")
        os.makedirs(db_dir, exist_ok=True)
    
    if not initialize:
        # For commands that just need to check if the DB exists
        if not os.path.exists(config.db_path):
            console.print("[yellow]Database does not exist. Run 'rssidian init' first.[/yellow]")
            return None
        # Import here to avoid loading at startup
        from .models import get_db_session
        return get_db_session(config.db_path)
    
    # For commands that need a fully initialized database
    # Import here to avoid loading at startup
    from .models import init_db, get_db_session, Base
    from sqlalchemy import create_engine
    
    if not os.path.exists(config.db_path):
        console.print("[yellow]Database does not exist. Creating it now...[/yellow]")
        return init_db(config.db_path)
    else:
        # Even if the DB file exists, make sure the tables are created
        # This is safe as create_all() checks if tables exist before creating
        engine = create_engine(f"sqlite:///{config.db_path}")
        Base.metadata.create_all(engine)
        return get_db_session(config.db_path)


@click.group()
@click.version_option(version=__version__, prog_name="rssidian")
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
    
    # Initialize database - import here to avoid loading at startup
    from .models import init_db
    
    config = Config()
    db_session = init_db(config.db_path)
    db_session.close()
    
    console.print(Panel.fit(
        "[bold green]RSSidian initialized successfully[/bold green]", 
        border_style="green"
    ))


@cli.command()
@click.pass_context
def auth(ctx):
    """Authenticate with AI providers (currently supports Anthropic OAuth)."""
    config = ctx.obj["config"]
    
    if config.ai_provider == "anthropic":
        console.print("[bold blue]Starting Anthropic OAuth authentication...[/bold blue]")
        
        # Import the core module here to avoid loading at startup
        from .core import RSSProcessor
        from .models import init_db
        
        # Initialize database connection
        db_session = init_db(config.db_path)
        
        try:
            processor = RSSProcessor(config, db_session)
            success = processor.authenticate_anthropic()
            
            if success:
                console.print(Panel.fit(
                    "[bold green]Successfully authenticated with Anthropic![/bold green]\n"
                    "You can now use Anthropic models for content analysis.",
                    border_style="green"
                ))
            else:
                console.print(Panel.fit(
                    "[bold red]Authentication failed![/bold red]\n"
                    "Please check your OAuth credentials and try again.",
                    border_style="red"
                ))
        finally:
            db_session.close()
    
    elif config.ai_provider == "openrouter":
        console.print("[yellow]OpenRouter uses API key authentication.[/yellow]")
        console.print("Please set your API key in the configuration file or environment variable.")
        console.print("See the config.toml.example file for details.")
    
    else:
        console.print(f"[red]Unknown AI provider: {config.ai_provider}[/red]")
        console.print("Please check your configuration file.")


@cli.group()
def opml():
    """Import and export feeds in OPML format."""
    pass


@opml.command("import")
@click.argument("opml_file", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Force import even if feeds already exist")
@click.pass_context
def opml_import(ctx, opml_file, force):
    """Import feeds from OPML file."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)
    
    console.print(f"Importing feeds from [bold]{opml_file}[/bold]...")
    
    try:
        # Import modules here to avoid loading at startup
        from .opml import parse_opml, import_feeds_from_opml
        from .models import Feed
        
        # Parse the OPML file first to get a count of feeds
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


@opml.command("export")
@click.argument("output_file", type=click.Path())
@click.option("--exclude-muted", is_flag=True, 
              help="Exclude muted feeds from the export")
@click.pass_context
def opml_export(ctx, output_file, exclude_muted):
    """Export feeds to OPML file."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config, initialize=False)
    
    try:
        # Import modules here to avoid loading at startup
        from .models import Feed
        from .opml import export_feeds_to_opml
        
        # Query feeds to get a count
        query = db_session.query(Feed)
        if exclude_muted:
            query = query.filter_by(muted=False)
        
        feeds_count = query.count()
        
        if feeds_count == 0:
            console.print("[bold yellow]No feeds to export.[/bold yellow]")
            db_session.close()
            return
        
        # Generate OPML content
        opml_content = export_feeds_to_opml(db_session, not exclude_muted)
        
        # Write to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(opml_content)
        
        # Show summary
        summary_table = Table(show_header=False, box=box.SIMPLE)
        summary_table.add_column("Statistic", style="cyan")
        summary_table.add_column("Value")
        
        summary_table.add_row("Exported feeds", f"[green]{feeds_count}[/green]")
        summary_table.add_row("Output file", f"[bold]{output_file}[/bold]")
        summary_table.add_row("Excluded muted feeds", f"[{'green' if exclude_muted else 'yellow'}]{exclude_muted}[/{'green' if exclude_muted else 'yellow'}]")
        
        console.print(Panel.fit(
            "[bold green]Export Successful[/bold green]", 
            border_style="green"
        ))
        console.print(summary_table)
        
    except Exception as e:
        console.print(f"[bold red]Error exporting feeds:[/bold red] {str(e)}")
    finally:
        db_session.close()


@cli.group()
def subscriptions():
    """Manage feed subscriptions."""
    pass


@subscriptions.command("add")
@click.argument("feed_url")
@click.option("--title", help="Feed title (auto-detected if not provided)")
@click.option("--force", is_flag=True, help="Force update if feed already exists")
@click.pass_context
def add_subscription(ctx, feed_url, title, force):
    """Add a new RSS feed subscription."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)

    console.print(f"Adding feed: [bold]{feed_url}[/bold]")

    try:
        # Import modules here
        from .models import Feed
        import feedparser
        import requests

        # Check if feed already exists
        existing_feed = db_session.query(Feed).filter_by(url=feed_url).first()

        if existing_feed and not force:
            console.print(f"[yellow]Feed already exists:[/yellow] {existing_feed.title}")
            console.print(f"Use --force to update the existing feed")
            db_session.close()
            return

        # Fetch and parse the feed to validate it and get metadata
        console.print("Validating feed...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        try:
            response = requests.get(feed_url, headers=headers, timeout=30)
            response.raise_for_status()
            feed_data = feedparser.parse(response.content)
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error fetching feed:[/red] {str(e)}")
            db_session.close()
            return

        # Check if the feed was parsed successfully
        if hasattr(feed_data, 'bozo') and feed_data.bozo:
            if hasattr(feed_data, 'bozo_exception'):
                console.print(f"[yellow]Warning:[/yellow] Feed parsing issue: {str(feed_data.bozo_exception)}")

        # Extract feed metadata
        feed_title = title or feed_data.feed.get('title', 'Unknown Feed')
        feed_description = feed_data.feed.get('description', '')
        feed_website_url = feed_data.feed.get('link', '')

        # Create or update feed
        if existing_feed:
            console.print(f"Updating existing feed: [bold]{existing_feed.title}[/bold]")
            existing_feed.title = feed_title
            existing_feed.description = feed_description
            existing_feed.website_url = feed_website_url
            existing_feed.last_updated = datetime.utcnow()
            db_session.commit()

            console.print(Panel.fit(
                f"[bold green]Feed Updated Successfully[/bold green]\n\n"
                f"Title: {feed_title}\n"
                f"URL: {feed_url}\n"
                f"Website: {feed_website_url}",
                border_style="green"
            ))
        else:
            new_feed = Feed(
                title=feed_title,
                url=feed_url,
                description=feed_description,
                website_url=feed_website_url,
                last_updated=datetime.utcnow(),
                last_checked=datetime.utcnow()
            )
            db_session.add(new_feed)
            db_session.commit()

            console.print(Panel.fit(
                f"[bold green]Feed Added Successfully[/bold green]\n\n"
                f"Title: {feed_title}\n"
                f"URL: {feed_url}\n"
                f"Website: {feed_website_url}",
                border_style="green"
            ))

        # Show feed statistics
        total_feeds = db_session.query(Feed).count()
        console.print(f"\n[dim]Total feeds in database:[/dim] [bold]{total_feeds}[/bold]")

    except Exception as e:
        console.print(f"[bold red]Error adding feed:[/bold red] {str(e)}")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
    finally:
        db_session.close()


@subscriptions.command("list")
@click.option("--sort", type=click.Choice(["title", "updated", "articles", "rating"]), default="title",
              help="Sort subscriptions by field")
@click.option("--width", type=int, default=60,
              help="Limit the width of the feed title column")
@click.pass_context
def list_subscriptions(ctx, sort, width):
    """List all feed subscriptions."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)
    
    # Import models here
    from .models import Feed
    
    # Query feeds with different sorting
    if sort == "title":
        feeds = db_session.query(Feed).order_by(Feed.title).all()
    elif sort == "updated":
        feeds = db_session.query(Feed).order_by(Feed.last_updated.desc()).all()
    elif sort == "articles":
        # Sort by article count
        feeds = db_session.query(Feed).order_by(Feed.article_count.desc()).all()
    elif sort == "rating":
        # Sort by average quality score (None values last)
        feeds = db_session.query(Feed).order_by(
            # This puts NULL values at the end
            Feed.avg_quality_score.is_(None),
            Feed.avg_quality_score.desc()
        ).all()
    
    # Display feeds
    if not feeds:
        console.print("[bold red]No subscriptions found.[/bold red]")
        db_session.close()
        return
    
    console.print(f"[bold green]Total subscriptions:[/bold green] {len(feeds)}")
    console.print()
    
    # Create a table for better visualization
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    
    # Set row styles for alternating colors with background
    table.row_styles = ["none", "on dark_green"]
    
    # Get terminal width for better column sizing
    import shutil
    terminal_width = shutil.get_terminal_size().columns
    
    # Calculate column widths based on terminal width
    domain_width = min(30, max(15, int(terminal_width * 0.2)))
    status_width = 10
    articles_width = 8
    date_width = 12
    feed_width = min(width, terminal_width - domain_width - status_width - articles_width - date_width - 10)  # 10 for padding and borders
    
    # Add columns with appropriate width constraints
    table.add_column("Feed", width=feed_width, no_wrap=False)
    table.add_column("Status", width=status_width)
    table.add_column("Articles", justify="right", width=articles_width)
    table.add_column("Rating", justify="center", width=10)
    table.add_column("Last Updated", width=date_width)
    table.add_column("Domain", style="dim", width=domain_width, no_wrap=True)
    table.add_column("Errors", justify="right", width=8)
    table.add_column("Last Error", style="dim", width=30, no_wrap=True)
    
    for feed in feeds:
        # Use stored article count instead of querying each time
        article_count = feed.article_count or 0
        last_updated = feed.last_updated.strftime("%Y-%m-%d") if feed.last_updated else "Never"
        
        # Create status column with muted and peer-through indicators
        status_parts = []
        if feed.muted:
            status_parts.append("[dim italic]MUTED[/dim italic]")
        if feed.peer_through:
            status_parts.append("[green]PT[/green]")
        status = " ".join(status_parts) if status_parts else ""
        
        # Prepare rating display
        rating_display = ""
        if feed.quality_tier_counts:
            try:
                tier_counts = json.loads(feed.quality_tier_counts)
                # Find the highest tier with at least one article
                for tier in ["S", "A", "B", "C", "D"]:
                    if tier in tier_counts and tier_counts[tier] > 0:
                        # Determine color based on tier
                        if tier in ["S", "A"]:
                            color = "green"
                        elif tier == "B":
                            color = "yellow"
                        else:
                            color = "red"
                        rating_display = f"[bold {color}]{tier}[/bold {color}]"
                        break
                
                # Add avg score if available
                if feed.avg_quality_score is not None:
                    rating_display += f" {int(feed.avg_quality_score)}"
            except json.JSONDecodeError:
                pass
        
        # Extract domain name from feed URL
        import re
        from urllib.parse import urlparse
        domain = ""
        if feed.url:
            try:
                parsed_url = urlparse(feed.url)
                domain = parsed_url.netloc
                # Remove www. prefix if present
                domain = re.sub(r'^www\.', '', domain)
                
                # Special handling for domains that benefit from path inclusion
                path_domains = ["medium.com", "github.com", "substack.com", "wordpress.com", "blogspot.com", "feeds.feedburner.com"]
                
                if domain in path_domains and parsed_url.path:
                    path_parts = parsed_url.path.strip('/').split('/')
                    if path_parts and path_parts[0]:
                        # Skip feed, rss, atom in path
                        first_segment = path_parts[0]
                        if first_segment.lower() in ["feed", "rss", "atom", "feeds"]:
                            if len(path_parts) > 1:
                                first_segment = path_parts[1]
                            else:
                                first_segment = ""
                                
                        if first_segment:
                            domain = f"{domain}/{first_segment}"
                
                # Truncate very long domains if needed
                if len(domain) > domain_width - 3 and domain_width > 5:
                    domain = domain[:domain_width-3] + "..."
            except:
                domain = "unknown"
                
        # Prepare feed title - no need to truncate as the Rich table will handle it with ellipsis
        feed_title = feed.title
        
        # Format error count and last error
        error_count = str(feed.error_count) if feed.error_count > 0 else ""
        error_style = "[red]" if feed.error_count > 0 else ""
        last_error = feed.last_error if feed.last_error else ""
        if last_error and len(last_error) > 30:
            last_error = last_error[:27] + "..."

        table.add_row(
            feed_title,
            status,
            str(article_count),
            rating_display,
            last_updated,
            domain,
            f"{error_style}{error_count}[/red]" if error_count else "",
            last_error
        )
    
    console.print(table)
    
    db_session.close()


@subscriptions.command("mute")
@click.argument("feed_title")
@click.pass_context
def mute_subscription(ctx, feed_title):
    """Mute a feed subscription."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)
    
    # Import models here
    from .models import Feed
    
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
    db_session = ensure_db_exists(config)
    
    # Import models here
    from .models import Feed
    
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


@subscriptions.command("enable-peer-through")
@click.argument("feed_title")
@click.pass_context
def enable_peer_through(ctx, feed_title):
    """Enable peer-through for an aggregator feed to fetch origin article content."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)
    
    # Find feed by title
    feed = db_session.query(Feed).filter(Feed.title.like(f"%{feed_title}%")).first()
    
    if not feed:
        click.echo(f"No feed found matching title: {feed_title}", err=True)
        db_session.close()
        return
    
    feed.peer_through = True
    db_session.commit()
    click.echo(f"Enabled peer-through for feed: {feed.title}")
    db_session.close()


@subscriptions.command("disable-peer-through")
@click.argument("feed_title")
@click.pass_context
def disable_peer_through(ctx, feed_title):
    """Disable peer-through for a feed."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)
    
    # Find feed by title
    feed = db_session.query(Feed).filter(Feed.title.like(f"%{feed_title}%")).first()
    
    if not feed:
        click.echo(f"No feed found matching title: {feed_title}", err=True)
        db_session.close()
        return
    
    feed.peer_through = False
    db_session.commit()
    click.echo(f"Disabled peer-through for feed: {feed.title}")
    db_session.close()


@cli.command()
@click.option("--lookback", type=int, help="Number of days to look back for articles")
@click.option("--debug/--no-debug", default=False, help="Enable debug output")
@click.pass_context
def ingest(ctx, lookback, debug):
    """Ingest articles from RSS feeds."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)
    
    # Import modules only when needed
    from .core import RSSProcessor
    from .markdown import write_digest_to_obsidian
    from .cost_tracker import init_cost_tracker, format_cost_summary
    
    # Initialize cost tracker if enabled
    cost_tracking_enabled = config.cost_tracking_enabled
    if cost_tracking_enabled:
        init_cost_tracker()
    
    # Suppress RSS module logs unless debug is enabled
    if not debug and not os.environ.get("DEBUG"):
        logging.getLogger("rssidian.rss").setLevel(logging.WARNING)
    
    processor = RSSProcessor(config, db_session)
    
    try:
        # Step 1: Fetch articles from feeds
        result = processor.ingest_feeds(lookback_days=lookback, debug=debug)
        click.echo(f"Processed {result['feeds_processed']} feeds, found {result['new_articles']} new articles.")
        
        # Display information about auto-muted feeds if any
        if 'auto_muted' in result and result['auto_muted'] > 0:
            click.echo(f"Auto-muted {result['auto_muted']} feeds due to persistent errors. Use 'subscriptions list' to see details.")
        
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
        
        # Display cost summary if enabled
        if cost_tracking_enabled:
            click.echo("\n" + format_cost_summary())
        
    except Exception as e:
        click.echo(f"Error during ingestion: {str(e)}", err=True)
        if debug:
            import traceback
            click.echo(traceback.format_exc())
        # Still show cost summary on error if enabled
        if cost_tracking_enabled:
            click.echo("\n" + format_cost_summary())
    finally:
        db_session.close()


@cli.command()
@click.argument("query")
@click.option("--relevance", type=float, default=0.3, help="Minimum relevance threshold (0-1)")
@click.option("--refresh/--no-refresh", default=False, help="Refresh search index before searching")
@click.pass_context
def search(ctx, query, relevance, refresh):
    """Search through article content using natural language."""
    config = ctx.obj["config"]
    db_session = ensure_db_exists(config)
    
    # Import processor only when needed
    from .core import RSSProcessor
    from .cost_tracker import init_cost_tracker, format_cost_summary

    # Initialize cost tracker if enabled
    cost_tracking_enabled = config.cost_tracking_enabled
    if cost_tracking_enabled:
        init_cost_tracker()
    import os
    
    # Initialize cost tracker if enabled
    cost_tracking_enabled = config.cost_tracking_enabled
    if cost_tracking_enabled:
        init_cost_tracker()
    
    # Set logging level based on DEBUG environment variable
    if not os.environ.get("DEBUG"):
        import logging
        logging.getLogger("rssidian.core").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    
    processor = RSSProcessor(config, db_session)
    
    try:
        # Create a progress bar for the search process
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Searching..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            refresh_per_second=10,
        ) as progress:
            # Add a task for the search process
            task = progress.add_task("[cyan]Searching...", total=1)
            
            # Perform the search
            results = processor.search(query, relevance_threshold=relevance, refresh=refresh)
            
            # Complete the progress bar
            progress.update(task, completed=1)
        
        if not results:
            console.print("[yellow]No matching articles found.[/yellow]")
            db_session.close()
            return
        
        console.print(f"[bold green]Found {len(results)} matching articles:[/bold green]\n")
        
        # Group results by feed
        feeds = {}
        for result in results:
            feed = result["feed"]
            if feed not in feeds:
                feeds[feed] = []
            feeds[feed].append(result)
        
        # Display results grouped by feed
        for feed, feed_results in feeds.items():
            console.print(f"[bold cyan]{feed}[/bold cyan]")
            console.print("[dim]" + "─" * len(feed) + "[/dim]\n")
            
            for result in feed_results:
                title = result["title"]
                published = result["published_at"].strftime("%Y-%m-%d") if result["published_at"] else "Unknown date"
                relevance_pct = int(result["relevance"] * 100)
                
                # Format quality tier with appropriate color
                quality_tier = result["quality_tier"]
                if quality_tier:
                    tier_colors = {
                        "S": "bright_magenta",
                        "A": "green",
                        "B": "blue",
                        "C": "yellow",
                        "D": "red"
                    }
                    color = tier_colors.get(quality_tier, "white")
                    quality = f"[bold {color}]{quality_tier}-Tier[/bold {color}]" 
                else:
                    quality = ""
                
                # Format relevance with color based on percentage
                if relevance_pct >= 80:
                    relevance_str = f"[bold green]Relevance: {relevance_pct}%[/bold green]"
                elif relevance_pct >= 60:
                    relevance_str = f"[green]Relevance: {relevance_pct}%[/green]"
                elif relevance_pct >= 40:
                    relevance_str = f"[yellow]Relevance: {relevance_pct}%[/yellow]"
                else:
                    relevance_str = f"[dim]Relevance: {relevance_pct}%[/dim]"
                
                # Print title and metadata
                console.print(f"[bold]{title}[/bold]")
                console.print(f"[dim]{published}[/dim] | {relevance_str} | {quality}")
                console.print(f"[blue underline]{result['url']}[/blue underline]\n")
                
                # Print excerpt if available
                if result["excerpt"]:
                    console.print(Panel.fit(
                        result["excerpt"],
                        title="[bold]Excerpt[/bold]",
                        border_style="dim",
                        padding=(0, 1)
                    ))
                
                # Print summary if available
                if result["summary"]:
                    console.print(Panel.fit(
                        result["summary"],
                        title="[bold]Summary[/bold]",
                        border_style="dim",
                        padding=(0, 1)
                    ))
                
                console.print()
        
        # Display cost summary if enabled
        if cost_tracking_enabled:
            console.print("\n" + format_cost_summary())
    
    except Exception as e:
        console.print(f"[bold red]Error during search:[/bold red] {str(e)}")
    finally:
        db_session.close()


@cli.command()
@click.option("--port", type=int, default=8080, help="Port to run the API server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to bind the API server to")
@click.option("--stdio", is_flag=True, help="Run in STDIO mode for direct integration with Claude Desktop")
@click.pass_context
def mcp(ctx, port, host, stdio):
    """Start the MCP (Model Context Protocol) API service."""
    config = ctx.obj["config"]
    
    if stdio:
        # Run in STDIO mode
        from .mcp_stdio import run_stdio_server
        click.echo("Starting MCP (Model Context Protocol) service in STDIO mode", err=True)
        run_stdio_server(config)
    else:
        # Run as HTTP server
        # Import modules only when needed
        import uvicorn
        from .api import create_api
        
        # Create the FastAPI app
        app = create_api(config)
        
        click.echo(f"Starting MCP (Model Context Protocol) service on http://{host}:{port}")
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
    db_session = ensure_db_exists(config)
    
    # Import database models here
    from .models import Article, Feed
    
    # Database stats
    article_count = db_session.query(Article).count()
    feed_count = db_session.query(Feed).count()
    muted_feed_count = db_session.query(Feed).filter_by(muted=True).count()
    processed_article_count = db_session.query(Article).filter_by(processed=True).count()
    with_embedding_count = db_session.query(Article).filter_by(embedding_generated=True).count()
    jina_enhanced_count = db_session.query(Article).filter_by(jina_enhanced=True).count()
    
    # Quality tier stats
    quality_counts = {
        "S": db_session.query(Article).filter_by(quality_tier="S").count(),
        "A": db_session.query(Article).filter_by(quality_tier="A").count(),
        "B": db_session.query(Article).filter_by(quality_tier="B").count(),
        "C": db_session.query(Article).filter_by(quality_tier="C").count(),
        "D": db_session.query(Article).filter_by(quality_tier="D").count(),
    }
    
    # Get vector index size
    vector_index_size_bytes = 0
    vector_index_size_human = "Not found"
    if os.path.exists(config.annoy_index_path):
        vector_index_size_bytes = os.path.getsize(config.annoy_index_path)
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if vector_index_size_bytes < 1024.0 or unit == 'GB':
                vector_index_size_human = f"{vector_index_size_bytes:.2f} {unit}"
                break
            vector_index_size_bytes /= 1024.0
    
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
    config_table.add_row("Vector index size", vector_index_size_human)
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
    stats_table.add_row("Jina.ai enhanced articles", str(jina_enhanced_count))
    
    console.print(stats_table)
    
    # Quality distribution
    console.print("\n", Panel.fit(
        "[bold blue]Quality Distribution[/bold blue]", 
        border_style="blue"
    ))
    
    tier_table = Table(box=box.SIMPLE)
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("Count", justify="right")
    tier_table.add_column("% of Total", justify="right")
    
    tier_colors = {
        "S": "bright_green",
        "A": "green",
        "B": "yellow",
        "C": "red",
        "D": "bright_red"
    }
    
    # Calculate total articles with quality tiers
    total_quality_articles = sum(quality_counts.values())
    
    for tier in ["S", "A", "B", "C", "D"]:
        count = quality_counts[tier]
        percentage = (count / total_quality_articles * 100) if total_quality_articles > 0 else 0
        tier_table.add_row(
            f"[{tier_colors[tier]}]{tier}[/{tier_colors[tier]}]",
            str(count),
            f"{percentage:.1f}%"
        )
    
    console.print(tier_table)
    
    db_session.close()


if __name__ == "__main__":
    cli()