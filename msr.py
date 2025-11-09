"""
Minimalist PR Metadata Mining Script
Mines PR metadata from GitHub Archive via BigQuery and saves to HuggingFace dataset.
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
from google.cloud import bigquery
import backoff

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_metadata"
PR_METADATA_REPO = "SWE-Arena/pr_metadata"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_metadata"  # For storing computed leaderboard data
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for mining new PRs

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_jsonl(filename):
    """Load JSONL file and return list of dictionaries."""
    if not os.path.exists(filename):
        return []

    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def get_bigquery_client():
    """
    Initialize BigQuery client using credentials from environment variable.

    Expects GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable containing
    the service account JSON credentials as a string.
    """
    # Get the JSON content from environment variable
    creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')

    if creds_json:
        # Create a temporary file to store credentials
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_file.write(creds_json)
            temp_path = temp_file.name

        # Set environment variable to point to temp file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path

        # Initialize BigQuery client
        client = bigquery.Client()

        # Clean up temp file
        os.unlink(temp_path)

        return client
    else:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")


def generate_table_union_statements(start_date, end_date):
    """
    Generate UNION ALL statements for githubarchive.day tables in date range.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        String with UNION ALL SELECT statements for all tables in range
    """
    table_names = []
    current_date = start_date

    while current_date < end_date:
        table_name = f"`githubarchive.day.{current_date.strftime('%Y%m%d')}`"
        table_names.append(table_name)
        current_date += timedelta(days=1)

    # Create UNION ALL chain
    union_parts = [f"SELECT * FROM {table}" for table in table_names]
    return " UNION ALL ".join(union_parts)


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# HUGGINGFACE API RETRY WRAPPERS
# =============================================================================

def is_rate_limit_error(e):
    """Check if exception is a HuggingFace rate limit error (429)."""
    if isinstance(e, HfHubHTTPError):
        return e.response.status_code == 429
    return False


def backoff_handler(details):
    """Handler to print retry attempt information."""
    wait_time = details['wait']
    tries = details['tries']
    wait_minutes = wait_time / 60
    print(f"   ⏳ Rate limited. Retrying in {wait_minutes:.1f} minutes ({wait_time:.0f}s) - attempt {tries}/8...")


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,  # Start at 5 minutes (300 seconds)
    max_value=3600,  # Cap at 60 minutes (3600 seconds)
    jitter=backoff.full_jitter,
    on_backoff=backoff_handler
)
def upload_large_folder_with_backoff(api, **kwargs):
    """Wrapper for HfApi.upload_large_folder with exponential backoff on rate limits."""
    return api.upload_large_folder(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,  # Start at 5 minutes (300 seconds)
    max_value=3600,  # Cap at 60 minutes (3600 seconds)
    jitter=backoff.full_jitter,
    on_backoff=backoff_handler
)
def list_repo_files_with_backoff(api, **kwargs):
    """Wrapper for HfApi.list_repo_files with exponential backoff on rate limits."""
    return api.list_repo_files(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,  # Start at 5 minutes (300 seconds)
    max_value=3600,  # Cap at 60 minutes (3600 seconds)
    jitter=backoff.full_jitter,
    on_backoff=backoff_handler
)
def hf_hub_download_with_backoff(**kwargs):
    """Wrapper for hf_hub_download with exponential backoff on rate limits."""
    return hf_hub_download(**kwargs)


@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,  # Start at 5 minutes (300 seconds)
    max_value=3600,  # Cap at 60 minutes (3600 seconds)
    jitter=backoff.full_jitter,
    on_backoff=backoff_handler
)
def upload_file_with_backoff(api, **kwargs):
    """Wrapper for HfApi.upload_file with exponential backoff on rate limits."""
    return api.upload_file(**kwargs)


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

def fetch_issue_metadata_batched(client, identifiers, start_date, end_date, batch_size=100, upload_immediately=True):
    """
    Fetch issue metadata for ALL agents using BATCHED BigQuery queries.
    Splits agents into smaller batches to avoid performance issues with large numbers of agents.

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)
        batch_size: Number of agents to process per batch (default: 100)
        upload_immediately: If True, upload each batch's results to HuggingFace immediately (default: True)

    Returns:
        Dictionary mapping agent identifier to list of issue metadata
    """
    # Split identifiers into batches
    batches = [identifiers[i:i + batch_size] for i in range(0, len(identifiers), batch_size)]
    total_batches = len(batches)

    print(f"\n🔍 Using BATCHED approach for {len(identifiers)} agents")
    print(f"   Total batches: {total_batches} (batch size: {batch_size})")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    if upload_immediately:
        print(f"   Upload mode: Immediate (after each batch)")
    else:
        print(f"   Upload mode: Deferred (all at once)")

    # Collect results from all batches
    all_metadata = {}

    for batch_num, batch_identifiers in enumerate(batches, 1):
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch_identifiers)} agents)...")

        try:
            # Query each batch
            batch_results = fetch_all_pr_metadata_single_query(
                client, batch_identifiers, start_date, end_date
            )

            # Merge results
            for identifier, metadata_list in batch_results.items():
                if identifier in all_metadata:
                    all_metadata[identifier].extend(metadata_list)
                else:
                    all_metadata[identifier] = metadata_list

            print(f"   ✓ Batch {batch_num}/{total_batches} complete")

            # Upload immediately after this batch if enabled
            if upload_immediately and batch_results:
                print(f"\n   📤 Uploading batch {batch_num}/{total_batches} results to HuggingFace...")
                upload_success = 0
                upload_errors = 0

                for identifier, metadata_list in batch_results.items():
                    if metadata_list:
                        if save_pr_metadata_to_hf(metadata_list, identifier):
                            upload_success += 1
                        else:
                            upload_errors += 1

                print(f"   ✓ Batch {batch_num}/{total_batches} upload complete ({upload_success} agents uploaded, {upload_errors} errors)")

        except Exception as e:
            print(f"   ✗ Batch {batch_num}/{total_batches} failed: {str(e)}")
            print(f"   Continuing with remaining batches...")
            continue

    total_prs = sum(len(metadata_list) for metadata_list in all_metadata.values())
    print(f"\n✓ All batches complete! Found {total_prs} total PRs across {len(all_metadata)} agents")

    return all_metadata


def fetch_all_pr_metadata_single_query(client, identifiers, start_date, end_date):
    """
    Fetch PR metadata for a BATCH of agents using ONE comprehensive BigQuery query.

    NOTE: This function is designed for smaller batches (~100 agents).
    For large numbers of agents, use fetch_issue_metadata_batched() instead.

    This query fetches:
    1. PRs authored by agents (user.login matches identifier)

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        Dictionary mapping agent identifier to list of PR metadata:
        {
            'agent-identifier': [
                {
                    'url': PR URL,
                    'created_at': Creation timestamp,
                    'merged_at': Merge timestamp (if merged, else None),
                    'closed_at': Close timestamp (if closed but not merged, else None)
                },
                ...
            ],
            ...
        }
    """
    print(f"   Querying BigQuery for {len(identifiers)} agents in this batch...")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Generate table UNION statements for the time range
    table_union = generate_table_union_statements(start_date, end_date)

    # Build identifier list for SQL IN clause (author matching only)
    author_list = ', '.join([f"'{id}'" for id in identifiers])

    # Build comprehensive query with CTE
    query = f"""
    WITH pr_events AS (
      -- Get all PR events (opened, closed) for all agents
      SELECT
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as pr_author,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at') as created_at,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS BOOL) as is_merged,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at') as merged_at,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
        JSON_EXTRACT_SCALAR(payload, '$.action') as action,
        created_at as event_time
      FROM (
        {table_union}
      ) t
      WHERE
        type = 'PullRequestEvent'
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IS NOT NULL
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') IN ({author_list})
    ),

    pr_latest_state AS (
      -- Get the latest state for each PR (most recent event)
      SELECT
        url,
        pr_author,
        created_at,
        merged_at,
        closed_at,
        ROW_NUMBER() OVER (PARTITION BY url ORDER BY event_time DESC) as row_num
      FROM pr_events
    )

    -- Return deduplicated PR metadata
    SELECT DISTINCT
      url,
      pr_author,
      created_at,
      merged_at,
      closed_at
    FROM pr_latest_state
    WHERE row_num = 1
    ORDER BY created_at DESC
    """

    print(f"   Scanning {(end_date - start_date).days} days of GitHub Archive data...")
    print(f"   Batch agents: {', '.join(identifiers[:5])}{'...' if len(identifiers) > 5 else ''}")

    try:
        query_job = client.query(query)
        results = list(query_job.result())

        print(f"   ✓ Found {len(results)} PRs in this batch")

        # Group results by agent
        metadata_by_agent = defaultdict(list)

        for row in results:
            # Convert datetime objects to ISO strings
            created_at = row.created_at
            if hasattr(created_at, 'isoformat'):
                created_at = created_at.isoformat()

            merged_at = row.merged_at
            if hasattr(merged_at, 'isoformat'):
                merged_at = merged_at.isoformat()

            closed_at = row.closed_at
            if hasattr(closed_at, 'isoformat'):
                closed_at = closed_at.isoformat()

            pr_data = {
                'html_url': row.url,
                'created_at': created_at,
                'merged_at': merged_at,
                'closed_at': closed_at,
            }

            # Assign to agent based on author
            pr_author = row.pr_author
            if pr_author and pr_author in identifiers:
                metadata_by_agent[pr_author].append(pr_data)

        # Print breakdown by agent (only show agents with PRs)
        print(f"   📊 Batch breakdown:")
        for identifier in identifiers:
            count = len(metadata_by_agent.get(identifier, []))
            if count > 0:
                metadata = metadata_by_agent[identifier]
                merged_count = sum(1 for m in metadata if m['merged_at'] is not None)
                closed_count = sum(1 for m in metadata if m['closed_at'] is not None and m['merged_at'] is None)
                open_count = count - merged_count - closed_count
                print(f"      {identifier}: {count} PRs ({merged_count} merged, {closed_count} closed, {open_count} open)")

        # Convert defaultdict to regular dict
        return dict(metadata_by_agent)

    except Exception as e:
        print(f"   ✗ BigQuery error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# HUGGINGFACE STORAGE FUNCTIONS
# =============================================================================

def group_metadata_by_date(metadata_list):
    """
    Group PR metadata by exact date (year.month.day) for efficient daily storage.
    Returns dict: {(year, month, day): [metadata_list]}
    """
    grouped = defaultdict(list)

    for pr_meta in metadata_list:
        created_at = pr_meta.get('created_at')
        if not created_at:
            continue

        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            key = (dt.year, dt.month, dt.day)
            grouped[key].append(pr_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{created_at}': {e}")

    return dict(grouped)


def save_pr_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save PR metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's PRs.

    This function OVERWRITES existing files completely with fresh data from BigQuery.
    Uses batch upload to avoid rate limit (uploads entire folder in single operation).

    Args:
        metadata_list: List of PR metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    import shutil

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)

        # Group by date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        if not grouped:
            print(f"   No valid metadata to save for {agent_identifier}")
            return False

        # Create a temporary directory for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        try:
            print(f"   📦 Preparing batch upload for {len(grouped)} daily files...")

            # Process each daily file
            for (pr_year, month, day), day_metadata in grouped.items():
                filename = f"{agent_identifier}/{pr_year}.{month:02d}.{day:02d}.jsonl"
                local_filename = os.path.join(agent_folder, f"{pr_year}.{month:02d}.{day:02d}.jsonl")

                # Sort by created_at for better organization
                day_metadata.sort(key=lambda x: x.get('created_at', ''), reverse=True)

                # Save to temp directory (complete overwrite, no merging)
                save_jsonl(local_filename, day_metadata)
                print(f"      Prepared {len(day_metadata)} PRs for {filename}")

            # Upload entire folder using upload_large_folder (optimized for large files)
            print(f"   📤 Uploading {len(grouped)} files ({len(metadata_list)} total PRs)...")
            upload_large_folder_with_backoff(
                api,
                folder_path=temp_dir,
                repo_id=PR_METADATA_REPO,
                repo_type="dataset"
            )
            print(f"   ✓ Batch upload complete for {agent_identifier}")

            return True

        finally:
            # Always clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"   ✗ Error saving PR metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_agents_from_hf():
    """
    Load all agent metadata JSON files from HuggingFace dataset.

    The github_identifier is extracted from the filename (e.g., 'agent-name[bot].json' -> 'agent-name[bot]')
    """
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = list_repo_files_with_backoff(api, repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        print(f"Found {len(json_files)} agent files in {AGENTS_REPO}")

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download_with_backoff(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)

                    # Only process agents with status == "public"
                    if agent_data.get('status') != 'public':
                        print(f"Skipping {json_file}: status is not 'public'")
                        continue

                    # Extract github_identifier from filename (remove .json extension)
                    github_identifier = json_file.replace('.json', '')
                    agent_data['github_identifier'] = github_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return []


# =============================================================================
# LEADERBOARD DATA COMPUTATION & STORAGE
# =============================================================================

def calculate_pr_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of PR metadata.

    Returns a dictionary with comprehensive PR metrics.
    Acceptance rate = merged PRs / (merged PRs + closed but not merged PRs) * 100
    """
    total_prs = len(metadata_list)
    merged = sum(1 for pr_meta in metadata_list if pr_meta.get('merged_at'))

    # Count closed PRs (rejected) - those with closed_at but no merged_at
    closed_not_merged = sum(1 for pr_meta in metadata_list
                           if pr_meta.get('closed_at') and not pr_meta.get('merged_at'))

    # Total decisions made = merged + closed (rejected)
    total_decisions = merged + closed_not_merged

    # Calculate acceptance rate based on decisions made
    acceptance_rate = (merged / total_decisions * 100) if total_decisions > 0 else 0

    return {
        'total_prs': total_prs,
        'merged_prs': merged,
        'acceptance_rate': round(acceptance_rate, 2),
    }


def calculate_monthly_metrics(all_metadata, agents):
    """
    Calculate monthly metrics for all agents for visualization.

    Args:
        all_metadata: List of all PR metadata with agent_identifier field
        agents: List of agent data dictionaries

    Returns:
        dict with monthly metrics organized by agent
    """
    from datetime import datetime, timezone

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {
        agent.get('github_identifier'): agent.get('name', 'Unknown')
        for agent in agents
        if agent.get('github_identifier')
    }

    # Group by agent and month
    agent_month_data = defaultdict(lambda: defaultdict(list))

    for pr_meta in all_metadata:
        agent_identifier = pr_meta.get('agent_identifier')
        created_at = pr_meta.get('created_at')

        if not agent_identifier or not created_at:
            continue

        # Get agent_name from identifier
        agent_name = identifier_to_name.get(agent_identifier, agent_identifier)

        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            month_key = f"{dt.year}-{dt.month:02d}"
            agent_month_data[agent_name][month_key].append(pr_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{created_at}': {e}")
            continue

    # Get all unique months and sort them
    all_months = set()
    for agent_data in agent_month_data.values():
        all_months.update(agent_data.keys())
    months = sorted(list(all_months))

    # Calculate metrics for each agent and month
    result_data = {}
    for agent_name, month_dict in agent_month_data.items():
        acceptance_rates = []
        total_prs = []
        merged_prs = []
        closed_not_merged_list = []

        for month in months:
            prs_in_month = month_dict.get(month, [])

            # Count merged PRs
            merged_count = sum(1 for pr in prs_in_month if pr.get('merged_at'))

            # Count closed but not merged
            closed_not_merged_count = sum(1 for pr in prs_in_month
                                         if pr.get('closed_at') and not pr.get('merged_at'))

            # Total PRs created in this month
            total_count = len(prs_in_month)

            # Calculate acceptance rate
            total_decisions = merged_count + closed_not_merged_count
            acceptance_rate = (merged_count / total_decisions * 100) if total_decisions > 0 else None

            acceptance_rates.append(acceptance_rate)
            total_prs.append(total_count)
            merged_prs.append(merged_count)
            closed_not_merged_list.append(closed_not_merged_count)

        result_data[agent_name] = {
            'acceptance_rates': acceptance_rates,
            'total_prs': total_prs,
            'merged_prs': merged_prs,
            'closed_not_merged': closed_not_merged_list
        }

    agents_list = sorted(list(agent_month_data.keys()))

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


def load_all_pr_metadata_from_hf(agents):
    """
    Load all PR metadata from HuggingFace dataset for all agents.

    Args:
        agents: List of agent dictionaries with github_identifier

    Returns:
        List of PR metadata with agent_identifier field added
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

        # List all files in the repository
        files = list_repo_files_with_backoff(api, repo_id=PR_METADATA_REPO, repo_type="dataset")

        # Filter for files within the time frame
        relevant_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:
                    filename = parts[1]
                    try:
                        date_part = filename.replace('.jsonl', '')
                        date_components = date_part.split('.')
                        if len(date_components) == 3:
                            file_year, file_month, file_day = map(int, date_components)
                            file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                            if file_date >= cutoff_date:
                                relevant_files.append(f)
                    except Exception:
                        continue

        print(f"\n📥 Loading PR metadata from {len(relevant_files)} daily files...")

        all_metadata = []
        for filename in relevant_files:
            try:
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                agent_identifier = parts[0]

                file_path = hf_hub_download_with_backoff(
                    repo_id=PR_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                day_metadata = load_jsonl(file_path)

                # Add agent_identifier to each PR
                for pr_meta in day_metadata:
                    created_at = pr_meta.get('created_at')
                    if created_at:
                        try:
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if dt >= cutoff_date:
                                pr_meta['agent_identifier'] = agent_identifier
                                all_metadata.append(pr_meta)
                        except Exception:
                            continue

            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total PRs")
        return all_metadata

    except Exception as e:
        print(f"✗ Error loading PR metadata: {str(e)}")
        return []


def construct_leaderboard_from_metadata(all_metadata, agents):
    """
    Construct leaderboard data from PR metadata.

    Args:
        all_metadata: List of PR metadata with agent_identifier field
        agents: List of agent dictionaries

    Returns:
        Dictionary mapping agent identifier to stats
    """
    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('name', 'Unknown')

        # Filter metadata for this agent
        bot_metadata = [pr for pr in all_metadata if pr.get('agent_identifier') == identifier]

        # Calculate stats
        stats = calculate_pr_stats_from_metadata(bot_metadata)

        cache_dict[identifier] = {
            'name': agent_name,
            'website': agent.get('website', 'Unknown'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


def save_leaderboard_data_to_hf(leaderboard_data, monthly_metrics):
    """
    Save computed leaderboard and monthly metrics to HuggingFace dataset as swe-pr.json.

    Args:
        leaderboard_data: Dictionary with agent stats (from construct_leaderboard)
        monthly_metrics: Dictionary with monthly metrics (from calculate_monthly_metrics)

    Returns:
        True if successful, False otherwise
    """
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)

        # Combine data into single JSON structure
        combined_data = {
            'leaderboard': leaderboard_data,
            'monthly_metrics': monthly_metrics,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        try:
            json.dump(combined_data, temp_file, indent=2)
            temp_file.close()

            # Upload to HuggingFace
            print(f"\n📤 Uploading leaderboard data to {LEADERBOARD_REPO}/swe-pr.json...")
            upload_file_with_backoff(
                api,
                path_or_fileobj=temp_file.name,
                path_in_repo="swe-pr.json",
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset"
            )
            print(f"✓ Leaderboard data uploaded successfully")
            return True

        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    except Exception as e:
        print(f"✗ Error saving leaderboard data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine PR metadata for all agents within LEADERBOARD_TIME_FRAME_DAYS and save to HuggingFace.
    Uses ONE BigQuery query for ALL agents (most efficient approach).
    """
    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    # Extract all identifiers
    identifiers = [agent['github_identifier'] for agent in agents if agent.get('github_identifier')]
    if not identifiers:
        print("No valid agent identifiers found")
        return

    print(f"\n{'='*80}")
    print(f"Starting PR metadata mining for {len(identifiers)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"Data source: BigQuery + GitHub Archive (BATCHED QUERIES)")
    print(f"{'='*80}\n")

    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        # Use batched approach for better performance
        # upload_immediately=True means each batch uploads to HuggingFace right after BigQuery completes
        all_metadata = fetch_issue_metadata_batched(
            client, identifiers, start_date, end_date, batch_size=100, upload_immediately=True
        )

        # Calculate summary statistics
        total_prs = sum(len(metadata_list) for metadata_list in all_metadata.values())
        agents_with_data = sum(1 for metadata_list in all_metadata.values() if metadata_list)

        print(f"\n{'='*80}")
        print(f"✅ BigQuery mining and upload complete!")
        print(f"   Total agents: {len(agents)}")
        print(f"   Agents with data: {agents_with_data}")
        print(f"   Total PRs found: {total_prs}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"✗ Error during BigQuery fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Compute and save leaderboard data
    print(f"\n{'='*80}")
    print(f"📊 Computing leaderboard and monthly metrics...")
    print(f"{'='*80}\n")

    try:
        # Load all PR metadata from HuggingFace
        all_pr_metadata = load_all_pr_metadata_from_hf(agents)

        if all_pr_metadata:
            # Construct leaderboard
            leaderboard_data = construct_leaderboard_from_metadata(all_pr_metadata, agents)
            print(f"✓ Computed leaderboard for {len(leaderboard_data)} agents")

            # Calculate monthly metrics
            monthly_metrics = calculate_monthly_metrics(all_pr_metadata, agents)
            print(f"✓ Computed monthly metrics for {len(monthly_metrics['agents'])} agents across {len(monthly_metrics['months'])} months")

            # Save to HuggingFace
            if save_leaderboard_data_to_hf(leaderboard_data, monthly_metrics):
                print(f"\n{'='*80}")
                print(f"✅ Leaderboard data saved successfully!")
                print(f"{'='*80}\n")
            else:
                print(f"\n{'='*80}")
                print(f"⚠️  Warning: Failed to save leaderboard data")
                print(f"{'='*80}\n")
        else:
            print(f"⚠️  No PR metadata found to compute leaderboard")

    except Exception as e:
        print(f"✗ Error computing/saving leaderboard data: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
