import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter
import json
import os
import time
import tempfile
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"  # HuggingFace dataset for agent metadata
PR_METADATA_REPO = "SWE-Arena/pr_metadata"  # HuggingFace dataset for PR metadata
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for constructing leaderboard
UPDATE_TIME_FRAME_DAYS = 30  # Time frame for mining new PRs

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
    ("Total PRs", "number"),
    ("Merged PRs", "number"),
    ("Acceptance Rate (%)", "number"),
]

# =============================================================================
# JSONL FILE OPERATIONS
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
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def parse_date_string(date_string):
    """
    Parse date string to datetime object, handling various formats.

    Handles:
    - ISO format with 'T' or space between date and time
    - Timezone with 'Z' or incomplete offset (+00, -00)
    - Complete timezone offset (+00:00, -00:00)

    Args:
        date_string: Date string in various formats

    Returns:
        datetime object or raises exception
    """
    if not date_string:
        raise ValueError("Empty date string")

    # Replace space with 'T' for ISO format compatibility
    date_string = date_string.replace(' ', 'T')

    # Fix incomplete timezone offset (+00 or -00 -> +00:00 or -00:00)
    if date_string[-3:-2] in ('+', '-') and ':' not in date_string[-3:]:
        date_string = date_string + ':00'

    # Parse the date string (handles both with and without microseconds)
    dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))

    return dt


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

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


def fetch_all_pr_metadata_single_query(client, identifiers, start_date, end_date):
    """
    Fetch PR metadata for ALL agents using ONE comprehensive BigQuery query.

    This query fetches:
    1. PRs authored by agents (user.login matches identifier)
    2. PRs with co-authored-by (search in body for co-authored-by)
    3. PRs from branches starting with agent identifier (head.ref pattern)

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        Dictionary mapping agent identifier to list of PR metadata
    """
    print(f"\n🔍 Querying BigQuery for ALL {len(identifiers)} agents in ONE QUERY")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Generate table UNION statements for the time range
    table_union = generate_table_union_statements(start_date, end_date)

    # Build identifier lists for SQL IN clauses
    # For author matching, include identifiers with [bot]
    author_list = ', '.join([f"'{id}'" for id in identifiers if '[bot]' in id])

    # For branch matching and co-author, use stripped identifiers (without [bot])
    stripped_identifiers = [id.replace('[bot]', '') for id in identifiers]

    # Build co-author pattern (search in body)
    coauthor_patterns = ' OR '.join([f"LOWER(JSON_EXTRACT_SCALAR(payload, '$.pull_request.body')) LIKE '%co-authored-by: {id.lower()}%'"
                                      for id in stripped_identifiers if id])

    # Build branch pattern
    branch_patterns = ' OR '.join([f"JSON_EXTRACT_SCALAR(payload, '$.pull_request.head.ref') LIKE '{id}/%'"
                                   for id in stripped_identifiers if id])

    # Build comprehensive query with CTE
    query = f"""
    WITH pr_events AS (
      -- Get all PR events (opened, closed) for all agents
      SELECT
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as html_url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as pr_author,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.head.ref') as branch_name,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.body') as pr_body,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at') as created_at,
        CAST(JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged') AS BOOL) as is_merged,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.merged_at') as merged_at,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
        JSON_EXTRACT_SCALAR(payload, '$.action') as action,
        created_at as event_time
      FROM (
        {table_union}
      )
      WHERE
        type = 'PullRequestEvent'
        AND JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') IS NOT NULL
        AND (
          -- Match PRs authored by agents with [bot] suffix
          {f"JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') IN ({author_list})" if author_list else "FALSE"}
          {" OR " if author_list and (coauthor_patterns or branch_patterns) else ""}
          -- Match PRs with co-authored-by in body
          {f"({coauthor_patterns})" if coauthor_patterns else ""}
          {" OR " if coauthor_patterns and branch_patterns else ""}
          -- Match PRs with branch names starting with agent identifier
          {f"({branch_patterns})" if branch_patterns else ""}
        )
    ),

    pr_latest_state AS (
      -- Get the latest state for each PR (most recent event)
      SELECT
        html_url,
        pr_author,
        branch_name,
        pr_body,
        created_at,
        merged_at,
        closed_at,
        ROW_NUMBER() OVER (PARTITION BY html_url ORDER BY event_time DESC) as row_num
      FROM pr_events
    )

    -- Return deduplicated PR metadata
    SELECT DISTINCT
      html_url,
      pr_author,
      branch_name,
      pr_body,
      created_at,
      merged_at,
      -- Only include closed_at if PR is closed but not merged
      CASE
        WHEN merged_at IS NOT NULL THEN NULL
        ELSE closed_at
      END as closed_at
    FROM pr_latest_state
    WHERE row_num = 1
    ORDER BY created_at DESC
    """

    print(f"   Querying {(end_date - start_date).days} days of GitHub Archive data...")
    print(f"   Agents: {', '.join(identifiers[:5])}{'...' if len(identifiers) > 5 else ''}")

    try:
        query_job = client.query(query)
        results = list(query_job.result())

        print(f"   ✓ Found {len(results)} total PRs across all agents")

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
                'html_url': row.html_url,
                'created_at': created_at,
                'merged_at': merged_at,
                'closed_at': closed_at,
            }

            # Assign to agent based on author, co-author, or branch pattern
            pr_author = row.pr_author
            branch_name = row.branch_name or ''
            pr_body = (row.pr_body or '').lower()

            # First, try to match by author
            if pr_author and pr_author in identifiers:
                metadata_by_agent[pr_author].append(pr_data)
            else:
                # Try to match by co-author or branch pattern
                for identifier in identifiers:
                    stripped_id = identifier.replace('[bot]', '')
                    if not stripped_id:
                        continue

                    # Check co-author
                    if f'co-authored-by: {stripped_id.lower()}' in pr_body:
                        metadata_by_agent[identifier].append(pr_data)
                        break

                    # Check branch pattern
                    if branch_name.startswith(f"{stripped_id}/"):
                        metadata_by_agent[identifier].append(pr_data)
                        break

        # Print breakdown by agent
        print(f"\n   📊 Results breakdown by agent:")
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
# GITHUB API OPERATIONS (Minimal - Only for Validation)
# =============================================================================

def get_github_token():
    """Get first GitHub token from environment variables."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. Validation will be limited.")
    return token


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists (simple validation)."""
    try:
        token = get_github_token()
        headers = {'Authorization': f'token {token}'} if token else {}
        url = f'https://api.github.com/users/{identifier}'

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub identifier not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# PR STATISTICS
# =============================================================================

def calculate_pr_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of PR metadata (lightweight objects).
    Works with minimal metadata: html_url, created_at, merged_at, closed_at.

    Returns a dictionary with comprehensive PR metrics.

    Acceptance rate is calculated as:
        merged PRs / (merged PRs + closed but not merged PRs) * 100

    This only counts PRs where a decision has been made (either merged or rejected/closed).
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


def calculate_monthly_metrics_by_agent(top_n=None):
    """
    Calculate monthly metrics for all agents (or top N agents) for visualization.
    Loads data directly from SWE-Arena/pr_metadata dataset.

    Args:
        top_n: If specified, only return metrics for the top N agents by total PRs.
               Agents are ranked by their total PR count across all months.

    Returns:
        dict: {
            'agents': list of agent names,
            'months': list of month labels (e.g., '2025-01'),
            'data': {
                agent_name: {
                    'acceptance_rates': list of acceptance rates by month,
                    'total_prs': list of PR counts by month,
                    'merged_prs': list of merged PR counts by month,
                    'closed_not_merged': list of closed but not merged PR counts by month
                }
            }
        }
    """
    # Load ALL agents from HuggingFace agents repo
    agents = load_agents_from_hf()

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {agent.get('github_identifier'): agent.get('agent_name') for agent in agents if agent.get('github_identifier')}

    # Load all PR metadata from pr_metadata dataset
    all_metadata = load_pr_metadata()

    if not all_metadata:
        return {'agents': [], 'months': [], 'data': {}}

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
            dt = parse_date_string(created_at)
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

    # Filter to top N agents if specified
    agents_list = sorted(list(agent_month_data.keys()))
    if top_n is not None and top_n > 0:
        # Calculate total PRs for each agent across all months
        agent_totals = []
        for agent_name in agents_list:
            total_pr_count = sum(result_data[agent_name]['total_prs'])
            agent_totals.append((agent_name, total_pr_count))

        # Sort by total PRs (descending) and take top N
        agent_totals.sort(key=lambda x: x[1], reverse=True)
        top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

        # Filter result_data to only include top agents
        result_data = {agent: result_data[agent] for agent in top_agents if agent in result_data}
        agents_list = top_agents

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


# =============================================================================
# PR METADATA STORAGE & RETRIEVAL
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
            dt = parse_date_string(created_at)
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
            api.upload_large_folder(
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


def load_pr_metadata():
    """
    Loads PR metadata from the last LEADERBOARD_TIME_FRAME_DAYS only.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Returns:
        List of dictionaries with 'agent_identifier' added to each PR metadata.
        Only includes PRs within the last LEADERBOARD_TIME_FRAME_DAYS.
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate cutoff date for filtering
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

        # List all files in the repository
        files = api.list_repo_files(repo_id=PR_METADATA_REPO, repo_type="dataset")

        # Filter for files within the time frame: [agent_identifier]/YYYY.MM.DD.jsonl
        # Parse date from filename and only include files within LEADERBOARD_TIME_FRAME_DAYS
        relevant_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:  # [agent_identifier]/YYYY.MM.DD.jsonl
                    filename = parts[1]
                    try:
                        # Parse date from filename: YYYY.MM.DD.jsonl
                        date_part = filename.replace('.jsonl', '')  # Get YYYY.MM.DD
                        date_components = date_part.split('.')
                        if len(date_components) == 3:
                            file_year, file_month, file_day = map(int, date_components)
                            file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                            # Only include files within the time frame
                            if file_date >= cutoff_date:
                                relevant_files.append(f)
                    except Exception:
                        # If date parsing fails, skip this file
                        continue

        total_months = LEADERBOARD_TIME_FRAME_DAYS // 30
        print(f"📥 Loading PR metadata from last {total_months} months ({len(relevant_files)} daily files across all agents)...")

        all_metadata = []
        for filename in relevant_files:
            try:
                # Extract agent_identifier from path (first part)
                # Format: agent_identifier/YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    print(f"   Warning: Unexpected filename format: {filename}")
                    continue

                agent_identifier = parts[0]

                file_path = hf_hub_download(
                    repo_id=PR_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                day_metadata = load_jsonl(file_path)

                # Filter individual PRs by created_at date as a double-check
                for pr_meta in day_metadata:
                    created_at = pr_meta.get('created_at')
                    if created_at:
                        try:
                            dt = parse_date_string(created_at)
                            if dt >= cutoff_date:
                                pr_meta['agent_identifier'] = agent_identifier
                                all_metadata.append(pr_meta)
                        except Exception:
                            # If date parsing fails, skip this PR
                            continue
                    else:
                        # If no created_at, skip this PR
                        continue

                print(f"   ✓ Loaded PRs from {filename}")
            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total PRs from last {total_months} months")
        return all_metadata

    except Exception as e:
        total_months = LEADERBOARD_TIME_FRAME_DAYS // 30
        print(f"✗ Error loading PR metadata from last {total_months} months: {str(e)}")
        return []


def get_daily_files_last_time_frame(agent_identifier):
    """
    Get list of daily file paths for an agent from the configured time frame.

    Args:
        agent_identifier: GitHub identifier of the agent

    Returns:
        List of file paths in format: [agent_identifier]/YYYY.MM.DD.jsonl
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate date range using configured time frame
        today = datetime.now(timezone.utc)
        cutoff_date = today - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

        # List all files in the repository
        files = api.list_repo_files(repo_id=PR_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        # Filter by date range (extract date from filename)
        recent_files = []
        for filename in agent_files:
            try:
                # Extract date from filename: YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                date_part = parts[1].replace('.jsonl', '')  # Get YYYY.MM.DD
                date_components = date_part.split('.')
                if len(date_components) != 3:
                    continue

                file_year, file_month, file_day = map(int, date_components)
                file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                # Include if within configured time frame
                if cutoff_date <= file_date <= today:
                    recent_files.append(filename)
            except Exception:
                continue

        return recent_files

    except Exception as e:
        print(f"Error getting daily files: {str(e)}")
        return []


# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def load_agents_from_hf():
    """Load all agent metadata JSON files from HuggingFace dataset."""
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = api.list_repo_files(repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        print(f"Found {len(json_files)} agent files in {AGENTS_REPO}")

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)

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
        return None


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.

    Args:
        api: HfApi instance
        path_or_fileobj: Local file path to upload
        path_in_repo: Target path in the repository
        repo_id: Repository ID
        repo_type: Type of repository (e.g., "dataset")
        token: HuggingFace token
        max_retries: Maximum number of retry attempts

    Returns:
        True if upload succeeded, raises exception if all retries failed
    """
    delay = 2.0  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            if attempt > 0:
                print(f"   ✓ Upload succeeded on attempt {attempt + 1}/{max_retries}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay + random.uniform(0, 1.0)
                print(f"   ⚠️ Upload failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"   ⏳ Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_agent_to_hf(data):
    """Save a new agent to HuggingFace dataset as {identifier}.json in root."""
    try:
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        identifier = data['github_identifier']
        filename = f"{identifier}.json"

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            # Upload to HuggingFace (root directory)
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=AGENTS_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"✓ Saved agent to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"✗ Error saving agent: {str(e)}")
        return False


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def mine_all_agents():
    """
    Mine PR metadata for all agents within UPDATE_TIME_FRAME_DAYS and save to HuggingFace.
    Uses ONE BigQuery query for ALL agents (most efficient approach).

    This runs weekly to refresh the data with the latest PRs from the past UPDATE_TIME_FRAME_DAYS.
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
    print(f"Time frame: Last {UPDATE_TIME_FRAME_DAYS} days")
    print(f"Data source: BigQuery + GitHub Archive (ONE QUERY FOR ALL AGENTS)")
    print(f"{'='*80}\n")

    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return

    # Define time range: past UPDATE_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=UPDATE_TIME_FRAME_DAYS)

    try:
        all_metadata = fetch_all_pr_metadata_single_query(
            client, identifiers, start_date, end_date
        )
    except Exception as e:
        print(f"✗ Error during BigQuery fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Save results for each agent
    print(f"\n{'='*80}")
    print(f"💾 Saving results to HuggingFace for each agent...")
    print(f"{'='*80}\n")

    success_count = 0
    error_count = 0
    no_data_count = 0

    for i, agent in enumerate(agents, 1):
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        if not identifier:
            print(f"[{i}/{len(agents)}] Skipping agent without identifier")
            error_count += 1
            continue

        metadata = all_metadata.get(identifier, [])

        print(f"[{i}/{len(agents)}] {agent_name} ({identifier}):")

        try:
            if metadata:
                print(f"   💾 Saving {len(metadata)} PR records...")
                if save_pr_metadata_to_hf(metadata, identifier):
                    success_count += 1
                else:
                    error_count += 1
            else:
                print(f"   No PRs found")
                no_data_count += 1

        except Exception as e:
            print(f"   ✗ Error saving {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print(f"\n{'='*80}")
    print(f"✅ Mining complete!")
    print(f"   Total agents: {len(agents)}")
    print(f"   Successfully saved: {success_count}")
    print(f"   No data (skipped): {no_data_count}")
    print(f"   Errors: {error_count}")
    print(f"   BigQuery queries executed: 1")
    print(f"{'='*80}\n")


def construct_leaderboard_from_metadata():
    """
    Construct leaderboard from stored PR metadata instead of fetching all PRs.
    Much more memory-efficient and faster.

    Returns dictionary of agent stats.
    """
    print("📊 Constructing leaderboard from PR metadata...")
    # Load agents
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found")
        return {}

    # Load all PR metadata
    all_metadata = load_pr_metadata()

    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        # Filter metadata for this agent
        agent_metadata = [pr for pr in all_metadata if pr.get('agent_identifier') == identifier]

        # Calculate stats
        stats = calculate_pr_stats_from_metadata(agent_metadata)

        cache_dict[identifier] = {
            'agent_name': agent_name,
            'website': agent.get('website', 'Unknown'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def generate_color(index, total):
    """Generate distinct colors using HSL color space for better distribution"""
    hue = (index * 360 / total) % 360
    saturation = 70 + (index % 3) * 10  # Vary saturation slightly
    lightness = 45 + (index % 2) * 10   # Vary lightness slightly
    return f'hsl({hue}, {saturation}%, {lightness}%)'


def create_monthly_metrics_plot(top_n=5):
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Acceptance rate (%) as line curves
    - Right y-axis: Total PRs created as bar charts

    Each agent gets a unique color for both their line and bars.

    Args:
        top_n: Number of top agents to show (default: 5)
    """
    metrics = calculate_monthly_metrics_by_agent(top_n=top_n)

    if not metrics['agents'] or not metrics['months']:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Generate colors for all agents using HSL
    agent_colors = {agent: generate_color(idx, len(agents)) for idx, agent in enumerate(agents)}

    # Add traces for each agent
    for idx, agent_name in enumerate(agents):
        color = agent_colors[agent_name]
        agent_data = data[agent_name]

        # Add line trace for acceptance rate (left y-axis)
        acceptance_rates = agent_data['acceptance_rates']
        # Filter out None values for plotting
        x_acceptance = [month for month, rate in zip(months, acceptance_rates) if rate is not None]
        y_acceptance = [rate for rate in acceptance_rates if rate is not None]

        if x_acceptance and y_acceptance:  # Only add trace if there's data
            fig.add_trace(
                go.Scatter(
                    x=x_acceptance,
                    y=y_acceptance,
                    name=agent_name,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup=agent_name,
                    showlegend=True,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Acceptance Rate: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=False
            )

        # Add bar trace for total PRs (right y-axis)
        # Only show bars for months where agent has PRs
        x_bars = []
        y_bars = []
        for month, count in zip(months, agent_data['total_prs']):
            if count > 0:  # Only include months with PRs
                x_bars.append(month)
                y_bars.append(count)

        if x_bars and y_bars:  # Only add trace if there's data
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=f"{agent_name} (PRs)",
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=agent_name,
                    showlegend=False,  # Don't show in legend (already shown for line)
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Total PRs: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name  # Group bars by agent for proper spacing
                ),
                secondary_y=True
            )

    # Update axes labels
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text="<b>Acceptance Rate (%)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Total PRs</b>", secondary_y=True)

    # Update layout
    fig.update_layout(
        title=None,
        hovermode='closest',
        barmode='group',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    return fig


def get_leaderboard_dataframe():
    """
    Construct leaderboard data from PR metadata and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by total PRs.
    """
    # Construct leaderboard from PR metadata
    cache_dict = construct_leaderboard_from_metadata()

    if not cache_dict:
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    for identifier, data in cache_dict.items():
        # Filter out agents with zero total PRs
        if data.get('total_prs', 0) > 0:
            # Only include display-relevant fields
            rows.append([
                data.get('agent_name', 'Unknown'),
                data.get('website', 'Unknown'),
                data.get('total_prs', 0),
                data.get('merged_prs', 0),
                data.get('acceptance_rate', 0.0),
            ])

    # Create DataFrame
    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Total PRs", "Merged PRs", "Acceptance Rate (%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Total PRs descending
    if "Total PRs" in df.columns and not df.empty:
        df = df.sort_values(by="Total PRs", ascending=False).reset_index(drop=True)

    return df


def submit_agent(identifier, agent_name, organization, description, website):
    """
    Submit a new agent to the leaderboard.
    Validates input and saves submission.
    PR data will be populated by the weekly mining task.
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "❌ GitHub identifier is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not organization or not organization.strip():
        return "❌ Organization name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not website or not website.strip():
        return "❌ Website URL is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    description = description.strip()
    website = website.strip()

    # Validate GitHub identifier
    is_valid, message = validate_github_username(identifier)
    if not is_valid:
        return f"❌ {message}", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Check for duplicates by loading agents from HuggingFace
    agents = load_agents_from_hf()
    if agents:
        existing_names = {agent['github_identifier'] for agent in agents}
        if identifier in existing_names:
            return f"⚠️ Agent with identifier '{identifier}' already exists", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Create submission
    submission = {
        'agent_name': agent_name,
        'organization': organization,
        'github_identifier': identifier,
        'description': description,
        'website': website,
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "❌ Failed to save submission", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    success_msg = f"✅ Successfully submitted {agent_name}!\n\nPR data will be populated by the weekly mining task (runs every Monday at 12:00 AM UTC)."
    return success_msg, get_leaderboard_dataframe(), create_monthly_metrics_plot()


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

print(f"\n🚀 Starting SWE Agent PR Leaderboard")
print(f"   Leaderboard time frame: {LEADERBOARD_TIME_FRAME_DAYS} days ({LEADERBOARD_TIME_FRAME_DAYS // 30} months)")
print(f"   Mining update frequency: Every {UPDATE_TIME_FRAME_DAYS} days\n")

# Start APScheduler for weekly PR mining at 12:00 AM UTC every Monday
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    mine_all_agents,
    trigger=CronTrigger(day_of_week='mon', hour=0, minute=0),  # 12:00 AM UTC every Monday
    id='weekly_pr_mining',
    name='Weekly PR Mining',
    replace_existing=True
)
scheduler.start()
print(f"\n{'='*80}")
print(f"✓ Scheduler initialized successfully")
print(f"⛏️  Mining schedule: Every Monday at 12:00 AM UTC")
print(f"📥 On startup: Only loads cached data from HuggingFace (no mining)")
print(f"{'='*80}\n")

# Create Gradio interface
with gr.Blocks(title="SWE Agent PR Leaderboard", theme=gr.themes.Soft()) as app:
    total_months = LEADERBOARD_TIME_FRAME_DAYS // 30

    gr.Markdown("# 🏆 SWE Agent PR Leaderboard")
    gr.Markdown(f"Track and compare GitHub pull request statistics for SWE agents")

    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            gr.Markdown(f"*All statistics are based on PRs from the last {total_months} months*")

            leaderboard_table = Leaderboard(
                value=get_leaderboard_dataframe(),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=[
                    ColumnFilter(
                        "Acceptance Rate (%)",
                        min=0,
                        max=100,
                        default=[0, 100],
                        type="slider",
                        label="Acceptance Rate (%)"
                    )
                ]
            )

            gr.Markdown("### Monthly Metrics - Top 5 Agents")
            gr.Markdown("Track acceptance rates and PR activity over time for the most active agents")

            monthly_plot = gr.Plot(
                value=create_monthly_metrics_plot(top_n=5),
                label="Monthly PR Metrics"
            )

        # Submit Agent Tab
        with gr.Tab("➕ Submit Agent"):

            gr.Markdown("### Submit Your Agent")
            gr.Markdown("Fill in the details below to add your agent to the leaderboard.")

            with gr.Row():
                with gr.Column():
                    github_input = gr.Textbox(
                        label="GitHub Identifier*",
                        placeholder="Your agent username (e.g., my-agent-bot)"
                    )
                    name_input = gr.Textbox(
                        label="Agent Name*",
                        placeholder="Your agent's display name"
                    )

                with gr.Column():
                    organization_input = gr.Textbox(
                        label="Organization*",
                        placeholder="Your organization or team name"
                    )
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of your agent",
                        lines=3
                    )
                    website_input = gr.Textbox(
                        label="Website*",
                        placeholder="https://your-agent-website.com"
                    )

            submit_button = gr.Button(
                "Submit Agent",
                variant="primary"
            )
            submission_status = gr.Textbox(
                label="Submission Status",
                interactive=False
            )

            # Event handler
            submit_button.click(
                fn=submit_agent,
                inputs=[github_input, name_input, organization_input, description_input, website_input],
                outputs=[submission_status, leaderboard_table, monthly_plot]
            )


# Launch application
if __name__ == "__main__":
    app.launch()
