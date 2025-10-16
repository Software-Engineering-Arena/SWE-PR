import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import threading
from dotenv import load_dotenv
import pandas as pd
import random
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='SWE Agent PR Leaderboard')
parser.add_argument('--debug', '--DEBUG', action='store_true',
                    help='Enable debug mode (limits PR retrieval to 10 per query pattern)')
parser.add_argument('--no-debug', '--production', action='store_true',
                    help='Explicitly disable debug mode (force production mode)')
args = parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

# DEBUG MODE: Set to True to limit PR retrieval for testing
# When enabled, only fetches up to 10 PRs per query pattern per agent
# Priority: 1) Command-line args, 2) Environment variable, 3) Default (False)
if args.no_debug:
    DEBUG_MODE = False
elif args.debug:
    DEBUG_MODE = True
else:
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 'yes')

# In-memory cache for debug mode (data persists during session but NOT saved to HF)
DEBUG_LEADERBOARD_CACHE = {}
DEBUG_PR_METADATA_CACHE = defaultdict(list)

AGENTS_REPO = "SWE-Arena/pr_agents"  # HuggingFace dataset for agent metadata
LEADERBOARD_REPO = "SWE-Arena/pr_leaderboard"
PR_METADATA_REPO = "SWE-Arena/pr_metadata"  # HuggingFace dataset for PR metadata
UPDATE_INTERVAL = 86400  # 24 hours in seconds

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Organization", "string"),
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
                    # Migration: rename 'developer' to 'organization' if needed
                    if 'developer' in entry and 'organization' not in entry:
                        entry['organization'] = entry.pop('developer')
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def cache_to_dict(cache_list):
    """Convert list of cache entries to dictionary by identifier."""
    return {entry['github_identifier']: entry for entry in cache_list}


def dict_to_cache(cache_dict):
    """Convert dictionary back to list of values."""
    return list(cache_dict.values())


def normalize_date_format(date_string):
    """
    Convert date strings to standardized ISO 8601 format with Z suffix.
    Handles both old format (2025-10-15T23:23:47.983068) and new format (2025-10-15T23:23:47Z).
    """
    if not date_string or date_string == 'N/A':
        return 'N/A'
    
    try:
        # Parse the date string (handles both with and without microseconds)
        if '.' in date_string:
            # Old format with microseconds
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        else:
            # Already in correct format or GitHub format
            return date_string
        
        # Convert to standardized format
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


# =============================================================================
# GITHUB API OPERATIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Returns the final requests.Response on success or non-retryable status, or None after exhausting retries.
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers or {},
                params=params,
                json=json_body,
                data=data,
                timeout=timeout
            )

            status = resp.status_code

            # Success
            if 200 <= status < 300:
                return resp

            # Rate limits or server errors -> retry with backoff
            if status in (403, 429) or 500 <= status < 600:
                wait = None

                # Prefer Retry-After when present
                retry_after = resp.headers.get('Retry-After') or resp.headers.get('retry-after')
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = None

                # Fallback to X-RateLimit-Reset when 403/429
                if wait is None and status in (403, 429):
                    reset_hdr = resp.headers.get('X-RateLimit-Reset') or resp.headers.get('x-ratelimit-reset')
                    if reset_hdr:
                        try:
                            reset_ts = int(float(reset_hdr))
                            wait = max(reset_ts - time.time() + 2, 1)
                        except Exception:
                            wait = None

                # Final fallback: exponential backoff with jitter
                if wait is None:
                    wait = delay + random.uniform(0, 0.5)

                # Cap individual wait to avoid extreme sleeps
                wait = max(1.0, min(wait, 120.0))
                print(f"GitHub API {status}. Backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            # Non-retryable error; return response for caller to handle
            return resp

        except requests.RequestException as e:
            # Network error -> retry with backoff
            wait = delay + random.uniform(0, 0.5)
            wait = max(1.0, min(wait, 60.0))
            print(f"Request error: {e}. Retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)

    print(f"Exceeded max retries for {url}")
    return None

def get_github_token():
    """Get GitHub token from environment variables."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. API rate limits: 60/hour (authenticated: 5000/hour)")
    return token


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists with backoff-aware requests."""
    try:
        token = get_github_token()
        headers = {'Authorization': f'token {token}'} if token else {}
        url = f'https://api.github.com/users/{identifier}'
        response = request_with_backoff('GET', url, headers=headers, max_retries=1)
        if response is None:
            return False, "Validation error: network/rate limit exhausted"
        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub identifier not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def fetch_prs_with_time_partition(base_query, start_date, end_date, headers, prs_by_id, debug_limit=None):
    """
    Fetch PRs within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.

    Args:
        debug_limit: If set, stops fetching after this many PRs (for testing)

    Returns the number of PRs found in this time partition.
    """
    # Format dates for GitHub search (YYYY-MM-DD)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Add date range to query
    query = f'{base_query} created:{start_str}..{end_str}'

    print(f"  Searching range {start_str} to {end_str}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
        # Check debug limit
        if debug_limit is not None and total_in_partition >= debug_limit:
            print(f"    🐛 DEBUG MODE: Reached limit of {debug_limit} PRs, stopping...")
            return total_in_partition
        url = 'https://api.github.com/search/issues'
        params = {
            'q': query,
            'per_page': per_page,
            'page': page,
            'sort': 'created',
            'order': 'asc'
        }

        try:
            response = request_with_backoff('GET', url, headers=headers, params=params)
            if response is None:
                print(f"    Error: retries exhausted for range {start_str} to {end_str}")
                return total_in_partition

            if response.status_code != 200:
                print(f"    Error: HTTP {response.status_code} for range {start_str} to {end_str}")
                return total_in_partition

            data = response.json()
            total_count = data.get('total_count', 0)
            items = data.get('items', [])

            if not items:
                break

            # Add PRs to global dict
            for pr in items:
                pr_id = pr.get('id')
                if pr_id and pr_id not in prs_by_id:
                    prs_by_id[pr_id] = pr
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"    ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                # Calculate midpoint
                time_diff = end_date - start_date
                mid_date = start_date + time_diff / 2

                # Recursively fetch both halves
                count1 = fetch_prs_with_time_partition(base_query, start_date, mid_date, headers, prs_by_id, debug_limit)
                count2 = fetch_prs_with_time_partition(base_query, mid_date + timedelta(days=1), end_date, headers, prs_by_id, debug_limit)

                return count1 + count2

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"    Error fetching range {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"    ✓ Found {total_in_partition} PRs in range {start_str} to {end_str}")

    return total_in_partition


def extract_pr_metadata(pr):
    """
    Extract minimal PR metadata for efficient storage.
    Only keeps essential fields: html_url, created_at, merged_at, closed_at.
    Note: agent_name is not stored as it's inferred from the folder structure.
    """
    pull_request = pr.get('pull_request', {})

    # Extract dates
    created_at = pr.get('created_at')
    merged_at = pull_request.get('merged_at')
    closed_at = pr.get('closed_at')

    # Only store closed_at if PR is closed but not merged
    if merged_at:
        closed_at = None  # Don't store redundant info

    return {
        'html_url': pr.get('html_url'),
        'created_at': created_at,
        'merged_at': merged_at,
        'closed_at': closed_at
    }


def fetch_all_prs_metadata(identifier, agent_name, token=None, start_from_date=None, year=None):
    """
    Fetch pull requests associated with a GitHub user/bot for the past 6 months.
    Returns lightweight metadata instead of full PR objects.

    Uses time-based partitioning to bypass GitHub's 1000-result limit per query.
    Searches using multiple query patterns:
    - is:pr author:{identifier} (authored by the user)
    - is:pr "co-authored-by: {identifier}" (co-authored commits)
    - is:pr head:{identifier}/ (branch names starting with identifier)

    Args:
        identifier: GitHub username/bot identifier
        agent_name: Human-readable agent name for metadata
        token: GitHub API token
        start_from_date: Only fetch PRs created after this date (for incremental updates)
        year: Year parameter (deprecated, kept for compatibility but not used)

    Returns:
        List of minimal PR metadata dictionaries
    """
    headers = {'Authorization': f'token {token}'} if token else {}

    # Debug mode: limit PR retrieval for testing
    debug_limit_per_pattern = 10 if DEBUG_MODE else None

    if DEBUG_MODE:
        print(f"\n🐛 DEBUG MODE ENABLED: Limiting to {debug_limit_per_pattern} PRs per query pattern")

    # Define all query patterns to search
    query_patterns = [
        f'is:pr author:{identifier}',
        f'is:pr "co-authored-by: {identifier}"',
        f'is:pr head:{identifier}/',
    ]

    # Use a dict to deduplicate PRs by ID
    prs_by_id = {}

    # Define time range: past 6 months only (or from start_from_date if specified)
    current_time = datetime.now(timezone.utc)
    six_months_ago = current_time - timedelta(days=180)  # ~6 months

    if start_from_date:
        # Use start_from_date but ensure it's not older than 6 months
        start_date = max(start_from_date, six_months_ago)
    else:
        start_date = six_months_ago

    # End date is current time
    end_date = current_time

    for query_pattern in query_patterns:
        print(f"\n🔍 Searching with query: {query_pattern}")
        print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        pattern_start_time = time.time()
        initial_count = len(prs_by_id)

        # Fetch with time partitioning
        prs_found = fetch_prs_with_time_partition(
            query_pattern,
            start_date,
            end_date,
            headers,
            prs_by_id,
            debug_limit_per_pattern
        )

        pattern_duration = time.time() - pattern_start_time
        new_prs = len(prs_by_id) - initial_count

        print(f"   ✓ Pattern complete: {new_prs} new PRs found ({prs_found} total fetched, {len(prs_by_id) - initial_count - (prs_found - new_prs)} duplicates)")
        print(f"   ⏱️ Time taken: {pattern_duration:.1f} seconds")

        # Delay between different query patterns (shorter in debug mode)
        time.sleep(0.2 if DEBUG_MODE else 1.0)

    # Convert to lightweight metadata
    all_prs = list(prs_by_id.values())
    if DEBUG_MODE:
        print(f"\n✅ COMPLETE (DEBUG MODE): Found {len(all_prs)} unique PRs for {identifier}")
        print(f"   Note: In production mode, this would fetch ALL PRs")
    else:
        print(f"\n✅ COMPLETE: Found {len(all_prs)} unique PRs for {identifier}")
    print(f"📦 Extracting minimal metadata...")

    metadata_list = [extract_pr_metadata(pr) for pr in all_prs]

    # Calculate memory savings
    import sys
    original_size = sys.getsizeof(str(all_prs))
    metadata_size = sys.getsizeof(str(metadata_list))
    savings_pct = ((original_size - metadata_size) / original_size * 100) if original_size > 0 else 0

    print(f"💾 Memory efficiency: {original_size // 1024}KB → {metadata_size // 1024}KB (saved {savings_pct:.1f}%)")

    return metadata_list


def calculate_pr_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of PR metadata (lightweight objects).
    Works with minimal metadata: html_url, created_at, merged_at, closed_at, agent_name.

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
        'merged': merged,
        'acceptance_rate': round(acceptance_rate, 2),
    }


def calculate_monthly_metrics_by_agent():
    """
    Calculate monthly metrics for all agents for visualization.
    Loads data directly from SWE-Arena/pr_metadata dataset for the current year.

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
    # Get current year for loading metadata
    current_year = datetime.now().year
    
    # Load ALL agents from HuggingFace agents repo
    agents = load_agents_from_hf()

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {agent.get('github_identifier'): agent.get('agent_name') for agent in agents if agent.get('github_identifier')}

    # Load all PR metadata for current year from pr_metadata dataset
    all_metadata = load_pr_metadata_for_year(current_year)

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

            # Count merged PRs (those with merged_at during this time)
            # Note: We're filtering by created_at, but counting based on merged_at/closed_at
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

    return {
        'agents': sorted(list(agent_month_data.keys())),
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
    In debug mode, saves to in-memory cache only.

    This function APPENDS new metadata and DEDUPLICATES by html_url.

    Args:
        metadata_list: List of PR metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    # Skip saving to HF in debug mode - use in-memory cache instead
    if DEBUG_MODE:
        global DEBUG_PR_METADATA_CACHE
        # Merge with existing cache, deduplicating by html_url
        existing = {pr['html_url']: pr for pr in DEBUG_PR_METADATA_CACHE[agent_identifier] if pr.get('html_url')}
        new = {pr['html_url']: pr for pr in metadata_list if pr.get('html_url')}
        existing.update(new)
        DEBUG_PR_METADATA_CACHE[agent_identifier] = list(existing.values())
        print(f"🐛 DEBUG MODE: Saved to in-memory cache only ({len(metadata_list)} PRs) - NOT saved to HuggingFace")
        return True

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        for (pr_year, month, day), day_metadata in grouped.items():
            # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
            filename = f"{agent_identifier}/{pr_year}.{month:02d}.{day:02d}.jsonl"
            local_filename = f"{pr_year}.{month:02d}.{day:02d}.jsonl"
            print(f"📤 Uploading {len(day_metadata)} PRs to {filename}...")

            # Download existing file if it exists
            existing_metadata = []
            try:
                file_path = hf_hub_download(
                    repo_id=PR_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                existing_metadata = load_jsonl(file_path)
                print(f"   Found {len(existing_metadata)} existing PRs in {filename}")
            except Exception:
                print(f"   No existing file found for {filename}, creating new")

            # Merge and deduplicate by html_url
            existing_by_url = {meta['html_url']: meta for meta in existing_metadata if meta.get('html_url')}
            new_by_url = {meta['html_url']: meta for meta in day_metadata if meta.get('html_url')}

            # Update with new data (new data overwrites old)
            existing_by_url.update(new_by_url)
            merged_metadata = list(existing_by_url.values())

            # Save locally
            save_jsonl(local_filename, merged_metadata)

            # Upload to HuggingFace with folder path
            api.upload_file(
                path_or_fileobj=local_filename,
                path_in_repo=filename,
                repo_id=PR_METADATA_REPO,
                repo_type="dataset",
                token=token
            )

            # Clean up local file
            os.remove(local_filename)

            print(f"   ✓ Saved {len(merged_metadata)} total PRs to {filename}")

        return True

    except Exception as e:
        print(f"✗ Error saving PR metadata: {str(e)}")
        return False


def load_pr_metadata_for_year(year):
    """
    Load all PR metadata for a specific year from HuggingFace.
    Scans all agent folders and loads daily files matching the year.
    In debug mode, loads from in-memory cache if available.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Returns:
        List of dictionaries with 'agent_identifier' added to each PR metadata.
    """
    # In debug mode, check in-memory cache first
    if DEBUG_MODE and DEBUG_PR_METADATA_CACHE:
        all_metadata = []
        for agent_identifier, metadata_list in DEBUG_PR_METADATA_CACHE.items():
            for pr_meta in metadata_list:
                pr_with_agent = pr_meta.copy()
                pr_with_agent['agent_identifier'] = agent_identifier
                all_metadata.append(pr_with_agent)
        if all_metadata:
            print(f"🐛 DEBUG MODE: Loading PR metadata from in-memory cache ({len(all_metadata)} PRs)")
            return all_metadata

    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = api.list_repo_files(repo_id=PR_METADATA_REPO, repo_type="dataset")

        # Filter for files matching the year pattern: [agent_identifier]/YYYY.MM.DD.jsonl
        # Extract year from filename
        year_str = str(year)
        year_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:  # [agent_identifier]/YYYY.MM.DD.jsonl
                    filename = parts[1]
                    if filename.startswith(year_str + '.'):
                        year_files.append(f)

        print(f"📥 Loading PR metadata for {year} ({len(year_files)} daily files across all agents)...")

        all_metadata = []
        for filename in year_files:
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

                # Add agent_identifier to each PR metadata for processing
                for pr_meta in day_metadata:
                    pr_meta['agent_identifier'] = agent_identifier

                all_metadata.extend(day_metadata)
                print(f"   ✓ Loaded {len(day_metadata)} PRs from {filename}")
            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total PRs for {year}")
        return all_metadata

    except Exception as e:
        print(f"✗ Error loading PR metadata for {year}: {str(e)}")
        return []


def get_latest_pr_date_for_agent(agent_identifier):
    """
    Get the latest PR creation date for an agent from stored metadata.
    Used for incremental updates - only fetch PRs newer than this date.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Args:
        agent_identifier: GitHub identifier of the agent

    Returns:
        datetime or None if no existing PRs found.
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = api.list_repo_files(repo_id=PR_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        if not agent_files:
            return None

        # Find latest created_at across all files
        latest_date = None
        for filename in agent_files:
            try:
                file_path = hf_hub_download(
                    repo_id=PR_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                metadata = load_jsonl(file_path)

                for pr in metadata:
                    created_at = pr.get('created_at')
                    if created_at:
                        try:
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if latest_date is None or dt > latest_date:
                                latest_date = dt
                        except Exception:
                            continue
            except Exception:
                continue

        return latest_date

    except Exception:
        return None


def get_daily_files_last_n_months(agent_identifier, n_months=6):
    """
    Get list of daily file paths for an agent from the last N months.

    Args:
        agent_identifier: GitHub identifier of the agent
        n_months: Number of months to look back (default: 6)

    Returns:
        List of file paths in format: [agent_identifier]/YYYY.MM.DD.jsonl
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate date range
        today = datetime.now(timezone.utc)
        n_months_ago = today - timedelta(days=30 * n_months)

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

                # Include if within last n_months
                if n_months_ago <= file_date <= today:
                    recent_files.append(filename)
            except Exception:
                continue

        return recent_files

    except Exception as e:
        print(f"Error getting daily files: {str(e)}")
        return []


def fetch_pr_current_status(pr_url, token):
    """
    Fetch the current status of a single PR from GitHub API.

    Args:
        pr_url: PR HTML URL (e.g., https://github.com/owner/repo/pull/123)
        token: GitHub API token

    Returns:
        Dictionary with updated merged_at and closed_at, or None if failed
    """
    try:
        # Convert HTML URL to API URL
        # https://github.com/owner/repo/pull/123 -> https://api.github.com/repos/owner/repo/pulls/123
        parts = pr_url.replace('https://github.com/', '').split('/')
        if len(parts) < 4:
            return None

        owner, repo, pull_word, pr_number = parts[0], parts[1], parts[2], parts[3]
        api_url = f'https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}'

        headers = {'Authorization': f'token {token}'} if token else {}
        response = request_with_backoff('GET', api_url, headers=headers, max_retries=3)

        if response is None or response.status_code != 200:
            return None

        pr_data = response.json()
        merged_at = pr_data.get('merged_at')
        closed_at = pr_data.get('closed_at')

        # Only store closed_at if not merged
        if merged_at:
            closed_at = None

        return {
            'merged_at': merged_at,
            'closed_at': closed_at
        }

    except Exception as e:
        print(f"   Error fetching PR status for {pr_url}: {str(e)}")
        return None


def refresh_open_prs_for_agent(agent_identifier, token):
    """
    Refresh status for all open PRs from the last 6 months for an agent.
    Only updates PRs that are still open (no merged_at, no closed_at).

    This implements the smart update strategy:
    - Skip PRs that are already closed/merged
    - Fetch current status for open PRs
    - Update and save back to daily files

    Args:
        agent_identifier: GitHub identifier of the agent
        token: GitHub API token

    Returns:
        Tuple: (total_checked, updated_count)
    """
    print(f"\n🔄 Refreshing open PRs for {agent_identifier} (last 6 months)...")

    try:
        # Get daily files from last 6 months
        recent_files = get_daily_files_last_n_months(agent_identifier, n_months=6)

        if not recent_files:
            print(f"   No recent files found for {agent_identifier}")
            return (0, 0)

        print(f"   Found {len(recent_files)} daily files to check")

        total_checked = 0
        updated_count = 0

        # Process each file
        for filename in recent_files:
            try:
                # Download file
                file_path = hf_hub_download(
                    repo_id=PR_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=get_hf_token()
                )
                prs = load_jsonl(file_path)

                if not prs:
                    continue

                updated_prs = []
                file_had_updates = False

                # Check each PR
                for pr in prs:
                    # Skip if already closed or merged
                    if pr.get('merged_at') or pr.get('closed_at'):
                        updated_prs.append(pr)
                        continue

                    # PR is open, fetch current status
                    total_checked += 1
                    pr_url = pr.get('html_url')

                    if not pr_url:
                        updated_prs.append(pr)
                        continue

                    current_status = fetch_pr_current_status(pr_url, token)

                    if current_status:
                        # Check if status changed
                        if current_status['merged_at'] or current_status['closed_at']:
                            print(f"   ✓ PR status changed: {pr_url}")
                            pr['merged_at'] = current_status['merged_at']
                            pr['closed_at'] = current_status['closed_at']
                            updated_count += 1
                            file_had_updates = True

                    updated_prs.append(pr)
                    time.sleep(0.1)  # Rate limiting courtesy delay

                # Save file if there were updates
                if file_had_updates:
                    # Extract filename components for local save
                    parts = filename.split('/')
                    local_filename = parts[-1]  # Just YYYY.MM.DD.jsonl

                    # Save locally
                    save_jsonl(local_filename, updated_prs)

                    # Upload back to HuggingFace
                    api = HfApi()
                    api.upload_file(
                        path_or_fileobj=local_filename,
                        path_in_repo=filename,
                        repo_id=PR_METADATA_REPO,
                        repo_type="dataset",
                        token=get_hf_token()
                    )

                    # Clean up local file
                    os.remove(local_filename)
                    print(f"   💾 Updated {filename}")

            except Exception as e:
                print(f"   Warning: Could not process {filename}: {str(e)}")
                continue

        print(f"   ✅ Refresh complete: {total_checked} open PRs checked, {updated_count} updated")
        return (total_checked, updated_count)

    except Exception as e:
        print(f"   ✗ Error refreshing PRs for {agent_identifier}: {str(e)}")
        return (0, 0)


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
                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return None


def load_leaderboard_dataset():
    """Load leaderboard data from HuggingFace dataset for current year.
    In debug mode, loads from in-memory cache if available."""
    # In debug mode, check in-memory cache first
    if DEBUG_MODE and DEBUG_LEADERBOARD_CACHE:
        print(f"🐛 DEBUG MODE: Loading leaderboard from in-memory cache ({len(DEBUG_LEADERBOARD_CACHE)} entries)")
        return list(DEBUG_LEADERBOARD_CACHE.values())

    try:
        year = datetime.now().year
        filename = f"{year}.csv"

        # Try to download the CSV file for current year
        file_path = hf_hub_download(
            repo_id=LEADERBOARD_REPO,
            filename=filename,
            repo_type="dataset"
        )

        # Load CSV into list of dicts
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
        print(f"✓ Loaded {len(data)} entries from {filename}")
        return data

    except Exception as e:
        print(f"Could not load leaderboard dataset for year {datetime.now().year}: {str(e)}")
        return None


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


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

        # Upload to HuggingFace (root directory)
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=AGENTS_REPO,
            repo_type="dataset",
            token=token
        )

        # Clean up local file
        os.remove(filename)

        print(f"✓ Saved agent to HuggingFace: {filename}")
        return True

    except Exception as e:
        print(f"✗ Error saving agent: {str(e)}")
        return False


def save_leaderboard_to_hf(cache_dict):
    """Save complete leaderboard to HuggingFace dataset as CSV.
    In debug mode, saves to in-memory cache only."""
    # Skip saving in debug mode - use in-memory cache instead
    if DEBUG_MODE:
        global DEBUG_LEADERBOARD_CACHE
        DEBUG_LEADERBOARD_CACHE = cache_dict.copy()
        data_list = dict_to_cache(cache_dict)
        print(f"🐛 DEBUG MODE: Saved to in-memory cache only ({len(data_list)} entries) - NOT saved to HuggingFace")
        return True

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        # Convert to DataFrame
        data_list = dict_to_cache(cache_dict)
        df = pd.DataFrame(data_list)

        # Save to CSV with year as filename
        year = datetime.now().year
        filename = f"{year}.csv"
        df.to_csv(filename, index=False)

        # Upload to HuggingFace
        api = HfApi()
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=LEADERBOARD_REPO,
            repo_type="dataset",
            token=token
        )

        # Clean up local file
        os.remove(filename)

        print(f"✓ Saved leaderboard to HuggingFace as {filename} ({len(data_list)} entries)")
        return True

    except Exception as e:
        print(f"✗ Error saving leaderboard: {str(e)}")
        return False


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def update_all_agents_incremental():
    """
    Memory-efficient incremental update of PR statistics for all agents.

    Strategy:
    1. For each agent, check latest PR date from stored metadata
    2. Only fetch NEW PRs created after that date
    3. Store minimal metadata (not full PR objects) to avoid storage limits
    4. Construct leaderboard from stored metadata

    Returns dictionary of all agent data with current stats.
    """
    token = get_github_token()
    current_year = datetime.now().year

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return {}

    cache_dict = {}

    # Update each agent
    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        if not identifier:
            print(f"Warning: Skipping agent without identifier: {agent}")
            continue

        try:
            print(f"\n{'='*80}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*80}")

            # Check for existing metadata to determine incremental update date
            latest_pr_date = get_latest_pr_date_for_agent(identifier)

            if latest_pr_date:
                print(f"📅 Latest PR found: {latest_pr_date.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Fetching only PRs created after this date...")
                start_from = latest_pr_date + timedelta(seconds=1)  # Start 1 second after
            else:
                print(f"📅 No existing PRs found. Fetching all PR metadata...")
                start_from = None

            # Fetch PR metadata (lightweight, memory-efficient)
            new_metadata = fetch_all_prs_metadata(
                identifier,
                agent_name,
                token,
                start_from_date=start_from
            )

            if new_metadata:
                # Save new metadata to HuggingFace (organized by agent_identifier/year.month)
                print(f"💾 Saving {len(new_metadata)} new PR records...")
                save_pr_metadata_to_hf(new_metadata, identifier)

            # Load all metadata for current year to calculate stats
            print(f"📊 Calculating statistics from stored metadata...")
            all_year_metadata = load_pr_metadata_for_year(current_year)

            # Filter for this specific agent
            agent_metadata = [pr for pr in all_year_metadata if pr.get('agent_identifier') == identifier]

            # Calculate stats from metadata
            stats = calculate_pr_stats_from_metadata(agent_metadata)

            # Merge metadata with stats
            cache_dict[identifier] = {
                'agent_name': agent_name,
                'organization': agent.get('organization', 'Unknown'),
                'github_identifier': identifier,
                **stats
            }

            print(f"✓ Updated {identifier}: {stats['total_prs']} PRs, {stats['acceptance_rate']}% acceptance")

        except Exception as e:
            print(f"✗ Error updating {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return cache_dict


def construct_leaderboard_from_metadata():
    """
    Construct leaderboard from stored PR metadata instead of fetching all PRs.
    Much more memory-efficient and faster.

    Returns dictionary of agent stats.
    """
    print("📊 Constructing leaderboard from PR metadata...")
    current_year = datetime.now().year

    # Load agents
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found")
        return {}

    # Load all PR metadata for current year
    all_metadata = load_pr_metadata_for_year(current_year)

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
            'organization': agent.get('organization', 'Unknown'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


def initialize_data():
    """
    Initialize data on application startup.
    Priority: 1) Leaderboard dataset, 2) PR metadata (if available), 3) Full GitHub mining

    In DEBUG MODE:
    - If no data available, automatically mine up to 10 PRs per query per agent
    - Does NOT save to HuggingFace datasets
    """
    print("🚀 Initializing leaderboard data...")

    # Try loading existing leaderboard
    leaderboard_data = load_leaderboard_dataset()
    if leaderboard_data:
        print("✓ Initialized from leaderboard dataset")
        return

    # Try constructing from PR metadata (fast, memory-efficient)
    try:
        cache_dict = construct_leaderboard_from_metadata()
        # Check if there's actually meaningful data (at least one agent with PRs)
        has_data = any(entry.get('total_prs', 0) > 0 for entry in cache_dict.values())
        if cache_dict and has_data:
            save_leaderboard_to_hf(cache_dict)
            print("✓ Initialized from PR metadata")
            return
    except Exception as e:
        print(f"Could not construct from metadata: {e}")

    # If in debug mode and no data available, mine immediately
    if DEBUG_MODE:
        print("\n🐛 DEBUG MODE: No data available, mining immediately (up to 10 PRs per query per agent)...")
        agents = load_agents_from_hf()
        if agents:
            print(f"✓ Loaded {len(agents)} agents from HuggingFace")
            print("⛏️ Mining GitHub data in debug mode (limited to 10 PRs per query)...")
            cache_dict = update_all_agents_incremental()
            if cache_dict:
                # In debug mode, this won't actually save to HF
                save_leaderboard_to_hf(cache_dict)
                print("✓ Debug mining complete (data NOT saved to HuggingFace)")
            return
        else:
            print("⚠️ No agents found. Waiting for first submission...")
            return

    # Production mode: Fallback to full incremental mining from GitHub
    agents = load_agents_from_hf()
    if agents:
        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        print("⛏️ Mining GitHub data (this may take a while)...")
        cache_dict = update_all_agents_incremental()
        if cache_dict:
            save_leaderboard_to_hf(cache_dict)
        return

    # No data available
    print("⚠️ No data sources available. Waiting for first submission...")


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot():
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Acceptance rate (%) as line curves
    - Right y-axis: Total PRs created as bar charts

    Each agent gets a unique color for both their line and bars.
    """
    metrics = calculate_monthly_metrics_by_agent()

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

    # Define colors for agents (using a color palette)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Add traces for each agent
    for idx, agent_name in enumerate(agents):
        color = colors[idx % len(colors)]
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
                                 'Month: %{x}<br>' +
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
                                 'Month: %{x}<br>' +
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
        hovermode='x unified',
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
    Load leaderboard data from HuggingFace and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by acceptance rate.
    """
    # Load leaderboard data from HuggingFace
    leaderboard_data = load_leaderboard_dataset()

    if not leaderboard_data:
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    for data in leaderboard_data:
        # Only include display-relevant fields
        rows.append([
            data.get('agent_name', 'Unknown'),
            data.get('organization', 'Unknown'),
            data.get('total_prs', 0),
            data.get('merged', 0),
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

    # Sort by Acceptance Rate (%) descending
    if "Acceptance Rate (%)" in df.columns and not df.empty:
        df = df.sort_values(by="Acceptance Rate (%)", ascending=False).reset_index(drop=True)

    return df


def refresh_leaderboard():
    """Manually trigger data refresh for all agents using incremental updates."""
    try:
        print("🔄 Manual refresh initiated (incremental mode)")
        cache_dict = update_all_agents_incremental()
        if cache_dict:
            save_leaderboard_to_hf(cache_dict)
        return "✅ Data refreshed successfully!", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    except Exception as e:
        error_msg = f"❌ Refresh failed: {str(e)}"
        print(error_msg)
        return error_msg, get_leaderboard_dataframe(), create_monthly_metrics_plot()


def submit_agent(identifier, agent_name, organization, description, website):
    """
    Submit a new agent to the leaderboard.
    Validates input, saves submission, and fetches PR metadata (memory-efficient).
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

    # Fetch PR metadata immediately (memory-efficient)
    token = get_github_token()
    try:
        print(f"Fetching PR metadata for {agent_name}...")

        # Fetch lightweight metadata
        metadata_list = fetch_all_prs_metadata(identifier, agent_name, token)

        if metadata_list:
            # Save metadata to HuggingFace
            save_pr_metadata_to_hf(metadata_list, identifier)

        # Calculate stats from metadata
        stats = calculate_pr_stats_from_metadata(metadata_list)

        # Load current leaderboard
        leaderboard_data = load_leaderboard_dataset()
        if not leaderboard_data:
            leaderboard_data = []

        # Convert to dict for easy updating
        cache_dict = {entry['github_identifier']: entry for entry in leaderboard_data}
        cache_dict[identifier] = {**submission, **stats}

        # Save to HuggingFace
        save_leaderboard_to_hf(cache_dict)

        return f"✅ Successfully submitted {agent_name}!", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    except Exception as e:
        error_msg = f"⚠️ Submitted {agent_name}, but failed to fetch PR data: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, get_leaderboard_dataframe(), create_monthly_metrics_plot()


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

def daily_update_task():
    """
    Daily scheduled task (runs at 12:00 AM UTC) for smart PR updates.

    Strategy:
    1. For each agent, refresh open PRs from last 6 months
    2. Skip PRs that are already closed/merged (no API calls)
    3. Only fetch status for open PRs to check if they've been closed/merged
    4. Update leaderboard with refreshed data

    This is much more efficient than fetching all PRs every time.
    """
    print(f"\n{'='*80}")
    print(f"🕛 Daily update started at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}")

    try:
        token = get_github_token()

        # Load all agents
        agents = load_agents_from_hf()
        if not agents:
            print("No agents found")
            return

        print(f"📋 Processing {len(agents)} agents...")

        total_checked = 0
        total_updated = 0

        # Refresh open PRs for each agent (last 6 months)
        for agent in agents:
            identifier = agent.get('github_identifier')
            agent_name = agent.get('agent_name', 'Unknown')

            if not identifier:
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*60}")

            # Refresh open PRs from last 6 months
            checked, updated = refresh_open_prs_for_agent(identifier, token)
            total_checked += checked
            total_updated += updated

        print(f"\n{'='*80}")
        print(f"📊 Refresh Summary:")
        print(f"   Total open PRs checked: {total_checked}")
        print(f"   PRs updated (closed/merged): {total_updated}")
        print(f"{'='*80}")

        # Reconstruct leaderboard from all stored metadata
        print(f"\n📈 Rebuilding leaderboard from refreshed data...")
        cache_dict = construct_leaderboard_from_metadata()

        if cache_dict:
            # Save leaderboard
            save_leaderboard_to_hf(cache_dict)
            print("✓ Leaderboard updated successfully")

        print(f"\n✅ Daily update completed at {datetime.now(timezone.utc).isoformat()}")

    except Exception as e:
        print(f"✗ Daily update failed: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

# Initialize data before creating UI
if DEBUG_MODE:
    print("\n" + "="*80)
    print("🐛 DEBUG MODE ENABLED 🐛")
    print("="*80)
    print("PR retrieval is limited to 10 PRs per query pattern per agent")

    # Show how debug mode was enabled
    if args.debug:
        print("Enabled via: command-line flag '--debug'")
        print("To disable: run without '--debug' flag")
    else:
        print("Enabled via: DEBUG_MODE environment variable")
        print("To disable: run with '--no-debug' flag or unset DEBUG_MODE")

    print("="*80 + "\n")
else:
    print("\n🚀 Starting in PRODUCTION MODE - full PR retrieval enabled")
    if args.no_debug:
        print("   (Explicitly set via '--no-debug' flag)")
    print()

initialize_data()

# Start APScheduler for daily updates at 12:00 AM UTC
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    daily_update_task,
    trigger=CronTrigger(hour=0, minute=0),  # 12:00 AM UTC daily
    id='daily_pr_refresh',
    name='Daily PR Status Refresh',
    replace_existing=True
)
scheduler.start()
print("✓ Scheduler started: Daily updates at 12:00 AM UTC")

# Create Gradio interface
with gr.Blocks(title="SWE Agent PR Leaderboard", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("# 🏆 SWE Agent PR Leaderboard")
    gr.Markdown("Track and compare GitHub pull request statistics for SWE agents")
    
    with gr.Tabs():
        
        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            with gr.Row():
                refresh_button = gr.Button("🔄 Refresh Data", variant="primary")
                status_display = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    scale=3
                )
            
            leaderboard_table = Leaderboard(
                value=get_leaderboard_dataframe(),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Organization"],
                filter_columns=["Acceptance Rate (%)"]
            )

            gr.Markdown("### Monthly Metrics")
            gr.Markdown("Track acceptance rates and PR activity over time")

            monthly_plot = gr.Plot(
                value=create_monthly_metrics_plot(),
                label="Monthly PR Metrics"
            )

            refresh_button.click(
                fn=refresh_leaderboard,
                outputs=[status_display, leaderboard_table, monthly_plot]
            )
        
        # Submit Agent Tab
        with gr.Tab("➕ Submit Agent"):
            
            gr.Markdown("### Submit Your Agent")
            gr.Markdown("Fill in the details below to add your agent to the leaderboard. Make sure you're logged in to HuggingFace CLI on your machine.")
            
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
                        label="Website",
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