"""
Minimalist PR Metadata Mining Script
Mines PR metadata from GitHub and saves to HuggingFace dataset.
"""

import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"
PR_METADATA_REPO = "SWE-Arena/pr_metadata"
LEADERBOARD_TIME_FRAME_DAYS = 180  # 6 months

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


def get_github_token():
    """Get GitHub token from environment variables."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. API rate limits: 60/hour (authenticated: 5000/hour)")
    return token


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GITHUB API FUNCTIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.
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


def fetch_prs_within_day_partition(base_query, start_date, end_date, headers, prs_by_id, depth=0, max_depth=8):
    """
    Recursively fetch PRs within a time range by subdividing into smaller granularities.
    This function handles INTRA-DAY partitioning (hours → minutes → seconds).
    
    Used when a single day query hits the 1000-result limit.
    Recursion is bounded to prevent stack overflow.

    Args:
        base_query: Base query string (already includes the day in created: clause)
        start_date: Start datetime
        end_date: End datetime
        headers: HTTP headers with auth token
        prs_by_id: Dict to store deduplicated PRs by ID
        depth: Current recursion depth
        max_depth: Maximum allowed recursion depth

    Returns:
        Total number of new PRs found in this partition
    """
    # Safety limit on recursion depth
    if depth >= max_depth:
        print(f"{'  ' * depth}⚠️  Max recursion depth ({max_depth}) reached. Some results may be missing.")
        return 0

    time_diff = end_date - start_date
    total_seconds = time_diff.total_seconds()

    # Determine granularity based on time range
    if total_seconds >= 3600:  # >= 1 hour - subdivide by hours
        start_str = start_date.strftime('%Y-%m-%dT%H:00:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:59:59Z')
        granularity = "hour"
    elif total_seconds >= 60:  # >= 1 minute - subdivide by minutes
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:59Z')
        granularity = "minute"
    else:  # < 1 minute - subdivide by seconds
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        granularity = "second"

    query = f'{base_query} created:{start_str}..{end_str}'
    indent = "  " * depth

    print(f"{indent}[Depth {depth}] Searching {granularity} range: {start_str} to {end_str}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
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
                print(f"{indent}  ✗ Retries exhausted for {start_str} to {end_str}")
                return total_in_partition

            if response.status_code != 200:
                print(f"{indent}  ✗ HTTP {response.status_code} for {start_str} to {end_str}")
                return total_in_partition

            data = response.json()
            total_count = data.get('total_count', 0)
            items = data.get('items', [])

            if not items:
                break

            # Add PRs to global dict, count new ones
            for pr in items:
                pr_id = pr.get('id')
                if pr_id and pr_id not in prs_by_id:
                    prs_by_id[pr_id] = pr
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"{indent}  ⚠️  Hit 1000-result limit ({total_count} total). Subdividing {granularity}...")

                # Check if we can subdivide further
                if total_seconds < 2:
                    print(f"{indent}  ⚠️  Cannot subdivide further (< 2 seconds). Some results may be missing.")
                    break

                # Subdivide based on current granularity
                if granularity == "hour":
                    # Split hour into 4 parts (15-minute intervals)
                    num_splits = 4
                elif granularity == "minute":
                    # Split minute into 4 parts (15-second intervals)
                    num_splits = 4
                else:  # granularity == "second"
                    # Can't subdivide seconds further meaningfully
                    print(f"{indent}  ⚠️  Already at second granularity. Cannot subdivide. Some results may be missing.")
                    break

                split_duration = time_diff / num_splits
                total_from_splits = 0

                for i in range(num_splits):
                    split_start = start_date + split_duration * i
                    split_end = start_date + split_duration * (i + 1)

                    count = fetch_prs_within_day_partition(
                        base_query, split_start, split_end, headers, prs_by_id, depth + 1, max_depth
                    )
                    total_from_splits += count

                return total_from_splits

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"{indent}  ✗ Error fetching {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"{indent}  ✓ Found {total_in_partition} PRs in {granularity} range")

    return total_in_partition


def fetch_prs_with_time_partition(base_query, start_date, end_date, headers, prs_by_id):
    """
    Iteratively fetch PRs by iterating through each day in the date range.
    For each day, query with daily granularity.
    If a single day hits the 1000-result limit, subdivide that day recursively (hours → minutes → seconds).

    This hybrid iterative-recursive approach prevents deep recursion by:
    - Using iteration for the outer loop (days)
    - Using recursion only for intra-day partitioning (hours/minutes/seconds)

    Args:
        base_query: Base query string (e.g., 'is:pr author:{identifier}')
        start_date: Start date
        end_date: End date (inclusive)
        headers: HTTP headers with auth token
        prs_by_id: Dict to store deduplicated PRs by ID

    Returns:
        Total number of new PRs found
    """
    current_date = start_date
    total_found = 0

    # Iterate through each day
    while current_date <= end_date:
        day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Ensure we don't go past end_date
        if day_end > end_date:
            day_end = end_date

        day_str = current_date.strftime('%Y-%m-%d')
        print(f"\n📅 Processing day: {day_str}")

        # First, try a simple daily query
        query = f'{base_query} created:{day_str}'
        url = 'https://api.github.com/search/issues'
        params = {
            'q': query,
            'per_page': 1,
            'page': 1,
            'sort': 'created',
            'order': 'asc'
        }

        try:
            response = request_with_backoff('GET', url, headers=headers, params=params)
            if response and response.status_code == 200:
                data = response.json()
                total_count = data.get('total_count', 0)

                if total_count > 1000:
                    print(f"  ⚠️  Day has {total_count} PRs (exceeds 1000-result limit). Subdividing by time of day...")
                    # Use recursive intra-day partitioning
                    count = fetch_prs_within_day_partition(
                        base_query, day_start, day_end, headers, prs_by_id, depth=0
                    )
                    total_found += count
                else:
                    # Normal case: fetch all PRs for this day
                    print(f"  Fetching {total_count} PRs...")
                    page = 1
                    per_page = 100
                    day_count = 0

                    while True:
                        params_page = {
                            'q': query,
                            'per_page': per_page,
                            'page': page,
                            'sort': 'created',
                            'order': 'asc'
                        }

                        response = request_with_backoff('GET', url, headers=headers, params=params_page)
                        if not response or response.status_code != 200:
                            break

                        items = response.json().get('items', [])
                        if not items:
                            break

                        for pr in items:
                            pr_id = pr.get('id')
                            if pr_id and pr_id not in prs_by_id:
                                prs_by_id[pr_id] = pr
                                day_count += 1

                        if len(items) < per_page:
                            break

                        page += 1
                        time.sleep(0.5)

                    if day_count > 0:
                        print(f"  ✓ Found {day_count} new PRs for {day_str}")
                    total_found += day_count

            time.sleep(0.5)  # Courtesy delay between days

        except Exception as e:
            print(f"  ✗ Error processing {day_str}: {str(e)}")
            continue

        # Move to next day
        current_date += timedelta(days=1)

    return total_found


def extract_pr_metadata(pr):
    """
    Extract minimal PR metadata for efficient storage.
    Only keeps essential fields: html_url, created_at, merged_at, closed_at.
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


def fetch_all_prs_metadata(identifier, agent_name, token=None):
    """
    Fetch pull requests associated with a GitHub user or bot for the past LEADERBOARD_TIME_FRAME_DAYS.
    Returns lightweight metadata instead of full PR objects.

    This function employs time-based partitioning to navigate GitHub's 1000-result limit per query.
    It searches using multiple query patterns:
    - is:pr author:{identifier} (PRs authored by the bot)
    - is:pr "co-authored-by: {identifier}" (PRs with commits co-authored by the bot)
    - is:pr head:{identifier}/ (PRs with branch names starting with the bot identifier)

    Args:
        identifier: GitHub username or bot identifier
        agent_name: Human-readable name of the agent for metadata purposes
        token: GitHub API token for authentication

    Returns:
        List of dictionaries containing minimal PR metadata
    """
    headers = {'Authorization': f'token {token}'} if token else {}

    # Define query patterns per rules:
    # 1) author pattern only if identifier contains "[bot]"
    # 2) co-author and head patterns use identifier with "[bot]" removed
    stripped_id = identifier.replace('[bot]', '')
    query_patterns = []
    if '[bot]' in identifier:
        query_patterns.append(f'is:pr author:{identifier}')
    if stripped_id:
        query_patterns.append(f'is:pr "co-authored-by: {stripped_id}"')
        query_patterns.append(f'is:pr head:{stripped_id}/')

    # Use a dict to deduplicate PRs by ID
    prs_by_id = {}

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)  # 12:00 AM today (UTC)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    for query_pattern in query_patterns:
        print(f"\n🔍 Searching with query: {query_pattern}")
        print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (today excluded)")

        pattern_start_time = time.time()
        initial_count = len(prs_by_id)

        # Fetch with time partitioning
        prs_found = fetch_prs_with_time_partition(
            query_pattern,
            start_date,
            end_date,
            headers,
            prs_by_id
        )

        pattern_duration = time.time() - pattern_start_time
        new_prs = len(prs_by_id) - initial_count

        print(f"   ✓ Pattern complete: {new_prs} new PRs found ({prs_found} total fetched, {len(prs_by_id) - initial_count - (prs_found - new_prs)} duplicates)")
        print(f"   ⏱️ Time taken: {pattern_duration:.1f} seconds")

        time.sleep(1.0)

    # Convert to lightweight metadata
    all_prs = list(prs_by_id.values())

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


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.
    """
    delay = 2.0

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
                delay = min(delay * 2, 60.0)
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_pr_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save PR metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's PRs.

    This function APPENDS new metadata and DEDUPLICATES by html_url.

    Args:
        metadata_list: List of PR metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
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

            try:
                # Upload to HuggingFace with folder path
                upload_with_retry(
                    api=api,
                    path_or_fileobj=local_filename,
                    path_in_repo=filename,
                    repo_id=PR_METADATA_REPO,
                    repo_type="dataset",
                    token=token
                )
                print(f"   ✓ Saved {len(merged_metadata)} total PRs to {filename}")
            finally:
                # Always clean up local file, even if upload fails
                if os.path.exists(local_filename):
                    os.remove(local_filename)

        return True

    except Exception as e:
        print(f"✗ Error saving PR metadata: {str(e)}")
        return False


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
        return []


# =============================================================================
# MAIN MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine PR metadata for all agents within LEADERBOARD_TIME_FRAME_DAYS and save to HuggingFace.
    """
    token = get_github_token()

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    print(f"\n{'='*80}")
    print(f"Starting PR metadata mining for {len(agents)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"{'='*80}\n")

    # Mine each agent
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

            # Fetch PR metadata
            metadata = fetch_all_prs_metadata(identifier, agent_name, token)

            if metadata:
                print(f"💾 Saving {len(metadata)} PR records...")
                save_pr_metadata_to_hf(metadata, identifier)
                print(f"✓ Successfully processed {agent_name}")
            else:
                print(f"   No PRs found for {agent_name}")

        except Exception as e:
            print(f"✗ Error processing {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ Mining complete for all agents")
    print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
