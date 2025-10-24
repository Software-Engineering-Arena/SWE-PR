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


def get_github_tokens():
    """Get all GitHub tokens from environment variables (all vars starting with GITHUB_TOKEN)."""
    tokens = []
    for key, value in os.environ.items():
        if key.startswith('GITHUB_TOKEN') and value:
            tokens.append(value)

    if not tokens:
        print("Warning: No GITHUB_TOKEN* found. API rate limits: 60/hour (authenticated: 5000/hour)")
    else:
        print(f"✓ Loaded {len(tokens)} GitHub token(s) for token pool")

    return tokens


def get_github_token():
    """Get primary GitHub token from environment variables (for backward compatibility)."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. API rate limits: 60/hour (authenticated: 5000/hour)")
    return token


class TokenPool:
    """
    Hybrid token pool that manages GitHub tokens with parallel execution and round-robin fallback.

    Strategy:
    - 50% of tokens allocated to parallel pool (for concurrent API calls)
    - 50% of tokens allocated to round-robin pool (for rate limit fallback)
    - Automatically switches to round-robin when parallel tokens hit rate limits
    - Thread-safe for concurrent access
    """
    def __init__(self, tokens):
        import threading

        self.all_tokens = tokens if tokens else [None]
        self.lock = threading.Lock()

        # Split tokens into parallel and round-robin pools (50/50)
        total_tokens = len(self.all_tokens)
        split_point = max(1, total_tokens // 2)  # At least 1 token in each pool

        self.parallel_tokens = self.all_tokens[:split_point]
        self.roundrobin_tokens = self.all_tokens[split_point:]

        # If only 1 token, use it in both pools
        if total_tokens == 1:
            self.parallel_tokens = self.all_tokens
            self.roundrobin_tokens = self.all_tokens

        # Track rate-limited tokens with reset times
        self.rate_limited_parallel = {}  # {token: reset_timestamp}
        self.rate_limited_roundrobin = {}  # {token: reset_timestamp}

        # Round-robin index for fallback pool
        self.roundrobin_index = 0

        # Statistics
        self.parallel_calls = 0
        self.roundrobin_calls = 0
        self.fallback_triggers = 0

        print(f"🔄 Hybrid Token Pool initialized:")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Parallel pool: {len(self.parallel_tokens)} token(s)")
        print(f"   Round-robin pool: {len(self.roundrobin_tokens)} token(s)")

    def _clean_expired_rate_limits(self):
        """Remove tokens from rate limit tracking if their reset time has passed."""
        current_time = time.time()

        # Clean parallel pool
        expired_parallel = [token for token, reset_time in self.rate_limited_parallel.items()
                           if current_time >= reset_time]
        for token in expired_parallel:
            del self.rate_limited_parallel[token]

        # Clean round-robin pool
        expired_roundrobin = [token for token, reset_time in self.rate_limited_roundrobin.items()
                             if current_time >= reset_time]
        for token in expired_roundrobin:
            del self.rate_limited_roundrobin[token]

    def get_parallel_token(self):
        """
        Get a token from the parallel pool for concurrent execution.
        Returns None if all parallel tokens are rate-limited.
        """
        with self.lock:
            self._clean_expired_rate_limits()

            # Find first non-rate-limited token in parallel pool
            for token in self.parallel_tokens:
                if token not in self.rate_limited_parallel:
                    self.parallel_calls += 1
                    return token

            return None  # All parallel tokens are rate-limited

    def get_available_parallel_tokens(self):
        """
        Get all available tokens from parallel pool (not rate-limited).
        Used for batch parallel execution.
        """
        with self.lock:
            self._clean_expired_rate_limits()
            available = [token for token in self.parallel_tokens
                        if token not in self.rate_limited_parallel]
            return available

    def get_roundrobin_token(self):
        """
        Get the next token from round-robin pool (fallback mechanism).
        Skips rate-limited tokens and rotates to the next available one.
        """
        with self.lock:
            self._clean_expired_rate_limits()

            attempts = 0
            max_attempts = len(self.roundrobin_tokens)

            while attempts < max_attempts:
                token = self.roundrobin_tokens[self.roundrobin_index]
                self.roundrobin_index = (self.roundrobin_index + 1) % len(self.roundrobin_tokens)

                if token not in self.rate_limited_roundrobin:
                    self.roundrobin_calls += 1
                    return token

                attempts += 1

            # All round-robin tokens are rate-limited
            return None

    def get_next_token(self):
        """
        Get the next available token (try parallel first, fallback to round-robin).
        This is the main method for backwards compatibility.
        """
        # Try parallel pool first
        token = self.get_parallel_token()
        if token:
            return token

        # Fallback to round-robin
        with self.lock:
            self.fallback_triggers += 1

        token = self.get_roundrobin_token()
        if token:
            return token

        # All tokens exhausted - return first parallel token anyway (will hit rate limit)
        return self.parallel_tokens[0] if self.parallel_tokens else None

    def get_headers(self):
        """Get headers with the next available token."""
        token = self.get_next_token()
        return {'Authorization': f'token {token}'} if token else {}

    def mark_rate_limited(self, token, reset_timestamp=None):
        """
        Mark a token as rate-limited with optional reset timestamp.

        Args:
            token: The token that hit rate limit
            reset_timestamp: Unix timestamp when rate limit resets (optional)
        """
        with self.lock:
            # Default to 1 hour from now if no reset time provided
            if reset_timestamp is None:
                reset_timestamp = time.time() + 3600

            # Mark in appropriate pool
            if token in self.parallel_tokens:
                self.rate_limited_parallel[token] = reset_timestamp
                print(f"   ⚠️ Parallel token marked as rate-limited until {datetime.fromtimestamp(reset_timestamp, timezone.utc).strftime('%H:%M:%S UTC')}")

            if token in self.roundrobin_tokens:
                self.rate_limited_roundrobin[token] = reset_timestamp
                print(f"   ⚠️ Round-robin token marked as rate-limited until {datetime.fromtimestamp(reset_timestamp, timezone.utc).strftime('%H:%M:%S UTC')}")

    def get_stats(self):
        """Get usage statistics for monitoring."""
        with self.lock:
            return {
                'parallel_calls': self.parallel_calls,
                'roundrobin_calls': self.roundrobin_calls,
                'fallback_triggers': self.fallback_triggers,
                'parallel_rate_limited': len(self.rate_limited_parallel),
                'roundrobin_rate_limited': len(self.rate_limited_roundrobin)
            }

    def print_stats(self):
        """Print usage statistics."""
        stats = self.get_stats()
        total_calls = stats['parallel_calls'] + stats['roundrobin_calls']

        if total_calls > 0:
            print(f"\n📊 Token Pool Statistics:")
            print(f"   Total API calls: {total_calls}")
            print(f"   Parallel calls: {stats['parallel_calls']} ({stats['parallel_calls']/total_calls*100:.1f}%)")
            print(f"   Round-robin calls: {stats['roundrobin_calls']} ({stats['roundrobin_calls']/total_calls*100:.1f}%)")
            print(f"   Fallback triggers: {stats['fallback_triggers']}")
            print(f"   Currently rate-limited: {stats['parallel_rate_limited']} parallel, {stats['roundrobin_rate_limited']} round-robin")


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GITHUB API FUNCTIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30, token_pool=None, token=None):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Args:
        token_pool: Optional TokenPool instance for marking rate-limited tokens
        token: Optional token string used for this request (for rate limit tracking)

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
                reset_timestamp = None

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
                            reset_timestamp = reset_ts
                            wait = max(reset_ts - time.time() + 2, 1)
                        except Exception:
                            wait = None

                # Final fallback: exponential backoff with jitter
                if wait is None:
                    wait = delay + random.uniform(0, 0.5)

                # Mark token as rate-limited if we have token pool and token info
                if status in (403, 429) and token_pool and token:
                    token_pool.mark_rate_limited(token, reset_timestamp)

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


def fetch_prs_within_day_partition(base_query, start_date, end_date, token_pool, prs_by_id, depth=0, max_depth=8):
    """
    Recursively fetch PRs within a time range by subdividing into smaller granularities.
    This function handles INTRA-DAY partitioning (hours → minutes → seconds).
    
    Used when a single day query hits the 1000-result limit.
    Recursion is bounded to prevent stack overflow.

    Args:
        base_query: Base query string (already includes the day in created: clause)
        start_date: Start datetime
        end_date: End datetime
        token_pool: TokenPool instance for rotating tokens
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
            # Get token for tracking
            token = token_pool.get_next_token()
            headers = {'Authorization': f'token {token}'} if token else {}

            response = request_with_backoff('GET', url, headers=headers, params=params,
                                           token_pool=token_pool, token=token)
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
                        base_query, split_start, split_end, token_pool, prs_by_id, depth + 1, max_depth
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


def fetch_prs_with_time_partition(base_query, start_date, end_date, token_pool, prs_by_id):
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
        token_pool: TokenPool instance for rotating tokens
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
            # Get token for tracking
            token = token_pool.get_next_token()
            headers = {'Authorization': f'token {token}'} if token else {}

            response = request_with_backoff('GET', url, headers=headers, params=params,
                                           token_pool=token_pool, token=token)
            if response and response.status_code == 200:
                data = response.json()
                total_count = data.get('total_count', 0)

                if total_count > 1000:
                    print(f"  ⚠️  Day has {total_count} PRs (exceeds 1000-result limit). Subdividing by time of day...")
                    # Use recursive intra-day partitioning
                    count = fetch_prs_within_day_partition(
                        base_query, day_start, day_end, token_pool, prs_by_id, depth=0
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

                        # Get token for tracking
                        token = token_pool.get_next_token()
                        headers = {'Authorization': f'token {token}'} if token else {}

                        response = request_with_backoff('GET', url, headers=headers, params=params_page,
                                                       token_pool=token_pool, token=token)
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


def fetch_prs_parallel(query_patterns, start_date, end_date, token_pool, max_workers=None):
    """
    Fetch PRs for multiple query patterns in parallel using available tokens.

    Args:
        query_patterns: List of query pattern strings
        start_date: Start date for PR search
        end_date: End date for PR search
        token_pool: TokenPool instance
        max_workers: Maximum number of concurrent workers (defaults to number of available parallel tokens)

    Returns:
        Dictionary mapping query pattern to list of PRs found
    """
    import concurrent.futures

    # Determine number of workers based on available parallel tokens
    available_tokens = token_pool.get_available_parallel_tokens()
    if not available_tokens:
        # Fall back to sequential if no parallel tokens available
        print("   ⚠️ No parallel tokens available, using sequential fallback")
        return None

    if max_workers is None:
        max_workers = len(available_tokens)

    print(f"   🚀 Starting parallel execution with {max_workers} worker(s)")

    results = {}

    def fetch_single_pattern(pattern):
        """Fetch PRs for a single query pattern."""
        prs_by_id = {}
        try:
            prs_found = fetch_prs_with_time_partition(
                pattern,
                start_date,
                end_date,
                token_pool,
                prs_by_id
            )
            return pattern, prs_by_id
        except Exception as e:
            print(f"   ✗ Error in parallel fetch for pattern '{pattern}': {str(e)}")
            return pattern, {}

    # Execute patterns in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pattern = {
            executor.submit(fetch_single_pattern, pattern): pattern
            for pattern in query_patterns
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_pattern):
            pattern = future_to_pattern[future]
            try:
                pattern_key, prs = future.result()
                results[pattern_key] = prs
                print(f"   ✓ Parallel fetch completed for pattern: {pattern_key}")
            except Exception as e:
                print(f"   ✗ Parallel fetch failed for pattern '{pattern}': {str(e)}")
                results[pattern] = {}

    return results


def fetch_all_prs_metadata(identifier, agent_name, token_pool=None, use_parallel=True):
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
        token_pool: TokenPool instance for rotating tokens

    Returns:
        List of dictionaries containing minimal PR metadata
    """

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

    # Try parallel execution first if enabled
    if use_parallel and len(query_patterns) > 1:
        print(f"\n🚀 Attempting parallel execution for {len(query_patterns)} query patterns...")
        parallel_start_time = time.time()

        parallel_results = fetch_prs_parallel(query_patterns, start_date, end_date, token_pool)

        if parallel_results is not None:
            # Merge results from parallel execution
            for pattern, pattern_prs in parallel_results.items():
                for pr_id, pr in pattern_prs.items():
                    if pr_id not in prs_by_id:
                        prs_by_id[pr_id] = pr

            parallel_duration = time.time() - parallel_start_time
            print(f"\n   ✅ Parallel execution complete: {len(prs_by_id)} unique PRs found")
            print(f"   ⏱️ Total time: {parallel_duration:.1f} seconds")

            # Print token pool statistics
            token_pool.print_stats()
        else:
            # Fallback to sequential execution
            print("   ⚠️ Parallel execution not available, falling back to sequential...")
            use_parallel = False

    # Sequential execution (fallback or if parallel disabled)
    if not use_parallel or len(query_patterns) <= 1:
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
                token_pool,
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
    Uses batch upload to avoid HuggingFace rate limits (256 commits/hour).

    Args:
        metadata_list: List of PR metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    import tempfile
    import shutil

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        # Create a temporary directory to prepare all files for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_dir = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_dir, exist_ok=True)

        try:
            print(f"📦 Preparing {len(grouped)} daily files for batch upload...")

            for (pr_year, month, day), day_metadata in grouped.items():
                # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
                filename = f"{agent_identifier}/{pr_year}.{month:02d}.{day:02d}.jsonl"
                local_path = os.path.join(agent_dir, f"{pr_year}.{month:02d}.{day:02d}.jsonl")

                print(f"   Preparing {len(day_metadata)} PRs for {filename}...")

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
                    print(f"      Found {len(existing_metadata)} existing PRs, merging...")
                except Exception:
                    print(f"      No existing file found, creating new...")

                # Merge and deduplicate by html_url
                existing_by_url = {meta['html_url']: meta for meta in existing_metadata if meta.get('html_url')}
                new_by_url = {meta['html_url']: meta for meta in day_metadata if meta.get('html_url')}

                # Update with new data (new data overwrites old)
                existing_by_url.update(new_by_url)
                merged_metadata = list(existing_by_url.values())

                # Save to temp directory
                save_jsonl(local_path, merged_metadata)
                print(f"      ✓ Prepared {len(merged_metadata)} total PRs")

            # Batch upload entire folder in a single commit
            print(f"\n📤 Uploading all files for {agent_identifier} in one batch...")
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=PR_METADATA_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"Update PR metadata for {agent_identifier}"
            )
            print(f"   ✓ Successfully uploaded {len(grouped)} files in 1 commit")

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

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
    # Initialize token pool
    tokens = get_github_tokens()
    token_pool = TokenPool(tokens)

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
            metadata = fetch_all_prs_metadata(identifier, agent_name, token_pool)

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
