import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import requests
from datetime import datetime, timezone
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import threading
from dotenv import load_dotenv
import pandas as pd
import random

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

CACHE_FILE = "agent_pr_cache.jsonl"
AGENTS_REPO = "SWE-Arena/pr_agents"  # HuggingFace dataset for agent metadata
LEADERBOARD_REPO = "SWE-Arena/pr_leaderboard"
UPDATE_INTERVAL = 86400  # 24 hours in seconds

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Organization", "string"),
    ("Total PRs", "number"),
    ("Merged PRs", "number"),
    ("Acceptance Rate (%)", "number"),
    ("Median Merge Duration (minutes)", "number"),
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
        response = request_with_backoff('GET', url, headers=headers, max_retries=6)
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


def fetch_all_prs(identifier, token=None):
    """
    Fetch all pull requests associated with a GitHub user/bot.
    Searches using multiple query patterns:
    - is:pr author:{identifier} (authored by the user)
    - is:pr head:{identifier}/ (branch names starting with identifier)
    - is:pr "co-authored-by: {identifier}" (co-authored commits)

    Uses pagination to retrieve all results and deduplicates by PR ID.
    """
    headers = {'Authorization': f'token {token}'} if token else {}

    # Define all query patterns to search
    query_patterns = [
        f'is:pr author:{identifier}',
        f'is:pr head:{identifier}/',
        f'is:pr "co-authored-by: {identifier}"'
    ]

    # Use a dict to deduplicate PRs by ID
    prs_by_id = {}

    for query in query_patterns:
        print(f"Searching with query: {query}")
        page = 1
        per_page = 100

        while True:
            url = f'https://api.github.com/search/issues'
            params = {
                'q': query,
                'per_page': per_page,
                'page': page
            }

            try:
                response = request_with_backoff('GET', url, headers=headers, params=params)
                if response is None:
                    print(f"Error fetching PRs for query '{query}': retries exhausted")
                    break

                if response.status_code != 200:
                    print(f"Error fetching PRs for query '{query}': HTTP {response.status_code}")
                    break

                data = response.json()
                items = data.get('items', [])

                if not items:
                    break

                # Add PRs to dict, using ID as key to avoid duplicates
                for pr in items:
                    pr_id = pr.get('id')
                    if pr_id and pr_id not in prs_by_id:
                        prs_by_id[pr_id] = pr

                # Check if there are more pages
                if len(items) < per_page:
                    break

                page += 1
                time.sleep(0.5)  # Courtesy delay between pages

            except Exception as e:
                print(f"Error fetching PRs for query '{query}': {str(e)}")
                break

        # Delay between different query patterns
        time.sleep(0.5)

    # Convert dict back to list
    all_prs = list(prs_by_id.values())
    print(f"Found {len(all_prs)} unique PRs for {identifier}")

    return all_prs


def calculate_pr_stats(prs):
    """
    Calculate statistics from a list of pull requests.
    Returns a dictionary with comprehensive PR metrics.
    """
    total_prs = len(prs)
    merged = 0
    repos = set()
    merged_times = []  # Store merged times in minutes for merged PRs
    
    for pr in prs:
        # Track repository information
        repo_url = pr.get('repository_url', '')
        if repo_url:
            repo_name = '/'.join(repo_url.split('/')[-2:])
            repos.add(repo_name)

        # Track PR status
        state = pr.get('state')
        if state == 'closed':
            pull_request = pr.get('pull_request', {})
            merged_at = pull_request.get('merged_at')
            if merged_at:
                merged += 1
                
                # Calculate merged time (creation to merge)
                try:
                    created_at = pr.get('created_at')
                    if created_at and merged_at:
                        created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        merged_dt = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                        merged_time_minutes = (merged_dt - created_dt).total_seconds() / 60  # Convert to minutes
                        merged_times.append(merged_time_minutes)
                except Exception as e:
                    print(f"Warning: Could not calculate merged time for PR: {e}")
    
    acceptance_rate = (merged / total_prs * 100) if total_prs > 0 else 0
    
    # Calculate median merged time
    median_merged_time = None
    if merged_times:
        merged_times.sort()
        n = len(merged_times)
        if n % 2 == 0:
            median_merged_time = (merged_times[n // 2 - 1] + merged_times[n // 2]) / 2
        else:
            median_merged_time = merged_times[n // 2]
        median_merged_time = round(median_merged_time, 2)
    
    return {
        'total_prs': total_prs,
        'merged': merged,
        'acceptance_rate': round(acceptance_rate, 2),
        'median_merged_time': median_merged_time,
    }


def fetch_agent_stats(identifier, token=None):
    """
    Fetch and calculate PR statistics for a single agent.
    Returns dictionary with all stats and metadata.
    """
    print(f"Fetching data for {identifier}...")
    prs = fetch_all_prs(identifier, token)
    stats = calculate_pr_stats(prs)
    stats['github_identifier'] = identifier
    return stats


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
    """Load leaderboard data from HuggingFace dataset for current year."""
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
    """Save complete leaderboard to HuggingFace dataset as CSV."""
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

def update_all_agents():
    """
    Update PR statistics for all agents from HuggingFace dataset.
    Returns dictionary of all agent data with current stats.
    """
    token = get_github_token()

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return {}

    # Load existing cache
    cache_list = load_jsonl(CACHE_FILE)
    cache_dict = cache_to_dict(cache_list)

    # Update each agent
    for agent in agents:
        identifier = agent.get('github_identifier')
        if not identifier:
            print(f"Warning: Skipping agent without identifier: {agent}")
            continue

        try:
            # Fetch fresh PR statistics
            stats = fetch_agent_stats(identifier, token)

            # Merge metadata with stats
            cache_dict[identifier] = {
                'agent_name': agent.get('agent_name', 'Unknown'),
                'organization': agent.get('organization', 'Unknown'),
                'github_identifier': identifier,
                **stats
            }

            # Progressive save
            save_jsonl(CACHE_FILE, dict_to_cache(cache_dict))
            print(f"✓ Updated {identifier}")

        except Exception as e:
            print(f"✗ Error updating {identifier}: {str(e)}")
            continue

    return cache_dict


def initialize_data():
    """
    Initialize data on application startup.
    Priority: Leaderboard dataset > HuggingFace agents dataset
    """
    print("🚀 Initializing leaderboard data...")

    # Try loading existing leaderboard
    leaderboard_data = load_leaderboard_dataset()
    if leaderboard_data:
        save_jsonl(CACHE_FILE, leaderboard_data)
        print("✓ Initialized from leaderboard dataset")
        return

    # Try loading agents from HuggingFace and mining GitHub data
    agents = load_agents_from_hf()
    if agents:
        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        print("⛏️ Mining GitHub data...")
        cache_dict = update_all_agents()
        if cache_dict:
            save_leaderboard_to_hf(cache_dict)
        return

    # No data available
    print("⚠️ No data sources available. Waiting for first submission...")
    save_jsonl(CACHE_FILE, [])


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def get_leaderboard_dataframe():
    """
    Convert cache data to pandas DataFrame for display.
    Returns formatted DataFrame sorted by merged PRs.
    """
    cache_list = load_jsonl(CACHE_FILE)
    cache_dict = cache_to_dict(cache_list)
    
    rows = []
    for identifier, data in cache_dict.items():
        # Only include display-relevant fields
        # Normalize date formats for consistent display
        rows.append([
            data.get('agent_name', 'Unknown'),
            data.get('organization', 'Unknown'),
            data.get('total_prs', 0),
            data.get('merged', 0),
            data.get('acceptance_rate', 0.0),
            data.get('median_merged_time', None),
        ])
    
    # Create DataFrame
    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)
    
    # Ensure numeric types
    numeric_cols = ["Total PRs", "Merged PRs", "Acceptance Rate (%)", "Median Merge Duration (minutes)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Sort by Acceptance Rate (%) descending
    if "Acceptance Rate (%)" in df.columns and not df.empty:
        df = df.sort_values(by="Acceptance Rate (%)", ascending=False).reset_index(drop=True)
    
    return df


def refresh_leaderboard():
    """Manually trigger data refresh for all agents."""
    try:
        print("🔄 Manual refresh initiated")
        cache_dict = update_all_agents()
        if cache_dict:
            save_leaderboard_to_hf(cache_dict)
        return "✅ Data refreshed successfully!", get_leaderboard_dataframe()
    except Exception as e:
        error_msg = f"❌ Refresh failed: {str(e)}"
        print(error_msg)
        return error_msg, get_leaderboard_dataframe()


def submit_agent(identifier, agent_name, organization, description, website):
    """
    Submit a new agent to the leaderboard.
    Validates input, saves submission, and fetches PR data.
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "❌ GitHub identifier is required", get_leaderboard_dataframe()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe()
    if not organization or not organization.strip():
        return "❌ Organization name is required", get_leaderboard_dataframe()
    if not website or not website.strip():
        return "❌ Website URL is required", get_leaderboard_dataframe()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    description = description.strip()
    website = website.strip()

    # Validate GitHub identifier
    is_valid, message = validate_github_username(identifier)
    if not is_valid:
        return f"❌ {message}", get_leaderboard_dataframe()

    # Check for duplicates by loading agents from HuggingFace
    agents = load_agents_from_hf()
    if agents:
        existing_names = {agent['github_identifier'] for agent in agents}
        if identifier in existing_names:
            return f"⚠️ Agent with identifier '{identifier}' already exists", get_leaderboard_dataframe()

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
        return "❌ Failed to save submission", get_leaderboard_dataframe()
    
    # Fetch PR data immediately
    token = get_github_token()
    try:
        stats = fetch_agent_stats(identifier, token)
        
        # Update cache
        cache_list = load_jsonl(CACHE_FILE)
        cache_dict = cache_to_dict(cache_list)
        cache_dict[identifier] = {**submission, **stats}
        save_jsonl(CACHE_FILE, dict_to_cache(cache_dict))
        
        # Save to HuggingFace
        save_leaderboard_to_hf(cache_dict)
        
        return f"✅ Successfully submitted {agent_name}!", get_leaderboard_dataframe()
        
    except Exception as e:
        error_msg = f"⚠️ Submitted {agent_name}, but failed to fetch PR data: {str(e)}"
        print(error_msg)
        return error_msg, get_leaderboard_dataframe()


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

def scheduled_update_task():
    """Background daemon thread for periodic data updates."""
    while True:
        time.sleep(UPDATE_INTERVAL)
        print(f"\n🔄 Scheduled update started at {datetime.now().isoformat()}")
        try:
            cache_dict = update_all_agents()
            if cache_dict:
                save_leaderboard_to_hf(cache_dict)
            print("✓ Scheduled update completed")
        except Exception as e:
            print(f"✗ Scheduled update failed: {str(e)}")


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

# Initialize data before creating UI
initialize_data()

# Start background update thread
update_thread = threading.Thread(target=scheduled_update_task, daemon=True)
update_thread.start()

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
                filter_columns=["Acceptance Rate (%)", "Median Merge Duration (minutes)"]
            )
            
            refresh_button.click(
                fn=refresh_leaderboard,
                outputs=[status_display, leaderboard_table]
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
                outputs=[submission_status, leaderboard_table]
            )


# Launch application
if __name__ == "__main__":
    app.launch()