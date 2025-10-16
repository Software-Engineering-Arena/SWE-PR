import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import requests
from datetime import datetime, timezone
from collections import defaultdict
from huggingface_hub import HfApi, HfFolder, hf_hub_download
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
AGENTS_FILE = "agent_metadata.jsonl"
SUBMISSIONS_REPO = "SWE-Arena/pr_submissions"
LEADERBOARD_REPO = "SWE-Arena/pr_leaderboard"
UPDATE_INTERVAL = 86400  # 24 hours in seconds

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Organization", "string"),
    ("GitHub Name", "string"),
    ("Total PRs", "number"),
    ("Merged PRs", "number"),
    ("Acceptance Rate (%)", "number"),
    ("First Contribution", "string"),
    ("Last Updated", "string")
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
    """Convert list of cache entries to dictionary keyed by github_name."""
    return {entry['github_name']: entry for entry in cache_list}


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

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None,
                         max_retries=6, timeout=30):
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


def validate_github_username(github_name):
    """Verify that a GitHub username exists with backoff-aware requests."""
    try:
        token = get_github_token()
        headers = {'Authorization': f'token {token}'} if token else {}
        url = f'https://api.github.com/users/{github_name}'
        response = request_with_backoff('GET', url, headers=headers, max_retries=6)
        if response is None:
            return False, "Validation error: network/rate limit exhausted"
        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub username not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def fetch_all_prs(github_name, token=None):
    """
    Fetch all pull requests authored by a GitHub user.
    Uses pagination to retrieve all results.
    """
    headers = {'Authorization': f'token {token}'} if token else {}
    all_prs = []
    page = 1
    per_page = 100
    
    while True:
        url = f'https://api.github.com/search/issues'
        params = {
            'q': f'is:pr author:{github_name}',
            'per_page': per_page,
            'page': page
        }

        try:
            response = request_with_backoff('GET', url, headers=headers, params=params, max_retries=6)
            if response is None:
                print(f"Error fetching PRs for {github_name}: retries exhausted")
                break

            if response.status_code != 200:
                print(f"Error fetching PRs for {github_name}: HTTP {response.status_code}")
                break

            data = response.json()
            items = data.get('items', [])

            if not items:
                break

            all_prs.extend(items)

            # Check if there are more pages
            if len(items) < per_page:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"Error fetching PRs for {github_name}: {str(e)}")
            break
    
    return all_prs


def calculate_pr_stats(prs):
    """
    Calculate statistics from a list of pull requests.
    Returns a dictionary with comprehensive PR metrics.
    """
    total_prs = len(prs)
    merged = 0
    repos = set()
    prs_by_repo = defaultdict(int)
    first_contribution = None
    last_contribution = None
    
    for pr in prs:
        # Track repository information
        repo_url = pr.get('repository_url', '')
        if repo_url:
            repo_name = '/'.join(repo_url.split('/')[-2:])
            repos.add(repo_name)
            prs_by_repo[repo_name] += 1
        
        # Track contribution timeline
        created_at = pr.get('created_at')
        if created_at:
            if not first_contribution or created_at < first_contribution:
                first_contribution = created_at
            if not last_contribution or created_at > last_contribution:
                last_contribution = created_at
        
        # Track PR status
        state = pr.get('state')
        if state == 'closed':
            pull_request = pr.get('pull_request', {})
            if pull_request.get('merged_at'):
                merged += 1
    
    acceptance_rate = (merged / total_prs * 100) if total_prs > 0 else 0
    
    return {
        'total_prs': total_prs,
        'merged': merged,
        'acceptance_rate': round(acceptance_rate, 2),
        'first_contribution': first_contribution or 'N/A',
        'last_contribution': last_contribution or 'N/A',
        'prs_by_repo': dict(prs_by_repo)
    }


def fetch_agent_stats(github_name, token=None):
    """
    Fetch and calculate PR statistics for a single agent.
    Returns dictionary with all stats and metadata.
    """
    print(f"Fetching data for {github_name}...")
    prs = fetch_all_prs(github_name, token)
    stats = calculate_pr_stats(prs)
    stats['github_name'] = github_name
    stats['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return stats


# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def load_submissions_dataset():
    """Load agent submissions from HuggingFace dataset."""
    try:
        dataset = load_dataset(SUBMISSIONS_REPO, split='train')
        submissions = []
        for row in dataset:
            submissions.append({
                'agent_name': row.get('agent_name', 'Unknown'),
                'github_name': row.get('github_name'),
                'organization': row.get('organization', 'Unknown'),
                'description': row.get('description', ''),
            })
        print(f"✓ Loaded {len(submissions)} submissions from HuggingFace")
        return submissions
    except Exception as e:
        print(f"Could not load submissions dataset: {str(e)}")
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


def save_submission_to_hf(data):
    """Save a new submission to HuggingFace dataset."""
    try:
        api = HfApi()
        token = HfFolder.get_token()
        
        if not token:
            raise Exception("No HuggingFace token found")
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        year = datetime.now().year
        github_name = data['github_name']
        filename = f"{year}/{timestamp}_{github_name}.json"
        
        # Save locally first
        os.makedirs(str(year), exist_ok=True)
        filepath = f"{year}/{timestamp}_{github_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Upload to HuggingFace
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=SUBMISSIONS_REPO,
            repo_type="dataset",
            token=token
        )
        
        # Clean up
        os.remove(filepath)
        os.rmdir(str(year))
        
        print(f"✓ Saved submission to HuggingFace: {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error saving submission: {str(e)}")
        return False


def save_leaderboard_to_hf(cache_dict):
    """Save complete leaderboard to HuggingFace dataset as CSV."""
    try:
        token = HfFolder.get_token()
        if not token:
            raise Exception("No HuggingFace token found")
        
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
    Update PR statistics for all agents in the metadata file.
    Returns dictionary of all agent data with current stats.
    """
    token = get_github_token()
    
    # Load agent metadata
    agents = load_jsonl(AGENTS_FILE)
    if not agents:
        print("No agents found in metadata file")
        return {}
    
    # Load existing cache
    cache_list = load_jsonl(CACHE_FILE)
    cache_dict = cache_to_dict(cache_list)
    
    # Update each agent
    for agent in agents:
        github_name = agent.get('github_name')
        if not github_name:
            print(f"Warning: Skipping agent without github_name: {agent}")
            continue
        
        try:
            # Fetch fresh PR statistics
            stats = fetch_agent_stats(github_name, token)
            
            # Merge metadata with stats
            cache_dict[github_name] = {
                'agent_name': agent.get('agent_name', 'Unknown'),
                'github_name': github_name,
                'organization': agent.get('organization', 'Unknown'),
                'description': agent.get('description', ''),
                **stats
            }
            
            # Progressive save
            save_jsonl(CACHE_FILE, dict_to_cache(cache_dict))
            print(f"✓ Updated {github_name}")
            
        except Exception as e:
            print(f"✗ Error updating {github_name}: {str(e)}")
            continue
    
    return cache_dict


def initialize_data():
    """
    Initialize data on application startup.
    Priority: Leaderboard dataset > Submissions dataset > Local files
    """
    print("🚀 Initializing leaderboard data...")
    
    # Try loading existing leaderboard
    leaderboard_data = load_leaderboard_dataset()
    if leaderboard_data:
        save_jsonl(CACHE_FILE, leaderboard_data)
        print("✓ Initialized from leaderboard dataset")
        return
    
    # Try loading submissions and mining GitHub data
    submissions_data = load_submissions_dataset()
    if submissions_data:
        save_jsonl(AGENTS_FILE, submissions_data)
        print("✓ Loaded metadata from submissions dataset")
        print("⛏️ Mining GitHub data...")
        cache_dict = update_all_agents()
        if cache_dict:
            save_leaderboard_to_hf(cache_dict)
        return
    
    # Check local metadata file
    agents = load_jsonl(AGENTS_FILE)
    if agents:
        print(f"✓ Found {len(agents)} agents in local metadata")
        print("⛏️ Mining GitHub data...")
        cache_dict = update_all_agents()
        if cache_dict:
            save_leaderboard_to_hf(cache_dict)
    else:
        print("⚠️ No data sources available. Waiting for first submission...")
        save_jsonl(AGENTS_FILE, [])
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
    for github_name, data in cache_dict.items():
        # Only include display-relevant fields
        # Normalize date formats for consistent display
        rows.append([
            data.get('agent_name', 'Unknown'),
            data.get('organization', 'Unknown'),
            github_name,
            data.get('total_prs', 0),
            data.get('merged', 0),
            data.get('acceptance_rate', 0.0),
            normalize_date_format(data.get('first_contribution', 'N/A')),
            normalize_date_format(data.get('last_updated', 'N/A'))
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


def submit_agent(github_name, agent_name, organization, description):
    """
    Submit a new agent to the leaderboard.
    Validates input, saves submission, and fetches PR data.
    """
    # Validate required fields
    if not github_name or not github_name.strip():
        return "❌ GitHub username is required", get_leaderboard_dataframe()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe()
    if not organization or not organization.strip():
        return "❌ Organization name is required", get_leaderboard_dataframe()
    
    # Clean inputs
    github_name = github_name.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    description = description.strip() if description else ""
    
    # Validate GitHub username
    is_valid, message = validate_github_username(github_name)
    if not is_valid:
        return f"❌ {message}", get_leaderboard_dataframe()
    
    # Check for duplicates
    agents = load_jsonl(AGENTS_FILE)
    existing_names = {agent['github_name'] for agent in agents}
    
    if github_name in existing_names:
        return f"⚠️ Agent with GitHub name '{github_name}' already exists", get_leaderboard_dataframe()
    
    # Create submission
    submission = {
        'agent_name': agent_name,
        'github_name': github_name,
        'organization': organization,
        'description': description,
    }
    
    # Save to HuggingFace
    if not save_submission_to_hf(submission):
        return "❌ Failed to save submission", get_leaderboard_dataframe()
    
    # Add to local metadata
    agents.append(submission)
    save_jsonl(AGENTS_FILE, agents)
    
    # Fetch PR data immediately
    token = get_github_token()
    try:
        stats = fetch_agent_stats(github_name, token)
        
        # Update cache
        cache_list = load_jsonl(CACHE_FILE)
        cache_dict = cache_to_dict(cache_list)
        cache_dict[github_name] = {**submission, **stats}
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
                search_columns=["Agent Name", "Organization", "GitHub Name"],
                filter_columns=["Total PRs", "Acceptance Rate (%)"]
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
                        label="GitHub Username*",
                        placeholder="octocat"
                    )
                    name_input = gr.Textbox(
                        label="Agent Name*",
                        placeholder="My Awesome Agent"
                    )
                
                with gr.Column():
                    organization_input = gr.Textbox(
                        label="Organization*",
                        placeholder="Acme Corp"
                    )
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of your agent",
                        lines=3
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
                inputs=[github_input, name_input, organization_input, description_input],
                outputs=[submission_status, leaderboard_table]
            )


# Launch application
if __name__ == "__main__":
    app.launch()