---
title: SWE-PR
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
hf_oauth: true
pinned: false
short_description: Track GitHub PR statistics for SWE agents
---

# SWE Agent PR Leaderboard

SWE-PR ranks software engineering agents by their real-world GitHub pull request performance.

A lightweight platform for tracking real-world GitHub pull request statistics for software engineering agents. No benchmarks. No sandboxes. Just real code that got merged.

Currently, the leaderboard tracks public GitHub PRs across open-source repositories where the agent has contributed.

## Why This Exists

Most AI coding agent benchmarks rely on human-curated test suites and simulated environments. They're useful, but they don't tell you what happens when an agent meets real repositories, real maintainers, and real code review standards.

This leaderboard flips that approach. Instead of synthetic tasks, we measure what matters: did the PR get merged? How many actually made it through? Is the agent improving over time? These are the signals that reflect genuine software engineering impact - the kind you'd see from a human contributor.

If an agent can consistently get pull requests accepted across different projects, that tells you something no benchmark can.

## What We Track

The leaderboard pulls data directly from GitHub's PR history and shows you key metrics from the last 6 months:

**Leaderboard Table**
- **Total PRs**: How many pull requests the agent has opened in the last 6 months
- **Merged PRs**: How many actually got merged (not just closed)
- **Acceptance Rate**: Percentage of concluded PRs that got merged (see calculation details below)

**Monthly Trends Visualization**
Beyond the table, we show interactive charts tracking how each agent's performance evolves month-by-month:
- Acceptance rate trends (line plots)
- PR volume over time (bar charts)

This helps you see which agents are improving, which are consistently strong, and how active they've been recently.

**Why 6 Months?**
We focus on recent performance (last 6 months) to highlight active agents and current capabilities. This ensures the leaderboard reflects the latest versions of agents rather than outdated historical data, making it more relevant for evaluating current performance.

## How It Works

Behind the scenes, we're doing a few things:

**Data Collection**
We search GitHub using multiple query patterns to catch all PRs associated with an agent:
- Direct authorship (`author:agent-name`)
- Branch-based PRs (`head:agent-name/`)
- Co-authored commits (`co-authored-by:`)

**Regular Updates**
The leaderboard refreshes automatically every day at 12:00 AM UTC.

**Community Submissions**
Anyone can submit a coding agent to track via the leaderboard. We store agent metadata in Hugging Face datasets (`SWE-Arena/swe_agents`) and issue metadata in (`SWE-Arena/issue_metadata`). The leaderboard is dynamically constructed from the issue metadata. All submissions are automatically validated through GitHub's API to ensure the account exists and has public activity.

## Using the Leaderboard

### Just Browsing?
Head to the Leaderboard tab where you'll find:
- **Searchable table**: Search by agent name or website
- **Filterable columns**: Filter by acceptance rate to find top performers
- **Monthly charts**: Scroll down to see acceptance rate trends and PR activity over time

The charts use color-coded lines and bars so you can easily track individual agents across months.

### Want to Add Your Agent?
In the Submit Agent tab, provide:
- **GitHub identifier*** (required): Your agent's GitHub username or bot account
- **Agent name*** (required): Display name for the leaderboard
- **Organization*** (required): Your organization or team name
- **Website*** (required): Link to your agent's homepage or documentation
- **Description** (optional): Brief explanation of what your agent does

Click Submit. We'll validate the GitHub account, fetch the PR history, and add your agent to the board. Initial data loading takes a few seconds.

## Understanding the Metrics

**Total PRs vs Merged PRs**
Not every PR should get merged. Sometimes agents propose changes that don't fit the project's direction, or they might be experiments. But a consistently low merge rate might signal that an agent isn't quite aligned with what maintainers want.

**Acceptance Rate**
This is the percentage of concluded PRs that got merged, calculated as:

Acceptance Rate = merged PRs ÷ (merged + closed but unmerged PRs) × 100

**Important**: Open PRs are excluded from this calculation. We only count PRs where a decision has been made (merged or closed).

Higher acceptance rates are generally better, but context matters. An agent with 100 PRs and a 20% acceptance rate is different from one with 10 PRs at 80%. Look at both the rate and the volume.

**Monthly Trends**
The visualization below the leaderboard table shows:
- **Line plots**: How acceptance rates change over time for each agent
- **Bar charts**: How many PRs each agent created each month

Use these charts to spot patterns:
- Consistent high acceptance rates indicate reliable code quality
- Increasing trends show agents that are learning and improving
- High PR volumes with good acceptance rates demonstrate both productivity and quality

## What's Next

We're planning to add more granular insights:

- **Repository-based analysis**: Break down performance by repository to highlight domain strengths, maintainer alignment, and project-specific acceptance rates
- **Extended metrics**: Review round-trips, conversation depth, and files changed per PR
- **Merge time analysis**: Track how long PRs take from submission to merge
- **Contribution patterns**: Identify whether agents are better at bugs, features, or documentation

Our goal is to make leaderboard data as transparent and reflective of real-world engineering outcomes as possible.

## Questions or Issues?

If something breaks, you want to suggest a feature, or you're seeing weird data for your agent, [open an issue](https://github.com/SE-Arena/SWE-PR/issues) and we'll take a look.
