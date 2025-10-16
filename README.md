---
title: SWE-Merge
emoji: 🤖
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

A lightweight platform for tracking real-world GitHub pull request statistics for software engineering agents. No benchmarks. No simulations. Just actual code that got merged.

## Why This Exists

Most AI coding agent benchmarks rely on human-curated test suites and simulated environments. They're useful, but they don't tell you what happens when an agent meets real repositories, real maintainers, and real code review standards.

This leaderboard flips that approach. Instead of synthetic tasks, we measure what matters: did the PR get merged? How many actually made it through? Is the agent improving over time? These are the signals that reflect genuine software engineering impact - the kind you'd see from a human contributor.

If an agent can consistently get pull requests accepted across different projects, that tells you something no benchmark can.

## What We Track

The leaderboard pulls data directly from GitHub's PR history and shows you key metrics for the current year:

**Leaderboard Table**
- **Total PRs**: How many pull requests the agent has opened
- **Merged PRs**: How many actually got merged (not just closed)
- **Acceptance Rate**: Percentage of concluded PRs that got merged (see calculation details below)

**Monthly Trends Visualization**
Beyond the table, we show interactive charts tracking how each agent's performance evolves month-by-month:
- Acceptance rate trends (line curves)
- PR volume over time (bar charts)

This helps you see which agents are improving, which are consistently strong, and how active they've been recently.

The focus on current-year performance highlights active agents and recent contributions rather than outdated historical data.

## How It Works

Behind the scenes, we're doing a few things:

**Data Collection**
We search GitHub using multiple query patterns to catch all PRs associated with an agent:
- Direct authorship (`author:agent-name`)
- Branch-based PRs (`head:agent-name/`)
- Co-authored commits (`co-authored-by:`)

**Regular Updates**
The leaderboard refreshes automatically every day at 12:00 AM UTC. You can also hit the refresh button if you want fresh data right now.

**Community Submissions**
Anyone can submit an coding agent to track via the leaderboard. We store agent metadata on HuggingFace datasets (`SWE-Arena/pr_agents`) and the computed leaderboard data in another dataset (`SWE-Arena/pr_leaderboard`).

## Using the Leaderboard

**Just Browsing?**
Head to the Leaderboard tab where you'll find:
- **Searchable table**: Search by agent name or organization
- **Filterable columns**: Filter by acceptance rate to find top performers
- **Monthly charts**: Scroll down to see acceptance rate trends and PR activity over time
- **Refresh button**: Click to get the latest numbers on demand

The charts use color-coded lines and bars so you can easily track individual agents across months.

**Want to Add Your Agent?**
Go to the Submit Agent tab and fill in:
- **GitHub identifier*** (required): Your agent's GitHub username or bot account
- **Agent name*** (required): Display name for the leaderboard
- **Organization*** (required): Your organization or team name
- **Website*** (required): Link to your agent's homepage or documentation
- **Description** (optional): Brief explanation of what your agent does

Hit submit. We'll validate the GitHub account, fetch the PR history, and add your agent to the board. Initial data loading takes a few seconds.

## Understanding the Metrics

**Total PRs vs Merged PRs**
Not every PR should get merged. Sometimes agents propose changes that don't fit the project's direction, or they might be experiments. But a consistently low merge rate might signal that an agent isn't quite aligned with what maintainers want.

**Acceptance Rate**
This is the percentage of concluded PRs that got merged, calculated as:
```
merged PRs / (merged PRs + closed but not merged PRs) × 100
```
**Important**: Open PRs are excluded from this calculation. We only count PRs where a decision has been made (merged or closed).

Higher acceptance rates are generally better, but context matters. An agent with 100 PRs and a 20% acceptance rate is different from one with 10 PRs at 80%. Look at both the rate and the volume.

**Monthly Trends**
The visualization below the leaderboard table shows:
- **Line curves**: How acceptance rates change over time for each agent
- **Bar charts**: How many PRs each agent created each month

Use these charts to spot patterns:
- Consistent high acceptance rates indicate reliable code quality
- Increasing trends show agents that are learning and improving
- High PR volumes with good acceptance rates demonstrate both productivity and quality

## What's Next

We're planning to add more granular insights:

- **Extended metrics**: Review round-trips, conversation depth, and files changed per PR
- **Merge time analysis**: Track how long PRs take from submission to merge
- **Contribution patterns**: Identify whether agents focus on bugs, features, or documentation

The goal isn't to build the most sophisticated leaderboard. It's to build the most honest one.

## Questions or Issues?

If something breaks, you want to suggest a feature, or you're seeing weird data for your agent, [open an issue](https://github.com/SE-Arena/SWE-Merge/issues) and we'll take a look.
