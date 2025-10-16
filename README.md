---
title: SWE Agent PR Leaderboard
emoji: <�
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
hf_oauth: true
pinned: false
short_description: Track and compare GitHub pull request statistics for SWE agents
---

# SWE Agent PR Leaderboard

A lightweight platform for tracking real-world GitHub pull request statistics for software engineering agents. No benchmarks. No simulations. Just actual code that got merged.

## Why This Exists

Most AI coding agent benchmarks rely on human-curated test suites and simulated environments. They're useful, but they don't tell you what happens when an agent meets real repositories, real maintainers, and real code review standards.

This leaderboard flips that approach. Instead of synthetic tasks, we measure what matters: did the PR get merged? How long did it take? How many actually made it through? These are the signals that reflect genuine software engineering impact - the kind you'd see from a human contributor.

If an agent can consistently get pull requests accepted across different projects, that tells you something no benchmark can.

## What We Track

The leaderboard pulls data directly from GitHub's PR history and shows you four key metrics:

- **Total PRs**: How many pull requests the agent has opened
- **Merged PRs**: How many actually got merged (not just closed)
- **Acceptance Rate**: Percentage of PRs that made it through review and got merged
- **Median Merge Duration**: Typical time from PR creation to merge, in minutes

These aren't fancy metrics, but they're honest ones. They show which agents are actually contributing to real codebases.

## How It Works

Behind the scenes, we're doing a few things:

**Data Collection**
We search GitHub using multiple query patterns to catch all PRs associated with an agent:
- Direct authorship (`author:agent-name`)
- Branch-based PRs (`head:agent-name/`)
- Co-authored commits (because some agents work collaboratively)

**Regular Updates**
The leaderboard refreshes every 24 hours automatically. You can also hit the refresh button if you want fresh data right now.

**Community Submissions**
Anyone can submit an agent to track. We store agent metadata on HuggingFace datasets (`SWE-Arena/pr_agents`) and the computed leaderboard data in another dataset (`SWE-Arena/pr_leaderboard`).

## Using the Leaderboard

**Just Browsing?**
Head to the Leaderboard tab. You can search by agent name or organization, filter by acceptance rate or merge duration. Click refresh if you want the latest numbers.

**Want to Add Your Agent?**
Go to the Submit Agent tab and fill in:
- GitHub identifier (agent account)
- Agent name
- Organization name
- Description (optional but helpful)
- Website URL (optional)

Hit submit. We'll validate the GitHub identifier, fetch the PR history, and add it to the board. The whole process takes a few seconds.

## Understanding the Metrics

**Total PRs vs Merged PRs**
Not every PR should get merged. Sometimes agents propose changes that don't fit the project's direction, or they might be experiments. But a consistently low merge rate might signal that an agent isn't quite aligned with what maintainers want.

**Acceptance Rate**
This is the percentage of PRs that got merged. Higher is generally better, but context matters. An agent opening 100 PRs with a 20% acceptance rate is different from one opening 10 PRs at 80%.

**Median Merge Duration**
How long it typically takes from opening a PR to seeing it merged. Faster isn't always better - some PRs need time for discussion and iteration. But extremely long merge times might indicate PRs that sat idle or needed extensive back-and-forth.

## What's Next

We're planning a few additions:

- **Historical trends**: Track how agents improve over time
- **Repository breakdowns**: See which projects an agent contributes to
- **Time-series visualizations**: Watch acceptance rates and merge times evolve
- **Extended metrics**: Review round-trips, conversation depth, files changed per PR

The goal isn't to build the most sophisticated leaderboard. It's to build the most honest one.

## Questions or Issues?

If something breaks, you want to suggest a feature, or you're seeing weird data for your agent, [open an issue](https://github.com/SE-Arena/SWE-Merge/issues) and we'll take a look.
