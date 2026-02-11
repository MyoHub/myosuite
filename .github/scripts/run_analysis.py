#!/usr/bin/env python3
"""
Entry point for the GitHub Actions issue analysis workflow.
Reads configuration from environment variables and runs the analysis.
"""

import os
import sys
import json

from github_issue_agent import GitHubIssueAgent


def main():
    github_token = os.getenv("GITHUB_TOKEN")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    issue_number = os.getenv("ISSUE_NUMBER")
    repo_owner = os.getenv("REPO_OWNER")
    repo_name = os.getenv("REPO_NAME")

    missing = []
    if not github_token:
        missing.append("GITHUB_TOKEN")
    if not anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if not issue_number:
        missing.append("ISSUE_NUMBER")
    if not repo_owner:
        missing.append("REPO_OWNER")
    if not repo_name:
        missing.append("REPO_NAME")

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    issue_number = int(issue_number)

    agent = GitHubIssueAgent(github_token, anthropic_api_key)

    print(f"Analyzing issue #{issue_number} in {repo_owner}/{repo_name}...")
    result = agent.analyze_issue(
        owner=repo_owner,
        repo=repo_name,
        issue_number=issue_number,
        extended_thinking=True,
    )

    print(f"\nAnalysis for issue #{result['issue_number']}: {result['issue_title']}")
    print("-" * 60)
    print(result["analysis"])

    is_test = os.getenv("IS_TEST", "false").lower() == "true"

    if is_test:
        print("\n[TEST MODE] Skipping posting comment to GitHub.")
    else:
        agent.post_analysis_comment(repo_owner, repo_name, issue_number, result)

    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as f:
            f.write(f"analysis_status=success\n")
            f.write(f"issue_number={issue_number}\n")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
