#!/usr/bin/env python3
"""
GitHub Issue Analysis Agent
Uses Claude API to analyze GitHub issues with repository context
and provide actionable resolution plans.
"""

import os
import anthropic
import requests
from typing import Dict, List, Optional
import json

class GitHubIssueAgent:
    def __init__(self, github_token: str, anthropic_api_key: str):
        """
        Initialize the GitHub Issue Agent
        
        Args:
            github_token: GitHub personal access token
            anthropic_api_key: Anthropic API key for Claude
        """
        self.github_token = github_token
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.github_headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def get_repository_context(self, owner: str, repo: str) -> Dict:
        """
        Gather repository context including README, structure, and recent commits
        """
        base_url = f"https://api.github.com/repos/{owner}/{repo}"
        
        context = {
            "repo_info": {},
            "readme": "",
            "structure": [],
            "recent_commits": []
        }
        
        # Get repository info
        response = requests.get(base_url, headers=self.github_headers)
        if response.status_code == 200:
            repo_data = response.json()
            context["repo_info"] = {
                "name": repo_data.get("name"),
                "description": repo_data.get("description"),
                "language": repo_data.get("language"),
                "topics": repo_data.get("topics", [])
            }
        
        # Get README
        try:
            readme_response = requests.get(
                f"{base_url}/readme",
                headers=self.github_headers
            )
            if readme_response.status_code == 200:
                readme_data = readme_response.json()
                # Decode base64 content
                import base64
                context["readme"] = base64.b64decode(
                    readme_data["content"]
                ).decode('utf-8')
        except Exception as e:
            print(f"Could not fetch README: {e}")
        
        # Get repository tree (file structure)
        try:
            tree_response = requests.get(
                f"{base_url}/git/trees/main?recursive=1",
                headers=self.github_headers
            )
            if tree_response.status_code == 200:
                tree_data = tree_response.json()
                context["structure"] = [
                    item["path"] for item in tree_data.get("tree", [])[:100]
                ]
        except Exception as e:
            print(f"Could not fetch repository structure: {e}")
        
        # Get recent commits
        try:
            commits_response = requests.get(
                f"{base_url}/commits?per_page=10",
                headers=self.github_headers
            )
            if commits_response.status_code == 200:
                commits_data = commits_response.json()
                context["recent_commits"] = [
                    {
                        "sha": commit["sha"][:7],
                        "message": commit["commit"]["message"],
                        "author": commit["commit"]["author"]["name"]
                    }
                    for commit in commits_data
                ]
        except Exception as e:
            print(f"Could not fetch recent commits: {e}")
        
        return context
    
    def get_issue_details(self, owner: str, repo: str, issue_number: int) -> Dict:
        """
        Fetch detailed information about a specific issue
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        response = requests.get(url, headers=self.github_headers)
        
        if response.status_code == 200:
            issue_data = response.json()
            
            # Get comments
            comments = []
            if issue_data.get("comments", 0) > 0:
                comments_url = issue_data["comments_url"]
                comments_response = requests.get(
                    comments_url,
                    headers=self.github_headers
                )
                if comments_response.status_code == 200:
                    comments_data = comments_response.json()
                    comments = [
                        {
                            "author": comment["user"]["login"],
                            "body": comment["body"],
                            "created_at": comment["created_at"]
                        }
                        for comment in comments_data
                    ]
            
            return {
                "number": issue_data["number"],
                "title": issue_data["title"],
                "body": issue_data["body"],
                "state": issue_data["state"],
                "labels": [label["name"] for label in issue_data.get("labels", [])],
                "assignees": [assignee["login"] for assignee in issue_data.get("assignees", [])],
                "created_at": issue_data["created_at"],
                "updated_at": issue_data["updated_at"],
                "comments": comments
            }
        else:
            raise Exception(f"Failed to fetch issue: {response.status_code}")
    
    def analyze_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        extended_thinking: bool = True
    ) -> Dict:
        """
        Analyze an issue using Claude and provide a resolution plan
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number to analyze
            extended_thinking: Use Claude's extended thinking capability
        
        Returns:
            Dictionary containing analysis and action plan
        """
        print(f"Fetching repository context for {owner}/{repo}...")
        repo_context = self.get_repository_context(owner, repo)
        
        print(f"Fetching issue #{issue_number} details...")
        issue = self.get_issue_details(owner, repo, issue_number)
        
        print("Analyzing issue with Claude...")
        
        # Construct the prompt for Claude
        prompt = f"""You are a GitHub issue analysis agent. Your task is to analyze the following issue in the context of the repository and provide a structured action plan.

REPOSITORY CONTEXT:
-------------------
Repository: {owner}/{repo}
Description: {repo_context['repo_info'].get('description', 'N/A')}
Primary Language: {repo_context['repo_info'].get('language', 'N/A')}
Topics: {', '.join(repo_context['repo_info'].get('topics', []))}

README Summary:
{repo_context['readme'][:2000] if repo_context['readme'] else 'No README available'}

Repository Structure (key files):
{chr(10).join(repo_context['structure'][:50])}

Recent Commits:
{chr(10).join([f"- {c['sha']}: {c['message'][:100]}" for c in repo_context['recent_commits']])}

ISSUE DETAILS:
--------------
Issue #{issue['number']}: {issue['title']}
State: {issue['state']}
Labels: {', '.join(issue['labels']) if issue['labels'] else 'None'}
Created: {issue['created_at']}

Description:
{issue['body']}

Comments ({len(issue['comments'])}):
{chr(10).join([f"- {c['author']}: {c['body'][:200]}" for c in issue['comments'][:5]])}

ANALYSIS TASK:
--------------
Please analyze this issue and provide:

1. **Issue Classification**: Categorize the issue (bug, feature request, documentation, question, etc.)

2. **Severity Assessment**: Rate the severity (critical, high, medium, low) and explain why

3. **Root Cause Analysis**: Based on the repository context, identify potential root causes

4. **Action Required**: Determine if this issue requires action (yes/no) and explain

5. **Resolution Plan**: Provide a detailed, step-by-step plan to resolve this issue, including:
   - Specific files that likely need to be modified
   - Code changes or implementation approach
   - Testing recommendations
   - Documentation updates needed

6. **Estimated Effort**: Estimate the effort required (hours/days)

7. **Related Issues**: Identify if this might be related to other common issues in similar projects

Please structure your response in a clear, actionable format."""

        # Call Claude API
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }
        if extended_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 3000
            }
        response = self.client.messages.create(**kwargs)
        
        # Extract the analysis from response
        analysis_text = ""
        thinking_text = ""
        
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                analysis_text = block.text
        
        result = {
            "issue_number": issue_number,
            "issue_title": issue['title'],
            "repository": f"{owner}/{repo}",
            "analysis": analysis_text,
            "thinking_process": thinking_text if extended_thinking else None,
            "model_used": response.model,
            "tokens_used": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens
            }
        }
        
        return result
    
    def post_analysis_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        analysis: Dict
    ):
        """
        Post the analysis as a comment on the GitHub issue
        """
        comment_body = f"""## 🤖 AI Analysis Report

{analysis['analysis']}

---
*Generated by Claude Issue Agent*
*Model: {analysis['model_used']} | Tokens: {analysis['tokens_used']['input']}→{analysis['tokens_used']['output']}*
"""
        
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        response = requests.post(
            url,
            headers=self.github_headers,
            json={"body": comment_body}
        )
        
        if response.status_code == 201:
            print(f"✓ Analysis posted to issue #{issue_number}")
            return True
        else:
            print(f"✗ Failed to post comment: {response.status_code}")
            return False


def main():
    """
    Example usage of the GitHub Issue Agent
    """
    # Get API keys from environment
    github_token = os.getenv("GITHUB_TOKEN")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not github_token or not anthropic_api_key:
        print("Error: Please set GITHUB_TOKEN and ANTHROPIC_API_KEY environment variables")
        return
    
    # Initialize agent
    agent = GitHubIssueAgent(github_token, anthropic_api_key)
    
    # Example: Analyze an issue
    owner = "your-username"  # Replace with repo owner
    repo = "your-repo"       # Replace with repo name
    issue_number = 1         # Replace with issue number
    
    try:
        # Analyze the issue
        result = agent.analyze_issue(
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            extended_thinking=True
        )
        
        # Print results
        print("\n" + "="*80)
        print(f"ANALYSIS FOR ISSUE #{result['issue_number']}: {result['issue_title']}")
        print("="*80)
        print(result['analysis'])
        print("\n" + "="*80)
        
        # Optionally post to GitHub
        post_to_github = input("\nPost this analysis to GitHub? (y/n): ")
        if post_to_github.lower() == 'y':
            agent.post_analysis_comment(owner, repo, issue_number, result)
        
        # Save to file
        with open(f"issue_{issue_number}_analysis.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Analysis saved to issue_{issue_number}_analysis.json")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
