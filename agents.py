from __future__ import annotations

from typing import List, Dict, Any
import json

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.firecrawl import FirecrawlTools


def build_orchestrator(model_name: str) -> Agent:
    """Orchestrator agent that manages workflow and provides final verdict."""
    return Agent(
        name="Orchestrator",
        role="Coordinate agents and manage job matching workflow",
        model=OpenAIChat(id=model_name),
        instructions=[
            "You coordinate the job matching process in this order:",
            "1. Resume Parser extracts candidate profile",
            "2. Job Scraper fetches each job posting details",
            "3. Job Scorer evaluates each job against candidate profile",
            "4. Summarizer generates final summary for top matches only",
            "Ensure each agent receives proper context from previous steps.",
            "Filter out jobs with match_score < 0.5 before summarization.",
            "Provide clear, structured output with all agent results.",
            "Never duplicate summaries - each job needs unique analysis.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def build_resume_parser(model_name: str) -> Agent:
    """Agent that extracts structured information from resume text."""
    return Agent(
        name="Resume Parser",
        role="Extract and structure all information from resume OCR text",
        model=OpenAIChat(id=model_name),
        instructions=[
            "Parse raw OCR text from resume and extract ALL information.",
            "You MUST return ONLY valid JSON (no markdown, no code blocks, no explanations).",
            "",
            "Extract these fields:",
            "- name: Full name of candidate",
            "- email: Email address",
            "- phone: Phone number",
            "- skills: Array of ALL technical skills, tools, frameworks, languages",
            "- experience_summary: String summary of work experience",
            "- total_years_experience: Number (calculate total years from all roles)",
            "- education: Array of objects with school, degree, dates",
            "- certifications: Array of certification names",
            "- interests: Array of interests/hobbies",
            "",
            "Example output format:",
            '{"name": "John Doe", "email": "john@example.com", "phone": "+1234567890", "skills": ["Python", "Java"], "experience_summary": "2 years in AI", "total_years_experience": 2, "education": [{"school": "MIT", "degree": "BS CS", "dates": "2020"}], "certifications": ["AWS"], "interests": ["AI", "ML"]}',
            "",
            "CRITICAL: Return ONLY the JSON object, nothing else.",
        ],
        show_tool_calls=False,
        markdown=False,
    )


def build_scraper() -> Agent:
    """Agent that scrapes individual job postings."""
    return Agent(
        name="Job Scraper",
        role="Extract complete job posting information from URLs",
        tools=[FirecrawlTools(scrape=True, crawl=False)],  # Fixed: use scrape for single URLs
        instructions=[
            "Given a job URL, extract ALL available information:",
            "- Job title (exact title from posting)",
            "- Company name",
            "- Complete job description",
            "- Required skills (list each skill separately)",
            "- Required experience (years and type)",
            "- Qualifications and education requirements",
            "- Responsibilities",
            "- Salary/compensation (if mentioned)",
            "- Location",
            "- Job type (full-time, internship, etc.)",
            "Return structured data with all fields clearly labeled.",
            "If a field is not found, mark it as 'Not specified'.",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def build_scorer(model_name: str) -> Agent:
    """Agent that evaluates job-candidate fit using reasoning."""
    return Agent(
        name="Job Scorer",
        role="Evaluate candidate-job match using AI reasoning and domain knowledge",
        model=OpenAIChat(id=model_name),
        instructions=[
            "You receive: candidate_profile (JSON) and job_details (JSON).",
            "Analyze the match holistically considering:",
            "1. Skills Match (40%): How many required skills does candidate have?",
            "2. Experience Match (30%): Does experience level and domain align?",
            "3. Role Fit (20%): Does past work match job responsibilities?",
            "4. Growth Potential (10%): Can candidate grow in this role?",
            "",
            "CRITICAL: Read the actual job requirements carefully!",
            "- If job is 'Billing Executive', score based on finance/billing skills",
            "- If job is 'Data Science', score based on ML/AI skills",
            "- Don't assume job type - read the scraped job_details",
            "",
            "Return JSON with:",
            "- match_score: 0.0 to 1.0 (be strict: <0.3=poor, 0.3-0.5=weak, 0.5-0.7=good, >0.7=excellent)",
            "- key_matches: list of specific matching qualifications",
            "- requirements_met: count of requirements candidate satisfies",
            "- total_requirements: total requirements in job posting",
            "- reasoning: 2-3 sentences explaining your score",
            "- mismatch_areas: list of gaps or missing qualifications",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def build_summarizer(model_name: str) -> Agent:
    """Agent that creates detailed job match summaries."""
    return Agent(
        name="Summarizer",
        role="Generate unique, detailed summaries for each matched job",
        model=OpenAIChat(id=model_name),
        instructions=[
            "You receive: candidate_profile, job_details, and match_score.",
            "Write a unique 150-200 word summary that:",
            "- States whether candidate is a good/poor fit (based on match_score)",
            "- Explains specific skills and experience that match",
            "- Highlights any experience gaps or concerns",
            "- Discusses growth opportunities if applicable",
            "- Mentions practical considerations (location, compensation, etc.)",
            "",
            "CRITICAL RULES:",
            "- Each summary must be unique - never reuse text from other jobs",
            "- Be honest about fit - don't oversell poor matches",
            "- Reference the actual job title and company name",
            "- If match_score < 0.5, explain why it's not a good fit",
            "- If match_score >= 0.5, explain why it's a strong match",
        ],
        show_tool_calls=True,
        markdown=True,
    )


# Usage example with proper workflow
def process_job_matching(
    resume_text: str,
    job_urls: List[str],
    model_name: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Complete job matching workflow.
    
    Args:
        resume_text: OCR-extracted text from resume
        job_urls: List of job posting URLs
        model_name: OpenAI model to use
    
    Returns:
        Dictionary with candidate profile and matched jobs
    """
    # Initialize agents
    parser = build_resume_parser(model_name)
    scraper = build_scraper()
    scorer = build_scorer(model_name)
    summarizer = build_summarizer(model_name)
    
    # Step 1: Parse resume
    print("=" * 80)
    print("📄 RESUME PARSER AGENT - Extracting candidate profile")
    print("=" * 80)
    candidate_profile = parser.run(resume_text)
    print(f"[PARSER OUTPUT]: {candidate_profile}")
    
    matched_jobs = []
    
    # Step 2-4: Process each job
    for idx, job_url in enumerate(job_urls, 1):
        print("\n" + "=" * 80)
        print(f"🔍 JOB SCRAPER AGENT - Processing job {idx}/{len(job_urls)}")
        print("=" * 80)
        
        # Scrape job details
        job_details = scraper.run(f"Scrape job posting from: {job_url}")
        print(f"[SCRAPER OUTPUT for {job_url}]:\n{job_details}\n")
        
        # Score the match
        print("=" * 80)
        print(f"🤖 JOB SCORER AGENT - Calculating match score for job {idx}")
        print("=" * 80)
        score_input = {
            "candidate_profile": candidate_profile,
            "job_details": job_details,
            "job_url": job_url
        }
        match_result = scorer.run(json.dumps(score_input))
        print(f"[SCORER OUTPUT]:\n{match_result}\n")
        
        # Parse score result
        try:
            score_data = json.loads(match_result) if isinstance(match_result, str) else match_result
            match_score = score_data.get("match_score", 0.0)
            
            # Only summarize if match score is reasonable
            if match_score >= 0.5:
                print("=" * 80)
                print(f"📝 SUMMARIZER AGENT - Generating summary for job {idx}")
                print("=" * 80)
                summary_input = {
                    "candidate_profile": candidate_profile,
                    "job_details": job_details,
                    "match_score": match_score,
                    "job_url": job_url
                }
                summary = summarizer.run(json.dumps(summary_input))
                print(f"[SUMMARIZER OUTPUT]:\n{summary}\n")
            else:
                summary = f"Low match score ({match_score:.1%}). Candidate skills don't align well with job requirements."
            
            matched_jobs.append({
                "rank": idx,
                "job_url": job_url,
                "match_score": match_score,
                "summary": summary,
                **score_data
            })
            
        except Exception as e:
            print(f"Error processing job {idx}: {e}")
            continue
    
    # Sort by match score
    matched_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    # Final output
    print("\n" + "=" * 80)
    print("✅ FINAL RESULTS")
    print("=" * 80)
    print(f"Analyzed {len(job_urls)} jobs")
    print(f"Good matches (≥50%): {len([j for j in matched_jobs if j.get('match_score', 0) >= 0.5])}")
    
    return {
        "candidate_profile": candidate_profile,
        "matched_jobs": matched_jobs,
        "jobs_analyzed": len(job_urls)
    }