"""
Extract and format job data from API response for Firebase storage.
Uses the exact same structure as test_firebase_simple.py
"""
from datetime import datetime
from typing import List, Dict, Any


def detect_portal(url: str) -> str:
    """
    Detect the job portal from URL domain.
    Same logic as in app.py
    """
    url_lower = url.lower()
    if 'linkedin.com' in url_lower:
        return 'LinkedIn'
    elif 'internshala.com' in url_lower:
        return 'Internshala'
    elif 'indeed.com' in url_lower:
        return 'Indeed'
    elif 'glassdoor.com' in url_lower:
        return 'Glassdoor'
    elif 'monster.com' in url_lower:
        return 'Monster'
    elif 'naukri.com' in url_lower:
        return 'Naukri'
    elif 'timesjobs.com' in url_lower:
        return 'TimesJobs'
    elif 'shine.com' in url_lower:
        return 'Shine'
    elif 'hired.com' in url_lower:
        return 'Hired'
    elif 'angel.co' in url_lower or 'angelist.com' in url_lower:
        return 'AngelList'
    elif 'stackoverflow.com' in url_lower or 'stackoverflowjobs.com' in url_lower:
        return 'Stack Overflow'
    elif 'github.com' in url_lower:
        return 'GitHub Jobs'
    elif 'dice.com' in url_lower:
        return 'Dice'
    elif 'ziprecruiter.com' in url_lower:
        return 'ZipRecruiter'
    elif 'simplyhired.com' in url_lower:
        return 'SimplyHired'
    else:
        # Extract domain name as fallback
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '').split('.')[0]
            return domain.capitalize()
        except:
            return 'Unknown'


def extract_jobs_from_response(api_response: dict) -> List[dict]:
    """
    Extract and format jobs from API response for Firebase.
    
    Maps API response fields to Firebase format exactly matching test_firebase_simple.py structure.
    
    Args:
        api_response: The JSON response from POST /api/match-jobs endpoint
            Expected structure:
            {
                "matched_jobs": [
                    {
                        "rank": 1,
                        "job_url": "...",
                        "job_title": "...",
                        "company": "...",
                        "match_score": 0.85,
                        "summary": "...",
                        "key_matches": [...],
                        "requirements_met": 7,
                        "total_requirements": 8,
                        "scraped_summary": "..."
                    }
                ]
            }
    
    Returns:
        List of job_data dictionaries ready for Firebase save_job_applications_batch()
        Format matches test_firebase_simple.py exactly:
        {
            "appliedDate": datetime.now(),
            "company": "...",
            "createdAt": datetime.now(),
            "interviewDate": "",
            "jobDescription": "...",
            "link": "...",
            "notes": "...",
            "portal": "...",
            "role": "...",
            "status": "Matched",
            "visaRequired": "No"
        }
    """
    jobs_data = []
    matched_jobs = api_response.get("matched_jobs", [])
    
    print(f"[EXTRACT] Processing {len(matched_jobs)} matched jobs from API response")
    
    for idx, job in enumerate(matched_jobs, 1):
        try:
            # Extract basic fields
            company = job.get("company", "").strip() or ""
            job_title = job.get("job_title", "").strip() or ""
            job_url = job.get("job_url", "").strip() or ""
            
            # Detect portal from URL
            portal = detect_portal(job_url) if job_url else "Unknown"
            
            # Extract match information
            match_score = job.get("match_score", 0.0)
            requirements_met = job.get("requirements_met", 0)
            total_requirements = job.get("total_requirements", 0)
            key_matches = job.get("key_matches", [])
            summary = (job.get("summary") or "").strip() or ""
            scraped_summary = (job.get("scraped_summary") or "").strip() or ""
            
            # Create jobDescription combining match info, summary, and key matches
            job_description_parts = []
            
            # Add match score info
            if match_score > 0:
                job_description_parts.append(f"Match Score: {match_score:.1%}")
            
            # Add requirements info
            if total_requirements > 0:
                req_percentage = (requirements_met / total_requirements) * 100
                job_description_parts.append(
                    f"Requirements Met: {requirements_met}/{total_requirements} ({req_percentage:.0f}%)"
                )
            
            # Add summary (prefer scraped_summary if available, otherwise summary)
            description_text = scraped_summary if scraped_summary else summary
            if description_text:
                # Truncate description if too long (keep first 1000 chars)
                if len(description_text) > 1000:
                    description_text = description_text[:1000] + "..."
                job_description_parts.append(description_text)
            
            # Add key matches
            if key_matches:
                matches_str = ", ".join(key_matches[:10])  # Limit to first 10 matches
                job_description_parts.append(f"Key Matches: {matches_str}")
            
            job_description = "\n\n".join(job_description_parts) or ""
            
            # Create notes from summary (truncate to 500 chars)
            notes = summary[:500] if summary else ""
            
            # Prepare job_data in EXACT format from test_firebase_simple.py
            # Using datetime.now() objects, not formatted strings
            job_data = {
                "appliedDate": datetime.now(),  # EXACT same as test_firebase_simple.py line 88
                "company": company,
                "createdAt": datetime.now(),  # EXACT same as test_firebase_simple.py line 90
                "interviewDate": "",  # EXACT same as test_firebase_simple.py line 91
                "jobDescription": job_description,  # Combined match info + summary
                "link": job_url,  # EXACT same as test_firebase_simple.py line 93
                "notes": notes,  # Summary truncated to 500 chars
                "portal": portal,  # EXACT same as test_firebase_simple.py line 95
                "role": job_title,  # EXACT same as test_firebase_simple.py line 96
                "status": "Matched",  # Use "Matched" for auto-matched jobs (not "Applied")
                "visaRequired": "No"  # EXACT same as test_firebase_simple.py line 98
            }
            
            jobs_data.append(job_data)
            
            print(f"[EXTRACT] Job {idx}: {job_title} at {company} ({portal}) - Match: {match_score:.1%}")
            
        except Exception as e:
            print(f"[EXTRACT] [ERROR] Failed to extract job {idx}: {str(e)}")
            import traceback
            print(f"[EXTRACT] Traceback: {traceback.format_exc()}")
            continue
    
    print(f"[EXTRACT] Successfully extracted {len(jobs_data)}/{len(matched_jobs)} jobs")
    return jobs_data


# Example usage
if __name__ == "__main__":
    # Example API response structure
    sample_api_response = {
        "candidate_profile": {
            "name": "John Doe",
            "skills": ["Python", "FastAPI"]
        },
        "matched_jobs": [
            {
                "rank": 1,
                "job_url": "https://internshala.com/job/detail/123456",
                "job_title": "Data Science AI & ML Research Associate Fresher Job",
                "company": "Megaminds IT Services",
                "match_score": 0.85,
                "summary": "John Doe is an excellent fit for this position. His experience with Python and machine learning frameworks aligns perfectly with the job requirements.",
                "key_matches": ["Python", "TensorFlow", "Keras", "Machine Learning"],
                "requirements_met": 7,
                "total_requirements": 8,
                "scraped_summary": "We are looking for a Data Science Research Associate with experience in AI/ML..."
            },
            {
                "rank": 2,
                "job_url": "https://linkedin.com/jobs/view/789012",
                "job_title": "Software Engineer - Python",
                "company": "Tech Corp",
                "match_score": 0.75,
                "summary": "Good match for Python development role.",
                "key_matches": ["Python", "FastAPI"],
                "requirements_met": 6,
                "total_requirements": 10,
                "scraped_summary": None
            }
        ],
        "jobs_analyzed": 2,
        "request_id": "abc123"
    }
    
    # Extract jobs
    print("="*70)
    print("EXAMPLE: Extracting jobs from API response")
    print("="*70)
    jobs = extract_jobs_from_response(sample_api_response)
    
    # Display extracted jobs
    print(f"\n[RESULT] Extracted {len(jobs)} jobs:")
    for i, job in enumerate(jobs, 1):
        print(f"\nJob {i}:")
        print(f"  Company: {job['company']}")
        print(f"  Role: {job['role']}")
        print(f"  Portal: {job['portal']}")
        print(f"  Link: {job['link']}")
        print(f"  Status: {job['status']}")
        print(f"  Notes (length): {len(job['notes'])} chars")
        print(f"  JobDescription (length): {len(job['jobDescription'])} chars")
        print(f"  appliedDate: {job['appliedDate']} (type: {type(job['appliedDate'])})")
        print(f"  createdAt: {job['createdAt']} (type: {type(job['createdAt'])})")
    
    # Example: Save to Firebase
    print(f"\n[INFO] To save to Firebase, use:")
    print(f"  from firebase_service import get_firebase_service")
    print(f"  firebase_service = get_firebase_service()")
    print(f"  doc_ids = firebase_service.save_job_applications_batch(user_id, jobs)")
    print("="*70)

