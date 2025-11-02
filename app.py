from __future__ import annotations

import asyncio
import json
import os
import time
import re
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from dotenv import load_dotenv
from pathlib import Path

from models import (
    MatchJobsJsonRequest,
    MatchJobsRequest,
    MatchJobsResponse,
    CandidateProfile,
    JobPosting,
    MatchedJob,
    ProgressStatus,
    Settings,
    FirebaseResume,
    FirebaseResumeListResponse,
    FirebaseResumeResponse,
    SavedCVResponse,
    GetUserResumesRequest,
    GetUserResumeRequest,
    GetUserResumePdfRequest,
    GetUserResumeBase64Request,
    GetUserSavedCvsRequest,
    ExtractJobInfoRequest,
    JobInfoExtracted,
)
from utils import decode_base64_pdf, extract_text_from_pdf_bytes, now_iso, make_request_id, redact_long_text, scrape_website_custom
from agents import build_resume_parser, build_scraper, build_scorer, build_summarizer, build_orchestrator
from firecrawl import FirecrawlApp
from pyngrok import ngrok, conf as ngrok_conf

# Optional imports for HTML parsing
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None


# Load environment from root .env and version2/.env if present
load_dotenv()  # project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# CRITICAL: Ensure GOOGLE_APPLICATION_CREDENTIALS is explicitly set from system environment
# This is needed because async context might not have access to system env vars
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    # Already loaded, keep it
    pass
else:
    # Try to get from system environment (Windows environment variables)
    import sys
    import subprocess
    try:
        # On Windows, try to get from system environment
        result = subprocess.run(
            ['powershell', '-Command', '[Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "User")'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = result.stdout.strip()
    except:
        pass  # Non-critical, continue anyway

app = FastAPI(title="Intelligent Job Matching API", version="0.1.0")

# Ngrok startup (optional)
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN") or os.getenv("NGROK_URL")
if NGROK_AUTHTOKEN and not os.getenv("DISABLE_NGROK"):
    try:
        ngrok_conf.get_default().auth_token = NGROK_AUTHTOKEN
        # Ensure no old tunnels keep port busy
        for t in ngrok.get_tunnels():
            try:
                ngrok.disconnect(t.public_url)
            except Exception:
                pass
        if NGROK_DOMAIN:
            ngrok.connect(addr="8000", proto="http", domain=NGROK_DOMAIN)
        else:
            ngrok.connect(addr="8000", proto="http")
    except Exception:
        # Non-fatal if ngrok fails
        pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory stores
REQUEST_PROGRESS: Dict[str, ProgressStatus] = {}
SCRAPE_CACHE: Dict[str, Dict[str, Any]] = {}
LAST_REQUESTS_BY_IP: Dict[str, List[float]] = {}


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120")),
        max_concurrent_scrapes=int(os.getenv("MAX_CONCURRENT_SCRAPES", "8")),
        rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
    )


async def rate_limit(request: Request, settings: Settings = Depends(get_settings)):
    ip = request.client.host if request.client else "unknown"
    window = 60.0
    max_req = settings.rate_limit_requests_per_minute
    now = time.time()
    bucket = LAST_REQUESTS_BY_IP.setdefault(ip, [])
    # prune
    while bucket and now - bucket[0] > window:
        bucket.pop(0)
    if len(bucket) >= max_req:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extract JSON from agent response, handling markdown code blocks and nested content."""
    if not text:
        return {}
    
    original_text = text
    text = text.strip()
    
    # Handle phi agent response objects and other response types
    if hasattr(text, 'content'):
        text = str(text.content)
    elif hasattr(text, 'messages') and text.messages:
        # Get last message content
        last_msg = text.messages[-1]
        text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
    else:
        text = str(text)
    
    text = text.strip()
    
    # Remove markdown code fences - more comprehensive matching
    if '```json' in text:
        # Extract content between ```json and ```
        match = re.search(r'```json\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif '```' in text:
        # Remove any code fence markers
        lines = text.split("\n")
        start_idx = 0
        end_idx = len(lines)
        
        # Find first non-code-fence line
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                start_idx = i + 1
                break
        
        # Find last code-fence line
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```" or lines[i].strip().startswith("```"):
                end_idx = i
                break
        
        text = "\n".join(lines[start_idx:end_idx]).strip()
    
    # Clean up common artifacts
    text = re.sub(r'^[^{]*', '', text)  # Remove leading non-JSON text
    text = re.sub(r'[^}]*$', '', text)  # Remove trailing non-JSON text
    text = text.strip()
    
    # Try direct JSON parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as e:
        pass
    
    # Try to fix common JSON issues and parse again
    fixed_text = text
    
    # Fix trailing commas before closing braces/brackets
    fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
    
    # Try parsing after trailing comma fix
    try:
        parsed = json.loads(fixed_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try to find the largest valid JSON object in the text
    # Find all potential JSON object boundaries
    start_positions = [m.start() for m in re.finditer(r'\{', text)]
    end_positions = [m.start() for m in re.finditer(r'\}', text)]
    
    # Try parsing from each opening brace
    best_match = None
    best_length = 0
    
    for start_pos in start_positions:
        # Find matching closing brace
        brace_count = 0
        for i in range(start_pos, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found matching brace
                    candidate = text[start_pos:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and len(parsed) > best_length:
                            best_match = parsed
                            best_length = len(parsed)
                    except json.JSONDecodeError:
                        pass
                    break
    
    if best_match:
        return best_match
    
    # Last resort: try to extract key-value pairs using regex
    result = {}
    # Extract quoted keys and values
    kv_pattern = r'"([^"]+)":\s*([^,}\]]+)'
    matches = re.finditer(kv_pattern, text)
    for match in matches:
        key = match.group(1)
        value = match.group(2).strip()
        # Try to parse value
        if value.startswith('"') and value.endswith('"'):
            result[key] = value[1:-1]
        elif value.startswith('['):
            # Try to parse array
            try:
                result[key] = json.loads(value)
            except:
                result[key] = value
        elif value.lower() in ('true', 'false'):
            result[key] = value.lower() == 'true'
        elif value.isdigit():
            result[key] = int(value)
        elif re.match(r'^\d+\.\d+$', value):
            result[key] = float(value)
        else:
            result[key] = value
    
    if result:
        print(f"⚠️  Partially parsed JSON using regex fallback. Got {len(result)} fields.")
        return result
    
    # If all else fails, log and return empty dict (workflow should handle this)
    print(f"⚠️  Failed to parse JSON from response")
    print(f"Response length: {len(original_text)} chars")
    print(f"Response preview: {original_text[:500]}...")
    return {}


def parse_experience_years(value: Any) -> Optional[float]:
    """Parse total years of experience from various formats."""
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Extract numbers from strings like "1 year", "2-3 years", "1.5 years"
        numbers = re.findall(r'\d+\.?\d*', value)
        if numbers:
            try:
                return float(numbers[0])
            except:
                pass
    
    return None


def detect_portal(url: str) -> str:
    """Detect the job portal from URL domain."""
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


def extract_json_ld_job_title(soup: BeautifulSoup) -> Optional[str]:
    """Extract job title from JSON-LD structured data."""
    try:
        for script in soup.find_all('script', type=lambda t: t and 'json' in str(t).lower() and 'ld' in str(t).lower()):
            try:
                json_data = json.loads(script.string or '{}')
                
                def extract_from_obj(obj):
                    if isinstance(obj, dict):
                        obj_type = obj.get('@type', '')
                        if 'JobPosting' in str(obj_type):
                            # Try different field names
                            for field in ['title', 'jobTitle', 'name', 'jobTitleText']:
                                if field in obj and obj[field]:
                                    return str(obj[field]).strip()
                        # Recursively search nested objects
                        for value in obj.values():
                            result = extract_from_obj(value)
                            if result:
                                return result
                    elif isinstance(obj, list):
                        for item in obj:
                            result = extract_from_obj(item)
                            if result:
                                return result
                    return None
                
                result = extract_from_obj(json_data)
                if result:
                    return result
            except (json.JSONDecodeError, AttributeError):
                continue
    except Exception:
        pass
    return None


def extract_job_info_from_url(url: str, firecrawl_api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract job title, company name from a job URL.
    Reuses the scraping logic from fetch_job function with enhanced extraction.
    
    Returns:
        Dictionary with 'job_title', 'company_name', 'portal', and 'success' fields
    """
    try:
        # Detect portal first
        portal = detect_portal(url)
        
        # Use Firecrawl SDK directly
        fc = scrape_website_custom(url, firecrawl_api_key)
        content = ''
        title = ''
        company = ''
        html_content = ''
        
        if isinstance(fc, dict) and 'error' not in fc:
            content = str(fc.get('content') or fc.get('markdown') or fc)
            md = fc.get('metadata') or {}
            title = md.get('title') or ''
            html_content = fc.get('html') or ''

        # Always parse HTML for better title/company extraction
        if not requests or not BeautifulSoup:
            return {
                'job_url': url,
                'job_title': None,
                'company_name': None,
                'portal': portal,
                'success': False,
                'error': 'requests and beautifulsoup4 are required for HTML parsing'
            }
        
        if not html_content:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            resp = requests.get(url, headers=headers, timeout=20)
            if resp.ok:
                html_content = resp.text
                soup = BeautifulSoup(html_content, 'lxml')
            else:
                return {
                    'job_url': url,
                    'job_title': None,
                    'company_name': None,
                    'portal': portal,
                    'success': False,
                    'error': f'Failed to fetch URL: {resp.status_code}'
                }
        else:
            soup = BeautifulSoup(html_content, 'lxml')
        
        # Enhanced title extraction - try multiple methods in order of accuracy
        
        # 1. JSON-LD structured data (most reliable)
        if not title:
            title = extract_json_ld_job_title(soup)
        
        # 2. Portal-specific selectors
        if not title:
            portal_lower = portal.lower()
            if portal_lower == 'internshala':
                # Internshala specific selectors
                title_elem = soup.select_one('.profile, .job_title, h1.profile_on_detail_page, .heading_4_5')
                if title_elem:
                    title = title_elem.get_text(strip=True)
            elif portal_lower == 'linkedin':
                # LinkedIn specific selectors
                title_elem = soup.select_one('.jobs-details-top-card__job-title, h1[data-test-id*="job-title"], .topcard__title')
                if title_elem:
                    title = title_elem.get_text(strip=True)
            elif portal_lower == 'indeed':
                # Indeed specific selectors
                title_elem = soup.select_one('.jobsearch-JobInfoHeader-title, h2.jobTitle')
                if title_elem:
                    title = title_elem.get_text(strip=True)
        
        # 3. Common job title selectors (expanded list)
        if not title:
            job_title_selectors = [
                # Class-based selectors
                'h1.job-title', 'h2.job-title', '.job-title', '.jobTitle', '.jobtitle',
                '[class*="job-title"]', '[class*="JobTitle"]', '[class*="jobTitle"]',
                '[data-testid*="job-title"]', '[data-testid*="jobTitle"]',
                '[data-cy*="job-title"]', '[data-job-title]',
                # ID-based selectors
                '#job-title', '#jobTitle', '#job_title',
                # Semantic selectors
                'h1[itemprop="title"]', '[itemprop="jobTitle"]',
                'h1[role="heading"]', '.heading-title',
                # Generic headings (check if they look like job titles)
                'h1', 'h2.title', '.title'
            ]
            for selector in job_title_selectors:
                try:
                    elem = soup.select_one(selector)
                    if elem:
                        title_text = elem.get_text(strip=True)
                        # Validate it looks like a job title
                        if title_text and len(title_text) < 150 and len(title_text) > 3:
                            # Exclude common non-job-title patterns
                            if not any(skip in title_text.lower() for skip in ['home', 'about', 'contact', 'login', 'sign up', 'menu', 'navigation']):
                                title = title_text
                                break
                except Exception:
                    continue
        
        # 4. Meta tags
        if not title or len(title) > 150:
            # Open Graph title
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                og_title_text = og_title.get('content').strip()
                if og_title_text and len(og_title_text) < 150:
                    title = og_title_text
            
            # Twitter card title
            if not title or len(title) > 150:
                twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
                if twitter_title and twitter_title.get('content'):
                    twitter_title_text = twitter_title.get('content').strip()
                    if twitter_title_text and len(twitter_title_text) < 150:
                        title = twitter_title_text
            
            # Schema.org itemprop
            if not title:
                itemprop_title = soup.find(attrs={'itemprop': 'title'})
                if itemprop_title:
                    title = itemprop_title.get_text(strip=True)
        
        # 5. Page title as fallback (with better cleaning)
        if not title:
            if soup.title and soup.title.string:
                page_title = soup.title.string.strip()
                # Clean common prefixes/suffixes
                page_title = re.sub(r'^\s*[-|]\s*', '', page_title)  # Remove leading separators
                page_title = re.sub(r'\s*[-|]\s*$', '', page_title)  # Remove trailing separators
                # Remove common website suffixes
                page_title = re.sub(r'\s*-?\s*(LinkedIn|Indeed|Glassdoor|Monster|Internshala).*$', '', page_title, flags=re.I)
                if page_title and len(page_title) < 150:
                    title = page_title
        
        # 6. Extract from first heading if still not found
        if not title:
            h1 = soup.find('h1')
            if h1:
                h1_text = h1.get_text(strip=True)
                if h1_text and len(h1_text) < 150 and len(h1_text) > 3:
                    title = h1_text
        
        # Extract company name - try multiple sources (same logic as fetch_job)
        def has_company_class(class_attr):
            if not class_attr:
                return False
            if isinstance(class_attr, list):
                return any('company' in str(c).lower() for c in class_attr)
            return 'company' in str(class_attr).lower()
        
        if not company:
            company_selectors = [
                '.company-name', '[class*="Company"]', 
                '[data-testid*="company"]', 'a[href*="/company/"]',
                'strong', '.employer'
            ]
            for selector in company_selectors:
                elem = soup.select_one(selector)
                if elem:
                    company_text = elem.get_text(strip=True)
                    if company_text and 3 <= len(company_text) <= 50:
                        company = company_text
                        break
            
            # Try elements with common class names
            if not company:
                for tag in ['span', 'div', 'a', 'p', 'h3', 'h4']:
                    elements = soup.find_all(tag, class_=has_company_class)
                    for elem in elements[:3]:
                        company_text = elem.get_text(strip=True)
                        if company_text and 3 <= len(company_text) <= 50:
                            company = company_text
                            break
                    if company:
                        break
            
            # Try strong tags with company/employer text
            if not company:
                strong_tags = soup.find_all('strong')
                for strong in strong_tags:
                    strong_text = strong.get_text(strip=True).lower()
                    if 'company' in strong_text or 'employer' in strong_text:
                        parent = strong.find_parent()
                        if parent:
                            parent_text = parent.get_text(strip=True)
                            if len(parent_text) < 100:
                                company = parent_text
                                break
            
            # Try meta tags
            if not company:
                meta_tags = soup.find_all('meta')
                for meta in meta_tags:
                    name_attr = meta.get('name', '').lower()
                    if name_attr and ('company' in name_attr or 'employer' in name_attr):
                        content = meta.get('content', '').strip()
                        if content and 3 <= len(content) <= 50:
                            company = content
                            break
            
            # Look in content text for "at [Company]" pattern
            if not company and content:
                company_match = re.search(r'\bat\s+([A-Z][A-Za-z\s&]{2,40})\b', content[:1000], re.I)
                if company_match:
                    company = company_match.group(1).strip()
        
        # Enhanced title cleaning and validation
        if title:
            # Remove common suffixes/prefixes
            title = re.sub(r'\s*[-–—|]\s*at\s+[^-]+$', '', title, flags=re.I)  # Remove " - at Company Name"
            title = re.sub(r'\s*[-–—|]\s*[^-]+(?:\.com|\.in|\.org).*$', '', title, flags=re.I)  # Remove website suffixes
            title = re.sub(r'\s*[-–—|]\s*.+$', '', title)  # Remove " - Company Name" (generic)
            title = re.sub(r'^[^:]*:\s*', '', title)  # Remove "Job Board: "
            title = re.sub(r'\s*[|]\s*', ' ', title)  # Replace pipe separators with space
            title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
            title = title.strip()
            
            # Validate title quality
            if title:
                # Remove if it's too short or looks like navigation
                if len(title) < 3 or len(title) > 150:
                    title = None
                elif any(bad in title.lower() for bad in ['home', 'menu', 'navigation', 'skip to', 'cookie', 'privacy policy']):
                    title = None
            
            if title:
                title = title[:100]  # Limit length
        
        if company:
            company = company.strip()[:50]  # Limit length
            company = re.sub(r'^at\s+', '', company, flags=re.I)
            company = company.strip()
        
        return {
            'job_url': url,
            'job_title': title or None,
            'company_name': company or None,
            'portal': portal,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'job_url': url,
            'job_title': None,
            'company_name': None,
            'portal': detect_portal(url),
            'success': False,
            'error': str(e)
        }


@app.get("/api/progress/{request_id}", response_model=ProgressStatus)
async def get_progress(request_id: str):
    status = REQUEST_PROGRESS.get(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Unknown request_id")
    return status


@app.post("/api/match-jobs", response_model=MatchJobsResponse, dependencies=[Depends(rate_limit)])
async def match_jobs(
    json_body: Optional[str] = Form(default=None),
    pdf_file: Optional[UploadFile] = File(default=None),
    settings: Settings = Depends(get_settings),
):
    request_id = make_request_id()
    REQUEST_PROGRESS[request_id] = ProgressStatus(
        request_id=request_id, status="queued", jobs_total=0, jobs_scraped=0, 
        jobs_cached=0, started_at=now_iso(), updated_at=now_iso()
    )

    try:
        # Parse input - support both new format and legacy format
        data: Optional[MatchJobsRequest] = None
        legacy_data: Optional[MatchJobsJsonRequest] = None
        
        if json_body:
            try:
                # Handle JSON that might be double-encoded or have extra quotes
                clean_json = json_body.strip()
                if clean_json.startswith('"') and clean_json.endswith('"'):
                    clean_json = clean_json[1:-1].replace('\\"', '"')
                payload = json.loads(clean_json)
                
                # Try new format first
                if "resume" in payload and "jobs" in payload:
                    data = MatchJobsRequest(**payload)
                else:
                    # Legacy format
                    legacy_data = MatchJobsJsonRequest(**payload)
            except Exception as e:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid JSON body: {e}. Received: {json_body[:100]}"
                )
        else:
            raise HTTPException(status_code=400, detail="Missing json_body field")

        REQUEST_PROGRESS[request_id].status = "parsing"
        REQUEST_PROGRESS[request_id].updated_at = now_iso()

        # Get resume text
        resume_bytes: Optional[bytes] = None
        if data and data.resume and data.resume.content:
            resume_bytes = decode_base64_pdf(data.resume.content)
        elif legacy_data and legacy_data.pdf:
            resume_bytes = decode_base64_pdf(legacy_data.pdf)
        elif pdf_file is not None:
            try:
                pdf_file.file.seek(0)
            except Exception:
                pass
            resume_bytes = await pdf_file.read()

        if not resume_bytes:
            raise HTTPException(
                status_code=400, 
                detail="Missing resume PDF (base64 or file upload)"
            )

        resume_text = extract_text_from_pdf_bytes(resume_bytes)

        # Set environment variables for agents
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")
        os.environ.setdefault("FIRECRAWL_API_KEY", settings.firecrawl_api_key or "")

        # STEP 1: Parse Resume
        print("\n" + "="*80)
        print("📄 RESUME PARSER AGENT - Extracting information from OCR text")
        print("="*80)
        
        resume_agent = build_resume_parser(settings.model_name)
        
        resume_prompt = f"""
Extract ALL information from this resume OCR text and return ONLY valid JSON.

Resume text:
{resume_text}

Return this exact structure (no markdown, no explanations):
{{
  "name": "Full name here",
  "email": "email@example.com",
  "phone": "+1234567890",
  "skills": ["Python", "TensorFlow", "Java", etc.],
  "experience_summary": "Brief work history summary",
  "total_years_experience": 1.5,
  "education": [{{"school": "University", "degree": "BS", "dates": "2027"}}],
  "certifications": ["Cert name"],
  "interests": ["Interest 1", "Interest 2"]
}}

Extract every skill, tool, and technology mentioned. Calculate total years from all work experience.
"""
        
        try:
            # Use synchronous run() for phi agents
            resume_response = resume_agent.run(resume_prompt)
            
            # Handle different response types
            if hasattr(resume_response, 'content'):
                response_text = str(resume_response.content)
            elif hasattr(resume_response, 'messages') and resume_response.messages:
                # Get last message content
                last_msg = resume_response.messages[-1]
                response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
            else:
                response_text = str(resume_response)
            
            response_text = response_text.strip()
            
            print("\n[RESUME PARSER RAW OUTPUT]:")
            print(response_text[:800])
            
            # Extract JSON from response
            resume_json = extract_json_from_response(response_text)
            
            # Validate we got something useful
            if not resume_json or not resume_json.get("name") or resume_json.get("name") == "Unknown":
                print("⚠️  Warning: Resume parsing returned incomplete data, attempting fallback extraction")
                
                # Fallback: Extract basic info using regex
                name_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', resume_text, re.MULTILINE)
                email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)
                phone_match = re.search(r'\+?\d[\d\s-]{8,}\d', resume_text)
                
                # Extract skills from common keywords
                skill_keywords = [
                    'Python', 'Java', 'C++', 'JavaScript', 'TypeScript', 'React', 'Node',
                    'TensorFlow', 'PyTorch', 'Keras', 'OpenCV', 'SQL', 'MySQL', 'MongoDB',
                    'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Git', 'Linux',
                    'Machine Learning', 'Deep Learning', 'AI', 'Data Science', 'NLP'
                ]
                found_skills = [skill for skill in skill_keywords if skill.lower() in resume_text.lower()]
                
                resume_json = {
                    "name": name_match.group(1) if name_match else "Unknown Candidate",
                    "email": email_match.group() if email_match else None,
                    "phone": phone_match.group() if phone_match else None,
                    "skills": found_skills or [],
                    "experience_summary": resume_text[:500],
                    "total_years_experience": 1.0,  # Default assumption
                    "education": [],
                    "certifications": [],
                    "interests": []
                }
                print("\n[FALLBACK EXTRACTION]:")
                print(f"Name: {resume_json['name']}")
                print(f"Skills found: {len(resume_json['skills'])}")
                
        except Exception as e:
            print(f"❌ Error parsing resume: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Last resort fallback
            resume_json = {
                "name": "Unknown Candidate",
                "email": None,
                "phone": None,
                "skills": [],
                "experience_summary": resume_text[:500],
                "total_years_experience": None,
                "education": [],
                "certifications": [],
                "interests": []
            }

        # Handle experience_summary - convert to string if needed
        exp_summary = resume_json.get("experience_summary")
        if isinstance(exp_summary, (list, dict)):
            exp_summary = json.dumps(exp_summary, indent=2)
        elif exp_summary is None:
            exp_summary = "Not provided"
        
        # Parse total years of experience
        total_years = parse_experience_years(resume_json.get("total_years_experience"))
        
        candidate_profile = CandidateProfile(
            name=resume_json.get("name") or "Unknown",
            email=resume_json.get("email"),
            phone=resume_json.get("phone"),
            skills=resume_json.get("skills", []) or [],
            experience_summary=exp_summary,
            total_years_experience=total_years,
            interests=resume_json.get("interests", []) or [],
            education=resume_json.get("education", []) or [],
            certifications=resume_json.get("certifications", []) or [],
            raw_text_excerpt=redact_long_text(resume_text, 300),
        )

        # STEP 2: Prepare job URLs
        if data:
            urls = [str(job.url) for job in data.jobs]
            job_titles = {str(job.url): job.title for job in data.jobs}
            job_companies = {str(job.url): job.company for job in data.jobs}
        else:
            urls = [str(u) for u in legacy_data.urls]
            job_titles = {}
            job_companies = {}
            
        REQUEST_PROGRESS[request_id].status = "scraping"
        REQUEST_PROGRESS[request_id].jobs_total = len(urls)
        REQUEST_PROGRESS[request_id].updated_at = now_iso()

        # STEP 3: Scrape Jobs
        print("\n" + "="*80)
        print(f"🔍 JOB SCRAPER - Fetching {len(urls)} job postings")
        print("="*80)

        semaphore = asyncio.Semaphore(settings.max_concurrent_scrapes)

        async def fetch_job(url: str) -> Optional[JobPosting]:
            """Fetch and parse a single job posting."""
            if url in SCRAPE_CACHE:
                REQUEST_PROGRESS[request_id].jobs_cached += 1
                REQUEST_PROGRESS[request_id].updated_at = now_iso()
                cached = SCRAPE_CACHE[url]
                return JobPosting(url=cached.get('url', url), **{k: v for k, v in cached.items() if k != 'url'})
            
            async with semaphore:
                try:
                    # Use Firecrawl SDK directly
                    fc = scrape_website_custom(url, settings.firecrawl_api_key)
                    content = ''
                    title = ''
                    company = ''
                    html_content = ''
                    
                    if isinstance(fc, dict) and 'error' not in fc:
                        content = str(fc.get('content') or fc.get('markdown') or fc)
                        md = fc.get('metadata') or {}
                        title = md.get('title') or ''
                        html_content = fc.get('html') or ''

                    # Always parse HTML for better title/company extraction
                    if not requests or not BeautifulSoup:
                        raise ImportError("requests and beautifulsoup4 are required for HTML parsing")
                    
                    if not html_content:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept-Language': 'en-US,en;q=0.9',
                        }
                        resp = requests.get(url, headers=headers, timeout=20)
                        if resp.ok:
                            html_content = resp.text
                            soup = BeautifulSoup(html_content, 'lxml')
                            
                            # Extract title - try multiple sources
                            if not title:
                                # Try page title first
                                if soup.title and soup.title.string:
                                    title = soup.title.string.strip()
                                
                                # Try h1 tags (common for job titles)
                                if not title or len(title) > 100:
                                    h1 = soup.find('h1')
                                    if h1 and h1.get_text(strip=True):
                                        title = h1.get_text(strip=True)
                                
                                # Try h2 with common job title classes/ids
                                if not title or len(title) > 100:
                                    for h2 in soup.find_all('h2', limit=5):
                                        h2_text = h2.get_text(strip=True)
                                        if h2_text and len(h2_text) < 100:
                                            # Check if it looks like a job title
                                            if any(keyword in h2_text.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'executive', 'director', 'assistant', 'coordinator', 'officer']):
                                                title = h2_text
                                                break
                                
                                # Try meta tags
                                if not title or len(title) > 100:
                                    og_title = soup.find('meta', property='og:title')
                                    if og_title and og_title.get('content'):
                                        title = og_title.get('content').strip()
                            
                            # Extract company name - try multiple sources
                            if not company:
                                # Try elements with common class names that contain 'company' (without regex)
                                def has_company_class(class_attr):
                                    if not class_attr:
                                        return False
                                    if isinstance(class_attr, list):
                                        return any('company' in str(c).lower() for c in class_attr)
                                    return 'company' in str(class_attr).lower()
                                
                                # Search common elements
                                for tag in ['span', 'div', 'a', 'p', 'h3', 'h4']:
                                    elements = soup.find_all(tag, class_=has_company_class)
                                    for elem in elements[:3]:  # Limit per tag type
                                        company_text = elem.get_text(strip=True)
                                        if company_text and 3 <= len(company_text) <= 50:
                                            company = company_text
                                            break
                                    if company:
                                        break
                                
                                # Try strong tags with company/employer text
                                if not company:
                                    strong_tags = soup.find_all('strong')
                                    for strong in strong_tags:
                                        strong_text = strong.get_text(strip=True).lower()
                                        if 'company' in strong_text or 'employer' in strong_text:
                                            # Try to get company name from nearby text
                                            parent = strong.find_parent()
                                            if parent:
                                                parent_text = parent.get_text(strip=True)
                                                if len(parent_text) < 100:
                                                    company = parent_text
                                                    break
                                
                                # Try meta tags
                                if not company:
                                    meta_tags = soup.find_all('meta')
                                    for meta in meta_tags:
                                        name_attr = meta.get('name', '').lower()
                                        if name_attr and ('company' in name_attr or 'employer' in name_attr):
                                            content = meta.get('content', '').strip()
                                            if content and 3 <= len(content) <= 50:
                                                company = content
                                                break
                                
                                # Look in content text for "at [Company]" pattern
                                if not company and content:
                                    company_match = re.search(r'\bat\s+([A-Z][A-Za-z\s&]{2,40})\b', content[:1000], re.I)
                                    if company_match:
                                        company = company_match.group(1).strip()
                            
                            # Get content if not already extracted
                            if not content:
                                desc_tag = soup.find('meta', attrs={'name': 'description'})
                                meta_desc = (desc_tag['content'].strip() if desc_tag and desc_tag.has_attr('content') else '')
                                main = soup.find('main') or soup.find('body')
                                text = (main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True))
                                content = (meta_desc + "\n\n" + text)[:20000]
                    else:
                        # Parse HTML content from Firecrawl
                        soup = BeautifulSoup(html_content, 'lxml')
                        
                        # Extract title - try multiple sources
                        if not title:
                            # Try page title first
                            if soup.title and soup.title.string:
                                title = soup.title.string.strip()
                            
                            # Try h1 tags
                            if not title or len(title) > 100:
                                h1 = soup.find('h1')
                                if h1 and h1.get_text(strip=True):
                                    title = h1.get_text(strip=True)
                            
                            # Try common job title selectors
                            job_title_selectors = [
                                'h1.job-title', 'h2.job-title', '.job-title', 
                                '[data-testid*="job-title"]', '[class*="JobTitle"]',
                                'h1', 'h2'
                            ]
                            for selector in job_title_selectors:
                                elem = soup.select_one(selector)
                                if elem:
                                    title_text = elem.get_text(strip=True)
                                    if title_text and len(title_text) < 100:
                                        title = title_text
                                        break
                        
                        # Extract company name
                        if not company:
                            company_selectors = [
                                '.company-name', '[class*="Company"]', 
                                '[data-testid*="company"]', 'a[href*="/company/"]',
                                'strong', '.employer'
                            ]
                            for selector in company_selectors:
                                elem = soup.select_one(selector)
                                if elem:
                                    company_text = elem.get_text(strip=True)
                                    if company_text and 3 <= len(company_text) <= 50:
                                        company = company_text
                                        break
                        
                        # Ensure content is extracted if not already
                        if not content:
                            desc_tag = soup.find('meta', attrs={'name': 'description'})
                            meta_desc = (desc_tag['content'].strip() if desc_tag and desc_tag.has_attr('content') else '')
                            main = soup.find('main') or soup.find('body')
                            text = (main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True))
                            content = (meta_desc + "\n\n" + text)[:20000]

                    # Clean up extracted values
                    if title:
                        # Remove common suffixes/prefixes
                        title = re.sub(r'\s*[-–—]\s*.+$', '', title)  # Remove " - Company Name"
                        title = re.sub(r'^.+:\s*', '', title)  # Remove "Job Board: "
                        title = title.strip()[:100]  # Limit length
                    
                    if company:
                        company = company.strip()[:50]  # Limit length
                        # Remove common prefixes
                        company = re.sub(r'^at\s+', '', company, flags=re.I)
                        company = company.strip()

                    print(f"\n✓ Scraped {url} ({len(content)} chars)")
                    if title:
                        print(f"  Title extracted: {title}")
                    if company:
                        print(f"  Company extracted: {company}")

                    # Use provided titles/companies first, then fall back to extracted
                    final_title = job_titles.get(url) or title or "Unknown Position"
                    final_company = job_companies.get(url) or company or ''
                    
                    # If still unknown, try to extract from content text using AI or patterns
                    if final_title == "Unknown Position" and content:
                        # Try to find job title pattern in first few lines of content
                        first_lines = content[:500].split('\n')[:5]
                        for line in first_lines:
                            line = line.strip()
                            if line and any(keyword in line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'specialist', 'executive', 'director']):
                                # Extract potential title (first meaningful line with job keywords)
                                if len(line) < 100:
                                    final_title = line
                                    break
                    
                    job = JobPosting(
                        url=url,
                        description=content,
                        job_title=final_title,
                        company=final_company,
                    )
                    
                    # Cache
                    cache_data = job.dict()
                    cache_data['url'] = str(cache_data['url'])
                    cache_data['scraped_summary'] = content[:200] + "..." if len(content) > 200 else content
                    SCRAPE_CACHE[url] = cache_data
                    
                    REQUEST_PROGRESS[request_id].jobs_scraped += 1
                    REQUEST_PROGRESS[request_id].updated_at = now_iso()
                    return job
                    
                except Exception as e:
                    print(f"❌ Error scraping {url}: {e}")
                    return None

        jobs: List[JobPosting] = [
            j for j in await asyncio.gather(*[fetch_job(u) for u in urls]) 
            if j is not None
        ]

        if not jobs:
            raise HTTPException(status_code=500, detail="Failed to scrape any job postings")

        # STEP 4: Score Jobs
        REQUEST_PROGRESS[request_id].status = "matching"
        REQUEST_PROGRESS[request_id].updated_at = now_iso()
        
        print("\n" + "="*80)
        print("🤖 JOB SCORER AGENT - Calculating match scores")
        print("="*80)

        scorer_agent = build_scorer(settings.model_name)

        def score_job_sync(job: JobPosting) -> Optional[Dict[str, Any]]:
            """Score a single job using AI reasoning."""
            try:
                prompt = f"""
Analyze the match between candidate and job. Consider ALL requirements from the job description.

Candidate Profile:
{json.dumps(candidate_profile.dict(), indent=2)}

Job Details:
- Title: {job.job_title}
- Company: {job.company}
- URL: {str(job.url)}
- Description: {job.description[:2000]}

CRITICAL: Read the job description carefully. If this is a:
- Billing/Finance role: Score based on financial/accounting skills
- Tech/Engineering role: Score based on technical skills
- Sales/Marketing role: Score based on communication/business skills

Return ONLY valid JSON (no markdown) with:
{{
  "match_score": 0.75,
  "key_matches": ["skill1", "skill2"],
  "requirements_met": 5,
  "total_requirements": 8,
  "reasoning": "Brief explanation of score"
}}

Be strict with scoring:
- < 0.3: Poor fit (major skill gaps)
- 0.3-0.5: Weak fit (some alignment)
- 0.5-0.7: Good fit (strong alignment)
- > 0.7: Excellent fit (ideal candidate)
"""
                response = scorer_agent.run(prompt)
                
                # Handle different response types
                if hasattr(response, 'content'):
                    response_text = str(response.content)
                elif hasattr(response, 'messages') and response.messages:
                    # Get last message content
                    last_msg = response.messages[-1]
                    response_text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
                else:
                    response_text = str(response)
                
                response_text = response_text.strip()
                
                print(f"\n[SCORER RAW OUTPUT for {job.job_title}]:")
                print(response_text[:500])
                
                # Extract JSON from response
                data = extract_json_from_response(response_text)
                
                # Validate and extract score with defaults
                if not data or data.get("match_score") is None:
                    print(f"⚠️  Warning: Could not extract match_score from response, using default 0.5")
                    data = data or {}
                    data["match_score"] = 0.5
                
                score = float(data.get("match_score", 0.5))
                print(f"✓ Scored {job.job_title}: {score:.1%}")
                
                return {
                    "job": job,
                    "match_score": score,
                    "key_matches": data.get("key_matches", []) or [],
                    "requirements_met": int(data.get("requirements_met", 0)),
                    "total_requirements": int(data.get("total_requirements", 1)),
                    "reasoning": data.get("reasoning", "Score calculated based on candidate-job alignment"),
                }
            except Exception as e:
                print(f"❌ Error scoring {job.job_title}: {e}")
                return None

        # Score sequentially to avoid rate limits
        scored = []
        for job in jobs:
            result = score_job_sync(job)
            if result:
                scored.append(result)
            await asyncio.sleep(0.5)  # Rate limit protection

        # Sort by match score and take top matches
        scored.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Only summarize jobs with decent match scores
        top_matches = [s for s in scored if s["match_score"] >= 0.5][:10]
        
        if not top_matches:
            print("⚠️  No jobs with match score >= 50%, taking top 5")
            top_matches = scored[:5]

        # STEP 5: Generate Summaries
        REQUEST_PROGRESS[request_id].status = "summarizing"
        REQUEST_PROGRESS[request_id].updated_at = now_iso()
        
        print("\n" + "="*80)
        print(f"📝 SUMMARIZER AGENT - Generating summaries for {len(top_matches)} jobs")
        print("="*80)

        summarizer_agent = build_summarizer(settings.model_name)

        def summarize_sync(entry: Dict[str, Any], rank: int) -> MatchedJob:
            """Generate summary for a matched job."""
            job: JobPosting = entry["job"]
            score = entry["match_score"]
            
            prompt = f"""
Write a 150-200 word unique summary for this job-candidate match.

Candidate: {candidate_profile.name}
- Skills: {', '.join(candidate_profile.skills[:10])}
- Experience: {candidate_profile.total_years_experience} years

Job: {job.job_title} at {job.company}
Match Score: {score:.1%}
Description: {job.description[:1500]}

Explain:
- Why this is {'a strong' if score >= 0.7 else 'a good' if score >= 0.5 else 'a weak'} match
- Specific skills/experience that align
- Growth opportunities
- Important considerations

Be honest about the fit level based on the score.
"""
            try:
                response = summarizer_agent.run(prompt)
                
                # Handle different response types
                if hasattr(response, 'content'):
                    text = str(response.content)
                elif hasattr(response, 'messages') and response.messages:
                    last_msg = response.messages[-1]
                    text = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
                else:
                    text = str(response)
                
                text = text.strip()
                
                # Strip markdown code fences if present
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1])
                
                print(f"✓ Summarized rank {rank}: {job.job_title}")
                
            except Exception as e:
                print(f"❌ Error summarizing rank {rank}: {e}")
                text = f"Match score: {score:.1%}. {entry.get('reasoning', '')}"
            
            return MatchedJob(
                rank=rank,
                job_url=str(job.url),
                job_title=job.job_title or "Unknown",
                company=job.company or "Unknown",
                match_score=round(score, 3),
                summary=text[:1200],
                key_matches=entry["key_matches"],
                requirements_met=entry["requirements_met"],
                total_requirements=entry["total_requirements"],
                scraped_summary=SCRAPE_CACHE.get(str(job.url), {}).get('scraped_summary'),
            )

        # Generate summaries sequentially
        matched_jobs = []
        for i, entry in enumerate(top_matches):
            result = summarize_sync(entry, i + 1)
            matched_jobs.append(result)
            await asyncio.sleep(0.5)

        print("\n" + "="*80)
        print("✅ FINAL RESPONSE - All agents completed")
        print("="*80)
        print(f"Found {len(matched_jobs)} matched jobs out of {len(jobs)} analyzed")
        print(f"Top match: {matched_jobs[0].job_title} ({matched_jobs[0].match_score:.1%})")
        print(f"Request ID: {request_id}")
        print("="*80 + "\n")

        REQUEST_PROGRESS[request_id].status = "completed"
        REQUEST_PROGRESS[request_id].updated_at = now_iso()

        # Get user_id from request
        user_id = None
        if data:
            user_id = data.user_id
        elif legacy_data:
            user_id = legacy_data.user_id

        # Save job applications to Firestore if user_id is provided
        if user_id and matched_jobs:
            try:
                # IMPORTANT: Load environment variables again to ensure they're available
                # This is critical because environment might not be loaded in async context
                load_dotenv()
                load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
                
                # Check environment variable is available (try multiple methods)
                creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                print(f"\n[DEBUG] GOOGLE_APPLICATION_CREDENTIALS from os.getenv(): {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
                print(f"[DEBUG] GOOGLE_APPLICATION_CREDENTIALS from os.environ: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
                print(f"[DEBUG] Final creds_path: {creds_path}")
                
                from firebase_service import get_firebase_service
                from job_extractor import extract_jobs_from_response
                
                print(f"\n{'='*80}")
                print(f"[SAVE] SAVING JOB APPLICATIONS TO FIRESTORE")
                print(f"{'='*80}")
                print(f"User ID: {user_id}")
                print(f"Number of matched jobs: {len(matched_jobs)}")
                
                # Try to get Firebase service
                print(f"[DEBUG] Attempting to get Firebase service...")
                try:
                    firebase_service = get_firebase_service()
                    print(f"[DEBUG] Firebase service obtained successfully")
                    print(f"[DEBUG] Firebase DB is initialized: {firebase_service._db is not None}")
                except Exception as init_error:
                    print(f"[ERROR] Failed to get Firebase service: {str(init_error)}")
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise
                
                # Convert MatchedJob Pydantic objects to dictionaries for extraction function
                # This ensures we use the proper formatting with datetime objects
                print(f"[DEBUG] Converting {len(matched_jobs)} matched jobs to API response format...")
                api_response_format = {
                    "matched_jobs": [
                        {
                            "rank": job.rank,
                            "job_url": str(job.job_url),
                            "job_title": job.job_title,
                            "company": job.company,
                            "match_score": job.match_score,
                            "summary": job.summary,
                            "key_matches": job.key_matches,
                            "requirements_met": job.requirements_met,
                            "total_requirements": job.total_requirements,
                            "scraped_summary": job.scraped_summary
                        }
                        for job in matched_jobs
                    ]
                }
                
                print(f"[DEBUG] Extracting and formatting jobs using extract_jobs_from_response...")
                jobs_to_save = extract_jobs_from_response(api_response_format)
                
                # Save all job applications (multiple documents will be created)
                if jobs_to_save:
                    print(f"[INFO] Preparing to save {len(jobs_to_save)} job applications to Firestore...")
                    print(f"[INFO] Each job will be saved as a separate document in users/{user_id}/job_applications/")
                    saved_doc_ids = firebase_service.save_job_applications_batch(user_id, jobs_to_save)
                    print(f"\n{'='*80}")
                    print(f"[SUCCESS] Successfully saved {len(saved_doc_ids)} job applications to Firestore")
                    print(f"[INFO] Document IDs: {saved_doc_ids}")
                    print(f"[INFO] Each document is saved at: users/{user_id}/job_applications/{{doc_id}}")
                    print(f"[PATH] Collection: users/{user_id}/job_applications/")
                    print(f"{'='*80}\n")
                else:
                    print("[WARNING] No job applications to save (extraction returned empty list)")
                    
            except ImportError as e:
                print(f"\n[WARNING] Firebase service not available: {str(e)}")
                print("[INFO] Make sure firebase-admin is installed: pip install firebase-admin")
            except Exception as e:
                print(f"\n[ERROR] Failed to save job applications to Firestore (non-fatal): {str(e)}")
                import traceback
                print(traceback.format_exc())
                print("\n[INFO] Note: The API response will still be returned, but applications were not saved to Firestore.")

        response = MatchJobsResponse(
            candidate_profile=candidate_profile,
            matched_jobs=matched_jobs,
            processing_time="",
            jobs_analyzed=len(urls),
            request_id=request_id,
        )
        return response
        
    except HTTPException:
        REQUEST_PROGRESS[request_id].status = "error"
        REQUEST_PROGRESS[request_id].error = "HTTP error"
        REQUEST_PROGRESS[request_id].updated_at = now_iso()
        raise
    except Exception as e:
        REQUEST_PROGRESS[request_id].status = "error"
        REQUEST_PROGRESS[request_id].error = str(e)
        REQUEST_PROGRESS[request_id].updated_at = now_iso()
        import traceback
        print(f"Full error traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    return {"status": "ok", "version": "0.1.0"}


# Firebase Resume Endpoints
@app.post("/api/firebase/resumes", response_model=FirebaseResumeListResponse)
async def get_user_resumes(request: GetUserResumesRequest):
    """
    Fetch all resumes for a specific user from Firebase Firestore.
    
    Request Body:
        user_id: The user ID to fetch resumes for
        
    Returns:
        List of resumes for the user
    """
    try:
        from firebase_service import get_firebase_service
        
        firebase_service = get_firebase_service()
        resumes_data = firebase_service.get_user_resumes(request.user_id)
        
        # Convert to Pydantic models
        resumes = [
            FirebaseResume(**resume_data)
            for resume_data in resumes_data
        ]
        
        return FirebaseResumeListResponse(
            user_id=request.user_id,
            resumes=resumes,
            count=len(resumes)
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resumes: {str(e)}"
        )


@app.post("/api/firebase/resumes/get", response_model=FirebaseResumeResponse)
async def get_user_resume(request: GetUserResumeRequest):
    """
    Fetch a specific resume by ID for a user from Firebase Firestore.
    
    Request Body:
        user_id: The user ID
        resume_id: The resume document ID
        
    Returns:
        The resume document
    """
    try:
        from firebase_service import get_firebase_service
        
        firebase_service = get_firebase_service()
        resume_data = firebase_service.get_resume_by_id(request.user_id, request.resume_id)
        
        if not resume_data:
            raise HTTPException(
                status_code=404,
                detail=f"Resume {request.resume_id} not found for user {request.user_id}"
            )
        
        return FirebaseResumeResponse(
            user_id=request.user_id,
            resume=FirebaseResume(**resume_data)
        )
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resume: {str(e)}"
        )


@app.post("/api/firebase/resumes/pdf")
async def get_user_resume_pdf(request: GetUserResumePdfRequest):
    """
    Fetch a resume PDF as raw bytes (decoded from base64).
    
    Request Body:
        user_id: The user ID
        resume_id: The resume document ID
        
    Returns:
        PDF file as bytes with appropriate content-type
    """
    try:
        from fastapi.responses import Response
        from firebase_service import get_firebase_service
        
        firebase_service = get_firebase_service()
        pdf_bytes = firebase_service.get_resume_pdf_bytes(request.user_id, request.resume_id)
        
        if not pdf_bytes:
            raise HTTPException(
                status_code=404,
                detail=f"Resume PDF not found for user {request.user_id}, resume {request.resume_id}"
            )
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="resume_{request.resume_id}.pdf"'
            }
        )
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resume PDF: {str(e)}"
        )


@app.post("/api/firebase/resumes/base64")
async def get_user_resume_base64(request: GetUserResumeBase64Request):
    """
    Fetch a resume PDF as base64 string (with PDF_BASE64: prefix removed).
    
    Request Body:
        user_id: The user ID
        resume_id: The resume document ID
        
    Returns:
        JSON with base64 string
    """
    try:
        from firebase_service import get_firebase_service
        
        firebase_service = get_firebase_service()
        resume_data = firebase_service.get_resume_by_id(request.user_id, request.resume_id)
        
        if not resume_data:
            raise HTTPException(
                status_code=404,
                detail=f"Resume {request.resume_id} not found for user {request.user_id}"
            )
        
        base64_content = firebase_service.extract_pdf_base64(resume_data)
        
        if not base64_content:
            raise HTTPException(
                status_code=404,
                detail=f"Resume PDF content not found for user {request.user_id}, resume {request.resume_id}"
            )
        
        return {
            "user_id": request.user_id,
            "resume_id": request.resume_id,
            "base64": base64_content
        }
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch resume base64: {str(e)}"
        )


@app.post("/api/firebase/users/saved-cvs", response_model=SavedCVResponse)
async def get_user_saved_cvs(request: GetUserSavedCvsRequest):
    """
    Fetch the savedCVs array for a user from Firebase Firestore.
    
    This endpoint retrieves the savedCVs array stored at the user document level.
    
    Request Body:
        user_id: The user ID
        
    Returns:
        The savedCVs array for the user
    """
    try:
        from firebase_service import get_firebase_service
        
        firebase_service = get_firebase_service()
        saved_cvs = firebase_service.get_user_saved_cvs(request.user_id)
        
        return SavedCVResponse(
            user_id=request.user_id,
            saved_cvs=saved_cvs,
            count=len(saved_cvs)
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase service not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch savedCVs: {str(e)}"
        )


# Job Information Extraction Endpoint
@app.post("/api/extract-job-info", response_model=JobInfoExtracted)
async def extract_job_info(
    request: ExtractJobInfoRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Extract job title, company name, portal, and description from a job posting URL.
    Uses enhanced HTML parsing with multiple extraction methods including JSON-LD,
    portal-specific selectors, AI fallback, and agent-based description generation.
    
    Request Body:
        job_url: The job posting URL to extract information from
        
    Returns:
        Extracted job information including:
        - job_title: The job title (if found)
        - company_name: The company name (if found)
        - portal: The job portal (e.g., LinkedIn, Internshala, Indeed)
        - description: A concise job description generated by scraper and summarizer agents
        - success: Whether extraction was successful
        - error: Error message if extraction failed
    """
    try:
        # Set environment variables for agents
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")
        os.environ.setdefault("FIRECRAWL_API_KEY", settings.firecrawl_api_key or "")
        
        # Extract job info using existing scraping functions
        job_info = extract_job_info_from_url(str(request.job_url), settings.firecrawl_api_key)
        
        # Generate description using scraper and summarizer agents
        description = None
        if settings.openai_api_key:
            try:
                print(f"[AGENT] Generating job description for: {request.job_url}")
                
                # Step 1: Use scraper agent to get full job details
                from agents import build_scraper, build_summarizer
                scraper_agent = build_scraper()
                
                print(f"[AGENT] [SCRAPER] Scraping job posting...")
                scrape_prompt = f"Extract all job posting details from this URL: {request.job_url}\n\nProvide complete job description, requirements, responsibilities, and any other relevant information."
                scrape_response = scraper_agent.run(scrape_prompt)
                
                # Extract scraped content
                scraped_content = ""
                if hasattr(scrape_response, 'content'):
                    scraped_content = str(scrape_response.content)
                elif hasattr(scrape_response, 'messages') and scrape_response.messages:
                    last_msg = scrape_response.messages[-1]
                    scraped_content = str(last_msg.content if hasattr(last_msg, 'content') else last_msg)
                else:
                    scraped_content = str(scrape_response)
                
                print(f"[AGENT] [SCRAPER] Scraped {len(scraped_content)} characters")
                
                # Step 2: Use summarizer agent to create concise description
                if scraped_content:
                    summarizer_agent = build_summarizer(settings.model_name)
                    
                    print(f"[AGENT] [SUMMARIZER] Creating concise description...")
                    summary_prompt = f"""Create a concise, professional job description summary (150-250 words) from this scraped job posting content.

Job Title: {job_info.get('job_title', 'Not specified')}
Company: {job_info.get('company_name', 'Not specified')}

Scraped Content:
{scraped_content[:4000]}

Generate a clear, well-structured summary that includes:
- Key responsibilities
- Required qualifications and skills
- Preferred experience level
- Any notable benefits or details

Keep it professional and informative, suitable for displaying to job seekers."""
                    
                    summary_response = summarizer_agent.run(summary_prompt)
                    
                    # Extract summary from agent response
                    if hasattr(summary_response, 'content'):
                        description = str(summary_response.content).strip()
                    elif hasattr(summary_response, 'messages') and summary_response.messages:
                        last_msg = summary_response.messages[-1]
                        description = str(last_msg.content if hasattr(last_msg, 'content') else last_msg).strip()
                    else:
                        description = str(summary_response).strip()
                    
                    # Clean up description (remove markdown code blocks if present)
                    description = re.sub(r'^```[\w]*\n', '', description)
                    description = re.sub(r'\n```$', '', description)
                    description = description.strip()
                    
                    print(f"[AGENT] [SUMMARIZER] Generated description ({len(description)} characters)")
                else:
                    print(f"[AGENT] [WARNING] No scraped content received from scraper agent")
                    
            except Exception as agent_error:
                # Non-fatal - continue without description
                print(f"[AGENT] [ERROR] Failed to generate description (non-fatal): {agent_error}")
                import traceback
                print(f"[AGENT] Traceback: {traceback.format_exc()}")
        
        # Add description to job_info
        if description:
            job_info['description'] = description
        
        # If title extraction failed or looks inaccurate, try AI fallback
        if not job_info.get('job_title') or len(job_info.get('job_title', '')) < 3:
            # Try AI extraction as fallback if OpenAI API key is available
            if settings.openai_api_key:
                try:
                    # Get page content for AI analysis
                    if not requests or not BeautifulSoup:
                        return JobInfoExtracted(**job_info)
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept-Language': 'en-US,en;q=0.9',
                    }
                    resp = requests.get(str(request.job_url), headers=headers, timeout=20)
                    if resp.ok:
                        soup = BeautifulSoup(resp.text, 'lxml')
                        # Get main content
                        main_content = ''
                        main_elem = soup.find('main') or soup.find('article') or soup.find('body')
                        if main_elem:
                            main_content = main_elem.get_text(strip=True)[:3000]  # Limit to avoid token limits
                        
                        # Use AI to extract job title from content
                        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key or "")
                        from agents import build_resume_parser
                        
                        extractor_agent = build_resume_parser(settings.model_name)
                        prompt = f"""
Extract ONLY the job title from this job posting page content. Return ONLY the job title text, nothing else.

Page content:
{main_content[:2000]}

Return ONLY the job title (e.g., "Software Engineer", "Data Scientist", "Product Manager"), no explanations, no quotes, no markdown.
"""
                        ai_response = extractor_agent.run(prompt)
                        
                        if hasattr(ai_response, 'content'):
                            ai_title = str(ai_response.content).strip()
                        elif hasattr(ai_response, 'messages') and ai_response.messages:
                            last_msg = ai_response.messages[-1]
                            ai_title = str(last_msg.content if hasattr(last_msg, 'content') else last_msg).strip()
                        else:
                            ai_title = str(ai_response).strip()
                        
                        # Clean AI response
                        ai_title = re.sub(r'^["\']|["\']$', '', ai_title)  # Remove quotes
                        ai_title = re.sub(r'^.*title[:\s]*', '', ai_title, flags=re.I)
                        ai_title = ai_title.strip()
                        
                        # Validate AI extracted title
                        if ai_title and 3 <= len(ai_title) <= 100:
                            # Exclude common AI errors
                            if not any(bad in ai_title.lower() for bad in ['i cannot', 'i don\'t', 'unable to', 'sorry', 'error']):
                                job_info['job_title'] = ai_title
                                job_info['success'] = True
                except Exception as ai_error:
                    # Non-fatal - continue with HTML extraction result
                    print(f"AI fallback failed (non-fatal): {ai_error}")
        
        return JobInfoExtracted(**job_info)
        
    except Exception as e:
        return JobInfoExtracted(
            job_url=str(request.job_url),
            job_title=None,
            company_name=None,
            portal=detect_portal(str(request.job_url)),
            success=False,
            error=str(e)
        )