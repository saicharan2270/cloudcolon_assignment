"""
Script to generate sample resumes for testing the resume screening system.
"""

import os
import random
from typing import List, Dict
from datetime import datetime, timedelta

def generate_sample_resumes(num_resumes: int = 50) -> List[Dict[str, str]]:
    """
    Generate sample resumes with varying levels of relevance to Salesforce Engineer position.
    
    Args:
        num_resumes (int): Number of sample resumes to generate
        
    Returns:
        List[Dict[str, str]]: List of sample resume data
    """
    
    # Sample names
    first_names = [
        "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
        "James", "Jennifer", "William", "Jessica", "Richard", "Amanda", "Thomas",
        "Nicole", "Christopher", "Stephanie", "Daniel", "Melissa", "Matthew",
        "Rachel", "Anthony", "Laura", "Mark", "Michelle", "Donald", "Kimberly",
        "Steven", "Deborah", "Paul", "Dorothy", "Andrew", "Helen", "Joshua",
        "Sharon", "Kenneth", "Carol", "Kevin", "Ruth", "Brian", "Julie",
        "George", "Joyce", "Timothy", "Virginia", "Ronald", "Victoria",
        "Jason", "Kelly", "Edward", "Lauren", "Jeffrey", "Christine"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
        "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
        "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz"
    ]
    
    # Skills with relevance weights
    skills_data = {
        # High relevance skills
        'apex': 0.9, 'visualforce': 0.8, 'lightning': 0.8, 'soql': 0.7,
        'sosl': 0.7, 'salesforce': 0.6, 'javascript': 0.6, 'html': 0.5,
        'css': 0.5, 'rest': 0.6, 'soap': 0.6, 'api': 0.6, 'git': 0.5,
        'agile': 0.4, 'scrum': 0.4, 'testing': 0.5, 'integration': 0.6,
        'cpq': 0.7, 'service cloud': 0.7, 'marketing cloud': 0.7,
        'heroku': 0.6, 'dx': 0.6,
        
        # Medium relevance skills
        'python': 0.4, 'java': 0.4, 'sql': 0.5, 'c++': 0.3, 'c#': 0.3,
        'php': 0.3, 'ruby': 0.3, 'go': 0.3, 'rust': 0.3, 'swift': 0.2,
        'kotlin': 0.2, 'react': 0.4, 'angular': 0.4, 'vue': 0.4,
        'node.js': 0.4, 'express': 0.4, 'django': 0.3, 'spring': 0.3,
        'docker': 0.4, 'kubernetes': 0.4, 'aws': 0.4, 'azure': 0.4,
        
        # Low relevance skills
        'photoshop': 0.1, 'illustrator': 0.1, 'wordpress': 0.2, 'excel': 0.2,
        'powerpoint': 0.1, 'word': 0.1, 'outlook': 0.1, 'teams': 0.1,
        'slack': 0.1, 'trello': 0.1, 'asana': 0.1, 'jira': 0.3
    }
    
    # Certifications
    certifications = [
        "Platform Developer I", "Platform Developer II", "Administrator",
        "Sales Cloud Consultant", "Service Cloud Consultant", "CPQ Specialist",
        "Marketing Cloud Consultant", "Integration Architecture Designer"
    ]
    
    # Education degrees
    degrees = [
        "Bachelor of Science in Computer Science",
        "Master of Science in Software Engineering",
        "Bachelor of Engineering in Information Technology",
        "Master of Business Administration",
        "Bachelor of Arts in Computer Science",
        "Associate Degree in Programming"
    ]
    
    # Job titles
    job_titles = [
        "Salesforce Developer", "Senior Salesforce Developer", "Salesforce Engineer",
        "Salesforce Consultant", "Apex Developer", "Lightning Developer",
        "Software Engineer", "Full Stack Developer", "Backend Developer",
        "Frontend Developer", "DevOps Engineer", "System Administrator"
    ]
    
    # Companies
    companies = [
        "Salesforce", "Accenture", "Deloitte", "PwC", "KPMG", "EY",
        "Cognizant", "Infosys", "TCS", "Wipro", "Tech Mahindra",
        "Microsoft", "Google", "Amazon", "IBM", "Oracle", "SAP",
        "Adobe", "Workday", "ServiceNow", "HubSpot", "Zendesk"
    ]
    
    resumes = []
    
    for i in range(num_resumes):
        # Determine relevance level for this resume
        if i < num_resumes * 0.2:  # 20% highly relevant
            relevance_level = "high"
        elif i < num_resumes * 0.5:  # 30% moderately relevant
            relevance_level = "medium"
        else:  # 50% low relevance
            relevance_level = "low"
        
        # Generate resume data
        resume = generate_single_resume(
            first_names, last_names, skills_data, certifications,
            degrees, job_titles, companies, relevance_level, i
        )
        resumes.append(resume)
    
    return resumes

def generate_single_resume(first_names, last_names, skills_data, certifications,
                          degrees, job_titles, companies, relevance_level, index):
    """Generate a single sample resume."""
    
    # Generate name
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    name = f"{first_name} {last_name}"
    
    # Generate email
    email = f"{first_name.lower()}.{last_name.lower()}@email.com"
    
    # Generate phone
    phone = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    # Generate location
    cities = ["San Francisco", "New York", "Austin", "Seattle", "Boston", "Chicago"]
    states = ["CA", "NY", "TX", "WA", "MA", "IL"]
    city_idx = random.randint(0, len(cities) - 1)
    location = f"{cities[city_idx]}, {states[city_idx]}"
    
    # Generate years of experience based on relevance
    if relevance_level == "high":
        years_exp = random.randint(3, 8)
    elif relevance_level == "medium":
        years_exp = random.randint(1, 5)
    else:
        years_exp = random.randint(0, 3)
    
    # Generate skills based on relevance
    skills = generate_skills(skills_data, relevance_level)
    
    # Generate certifications based on relevance
    certs = generate_certifications(certifications, relevance_level)
    
    # Generate education
    education = random.choice(degrees)
    
    # Generate experience
    experience = generate_experience(job_titles, companies, years_exp, relevance_level)
    
    # Generate raw text
    raw_text = generate_raw_text(name, email, phone, location, years_exp, skills, certs, education, experience)
    
    return {
        'name': name,
        'email': [email],
        'phone': [phone],
        'location': location,
        'years_experience': years_exp,
        'skills': skills,
        'certifications': certs,
        'education': [{'degree': education, 'institution': '', 'year': ''}],
        'experience': experience,
        'languages': [skill for skill in skills if skill.lower() in ['python', 'java', 'javascript', 'html', 'css', 'sql', 'apex']],
        'raw_text': raw_text,
        'file_path': f"sample_resume_{index + 1}.txt"
    }

def generate_skills(skills_data, relevance_level):
    """Generate skills based on relevance level."""
    skills = []
    
    if relevance_level == "high":
        # High relevance skills with high probability
        for skill, weight in skills_data.items():
            if weight >= 0.6 and random.random() < 0.8:
                skills.append(skill.title())
            elif weight >= 0.4 and random.random() < 0.5:
                skills.append(skill.title())
            elif weight < 0.4 and random.random() < 0.2:
                skills.append(skill.title())
    
    elif relevance_level == "medium":
        # Mix of skills
        for skill, weight in skills_data.items():
            if weight >= 0.6 and random.random() < 0.5:
                skills.append(skill.title())
            elif weight >= 0.4 and random.random() < 0.6:
                skills.append(skill.title())
            elif weight < 0.4 and random.random() < 0.4:
                skills.append(skill.title())
    
    else:  # low relevance
        # Mostly low relevance skills
        for skill, weight in skills_data.items():
            if weight < 0.4 and random.random() < 0.7:
                skills.append(skill.title())
            elif weight >= 0.4 and random.random() < 0.3:
                skills.append(skill.title())
    
    return list(set(skills))  # Remove duplicates

def generate_certifications(certifications, relevance_level):
    """Generate certifications based on relevance level."""
    certs = []
    
    if relevance_level == "high":
        # High probability of Salesforce certifications
        for cert in certifications:
            if "Salesforce" in cert or "Platform" in cert or "Cloud" in cert:
                if random.random() < 0.7:
                    certs.append(cert)
            else:
                if random.random() < 0.3:
                    certs.append(cert)
    
    elif relevance_level == "medium":
        # Medium probability
        for cert in certifications:
            if random.random() < 0.4:
                certs.append(cert)
    
    else:  # low relevance
        # Low probability
        for cert in certifications:
            if random.random() < 0.2:
                certs.append(cert)
    
    return certs

def generate_experience(job_titles, companies, years_exp, relevance_level):
    """Generate work experience."""
    experience = []
    
    # Generate 1-3 job experiences
    num_jobs = random.randint(1, 3)
    
    for i in range(num_jobs):
        job_title = random.choice(job_titles)
        company = random.choice(companies)
        
        # Adjust job titles based on relevance
        if relevance_level == "low":
            # More likely to have non-Salesforce titles
            if "Salesforce" not in job_title and random.random() < 0.7:
                job_title = random.choice(["Software Engineer", "Developer", "Programmer"])
        
        experience.append({
            'title': job_title,
            'company': company,
            'duration': f"{random.randint(1, 3)} years",
            'description': f"Worked on {job_title.lower()} projects"
        })
    
    return experience

def generate_raw_text(name, email, phone, location, years_exp, skills, certs, education, experience):
    """Generate raw text for the resume."""
    
    text = f"""
{name}
{email[0]} | {phone[0]} | {location}

SUMMARY
Experienced professional with {years_exp} years of experience in software development and technology solutions.

SKILLS
{', '.join(skills)}

EXPERIENCE
"""
    
    for exp in experience:
        text += f"{exp['title']} at {exp['company']} - {exp['duration']}\n"
        text += f"{exp['description']}\n\n"
    
    text += f"""
EDUCATION
{education}

CERTIFICATIONS
{', '.join(certs) if certs else 'None'}
"""
    
    return text.strip()

def save_sample_resumes(resumes: List[Dict[str, str]], output_dir: str = "data/sample_resumes"):
    """Save sample resumes to text files."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for i, resume in enumerate(resumes):
        filename = f"sample_resume_{i + 1}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(resume['raw_text'])
        
        print(f"Generated: {filename}")
    
    print(f"\nGenerated {len(resumes)} sample resumes in {output_dir}/")

def main():
    """Main function to generate sample resumes."""
    print("Generating sample resumes for testing...")
    
    # Generate 50 sample resumes
    resumes = generate_sample_resumes(50)
    
    # Save to files
    save_sample_resumes(resumes)
    
    # Print statistics
    high_relevant = sum(1 for r in resumes if r['years_experience'] >= 3 and len([s for s in r['skills'] if s.lower() in ['apex', 'visualforce', 'lightning', 'salesforce']]) >= 2)
    medium_relevant = sum(1 for r in resumes if 1 <= r['years_experience'] < 3 or len([s for s in r['skills'] if s.lower() in ['apex', 'visualforce', 'lightning', 'salesforce']]) == 1)
    low_relevant = len(resumes) - high_relevant - medium_relevant
    
    print(f"\nResume Distribution:")
    print(f"Highly Relevant: {high_relevant}")
    print(f"Moderately Relevant: {medium_relevant}")
    print(f"Low Relevance: {low_relevant}")

if __name__ == "__main__":
    main() 