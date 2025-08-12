
# Patch for JobMatcher to use custom required skills
def add_required_skills_method(job_matcher_instance, required_skills):
    """Add required skills to the job matcher instance."""
    job_matcher_instance.required_skills = required_skills
    
    # Override the skill calculation method
    def custom_skill_score(resume_data):
        """Calculate score based on custom required skills."""
        if not resume_data or 'skills' not in resume_data:
            print("DEBUG: No resume data or skills")
            return 0.0
        
        if not hasattr(job_matcher_instance, 'required_skills') or not job_matcher_instance.required_skills:
            # Fallback to original method
            print("DEBUG: No custom required skills, using original")
            return job_matcher_instance._original_skill_score(resume_data)
        
        # Use custom required skills
        skill_weights = {}
        for skill in job_matcher_instance.required_skills:
            skill_lower = skill.lower().strip()
            skill_weights[skill_lower] = 1.0
        
        skills = [skill.lower() for skill in resume_data['skills']]
        total_score = 0.0
        max_possible = len(skill_weights)
        
        print(f"DEBUG: Required skills: {list(skill_weights.keys())}")
        print(f"DEBUG: Resume skills: {skills}")
        
        matched_skills = []
        for skill in skills:
            for skill_key, weight in skill_weights.items():
                if skill_key in skill or skill in skill_key:
                    total_score += weight
                    matched_skills.append(skill)
                    break
        
        final_score = min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0
        print(f"DEBUG: Matched skills: {matched_skills}")
        print(f"DEBUG: Skill score: {total_score}/{max_possible} = {final_score}")
        
        return final_score
    
    # Store original method and replace with custom one
    job_matcher_instance._original_skill_score = job_matcher_instance._calculate_skill_score
    job_matcher_instance._calculate_skill_score = custom_skill_score
