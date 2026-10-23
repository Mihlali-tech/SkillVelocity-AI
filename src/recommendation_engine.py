import json
import os

def load_course_catalogue(filepath="data/course_catalogue.json"):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return json.load(f)

def get_recommended_courses(selected_industries=None, user_skills=None, experience_level=None, learning_goal=None):
    """
    Transparent multi-industry recommendation logic: matches user industry selections
    and skill goals against course metadata tags with weighted relevance scoring.
    """
    catalogue = load_course_catalogue()
    recommendations = []
    
    selected_industries = [ind.strip().lower() for ind in (selected_industries or [])]
    user_skills_lower = [skill.strip().lower() for skill in (user_skills or [])]
    
    for course in catalogue:
        score = 0
        matched_skills = []
        
        course_industry = course.get("industry", "").lower()
        course_skills = [s.lower() for s in course.get("skills", [])]
        course_level = course.get("level", "").lower()
        
        # 1. Skill Matching (Highest weight)
        for s in course_skills:
            if s in user_skills_lower:
                matched_skills.append(s)
                score += 3
                
        # 2. Industry Matching (Medium weight)
        if selected_industries and course_industry in selected_industries:
            score += 2
            
        # 3. Experience Level Matching (Bonus weight)
        if experience_level and course_level == experience_level.strip().lower():
            score += 1
            
        # Include if there is a skill match OR industry match (or if no filters are applied)
        if matched_skills or (selected_industries and course_industry in selected_industries) or not user_skills_lower:
            recommendations.append({
                "course": course,
                "matched_skills": matched_skills,
                "relevance_score": score
            })
            
    # Sort recommendations by highest relevance score descending
    recommendations = sorted(recommendations, key=lambda x: x["relevance_score"], reverse=True)
    return recommendations