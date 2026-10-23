import json
import os

def load_course_catalogue(filepath="data/course_catalogue.json"):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return json.load(f)

def get_recommended_courses(user_skills):
    """
    Transparent recommendation logic: matches user skill goals
    against course metadata tags.
    """
    catalogue = load_course_catalogue()
    recommendations = []
    
    # Normalize user skills to lowercase for matching
    user_skills_lower = [skill.strip().lower() for skill in user_skills]
    
    for course in catalogue:
        course_skills = [s.lower() for s in course.get("skills", [])]
        # Check intersection
        matches = [s for s in course_skills if s in user_skills_lower]
        if matches or not user_skills_lower:
            recommendations.append({
                "course": course,
                "matched_skills": matches,
                "relevance_score": len(matches)
            })
            
    # Sort by highest match count
    recommendations = sorted(recommendations, key=lambda x: x["relevance_score"], reverse=True)
    return recommendations