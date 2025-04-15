#!/usr/bin/env python3

import re
from typing import List, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class QuestionType(Enum):
    MISSING_SKILL = "missing_skill"
    EXPERIENCE_DETAIL = "experience_detail"
    GENERAL = "general"

@dataclass
class Question:
    text: str
    type: QuestionType
    related_skill: str

class CVAnalyzer:
    def __init__(self, job_requirements: List[str]):
        self.job_requirements = [req.lower() for req in job_requirements]
        self.skill_weights = {skill: 1.0 for skill in job_requirements}  # Default equal weights

    def extract_cv_info(self, cv_text: str) -> Set[str]:
        """
        Extract key information from CV text using simple keyword matching.
        Returns a set of found keywords.
        """
        # Convert to lowercase and find all words
        words = re.findall(r"\w+", cv_text.lower())
        return set(words)

    def calculate_match_score(self, cv_info: Set[str]) -> Tuple[float, List[str], List[str]]:
        """
        Calculate matching score between CV and job requirements.
        Returns:
        - Match score (percentage)
        - List of present requirements
        - List of missing requirements
        """
        present_requirements = [req for req in self.job_requirements if req in cv_info]
        missing_requirements = [req for req in self.job_requirements if req not in cv_info]
        
        # Calculate weighted score
        total_weight = sum(self.skill_weights.values())
        present_weight = sum(self.skill_weights[req] for req in present_requirements)
        score = (present_weight / total_weight * 100) if total_weight > 0 else 0
        
        return score, present_requirements, missing_requirements

    def generate_questions(self, 
                         present_requirements: List[str], 
                         missing_requirements: List[str]) -> List[Question]:
        """
        Generate interview questions based on CV analysis.
        """
        questions = []
        
        # Questions about missing skills
        for skill in missing_requirements:
            questions.append(Question(
                text=f"Can you tell me about your experience with {skill}?",
                type=QuestionType.MISSING_SKILL,
                related_skill=skill
            ))
        
        # Questions to elaborate on present skills
        for skill in present_requirements:
            questions.append(Question(
                text=f"Could you provide specific examples of how you've used {skill} in your previous roles?",
                type=QuestionType.EXPERIENCE_DETAIL,
                related_skill=skill
            ))
        
        # Add some general questions
        questions.append(Question(
            text="What are your career goals and how do they align with this position?",
            type=QuestionType.GENERAL,
            related_skill=""
        ))
        
        return questions

    def analyze_cv(self, cv_text: str) -> dict:
        """
        Main analysis function that combines all steps.
        """
        cv_info = self.extract_cv_info(cv_text)
        score, present_reqs, missing_reqs = self.calculate_match_score(cv_info)
        questions = self.generate_questions(present_reqs, missing_reqs)
        
        return {
            "match_score": score,
            "present_requirements": present_reqs,
            "missing_requirements": missing_reqs,
            "questions": questions
        }

def main():
    # Example job requirements
    job_requirements = [
        "python",
        "project management",
        "english",
        "sql",
        "team leadership",
        "agile"
    ]
    
    analyzer = CVAnalyzer(job_requirements)
    
    print("Enter CV text (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    
    cv_text = "\n".join(lines)
    
    # Analyze CV
    results = analyzer.analyze_cv(cv_text)
    
    # Display results
    print("\nAnalysis Results:")
    print(f"Match Score: {results['match_score']:.2f}%")
    
    print("\nPresent Requirements:")
    for req in results['present_requirements']:
        print(f"- {req}")
    
    print("\nMissing Requirements:")
    for req in results['missing_requirements']:
        print(f"- {req}")
    
    print("\nSuggested Interview Questions:")
    for i, question in enumerate(results['questions'], 1):
        print(f"{i}. {question.text}")

if __name__ == "__main__":
    main() 