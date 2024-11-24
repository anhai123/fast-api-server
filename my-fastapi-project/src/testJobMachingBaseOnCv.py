import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

class CVJobRecommender:
    def __init__(self):
        # Initialize NLTK components
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)

    def extract_cv_features(self, cv_text):
        """Extract key features from CV text"""
        # Define key sections to look for
        sections = {
            'skills': r'skills:(.*?)(?=\n\n|\Z)',
            'experience': r'experience:(.*?)(?=\n\n|\Z)',
            'education': r'education:(.*?)(?=\n\n|\Z)',
            'projects': r'projects:(.*?)(?=\n\n|\Z)'
        }

        features = {}
        for section, pattern in sections.items():
            match = re.search(pattern, cv_text, re.IGNORECASE | re.DOTALL)
            if match:
                features[section] = self.preprocess_text(match.group(1))
            else:
                features[section] = ''

        # Combine all features into one text
        features['combined'] = ' '.join(features.values())
        return features

    def create_job_database(self, job_data):
        """Create and process job database"""
        self.jobs_df = pd.DataFrame(job_data)
        # Combine relevant columns for matching
        self.jobs_df['combined'] = self.jobs_df.apply(
            lambda x: ' '.join([
                str(x['title']),
                str(x['description']),
                str(x['requirements'])
            ]), axis=1
        )
        # Preprocess job descriptions
        self.jobs_df['processed_text'] = self.jobs_df['combined'].apply(self.preprocess_text)
        # Create TF-IDF matrix for jobs
        self.jobs_matrix = self.vectorizer.fit_transform(self.jobs_df['processed_text'])

    def get_recommendations(self, cv_text, num_recommendations=5):
        """Get job recommendations based on CV"""
        # Extract and process CV features
        cv_features = self.extract_cv_features(cv_text)

        # Transform CV text using the same vectorizer
        cv_vector = self.vectorizer.transform([cv_features['combined']])

        # Calculate similarity scores
        similarity_scores = cosine_similarity(cv_vector, self.jobs_matrix)

        # Get top recommendations
        top_indices = similarity_scores[0].argsort()[-num_recommendations:][::-1]

        # Prepare recommendations with scores
        recommendations = []
        for idx in top_indices:
            job = self.jobs_df.iloc[idx]
            recommendations.append({
                'title': job['title'],
                'company': job['company'],
                'score': similarity_scores[0][idx],
                'match_percentage': f"{similarity_scores[0][idx] * 100:.2f}%",
                'description': job['description']
            })

        return recommendations

    def analyze_skill_match(self, cv_text, job_id):
        """Analyze specific skill matches for a job"""
        cv_features = self.extract_cv_features(cv_text)
        job = self.jobs_df.loc[job_id]

        # Extract skills from job requirements
        job_skills = set(self.preprocess_text(job['requirements']).split())
        cv_skills = set(cv_features['skills'].split())

        # Find matching and missing skills
        matching_skills = cv_skills.intersection(job_skills)
        missing_skills = job_skills - cv_skills

        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'match_percentage': len(matching_skills) / len(job_skills) if job_skills else 0
        }

# Example usage
def main():
    # Sample job data
    job_data = [
        {
            'title': 'Senior Software Engineer',
            'company': 'Tech Corp',
            'description': 'Looking for an experienced software engineer...',
            'requirements': 'Python, Java, SQL, AWS'
        },
        # Add more jobs...
    ]

    # Initialize recommender
    recommender = CVJobRecommender()

    # Create job database
    recommender.create_job_database(job_data)

    # Example CV text
    cv_text = """
    skills: Python, JavaScript, SQL, Machine Learning
    experience: 5 years software development
    education: Masters in Computer Science
    projects: Built recommendation systems
    """

    # Get recommendations
    recommendations = recommender.get_recommendations(cv_text)

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"Title: {rec['title']}")
        print(f"Company: {rec['company']}")
        print(f"Match Score: {rec['match_percentage']}")

main()
