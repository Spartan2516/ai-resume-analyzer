from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume = """
Python developer with experience in HTML, CSS, and data structures.
"""

job_description = """
Looking for software developer with Python, Git, SQL, and algorithms knowledge.
"""

text = [resume, job_description]

vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(text)

similarity = cosine_similarity(matrix)[0][1]

print("Resume Match Score:", round(similarity * 100, 2), "%")