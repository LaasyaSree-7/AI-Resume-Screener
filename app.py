import os
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
import docx2txt
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load a lightweight semantic model for scoring
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        reader = PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        return text
    elif filename.endswith('.docx'):
        return docx2txt.process(file)
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    return ""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No resume uploaded"}), 400
    
    file = request.files['resume']
    job_desc = request.form.get('job_description', 'Professional software engineer with python, teamwork, and project management skills.')
    
    if file.filename == '':
        return jsonify({"error": "Selected file is empty"}), 400

    # Extract text from resume
    resume_text = extract_text_from_file(file)
    if not resume_text.strip():
        return jsonify({"error": "Could not extract text from file."}), 400

    # Compute Semantic Score
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(job_desc, convert_to_tensor=True)
    similarity = util.cos_sim(resume_emb, jd_emb).item()
    
    # Scale score between 45 and 95 for realistic resume scoring
    base_score = int(round((similarity * 50) + 45))
    overall_score = min(max(base_score, 30), 98)

    # Generate breakdown and feedback based on text analysis
    word_count = len(resume_text.split())
    has_email = '@' in resume_text
    has_phone = any(char.isdigit() for char in resume_text[:500]) # simple heuristic
    
    impact_score = min(overall_score + 4, 100)
    keyword_score = overall_score
    formatting_score = 90 if (has_email and has_phone and word_count > 150) else 65
    experience_score = min(overall_score - 5, 100)

    strengths = []
    improvements = []

    if has_email and has_phone:
        strengths.append("Contact details (Email & Phone) are clearly identifiable.")
    else:
        improvements.append("Ensure your email and phone number are clearly visible at the top.")

    if word_count > 200:
        strengths.append(f"Good content volume ({word_count} words detected).")
    else:
        improvements.append("Your resume appears a bit short. Add more details about your responsibilities.")

    if overall_score > 75:
        strengths.append("Strong semantic alignment with the target description.")
    else:
        improvements.append("Incorporate more industry-specific keywords from the job description.")

    response_data = {
        "score": overall_score,
        "breakdown": {
            "impact": impact_score,
            "keywords": keyword_score,
            "formatting": formatting_score,
            "experience": experience_score
        },
        "strengths": strengths if strengths else ["Clean structure and readable layout."],
        "improvements": improvements if improvements else ["Add more metric-driven achievements (e.g., 'increased sales by 20%')."]
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)