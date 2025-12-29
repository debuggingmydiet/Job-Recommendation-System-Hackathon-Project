from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import google.generativeai as genai
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///job_recommendation.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure Google GenAI
genai.configure(api_key='YOUR_GOOGLE_API_KEY_HERE')
model = genai.GenerativeModel('gemini-pro')

# Database Models
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    required_skills = db.Column(db.Text, nullable=False)
    experience_required = db.Column(db.Float, default=0)
    min_qualification = db.Column(db.String(100))
    location = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # prevent exact duplicate job posts for same company + title + location
    __table_args__ = (
        db.UniqueConstraint('title', 'company', 'location', name='uq_job_title_company_location'),
    )

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    skills = db.Column(db.Text, nullable=False)
    experience = db.Column(db.Float, default=0)
    education = db.Column(db.String(200))
    resume_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # block duplicate (name, email) combos
    __table_args__ = (
        db.UniqueConstraint('name', 'email', name='uq_candidate_name_email'),
    )

class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    match_score = db.Column(db.Float, default=0)
    ai_analysis = db.Column(db.Text)
    status = db.Column(db.String(50), default='pending')
    applied_at = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database
with app.app_context():
    db.create_all()

# AI-Powered Matching Function
def get_ai_match_analysis(job_desc, candidate_skills, candidate_exp):
    prompt = f"""
    Analyze this job-candidate match:
    
    Job Description: {job_desc}
    Candidate Skills: {candidate_skills}
    Candidate Experience: {candidate_exp} years
    
    Provide:
    1. Match percentage (0-100)
    2. Key strengths (2-3 points)
    3. Gaps or weaknesses (if any)
    
    Format as JSON with keys: match_percentage, strengths, gaps
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return '{"match_percentage": 0, "strengths": [], "gaps": []}'

# ML-based Skill Matching
def calculate_skill_match(job_skills, candidate_skills):
    vectorizer = TfidfVectorizer()
    skills_matrix = vectorizer.fit_transform([job_skills, candidate_skills])
    similarity = cosine_similarity(skills_matrix[0:1], skills_matrix[1:2])[0][0]
    return round(similarity * 100, 2)

# Candidate Shortlisting Algorithm
def shortlist_candidates(job_id, top_n=5):
    job = Job.query.get(job_id)
    if not job:
        return []
    
    applications = Application.query.filter_by(job_id=job_id).all()
    candidates_data = []
    
    for app in applications:
        candidate = Candidate.query.get(app.candidate_id)
        
        # Calculate skill match
        skill_match = calculate_skill_match(job.required_skills, candidate.skills)
        
        # Experience score
        exp_score = min((candidate.experience / max(job.experience_required, 1)) * 100, 100)
        
        # Overall score
        overall_score = (skill_match * 0.6) + (exp_score * 0.4)
        
        # Update application
        app.match_score = overall_score
        app.ai_analysis = get_ai_match_analysis(job.description, candidate.skills, candidate.experience)
        
        candidates_data.append({
            'application_id': app.id,
            'candidate_name': candidate.name,
            'email': candidate.email,
            'skills': candidate.skills,
            'experience': candidate.experience,
            'match_score': overall_score,
            'ai_analysis': app.ai_analysis
        })
    
    db.session.commit()
    
    # Sort by match score
    candidates_data.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Mark top candidates as shortlisted
    for i, candidate in enumerate(candidates_data[:top_n]):
        app = Application.query.get(candidate['application_id'])
        app.status = 'shortlisted'
    
    for candidate in candidates_data[top_n:]:
        app = Application.query.get(candidate['application_id'])
        app.status = 'rejected'
    
    db.session.commit()
    
    return candidates_data[:top_n]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/jobs', methods=['GET', 'POST'])
def manage_jobs():
    if request.method == 'POST':
        data = request.json
        job = Job(
            title=data['title'],
            company=data['company'],
            description=data['description'],
            required_skills=data['required_skills'],
            experience_required=float(data.get('experience_required', 0)),
            min_qualification=data.get('min_qualification', ''),
            location=data.get('location', '')
        )
        db.session.add(job)
        db.session.commit()
        return jsonify({'message': 'Job created successfully', 'job_id': job.id}), 201
    
    jobs = Job.query.all()
    return jsonify([{
        'id': j.id,
        'title': j.title,
        'company': j.company,
        'description': j.description,
        'required_skills': j.required_skills,
        'experience_required': j.experience_required,
        'location': j.location
    } for j in jobs])

@app.route('/api/candidates', methods=['GET', 'POST'])
def manage_candidates():
    if request.method == 'POST':
        data = request.json
        candidate = Candidate(
            name=data['name'],
            email=data['email'],
            skills=data['skills'],
            experience=float(data.get('experience', 0)),
            education=data.get('education', ''),
            resume_text=data.get('resume_text', '')
        )
        db.session.add(candidate)
        db.session.commit()
        return jsonify({'message': 'Candidate registered', 'candidate_id': candidate.id}), 201
    
    candidates = Candidate.query.all()
    return jsonify([{
        'id': c.id,
        'name': c.name,
        'email': c.email,
        'skills': c.skills,
        'experience': c.experience,
        'education': c.education
    } for c in candidates])

@app.route('/api/apply', methods=['POST'])
def apply_job():
    data = request.json
    application = Application(
        job_id=data['job_id'],
        candidate_id=data['candidate_id']
    )
    db.session.add(application)
    db.session.commit()
    return jsonify({'message': 'Application submitted', 'application_id': application.id}), 201

@app.route('/api/shortlist/<int:job_id>', methods=['GET'])
def shortlist(job_id):
    top_n = request.args.get('top_n', 5, type=int)
    shortlisted = shortlist_candidates(job_id, top_n)
    return jsonify(shortlisted)

@app.route('/api/recommendations/<int:candidate_id>', methods=['GET'])
def get_recommendations(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    if not candidate:
        return jsonify({'error': 'Candidate not found'}), 404
    
    jobs = Job.query.all()
    recommendations = []
    
    for job in jobs:
        match_score = calculate_skill_match(job.required_skills, candidate.skills)
        recommendations.append({
            'job_id': job.id,
            'title': job.title,
            'company': job.company,
            'match_score': match_score,
            'location': job.location
        })
    
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    return jsonify(recommendations[:10])

@app.route('/api/export/<int:job_id>', methods=['GET'])
def export_shortlisted(job_id):
    applications = Application.query.filter_by(job_id=job_id, status='shortlisted').all()
    
    data = []
    for app in applications:
        candidate = Candidate.query.get(app.candidate_id)
        data.append({
            'Name': candidate.name,
            'Email': candidate.email,
            'Skills': candidate.skills,
            'Experience': candidate.experience,
            'Match Score': app.match_score,
            'Status': app.status
        })
    
    df = pd.DataFrame(data)
    filename = f'shortlisted_job_{job_id}.csv'
    df.to_csv(filename, index=False)
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


try:
    ml_model = joblib.load('job_recommendation_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("✅ ML Model loaded successfully")
except:
    ml_model = None
    print("⚠️ ML Model not found. Run train_model.py first")

def calculate_features(job, candidate):
    """Calculate ML features for prediction"""
    # Skill match
    job_skills = set(s.strip().lower() for s in job.required_skills.split(','))
    cand_skills = set(s.strip().lower() for s in candidate.skills.split(','))
    skill_overlap = len(job_skills.intersection(cand_skills))
    skill_match = (skill_overlap / len(job_skills)) * 100 if len(job_skills) > 0 else 0
    
    # Experience match
    exp_diff = abs(candidate.experience - job.experience_required)
    exp_match = max(0, 100 - (exp_diff * 20))
    
    # CGPA score (assuming you add cgpa field to Candidate model)
    cgpa_score = 75.0  # Default, or fetch from candidate.cgpa
    
    # Projects score (assuming you add projects_count to Candidate model)
    projects_score = 50.0  # Default, or calculate from candidate.projects_count
    
    return {
        'skill_match_score': skill_match,
        'experience_match_score': exp_match,
        'cgpa_score': cgpa_score,
        'projects_score': projects_score
    }

def shortlist_candidates_ml(job_id, top_n=5):
    """ML-powered candidate shortlisting"""
    job = Job.query.get(job_id)
    if not job or ml_model is None:
        return []
    
    applications = Application.query.filter_by(job_id=job_id).all()
    candidates_data = []
    
    for app in applications:
        candidate = Candidate.query.get(app.candidate_id)
        
        # Calculate features
        features = calculate_features(job, candidate)
        
        # Prepare for ML prediction
        X_pred = pd.DataFrame([features])
        
        # Predict outcome and probability
        prediction = ml_model.predict(X_pred)[0]
        probabilities = ml_model.predict_proba(X_pred)[0]
        predicted_outcome = label_encoder.inverse_transform([prediction])[0]
        
        # Overall score based on probability of being hired/shortlisted
        hire_prob = probabilities[2] if len(probabilities) > 2 else 0  # hired
        shortlist_prob = probabilities[1] if len(probabilities) > 1 else 0  # shortlisted
        overall_score = (hire_prob * 100 * 0.7) + (shortlist_prob * 100 * 0.3)
        
        # Update application
        app.match_score = overall_score
        app.ai_analysis = get_ai_match_analysis(job.description, candidate.skills, candidate.experience)
        
        candidates_data.append({
            'application_id': app.id,
            'candidate_name': candidate.name,
            'email': candidate.email,
            'skills': candidate.skills,
            'experience': candidate.experience,
            'match_score': overall_score,
            'predicted_outcome': predicted_outcome,
            'ai_analysis': app.ai_analysis
        })
    
    db.session.commit()
    
    # Sort by match score
    candidates_data.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Mark top candidates as shortlisted
    for candidate in candidates_data[:top_n]:
        app = Application.query.get(candidate['application_id'])
        app.status = 'shortlisted'
    
    for candidate in candidates_data[top_n:]:
        app = Application.query.get(candidate['application_id'])
        app.status = 'rejected'
    
    db.session.commit()
    
    return candidates_data[:top_n]