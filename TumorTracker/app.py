import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
from models import db, Prediction,User, Patient
from utils import preprocess_image, load_model, make_prediction
import google.generativeai as genai
from flask import make_response
import requests
from flask import request, jsonify
import logging
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///tumor_classification.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure Gemini
# Configuration Gemini (en haut du fichier)
genai.configure(
    api_key="AIzaSyCWrMmt_7k_3_BRCLIr8OiH08sXwoOW27w",
    transport='rest',
    client_options={'api_endpoint': 'generativelanguage.googleapis.com'}
)
# Paramètres de sécurité corrigés
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]


gemini_model = genai.GenerativeModel('gemini')
print(genai.list_models())
# Initialize the database
db.init_app(app)

# Load model
model = None
Talisman(app, content_security_policy={
    'default-src': "'self'",
    'style-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
    'script-src': ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"]
})




def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize function will be called within the application context
def initialize():
    global model
    db.create_all()
    model = load_model()
    logging.debug("Model loaded successfully")


# Routes
@app.route('/')
@login_required
def index():
    return render_template('login.html')
@app.route('/Analyze')
def analyze():
    return render_template('index.html')

@app.route('/history')
def history():
    predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(10).all()
    return render_template('history.html', predictions=predictions)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        try:
            # Save the file temporarily
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(filepath)

            # Determine patient ID based on user role
            if current_user.role == 'doctor':
                patient_id = request.form.get('patient_id')
                if not patient_id:
                    flash('Please select a patient', 'danger')
                    return redirect(url_for('index'))
            else:  # patient
                if not current_user.patient_record:
                    flash('Patient record not found', 'danger')
                    return redirect(url_for('index'))
                patient_id = current_user.patient_record.id

            # Send to Node.js API
            with open(filepath, 'rb') as f:
                files = {'image': (new_filename, f, 'image/jpeg')}
                response = requests.post('http://localhost:3000/predict', files=files)

            if response.status_code != 200:
                flash('Error from prediction API', 'danger')
                return redirect(url_for('index'))

            result = response.json()
            
            # Save to database
            relative_path = os.path.join('uploads', new_filename)
            prediction = Prediction(
                user_id=current_user.id,
                prediction=result['prediction'],
                confidence=result['confidence'],
                image_path=relative_path,
                patient_id=patient_id
            )
            db.session.add(prediction)
            db.session.commit()

            return render_template('index.html',
                               prediction=result['prediction'],
                               confidence=result['confidence'],
                               image_path=relative_path)

        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('Allowed file types are png, jpg, jpeg', 'warning')
        return redirect(url_for('index'))

@app.route('/view-chat',methods=['GET'])
def view_chat():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():

    try:
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message vide"}), 400

        model = genai.GenerativeModel(
            'gemini-1.5-pro-latest',
            safety_settings=SAFETY_SETTINGS,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 2000
            }
        )

        # Prompt amélioré pour le contexte médical
        prompt = f"""Je suis un docteur spécialisé en oncologie. 
        Répondez en français de manière claire, précise et professionnelle scientifique 
        dans 3 lignes pour m'aider à diagnostiquer.
        Question: {user_message}
        Réponse:"""

        response = model.generate_content(prompt)
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({
            "error": "Désolé, je ne peux pas répondre pour le moment",
            "details": str(e) if app.debug else None
        }), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        Prediction.query.delete()
        db.session.commit()
        flash('History cleared successfully', 'success')
    except Exception as e:
        logging.error(f"Error clearing history: {str(e)}")
        flash(f'Error clearing history: {str(e)}', 'danger')

    return redirect(url_for('history'))
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        
        print(f"Attempting login for: {request.form['email']}")  # Debug
        
        if user and user.check_password(request.form['password']):
            print(f"Successful login: {user.username} (Role: '{user.role}')")  # Debug
            login_user(user)
            next_page = request.args.get('next')
            
            if user.role.strip().lower() == 'patient':  # More robust check
                print("Redirecting to patient portal")  # Debug
                return redirect(next_page or url_for('patient_portal'))
            elif user.role.strip().lower() == 'doctor':
                print("Redirecting to doctor dashboard")  # Debug
                return redirect(next_page or url_for('analyze'))  # Changed endpoint
            
            print(f"Unknown role: {user.role}")  # Debug
        
        flash('Invalid credentials', 'danger')
        return render_template('login.html')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        # Get all doctors to populate the dropdown
        doctors = User.query.filter_by(role='doctor').all()
        return render_template('register.html', doctors=doctors)
    
    if request.method == 'POST':
        # Existing validation checks...
        
        # Create user with role
        user = User(
            username=request.form['username'],
            email=request.form['email'],
            role=request.form['role']
        )
        user.set_password(request.form['password'])
        db.session.add(user)
        db.session.commit()
        
        # If registering as patient, create patient record
        if user.role == 'patient':
            doctor_id = request.form.get('doctor_id')
            if not doctor_id:
                flash('Please select a doctor', 'danger')
                return redirect(url_for('register'))
                
            patient = Patient(
                name=request.form['username'],  # Or add separate name field
                doctor_id=doctor_id,
                user_id=user.id
            )
            db.session.add(patient)
            db.session.commit()
        
        login_user(user)
        flash(user.role)
        if user.role=='patient':
            return redirect(url_for('patient_portal'))
        if user.role=='doctor':
            return redirect(url_for('analyze'))

@app.route('/patient-portal')
@login_required
def patient_portal():

    patient = Patient.query.filter_by(user_id=current_user.id).first()
    
    if patient:
        predictions = Prediction.query.filter_by(patient_id=patient.id)\
                        .order_by(Prediction.timestamp.desc())\
                        .all()
    else:
        predictions = []
    
    return render_template('patient_portal.html', predictions=predictions)

@app.route('/doctor-portal')
@login_required
def doctor_portal():
    if current_user.role != 'doctor':
        abort(403)
    patients = Patient.query.filter_by(doctor_id=current_user.id).all()
    return render_template('doctor_portal.html', patients=patients)

@app.route('/patient-analysis/<int:patient_id>')
@login_required
def patient_analysis(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    if patient.doctor_id != current_user.id:
        abort(403)
    predictions = Prediction.query.filter_by(patient_id=patient_id).all()
    return render_template('patient_analysis.html', patient=patient, predictions=predictions)



# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


# Initialize database and model within the application context
with app.app_context():
    db.create_all()
    # Initialize the model
    model = load_model()