from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Create base class
class Base(DeclarativeBase):
    pass

# Create db instance
db = SQLAlchemy(model_class=Base)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False)  # 'doctor' or 'patient'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships with explicit foreign keys
    doctor_patients = db.relationship(
        'Patient',
        foreign_keys='Patient.doctor_id',
        backref='assigned_doctor',
        lazy=True
    )
    
    patient_record = db.relationship(
        'Patient',
        foreign_keys='Patient.user_id',
        backref='user_account',
        uselist=False,
        lazy=True
    )
    
    predictions = db.relationship(
        'Prediction',
        backref='predicting_user',
        lazy=True
    )

    def set_password(self, password):
        """Create hashed password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check hashed password"""
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    
    patient_predictions = db.relationship(
        'Prediction',
        backref='patient_record',
        lazy=True
    )

class Prediction(db.Model):  # Fixed: Removed 'userdb.Model' typo
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)  # Changed from 'result' to match your usage
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

    def __repr__(self):
        return f'<Prediction {self.id}: {self.prediction}>'