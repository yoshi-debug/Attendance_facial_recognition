from datetime import datetime
from attendance import app, db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return f"User('{self.username}','{self.email}')"

class Class(db.Model):
    __tablename__ = "class"   

    id = db.Column(db.Integer, primary_key=True)
    classname = db.Column(db.String(120), nullable=False)
    coordinator = db.Column(db.String(200))
    co_email = db.Column(db.String(200))

    students = db.relationship("Student", backref="class_obj", lazy=True)

    def __repr__(self):
        return f"Class('{self.classname}', '{self.coordinator}', '{self.co_email}')"


class Student(db.Model):
    __tablename__ = "student"   

    id = db.Column(db.Integer, primary_key=True)
    stuname = db.Column(db.String(200), nullable=False)
    regno = db.Column(db.String(200), nullable=False)
    mobileno = db.Column(db.String(200))

    class_id = db.Column(db.Integer, db.ForeignKey("class.id"), nullable=False)

    def __repr__(self):
        return f"Student('{self.stuname}', '{self.regno}', '{self.mobileno}')"
    
class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    timestamp = db.Column(db.String)   # <-- COINCIDE EXACTAMENTE CON TU BD

    student = db.relationship('Student')




