from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from attendance.models import User, Class, Student

class RegistrationForm(FlaskForm):
	username = StringField('Username',validators=[DataRequired(),Length(min=2,max=20)])
	email = StringField('Email', validators=[DataRequired(),Email()])
	password = PasswordField('Password', validators=[DataRequired()])
	confirm_password = PasswordField('Confirm Password', validators=[DataRequired(),EqualTo('password')])
	submit = SubmitField('Register')
	def validate_username(self,username):
		user = User.query.filter_by(username=username.data).first()
		if user:
			raise ValidationError('That username is taken. Please choose a different one')
	
	def	validate_email(self,email):
		user = User.query.filter_by(email=email.data).first()
		if user:
			raise ValidationError('That email is taken. Please choose a different one')		

class LoginForm(FlaskForm):
	email = StringField('Email',validators=[DataRequired(),Email()])
	password = PasswordField('Password',validators=[DataRequired()])
	remember = BooleanField('Remember Me')
	submit = SubmitField('Login')

class AddForm(FlaskForm):
    classname = StringField('Class Name', validators=[DataRequired()])
    coordinator = StringField('Coordinator Name', validators=[DataRequired()])
    co_email = StringField('Coordinator Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Create')

	
class EditForm(FlaskForm):
	classname = StringField('Class Name',validators=[DataRequired()])
	students = IntegerField('No of Students',validators=[DataRequired()])
	coordinator = StringField('Coordinator Name',validators=[DataRequired(),Length(min=2,max=20)])
	co_email = StringField('Coordinator Email',validators=[DataRequired(),Email()])	
	stuname = StringField('Student Name',validators=[DataRequired(),Length(min=2,max=20)])
	regno = IntegerField('Reg No',validators=[DataRequired(),Length(min=12,max=12)])
	mobile_no = IntegerField('Parents Mobile No',validators=[DataRequired(),Length(min=10,max=10)])
	submit = SubmitField('Update')
