# forms.py


from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, MultipleFileField, TextAreaField, FloatField, DateTimeField
from wtforms.validators import DataRequired, Email
from flask_wtf import FlaskForm
from wtforms import FileField
from wtforms.validators import DataRequired


from flask_wtf import FlaskForm
from wtforms import FileField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed

from flask_wtf import FlaskForm

class EmailForm(FlaskForm):
    email = StringField(validators=[DataRequired(), Email()])
    submit = SubmitField('Send Magic Link')

class UploadForm(FlaskForm):
    files = MultipleFileField(
        'Select photos/videos to upload', 
        validators=[
            DataRequired(), 
            FileAllowed(['jpg', 'jpeg', 'png', 'gif', 'dng', 'heic', 'mp4', 'mov', 'avi', 'wmv', 'flv', 'mkv', 'hevc', 'gpx'], 'Only images, videos, and satellite data are allowed!')
        ]
    )
    submit = SubmitField('Upload')

class CommentForm(FlaskForm):
    content = TextAreaField('Add a memory', validators=[DataRequired()])
    submit = SubmitField('Post Comment')

class EditMetadataForm(FlaskForm):
    latitude = FloatField('Latitude')
    longitude = FloatField('Longitude')
    timestamp = DateTimeField('Timestamp (YYYY-MM-DD HH:MM:SS)', format='%Y-%m-%d %H:%M:%S')
    submit = SubmitField('Save Changes')


from flask_wtf import FlaskForm
from wtforms.fields import MultipleFileField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed

class GPXUploadForm(FlaskForm):
    gpx_files = MultipleFileField('Upload GPX Files', validators=[DataRequired(), FileAllowed(['gpx'])])
    submit = SubmitField('Upload')  # Add this line to create a submit button
