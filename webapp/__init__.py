from flask import Flask

app = Flask(__name__)

from webapp import views
#from app import admin_views
#... and so on ..