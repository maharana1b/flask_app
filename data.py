from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234@localhost/dvd'
db = SQLAlchemy(app)

@app.route('/')
def index():
    # Explicitly declare the SQL query using text()
    query = text('SELECT film_id, title, release_year FROM public.film limit 10;')
    result = db.session.execute(query)
    
    # Process the result as needed
    # for row in result:
    #     print(row)
    
    return render_template('table.html',result = result)

if __name__ == '__main__':
    app.run(debug=True)
