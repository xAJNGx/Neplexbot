from flask import Flask, request, jsonify, render_template,render_template, request, redirect, url_for, flash
from langchain import PromptTemplate, LLMChain
from flask_caching import Cache
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from collections import deque


app = Flask(__name__,static_url_path='/static')
cache = Cache(app)

# Initialize LLM and other components as in the original code
local_llm = "neural-chat-7b-v3-1.Q4_K_M.gguf"
config = {
    "max_length": 50,
    "temperature": 0.7,
    "top_k": 50,
    # configuration settings as in the original code
}

llm = CTransformers(
    model=local_llm,
    model_type="mistral",
    lib="avx2",
    **config
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question. 
If you don't know the answer, just say Sorry,I cant provide answer.Please try again., don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

load_vector_store = Chroma(persist_directory="stored/data", embedding_function=embeddings)

retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/neplexbot'
app.config['SECRET_KEY'] = 'neplexbot'

# Additional configurations (optional)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable Flask-SQLAlchemy modification tracking
app.config['SQLALCHEMY_ECHO'] = True  # Print SQL queries to the console for debugging (optional)
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # seconds

cache.init_app(app)

db = SQLAlchemy(app)

#data model for user
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    __table_args__ = {'schema': 'neplexbot'}
    uid = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def get_id(self):
        return str(self.uid)


#class for registrationform
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

login_manager = LoginManager(app)
login_manager.login_view = 'home'  # Specify the login route

#userlogin defination
@login_manager.user_loader
def load_user(user_id):
    user_id = int(user_id)

    user = User.query.get(user_id)
    if user:
        # Regular user
        print(f"Loaded user: {user.username}")
        return user

    print(f"User with ID {user_id} not found.")
    return None

#route for userlogin
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()

        if user:
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('chatbot'))
        else:
            flash('Login failed. Please check your username and password.', 'error')

    return render_template('login.html')

#route for registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        # Form is valid, process registration
        username = form.username.data
        email = form.email.data
        password = form.password.data
        new_user = User(username=username,email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    # Form is not valid or not submitted yet, render registration template with form
    return render_template('registration.html', title='Register', form=form)

# render Homepage
@app.route('/')
def home():
    title = 'Neplexbot - Home'
    return render_template('index.html',title=title)

#render about page
@app.route('/about')
def about():
    return render_template('about.html')

#render chatbot
@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot1.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form.get('query')
    cached_response = cache.get(query)
    if cached_response:
        return jsonify(cached_response)

    # Your logic to handle the query
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    response = qa(query)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}

    # Cache the response for future use
    cache.set(query, response_data)

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=False)