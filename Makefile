run:
	poetry run streamlit run app.py

export:
	poetry export -f requirements.txt --output requirements.txt
