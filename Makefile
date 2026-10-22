.PHONY: test demo
test:
	pytest

demo:
	streamlit run app/streamlit_app.py
