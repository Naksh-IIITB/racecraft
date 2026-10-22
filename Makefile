.PHONY: test demo demo-ipv6
test:
	pytest

demo:
	streamlit run app/streamlit_app.py

demo-ipv6:
	streamlit run app/streamlit_app.py --server.address ::1 --server.port 8501
