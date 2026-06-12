# Animated Confidence Interval Coverage Simulator

This Streamlit app adapts the confidence interval simulation notebook into an interactive web app.

## Files

- `app.py` — the Streamlit app
- `requirements.txt` — Python package requirements for Streamlit Community Cloud

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a public GitHub repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Go to Streamlit Community Cloud and create a new app from that repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Teaching flow

The app shows:

1. A sample drawn from a known population distribution.
2. The sample mean and t-based confidence interval forming from that sample.
3. A running stack of confidence intervals showing whether each interval covers the true mean.
4. A live coverage total that can be compared with the selected confidence level.
