FROM python:3.12.4-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==2.2.5 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    scipy==1.15.2 \
    shap==0.52.0 \
    matplotlib==3.10.1

COPY data /app/data

RUN mkdir -p \
    /app/data_processed \
    /app/results \
    /app/results/figures

CMD ["bash"]
