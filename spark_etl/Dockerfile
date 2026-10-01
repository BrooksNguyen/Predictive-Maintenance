FROM bitnami/spark:3.5

USER root

WORKDIR /app
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Submit the streaming job to the Spark cluster
CMD ["spark-submit", \
     "--master", "spark://spark-master:7077", \
     "--packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0", \
     "--conf", "spark.cassandra.connection.host=scylladb", \
     "--conf", "spark.cassandra.connection.port=9042", \
     "/app/streaming_job.py"]
