#!/usr/bin/env bash

# for debugging
# catch errors early and pick up the last error that occurred before exiting
set -eo pipefail

# gunicorn is the service that deploys the Flask app
# leave $PORT as an environment variable, never hardcode it, and GCP will pick up the port name
# main:app main is the name of the Python file that contains Flask app (main.py)
# app is the Flask app name I specified in main.py
exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

# Exit immediately when one of the background processes terminate.
wait -n