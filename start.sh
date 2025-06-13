#!/bin/bash

# This command will start the Gunicorn server.
# It binds to all network interfaces on the port provided by Render's $PORT variable.
gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
