#!/bin/bash
source activate voice-clone-venv
export FLASK_APP=voice_cloning_server.py
flask run --host 0.0.0.0 