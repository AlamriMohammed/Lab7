#!/bin/bash
curl -X POST "https://lab7-ue77.onrender.com/predict" -H "Content-Type: application/json" \
-d '{"current_value":300000, "goals": 0.5}'
